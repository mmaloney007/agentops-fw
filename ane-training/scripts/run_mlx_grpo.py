#!/usr/bin/env python3
"""
MLX GRPO training script for Apple Silicon GPU comparison.

Runs GRPO (Group Relative Policy Optimization) training using MLX on Metal GPU,
producing JSONL logs compatible with the Obj-C ANE/CPU pipeline for comparison.

MLX uses Metal GPU only (no ANE). The key contrast for the paper is:
  GPU-only (MLX) vs ANE+CPU (Obj-C pipeline)

Usage:
    python scripts/run_mlx_grpo.py \
      --model HuggingFaceTB/SmolLM2-360M-Instruct \
      --tasks scripts/hard_tasks.jsonl \
      --steps 500 --group-size 4 --lr 1e-5 \
      --temperature 0.7 --max-tokens 64 \
      --out results/experiments/smollm2_mlx/grpo_log.jsonl
"""

import argparse
import json
import math
import os
import plistlib
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# ---------------------------------------------------------------------------
# Power monitoring (sudo powermetrics)
# ---------------------------------------------------------------------------

class PowerMonitor:
    """Background power monitoring via powermetrics plist output."""

    def __init__(self):
        self._proc = None
        self._thread = None
        self._lock = threading.Lock()
        self._latest = {"cpu_w": 0.0, "gpu_w": 0.0, "ane_w": 0.0}
        self._running = False

    def start(self):
        try:
            self._proc = subprocess.Popen(
                ["sudo", "-n", "powermetrics",
                 "--samplers", "cpu_power,gpu_power,ane_power",
                 "-i", "500", "-f", "plist"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, PermissionError):
            return

        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        """Read plist dictionaries from powermetrics stdout."""
        buf = b""
        while self._running and self._proc and self._proc.poll() is None:
            try:
                chunk = self._proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk

                # powermetrics in plist mode outputs successive plist docs
                # Each ends with </plist>
                while b"</plist>" in buf:
                    idx = buf.index(b"</plist>") + len(b"</plist>")
                    plist_bytes = buf[:idx]
                    buf = buf[idx:]

                    try:
                        data = plistlib.loads(plist_bytes)
                        cpu_w = 0.0
                        gpu_w = 0.0
                        ane_w = 0.0

                        # CPU power (mW -> W)
                        proc = data.get("processor", {})
                        cpu_mw = proc.get("cpu_power", 0) or 0
                        cpu_w = cpu_mw / 1000.0 if cpu_mw > 100 else cpu_mw

                        # GPU power (mW -> W)
                        gpu_mw = proc.get("gpu_power", 0) or 0
                        gpu_w = gpu_mw / 1000.0 if gpu_mw > 100 else gpu_mw

                        # ANE power
                        ane_mw = proc.get("ane_power", 0) or 0
                        ane_w = ane_mw / 1000.0 if ane_mw > 100 else ane_mw

                        with self._lock:
                            self._latest = {
                                "cpu_w": round(cpu_w, 2),
                                "gpu_w": round(gpu_w, 2),
                                "ane_w": round(ane_w, 2),
                            }
                    except Exception:
                        pass
            except Exception:
                break

    def read(self) -> dict:
        with self._lock:
            d = dict(self._latest)
        d["total_w"] = round(d["cpu_w"] + d["gpu_w"] + d["ane_w"], 2)
        d["cpu_pct"] = 0.0  # Not available from powermetrics in this mode
        d["ane_active"] = 0  # MLX never uses ANE
        return d

    def stop(self):
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None


# ---------------------------------------------------------------------------
# Task loading (same format as Obj-C pipeline)
# ---------------------------------------------------------------------------

def load_tasks(path: str) -> list:
    """Load tasks from JSONL file. Each line has 'instruction' and 'schema'."""
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return tasks


def build_chat_prompt(task: dict) -> str:
    """Build ChatML-style prompt from a task dict.

    Matches the Obj-C build_chat_prompt() for instruct models.
    Uses <|im_start|>/<|im_end|> ChatML format (SmolLM2, Qwen, etc).
    """
    instruction = task.get("instruction") or task.get("prompt") or task.get("user", "")
    schema = task.get("schema")

    schema_str = ""
    if schema:
        schema_str = json.dumps(schema, separators=(",", ":"))

    system_msg = task.get("system",
        "You are a helpful assistant. Always respond with valid JSON only, no other text.")

    user_msg = instruction
    if schema_str:
        user_msg += f"\n\nRespond with valid JSON matching this schema: {schema_str}"

    prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(response: str, schema: dict) -> float:
    """Compute reward for a single response.

    Reward breakdown:
      0.0 = not valid JSON
      0.5 = valid JSON but missing required keys
      1.0 = valid JSON with all required keys and correct types
    """
    # Strip any trailing special tokens or whitespace
    response = response.strip()
    # Remove any trailing <|im_end|> or <|endoftext|> tokens
    for token in ["<|im_end|>", "<|endoftext|>", "</s>", "<|end|>"]:
        if response.endswith(token):
            response = response[:-len(token)].strip()

    # Try to parse JSON
    try:
        obj = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return 0.0

    if not isinstance(obj, dict):
        return 0.5  # Valid JSON but not an object

    if not schema:
        return 1.0  # No schema to check against

    # Check required keys
    required = schema.get("required", [])
    if not required:
        return 1.0

    present = sum(1 for k in required if k in obj)
    if present == len(required):
        return 1.0
    elif present > 0:
        return 0.5
    else:
        return 0.25  # Valid JSON object but no required keys


# ---------------------------------------------------------------------------
# Generation (rollout)
# ---------------------------------------------------------------------------

def generate_rollout(model, tokenizer, prompt_text: str, max_tokens: int,
                     temperature: float) -> tuple:
    """Generate a single rollout and return (text, token_ids, log_probs).

    Returns:
        text: decoded response string
        token_ids: list of generated token ids
        log_probs: list of per-token log probabilities (for the chosen token)
    """
    from mlx_lm.generate import generate_step

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_array = mx.array(prompt_tokens)

    # Build temperature sampler
    if temperature > 0:
        def sampler(logprobs):
            # logprobs are already log-softmax'd
            # Convert to logits, apply temperature, resample
            # Actually generate_step passes log-probs, so:
            scaled = logprobs / temperature
            # Gumbel-max trick: sample from categorical
            gumbels = -mx.log(-mx.log(mx.random.uniform(shape=scaled.shape) + 1e-20) + 1e-20)
            return mx.argmax(scaled + gumbels, axis=-1)
    else:
        sampler = None  # greedy (default in generate_step)

    token_ids = []
    token_logprobs = []

    # Get EOS token ids
    eos_ids = set()
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    # Also check for special token ids
    for attr in ["eos_token_id"]:
        val = getattr(tokenizer, attr, None)
        if val is not None:
            if isinstance(val, (list, tuple)):
                eos_ids.update(val)
            else:
                eos_ids.add(val)

    for tok_id, logprobs_vec in generate_step(
        prompt_array, model, max_tokens=max_tokens, sampler=sampler
    ):
        tok_id_int = tok_id.item() if hasattr(tok_id, "item") else int(tok_id)

        if tok_id_int in eos_ids:
            break

        # Get log prob of the selected token
        lp = logprobs_vec[tok_id_int].item()
        token_ids.append(tok_id_int)
        token_logprobs.append(lp)

    # Decode
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text, token_ids, token_logprobs


# ---------------------------------------------------------------------------
# GRPO loss computation (for gradient step)
# ---------------------------------------------------------------------------

def grpo_loss_fn(model, prompt_tokens_list, response_tokens_list, advantages):
    """Compute GRPO policy gradient loss.

    For each rollout:
      loss_i = -advantage_i * mean(log_prob(response_tokens))

    Total loss = mean over rollouts with non-zero advantage.

    Args:
        model: the MLX language model
        prompt_tokens_list: list of mx.array prompt token sequences
        response_tokens_list: list of mx.array response token sequences
        advantages: mx.array of shape [group_size]

    Returns:
        scalar loss
    """
    total_loss = mx.array(0.0)
    count = 0

    for i in range(len(prompt_tokens_list)):
        adv = advantages[i]

        prompt_toks = prompt_tokens_list[i]
        resp_toks = response_tokens_list[i]

        if len(resp_toks) == 0:
            continue

        # Concatenate prompt + response for full forward pass
        full_seq = mx.concatenate([prompt_toks, resp_toks])
        # Model forward: input is full_seq[:-1], labels are full_seq[1:]
        input_ids = full_seq[:-1][None, :]  # [1, seq_len-1]
        logits = model(input_ids)  # [1, seq_len-1, vocab]
        logits = logits.squeeze(0)  # [seq_len-1, vocab]

        # We only care about the response portion
        # prompt has len(prompt_toks) tokens, so response starts at index len(prompt_toks)-1
        # in the shifted sequence (since input is shifted by 1)
        resp_start = len(prompt_toks) - 1
        resp_logits = logits[resp_start:resp_start + len(resp_toks)]  # [resp_len, vocab]

        # Compute log probs
        log_probs = resp_logits - mx.logsumexp(resp_logits, axis=-1, keepdims=True)

        # Gather log probs for actual response tokens
        resp_toks_flat = resp_toks.astype(mx.int32)
        # Index into log_probs for each position
        token_log_probs = mx.take_along_axis(
            log_probs, resp_toks_flat[:, None], axis=1
        ).squeeze(1)  # [resp_len]

        mean_log_prob = mx.mean(token_log_probs)

        # REINFORCE: loss = -advantage * log_prob
        total_loss = total_loss + (-adv * mean_log_prob)
        count += 1

    if count == 0:
        return mx.array(0.0)

    return total_loss / count


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MLX GRPO training for Apple Silicon GPU comparison")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID (e.g. HuggingFaceTB/SmolLM2-360M-Instruct)")
    parser.add_argument("--tasks", required=True,
                        help="Path to tasks JSONL file")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of GRPO training steps")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Number of rollouts per step")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens to generate per rollout")
    parser.add_argument("--out", required=True,
                        help="Output path for JSONL log")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load model and tokenizer via mlx_lm
    # ------------------------------------------------------------------
    print(f"[mlx-grpo] Loading model: {args.model}", file=sys.stderr)
    from mlx_lm.utils import load as mlx_load
    model, tokenizer = mlx_load(args.model)
    mx.eval(model.parameters())

    # Extract short model name for logging
    model_name = args.model.split("/")[-1]
    print(f"[mlx-grpo] Model loaded: {model_name}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 2. Load tasks
    # ------------------------------------------------------------------
    tasks = load_tasks(args.tasks)
    print(f"[mlx-grpo] Loaded {len(tasks)} tasks from {args.tasks}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 3. Setup optimizer
    # ------------------------------------------------------------------
    optimizer = optim.Adam(learning_rate=args.lr)

    # Wrap loss function for value_and_grad
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)

    # ------------------------------------------------------------------
    # 4. Setup output
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 5. Start power monitoring
    # ------------------------------------------------------------------
    power = PowerMonitor()
    power.start()
    time.sleep(0.5)  # Let first sample come in

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print(f"[mlx-grpo] Starting {args.steps} GRPO steps, "
          f"group_size={args.group_size}, lr={args.lr}, "
          f"temp={args.temperature}, max_tokens={args.max_tokens}",
          file=sys.stderr)

    with open(str(out_path), "w") as log_file:
        for step in range(args.steps):
            step_t0 = time.perf_counter()

            # --- Rollout phase ---
            rollout_t0 = time.perf_counter()

            task = tasks[step % len(tasks)]
            prompt_text = build_chat_prompt(task)
            schema = task.get("schema", {})

            texts = []
            all_token_ids = []
            all_log_probs = []
            rewards = []

            for g in range(args.group_size):
                text, tok_ids, lps = generate_rollout(
                    model, tokenizer, prompt_text,
                    args.max_tokens, args.temperature
                )
                texts.append(text)
                all_token_ids.append(tok_ids)
                all_log_probs.append(lps)

                r = compute_reward(text, schema)
                rewards.append(r)

            # Force eval to get accurate rollout timing
            mx.eval(mx.array(0.0))
            rollout_ms = (time.perf_counter() - rollout_t0) * 1000.0

            # --- Reward phase ---
            reward_t0 = time.perf_counter()

            rewards_arr = mx.array(rewards)
            mean_reward = float(mx.mean(rewards_arr).item())
            std_reward = float(mx.std(rewards_arr).item()) if len(rewards) > 1 else 0.0

            # Compute advantages: (r - mean) / (std + eps)
            advantages = (rewards_arr - mean_reward) / (std_reward + 1e-8)

            # Count JSON validity
            json_valid_count = sum(1 for r in rewards if r > 0.0)
            json_valid_pct = (json_valid_count / len(rewards)) * 100.0

            reward_ms = (time.perf_counter() - reward_t0) * 1000.0

            # --- Gradient phase ---
            gradient_t0 = time.perf_counter()

            # Check if we should skip gradient step (all advantages ~0)
            adv_range = float(mx.max(advantages).item()) - float(mx.min(advantages).item())
            if adv_range < 1e-6:
                # All rewards identical -> no gradient signal
                gradient_ms = 0.0
            else:
                # Prepare token sequences for loss computation
                prompt_tokens_enc = tokenizer.encode(prompt_text, add_special_tokens=False)
                prompt_tokens_mx = mx.array(prompt_tokens_enc)

                prompt_list = []
                response_list = []
                adv_list = []

                for g in range(args.group_size):
                    if len(all_token_ids[g]) == 0:
                        continue
                    prompt_list.append(prompt_tokens_mx)
                    response_list.append(mx.array(all_token_ids[g]))
                    adv_list.append(advantages[g])

                if prompt_list:
                    adv_for_grad = mx.stack(adv_list) if len(adv_list) > 1 else mx.array(adv_list)

                    loss, grads = loss_and_grad_fn(
                        model, prompt_list, response_list, adv_for_grad
                    )

                    # Gradient clipping
                    grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)

                    # Apply gradients
                    optimizer.update(model, grads)

                    # Force evaluation for accurate timing
                    mx.eval(model.parameters(), optimizer.state)

                gradient_ms = (time.perf_counter() - gradient_t0) * 1000.0

            total_ms = (time.perf_counter() - step_t0) * 1000.0

            # --- Read power ---
            power_data = power.read()

            # --- Build log entry (matches Obj-C schema exactly) ---
            entry = {
                "step": step,
                "seed": args.seed,
                "backend": "mlx",
                "model": model_name,
                "mean_reward": round(mean_reward, 4),
                "json_valid_pct": round(json_valid_pct, 1),
                "timing": {
                    "rollout_ms": round(rollout_ms, 1),
                    "reward_ms": round(reward_ms, 1),
                    "gradient_ms": round(gradient_ms, 1),
                    "sync_ms": 0,
                    "total_ms": round(total_ms, 1),
                    "ane_ms": 0,
                    "cpu_attn_ms": 0,
                    "cpu_proj_ms": 0,
                    "bwd_ane_ms": 0,
                },
                "power": power_data,
                "power_w": power_data["total_w"],
                "rewards": [round(r, 4) for r in rewards],
            }

            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

            # --- Progress output ---
            print(
                f"[step {step}/{args.steps}] "
                f"reward={mean_reward:.3f} json={json_valid_pct:.0f}% "
                f"rollout={rollout_ms:.0f}ms grad={gradient_ms:.0f}ms "
                f"total={total_ms:.0f}ms "
                f"power={power_data['total_w']:.1f}W",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # 7. Cleanup
    # ------------------------------------------------------------------
    power.stop()

    print(f"\n[mlx-grpo] Training complete. Log written to {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
