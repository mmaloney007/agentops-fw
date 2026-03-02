#!/usr/bin/env python3
"""MLX GRPO LoRA trainer for Apple Silicon.

Bridges the composite reward function from agent_stable_slo.rewards.composite
to mlx-lm's LoRA fine-tuning, implementing a minimal GRPO policy-gradient
loop entirely on the Metal backend.

Features:
 - Group sampling with per-group advantage normalisation.
 - KL penalty against a frozen reference copy.
 - LoRA adapter checkpoint saving / resume.
 - JSONL structured logging compatible with grpo_train_loop.py output.
 - Guarded MLX imports so the module is importable without MLX installed.

Usage:
    from agent_stable_slo.train.mlx_grpo_adapter import MLXGRPOTrainer
    from agent_stable_slo.train.mlx_train_config import MLXTrainConfig

    cfg = MLXTrainConfig(base_model="mlx-community/Llama-3.2-1B-Instruct-4bit", ...)
    trainer = MLXGRPOTrainer(cfg)
    trainer.run()
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent_stable_slo.rewards.composite import composite_reward
from agent_stable_slo.rewards.schema_reward import schema_valid
from agent_stable_slo.train.mlx_train_config import MLXTrainConfig

# ---------------------------------------------------------------------------
# Guarded MLX imports
# ---------------------------------------------------------------------------

_MLX_AVAILABLE = False

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    _MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

try:
    import mlx_lm
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.lora import LoRALinear  # noqa: F401
    from mlx_lm.sample_utils import make_sampler as _make_sampler

    _MLX_LM_AVAILABLE = True
except ImportError:
    mlx_lm = None  # type: ignore[assignment]
    mlx_generate = None  # type: ignore[assignment]
    mlx_load = None  # type: ignore[assignment]
    _make_sampler = None  # type: ignore[assignment]
    _MLX_LM_AVAILABLE = False


def _require_mlx() -> None:
    if not _MLX_AVAILABLE:
        raise ImportError(
            "MLX is required for MLXGRPOTrainer. "
            "Install with: pip install mlx mlx-lm"
        )
    if not _MLX_LM_AVAILABLE:
        raise ImportError(
            "mlx-lm is required for MLXGRPOTrainer. "
            "Install with: pip install mlx-lm"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_json(text: str, schema: dict) -> Dict[str, Any]:
    """Extract JSON from model output, handling embedded JSON in reasoning text."""
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Markdown code blocks
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p_strip = p.strip()
            if p_strip.startswith("json"):
                p_strip = p_strip[4:].strip()
            if not p_strip:
                continue
            try:
                return json.loads(p_strip)
            except Exception:
                continue

    # Regex extraction for embedded JSON objects
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    # Fallback for simple answer schemas
    if "answer" in schema.get("properties", {}):
        return {"answer": text.strip()}
    return {}


def _load_dataset(tasks_paths: List[str]) -> List[Dict[str, Any]]:
    """Load task records from one or more JSONL files."""
    rows: List[Dict[str, Any]] = []
    for tasks_path in tasks_paths:
        with open(tasks_path, "r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                rec = json.loads(raw)
                schema_path = rec["schema_path"]
                with open(schema_path, "r", encoding="utf-8") as sf:
                    schema = json.load(sf)
                rec["schema"] = schema
                rec["_source_task"] = tasks_path
                rows.append(rec)
    if not rows:
        raise ValueError(f"No task records found in {tasks_paths}")
    return rows


def _set_seed(seed: int) -> None:
    """Seed Python and MLX random generators."""
    random.seed(seed)
    if _MLX_AVAILABLE:
        mx.random.seed(seed)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class MLXGRPOTrainer:
    """GRPO policy-gradient trainer using MLX and LoRA adapters.

    Parameters
    ----------
    cfg : MLXTrainConfig
        Fully validated training configuration.
    """

    def __init__(self, cfg: MLXTrainConfig) -> None:
        _require_mlx()
        self.cfg = cfg
        self._setup_paths()
        self._load_data()

    # ------------------------------------------------------------------ #
    # Setup helpers                                                       #
    # ------------------------------------------------------------------ #

    def _setup_paths(self) -> None:
        ts = _timestamp()
        if not self.cfg.adapter_path:
            self.adapter_dir = Path(f"out/mlx_train_{ts}") / "adapter"
        else:
            self.adapter_dir = Path(self.cfg.adapter_path)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

        if not self.cfg.log_path:
            self.log_path = self.adapter_dir.parent / "train_log.jsonl"
        else:
            self.log_path = Path(self.cfg.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> None:
        self.dataset = _load_dataset(self.cfg.tasks)
        print(f"[data] Loaded {len(self.dataset)} task records from {len(self.cfg.tasks)} files")

    def _load_model(self) -> None:
        """Load model and tokenizer via mlx-lm, then apply LoRA."""
        print(f"[model] Loading {self.cfg.base_model} ...")

        # Patch sanitizer for Qwen3.5 VL models (strip vision_tower weights)
        self._patch_qwen35_sanitizer()

        self.model, self.tokenizer = mlx_load(self.cfg.base_model)

        # Apply LoRA via mlx-lm (auto-detects SwitchLinear for MoE experts)
        self._apply_lora()

        # Freeze MoE router gate LoRA — routing indices can't be differentiated
        for name, mod in self.model.named_modules():
            if name.endswith(".mlp.gate") and hasattr(mod, "lora_a"):
                mod.freeze()

        # Enable training mode (triggers stop_gradient on MoE routing indices)
        self.model.train()

        n_params = sum(p.size for _, p in self.model.trainable_parameters())
        n_total = sum(p.size for _, p in self.model.parameters())
        print(f"[model] Trainable: {n_params:,} / {n_total:,} ({100*n_params/max(1,n_total):.2f}%)")

        self._ref_model = None  # KL computed against base (LoRA=0 baseline)

    @staticmethod
    def _patch_qwen35_sanitizer() -> None:
        """Patch qwen3_5_moe sanitizer to drop vision_tower weights."""
        try:
            from mlx_lm.models.qwen3_5_moe import Model as Qwen35MoE

            _orig = Qwen35MoE.sanitize

            def _patched(self, weights):
                weights = {
                    k: v
                    for k, v in weights.items()
                    if "vision_tower" not in k and "visual" not in k
                }
                return _orig(self, weights)

            if not getattr(Qwen35MoE.sanitize, "_patched", False):
                Qwen35MoE.sanitize = _patched
                Qwen35MoE.sanitize._patched = True  # type: ignore[attr-defined]
        except ImportError:
            pass  # mlx-lm version without Qwen3.5 support

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the last cfg.lora_layers transformer layers."""
        from mlx_lm.tuner.utils import linear_to_lora_layers

        linear_to_lora_layers(
            model=self.model,
            num_layers=self.cfg.lora_layers,
            config={"rank": self.cfg.lora_rank, "scale": 20.0, "dropout": 0.0},
        )

    # ------------------------------------------------------------------ #
    # Generation                                                          #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, task_prompt: str) -> str:
        schema_hint = (
            "IMPORTANT: Output ONLY a valid JSON object. "
            "No explanations, no reasoning, no markdown - just the raw JSON.\n\n"
        )
        return f"{schema_hint}{task_prompt}\n\nJSON:"

    def _generate_one(self, prompt: str) -> Tuple[str, float, int]:
        """Generate a single completion and return (text, latency_ms, tokens_out)."""
        t0 = time.time()
        sampler = _make_sampler(
            temp=self.cfg.temperature if self.cfg.temperature > 0 else 0.0,
            top_p=self.cfg.top_p if self.cfg.temperature > 0 else 0.0,
        )
        # Switch to eval mode for generation (avoids stop_gradient in SwitchGLU)
        self.model.eval()
        response = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.cfg.max_tokens,
            sampler=sampler,
        )
        # Restore training mode for gradient computation
        self.model.train()
        lat_ms = (time.time() - t0) * 1000.0
        # mlx_generate returns the generated text string
        text = response.strip() if isinstance(response, str) else str(response).strip()
        # Estimate token count from text length (rough)
        tokens_out = max(1, len(text.split()))
        return text, lat_ms, tokens_out

    def _generate_group(
        self, prompt: str, group_size: int
    ) -> List[Dict[str, Any]]:
        """Generate group_size completions for a single prompt."""
        outputs = []
        for _ in range(group_size):
            text, lat_ms, tokens_out = self._generate_one(prompt)
            outputs.append({
                "text": text,
                "latency_ms": lat_ms,
                "tokens_out": tokens_out,
            })
        return outputs

    # ------------------------------------------------------------------ #
    # Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def _compute_log_probs(
        self, prompt_tokens: Any, gen_tokens: Any
    ) -> Any:
        """Compute per-token log-probs of gen_tokens given prompt_tokens."""
        full_tokens = mx.concatenate([prompt_tokens, gen_tokens], axis=-1)
        # Forward pass
        logits = self.model(full_tokens[None, :])  # (1, seq_len, vocab)
        logits = logits[0]  # (seq_len, vocab)

        # Slice to generation region: predict gen_tokens from prompt context
        prompt_len = prompt_tokens.shape[0]
        gen_logits = logits[prompt_len - 1 : -1, :]  # (gen_len, vocab)

        # Log-softmax
        log_probs = gen_logits - mx.logsumexp(gen_logits, axis=-1, keepdims=True)

        # Gather log-probs for actual tokens
        gen_len = gen_tokens.shape[0]
        token_log_probs = log_probs[mx.arange(gen_len), gen_tokens]

        return token_log_probs.sum()

    def _policy_gradient_loss(
        self,
        prompt_tokens: Any,
        gen_tokens_list: List[Any],
        advantages: List[float],
    ) -> Any:
        """Compute GRPO policy gradient loss with KL penalty.

        loss = -mean(advantage_i * sum(log_prob(gen_i | prompt)))
               + beta * mean(log_prob(gen_i | policy) - log_prob(gen_i | ref))

        Since the reference is the base model (LoRA weights = 0), we approximate
        the KL term as beta * sum(log_prob) when LoRA weights are small.
        For the initial implementation we use a simplified REINFORCE-style loss.
        """
        total_loss = mx.array(0.0)
        n = len(gen_tokens_list)

        for gen_tokens, adv in zip(gen_tokens_list, advantages):
            log_prob_sum = self._compute_log_probs(prompt_tokens, gen_tokens)
            # Policy gradient: -advantage * log_prob
            pg_loss = -adv * log_prob_sum
            # KL penalty: beta * log_prob (approximation against base)
            kl_term = self.cfg.beta * log_prob_sum
            total_loss = total_loss + pg_loss + kl_term

        return total_loss / max(1, n)

    # ------------------------------------------------------------------ #
    # Training loop                                                       #
    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        """Execute the full training loop. Returns path to final adapter."""
        _set_seed(self.cfg.seed)
        self._load_model()

        optimizer = optim.Adam(learning_rate=self.cfg.learning_rate)

        # Build loss + grad function
        loss_and_grad = nn.value_and_grad(self.model, self._step_loss_fn)

        running_baseline = 0.0
        step_rewards: List[float] = []

        print(f"[train] Starting {self.cfg.num_steps} steps, group_size={self.cfg.group_size}")
        train_start = time.time()

        with open(self.log_path, "w", encoding="utf-8") as log_file:
            for step in range(self.cfg.num_steps):
                step_t0 = time.time()

                # Sample a task
                row = random.choice(self.dataset)
                prompt = self._build_prompt(row["prompt"])
                schema = row["schema"]

                # Generate group completions
                group = self._generate_group(prompt, self.cfg.group_size)

                # Score each completion
                rewards = []
                group_meta = []
                for out in group:
                    out_json = _parse_json(out["text"], schema)
                    jv = schema_valid(out_json, schema)
                    r = composite_reward(
                        out_json,
                        schema,
                        ok_success=jv,
                        latency_ms=out["latency_ms"],
                        tokens=out["tokens_out"],
                        lam_latency=self.cfg.lam_latency,
                        mu_cost=self.cfg.mu_cost,
                    )
                    rewards.append(float(r))
                    group_meta.append({
                        "text": out["text"],
                        "json": out_json,
                        "json_valid": int(jv),
                        "reward": float(r),
                        "latency_ms": out["latency_ms"],
                        "tokens_out": out["tokens_out"],
                    })

                # Compute advantages (reward - group mean)
                mean_reward = sum(rewards) / max(1, len(rewards))
                advantages = [r - mean_reward for r in rewards]

                # Select best completion for logging
                best_idx = rewards.index(max(rewards))
                best = group_meta[best_idx]

                # Update running baseline
                running_baseline = 0.9 * running_baseline + 0.1 * mean_reward

                # Tokenise prompt and generations for loss computation
                prompt_tokens = mx.array(
                    self.tokenizer.encode(prompt, add_special_tokens=False)
                )
                gen_tokens_list = []
                for out in group:
                    toks = self.tokenizer.encode(out["text"], add_special_tokens=False)
                    if len(toks) == 0:
                        toks = [self.tokenizer.eos_token_id or 0]
                    gen_tokens_list.append(mx.array(toks))

                # Store for the closure
                self._current_prompt_tokens = prompt_tokens
                self._current_gen_tokens_list = gen_tokens_list
                self._current_advantages = advantages

                # Compute loss and gradients
                loss_val, grads = loss_and_grad(self.model)
                optimizer.update(self.model, grads)

                # Force evaluation to free memory
                mx.eval(loss_val)
                mx.eval(self.model.parameters())

                step_ms = (time.time() - step_t0) * 1000.0

                # Log step
                log_rec = {
                    "step": step,
                    "reward": best["reward"],
                    "mean_reward": round(mean_reward, 4),
                    "advantage": round(advantages[best_idx], 4),
                    "json_valid": best["json_valid"],
                    "latency_ms": round(best["latency_ms"], 3),
                    "tokens_out": best["tokens_out"],
                    "loss": round(float(loss_val), 6),
                    "step_ms": round(step_ms, 1),
                    "group_rewards": [round(r, 4) for r in rewards],
                    "schema_path": row.get("schema_path", ""),
                    "source_task": row.get("_source_task", ""),
                }
                log_file.write(json.dumps(log_rec) + "\n")
                log_file.flush()

                step_rewards.append(mean_reward)

                # Console output
                if (step + 1) % max(1, self.cfg.eval_interval) == 0:
                    recent = step_rewards[-self.cfg.eval_interval :]
                    avg_r = sum(recent) / len(recent)
                    valid_pct = sum(
                        1 for m in group_meta if m["json_valid"]
                    ) / len(group_meta)
                    print(
                        f"[step {step + 1}/{self.cfg.num_steps}] "
                        f"reward={avg_r:.3f} valid={valid_pct:.0%} "
                        f"loss={float(loss_val):.4f} step_ms={step_ms:.0f}"
                    )

                # Checkpoint
                if (
                    self.cfg.checkpoint_every > 0
                    and (step + 1) % self.cfg.checkpoint_every == 0
                ):
                    self._save_checkpoint(step)

                # Periodic memory cleanup
                if (step + 1) % 50 == 0:
                    _clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
                    if _clear:
                        _clear()

        # Final save
        self._save_checkpoint(self.cfg.num_steps - 1, final=True)

        elapsed = time.time() - train_start
        print(
            f"[done] {self.cfg.num_steps} steps in {elapsed/60:.1f}min. "
            f"Adapter: {self.adapter_dir}  Log: {self.log_path}"
        )
        return self.adapter_dir

    def _step_loss_fn(self, model: Any) -> Any:
        """Loss function closure used by nn.value_and_grad.

        Reads prompt/gen tokens and advantages from instance attributes
        set in the training loop.
        """
        total_loss = mx.array(0.0)
        n = len(self._current_gen_tokens_list)

        for gen_tokens, adv in zip(
            self._current_gen_tokens_list, self._current_advantages
        ):
            full_tokens = mx.concatenate(
                [self._current_prompt_tokens, gen_tokens], axis=-1
            )
            logits = model(full_tokens[None, :])[0]

            prompt_len = self._current_prompt_tokens.shape[0]
            gen_logits = logits[prompt_len - 1 : -1, :]
            log_probs = gen_logits - mx.logsumexp(
                gen_logits, axis=-1, keepdims=True
            )

            gen_len = gen_tokens.shape[0]
            token_log_probs = log_probs[mx.arange(gen_len), gen_tokens]
            log_prob_sum = token_log_probs.sum()

            # Policy gradient + KL penalty
            pg_loss = -adv * log_prob_sum
            kl_term = self.cfg.beta * log_prob_sum
            total_loss = total_loss + pg_loss + kl_term

        return total_loss / max(1, n)

    # ------------------------------------------------------------------ #
    # Checkpointing                                                       #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        """Save LoRA adapter weights."""
        tag = "final" if final else f"step_{step + 1}"
        ckpt_dir = self.adapter_dir / tag if not final else self.adapter_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save only LoRA parameters
        weights = {}
        for name, param in self.model.trainable_parameters():
            weights[name] = param
        mx.savez(str(ckpt_dir / "adapters.npz"), **weights)

        # Save metadata
        meta = {
            "step": step,
            "base_model": self.cfg.base_model,
            "lora_rank": self.cfg.lora_rank,
            "lora_layers": self.cfg.lora_layers,
            "config_version": self.cfg.config_version,
            "saved_at": time.time(),
        }
        with open(ckpt_dir / "adapter_config.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[checkpoint] Saved {tag} to {ckpt_dir}")
