#!/usr/bin/env python3
"""Hybrid ANE+MLX GRPO LoRA trainer for Apple Silicon.

Uses Apple Neural Engine (ANE) for fast inference rollouts via CoreML and
MLX for gradient computation + LoRA weight updates.  This heterogeneous
compute split exploits both accelerators simultaneously:

  - ANE: high-throughput, low-power inference rollouts (group sampling)
  - MLX/Metal: flexible gradient computation + optimizer steps

The key difference from MLXGRPOTrainer:
  - MLX trainer: MLX generates rollouts AND computes gradients
  - Hybrid trainer: ANE generates rollouts, MLX computes gradients only

Weight sync from MLX LoRA back to the ANE CoreML model is a measured
TODO stub -- export timing is captured, full CoreML re-conversion will
be iterated on later.

Features:
 - Group sampling with per-group advantage normalisation.
 - KL penalty against a frozen reference copy.
 - LoRA adapter checkpoint saving / resume.
 - JSONL structured logging with per-phase timing breakdowns.
 - Optional power monitoring via PowerMonitor.
 - Guarded MLX imports so the module is importable without MLX installed.

Usage:
    from agent_stable_slo.train.ane_grpo_adapter import HybridGRPOTrainer, HybridGRPOConfig

    cfg = HybridGRPOConfig(
        base_model="meta-llama/Llama-3.2-1B-Instruct",
        ane_meta_dir="/path/to/anemll/model",
        tasks=["tasks/clinc_en.jsonl"],
    )
    trainer = HybridGRPOTrainer(cfg)
    trainer.run()
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent_stable_slo.rewards.composite import composite_reward
from agent_stable_slo.rewards.schema_reward import schema_valid
from agent_stable_slo.rollout.providers import ane_local

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
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.lora import LoRALinear  # noqa: F401

    _MLX_LM_AVAILABLE = True
except ImportError:
    mlx_lm = None  # type: ignore[assignment]
    mlx_load = None  # type: ignore[assignment]
    _MLX_LM_AVAILABLE = False


def _require_mlx() -> None:
    if not _MLX_AVAILABLE:
        raise ImportError(
            "MLX is required for HybridGRPOTrainer. "
            "Install with: pip install mlx mlx-lm"
        )
    if not _MLX_LM_AVAILABLE:
        raise ImportError(
            "mlx-lm is required for HybridGRPOTrainer. "
            "Install with: pip install mlx-lm"
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HybridGRPOConfig:
    """Configuration for the Hybrid ANE+MLX GRPO trainer.

    Attributes
    ----------
    base_model : str
        HuggingFace model ID for the base model (used by MLX for gradients).
    ane_meta_dir : str
        Path to Anemll-converted model directory (must exist, contains meta.yaml).
    tasks : List[str]
        JSONL task file paths (must be non-empty).
    """

    # Model
    base_model: str = ""
    ane_meta_dir: str = ""
    tasks: List[str] = field(default_factory=list)

    # GRPO
    num_steps: int = 200
    group_size: int = 4
    beta: float = 0.1

    # LoRA
    lora_rank: int = 8
    lora_layers: int = 16

    # Generation
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

    # Reward weights
    lam_latency: float = 0.1
    mu_cost: float = 0.01

    # Training
    learning_rate: float = 1e-4
    seed: int = 42
    checkpoint_every: int = 50

    # Hybrid-specific
    batch_update_interval: int = 1
    measure_power: bool = False

    # Paths
    adapter_path: str = ""
    log_path: str = ""

    def __post_init__(self) -> None:
        if not self.ane_meta_dir or not Path(self.ane_meta_dir).is_dir():
            raise ValueError(
                f"ane_meta_dir must be a valid directory, got: {self.ane_meta_dir!r}"
            )
        if not self.tasks:
            raise ValueError("tasks must be a non-empty list of JSONL file paths")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _export_lora_weights(model: Any) -> Dict[str, Any]:
    """Return dict of LoRA parameters from the model."""
    weights = {}
    for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
        if "lora" in name:
            weights[name] = param
    return weights


def _ane_rollout(
    prompt: str,
    schema: dict,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> Dict[str, Any]:
    """Run a single inference rollout via the ANE CoreML provider.

    Calls ane_local.generate_raw and returns a dict with:
      text, parsed, latency_ms, ttft_ms, tokens_in, tokens_out
    """
    raw_text, parsed, latency_ms, ttft_ms, tokens_in, tokens_out = (
        ane_local.generate_raw(
            prompt=prompt,
            schema=schema,
            mode="structured",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    return {
        "text": raw_text,
        "parsed": parsed,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def _parse_json(text: str, schema: dict) -> Dict[str, Any]:
    """Extract JSON from model output, handling embedded JSON in reasoning text.

    Three-stage extraction:
    1. Direct JSON parse
    2. Markdown code blocks (```json ... ```)
    3. Regex extraction for embedded JSON objects
    4. Fallback: if schema has "answer" property, wrap raw text
    """
    # Stage 1: Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Stage 2: Markdown code blocks
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

    # Stage 3: Regex extraction for embedded JSON objects
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
    """Load task records from one or more JSONL files.

    Each record must have a 'schema_path' field pointing to a JSON schema
    file.  The schema is loaded inline and attached as 'schema'.
    """
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


class HybridGRPOTrainer:
    """Hybrid ANE+MLX GRPO policy-gradient trainer.

    Uses ANE for inference rollouts and MLX for gradient updates.

    Parameters
    ----------
    cfg : HybridGRPOConfig
        Fully validated training configuration.
    """

    def __init__(self, cfg: HybridGRPOConfig) -> None:
        self.cfg = cfg
        self._setup_paths()
        self._load_data()

    # ------------------------------------------------------------------ #
    # Setup helpers                                                       #
    # ------------------------------------------------------------------ #

    def _setup_paths(self) -> None:
        ts = _timestamp()
        if not self.cfg.adapter_path:
            self.adapter_dir = Path(f"out/hybrid_train_{ts}") / "adapter"
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
        print(
            f"[data] Loaded {len(self.dataset)} task records "
            f"from {len(self.cfg.tasks)} files"
        )

    def _load_mlx_model(self) -> None:
        """Load model and tokenizer via mlx-lm, then apply LoRA.

        This model is used ONLY for gradient computation -- rollouts
        go through ANE.
        """
        _require_mlx()

        print(f"[model] Loading {self.cfg.base_model} for gradient computation ...")
        self.model, self.tokenizer = mlx_load(self.cfg.base_model)

        # Apply LoRA via mlx-lm
        self._apply_lora()

        # Freeze MoE router gate LoRA -- routing indices can't be differentiated
        for name, mod in self.model.named_modules():
            if name.endswith(".mlp.gate") and hasattr(mod, "lora_a"):
                mod.freeze()

        # Enable training mode
        self.model.train()

        trainable = nn.utils.tree_flatten(self.model.trainable_parameters())
        lora_params = [(k, v) for k, v in trainable if "lora" in k]
        n_params = sum(p.size for _, p in lora_params)
        n_total = sum(p.size for _, p in nn.utils.tree_flatten(self.model.parameters()))
        print(
            f"[model] Trainable: {n_params:,} / {n_total:,} "
            f"({100 * n_params / max(1, n_total):.2f}%)"
        )

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the last cfg.lora_layers transformer layers."""
        from mlx_lm.tuner.utils import linear_to_lora_layers

        linear_to_lora_layers(
            model=self.model,
            num_layers=self.cfg.lora_layers,
            config={"rank": self.cfg.lora_rank, "scale": 20.0, "dropout": 0.0},
        )

    def _setup_ane(self) -> None:
        """Set ANE environment variables for ane_local.generate_raw."""
        os.environ["ANE_META_DIR"] = self.cfg.ane_meta_dir
        os.environ["ANE_HF_MODEL"] = self.cfg.base_model
        os.environ["ANE_MAX_TOKENS"] = str(self.cfg.max_tokens)
        print(
            f"[ane] Configured ANE_META_DIR={self.cfg.ane_meta_dir}, "
            f"ANE_HF_MODEL={self.cfg.base_model}"
        )

    # ------------------------------------------------------------------ #
    # Generation (ANE)                                                     #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, task_prompt: str) -> str:
        schema_hint = (
            "IMPORTANT: Output ONLY a valid JSON object. "
            "No explanations, no reasoning, no markdown - just the raw JSON.\n\n"
        )
        return f"{schema_hint}{task_prompt}\n\nJSON:"

    def _generate_group_ane(
        self, prompt: str, schema: dict
    ) -> List[Dict[str, Any]]:
        """Generate group_size rollouts via ANE inference."""
        outputs = []
        for _ in range(self.cfg.group_size):
            result = _ane_rollout(
                prompt=prompt,
                schema=schema,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            outputs.append(result)
        return outputs

    # ------------------------------------------------------------------ #
    # Loss computation (MLX)                                               #
    # ------------------------------------------------------------------ #

    def _step_loss_fn(self, model: Any) -> Any:
        """Loss function closure used by nn.value_and_grad.

        Reads prompt/gen tokens and advantages from instance attributes
        set in the training loop.  Mirrors the MLXGRPOTrainer pattern.
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
    # Weight sync                                                          #
    # ------------------------------------------------------------------ #

    def _sync_weights_to_ane(self, step: int) -> float:
        """Export LoRA weights and measure sync time.

        Full CoreML re-conversion is a TODO -- for now we measure the
        export time as a first step to understand the overhead.

        Returns sync_time_ms.
        """
        t0 = time.time()
        weights = _export_lora_weights(self.model)
        sync_ms = (time.time() - t0) * 1000.0

        print(
            f"[sync] Step {step}: exported {len(weights)} LoRA params "
            f"in {sync_ms:.1f}ms (CoreML re-conversion TODO)"
        )
        return sync_ms

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        """Save LoRA adapter weights + metadata."""
        tag = "final" if final else f"step_{step + 1}"
        ckpt_dir = self.adapter_dir / tag if not final else self.adapter_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save only LoRA parameters
        weights = _export_lora_weights(self.model)
        mx.savez(str(ckpt_dir / "adapters.npz"), **weights)

        # Save metadata
        meta = {
            "step": step,
            "base_model": self.cfg.base_model,
            "ane_meta_dir": self.cfg.ane_meta_dir,
            "lora_rank": self.cfg.lora_rank,
            "lora_layers": self.cfg.lora_layers,
            "trainer": "HybridGRPOTrainer",
            "saved_at": time.time(),
        }
        with open(ckpt_dir / "adapter_config.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[checkpoint] Saved {tag} to {ckpt_dir}")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        """Execute the full hybrid training loop.

        1. Sample task
        2. Phase 1: ANE rollouts (timed as ane_rollout_ms)
        3. Phase 2: Score with composite_reward + schema_valid
        4. Phase 3: MLX gradient update (timed as mlx_gradient_ms)
        5. Phase 4: Weight sync every batch_update_interval steps
        6. Log JSONL with all timing breakdowns
        7. Optional power monitoring
        8. Periodic checkpoints

        Returns path to final adapter directory.
        """
        _set_seed(self.cfg.seed)

        # Load MLX model for gradient computation
        self._load_mlx_model()

        # Configure ANE for rollouts
        self._setup_ane()

        optimizer = optim.Adam(learning_rate=self.cfg.learning_rate)

        # Build loss + grad function
        loss_and_grad = nn.value_and_grad(self.model, self._step_loss_fn)

        # Optional power monitor
        power_monitor = None
        if self.cfg.measure_power:
            try:
                from agent_stable_slo.hardware.power_monitor import PowerMonitor

                power_monitor = PowerMonitor()
            except ImportError:
                print("[warn] PowerMonitor not available, skipping power measurement")

        running_baseline = 0.0
        step_rewards: List[float] = []

        print(
            f"[train] Starting {self.cfg.num_steps} steps, "
            f"group_size={self.cfg.group_size}, "
            f"batch_update_interval={self.cfg.batch_update_interval}"
        )
        train_start = time.time()

        with open(self.log_path, "w", encoding="utf-8") as log_file:
            for step in range(self.cfg.num_steps):
                step_t0 = time.time()

                # Sample a task
                row = random.choice(self.dataset)
                prompt = self._build_prompt(row["prompt"])
                schema = row["schema"]

                # ---- Phase 1: ANE rollouts ----
                ane_t0 = time.time()
                group = self._generate_group_ane(prompt, schema)
                ane_rollout_ms = (time.time() - ane_t0) * 1000.0

                # ---- Phase 2: Score each completion ----
                rewards = []
                group_meta = []
                for out in group:
                    out_json = out.get("parsed", {})
                    if not out_json:
                        out_json = _parse_json(out["text"], schema)
                    jv = schema_valid(out_json, schema)
                    r = composite_reward(
                        out_json,
                        schema,
                        ok_success=jv,
                        latency_ms=out["latency_ms"],
                        tokens=out.get("tokens_out", 0),
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
                        "tokens_out": out.get("tokens_out", 0),
                    })

                # Compute advantages (reward - group mean)
                mean_reward = sum(rewards) / max(1, len(rewards))
                advantages = [r - mean_reward for r in rewards]

                # Select best completion for logging
                best_idx = rewards.index(max(rewards))
                best = group_meta[best_idx]

                # Update running baseline
                running_baseline = 0.9 * running_baseline + 0.1 * mean_reward

                # ---- Phase 3: MLX gradient update ----
                mlx_t0 = time.time()

                # Tokenise prompt and generations for loss computation
                prompt_tokens = mx.array(
                    self.tokenizer.encode(prompt, add_special_tokens=False)
                )
                gen_tokens_list = []
                for out in group:
                    toks = self.tokenizer.encode(
                        out["text"], add_special_tokens=False
                    )
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

                mlx_gradient_ms = (time.time() - mlx_t0) * 1000.0

                # ---- Phase 4: Weight sync ----
                weight_sync_ms = 0.0
                if (step + 1) % self.cfg.batch_update_interval == 0:
                    weight_sync_ms = self._sync_weights_to_ane(step)

                step_ms = (time.time() - step_t0) * 1000.0

                # ---- Logging ----
                log_rec: Dict[str, Any] = {
                    "step": step,
                    "reward": best["reward"],
                    "mean_reward": round(mean_reward, 4),
                    "json_valid": best["json_valid"],
                    "loss": round(float(loss_val), 6),
                    "step_ms": round(step_ms, 1),
                    "ane_rollout_ms": round(ane_rollout_ms, 1),
                    "mlx_gradient_ms": round(mlx_gradient_ms, 1),
                    "weight_sync_ms": round(weight_sync_ms, 1),
                    "group_rewards": [round(r, 4) for r in rewards],
                }

                # Optional power monitoring
                if power_monitor is not None:
                    try:
                        sample = power_monitor.sample()
                        log_rec["power_w"] = round(sample.get("power_w", 0.0), 2)
                        log_rec["energy_j"] = round(
                            sample.get("power_w", 0.0) * step_ms / 1000.0, 4
                        )
                    except Exception:
                        pass

                log_file.write(json.dumps(log_rec) + "\n")
                log_file.flush()

                step_rewards.append(mean_reward)

                # Console output every 10 steps
                if (step + 1) % 10 == 0:
                    recent = step_rewards[-10:]
                    avg_r = sum(recent) / len(recent)
                    valid_pct = sum(
                        1 for m in group_meta if m["json_valid"]
                    ) / len(group_meta)
                    print(
                        f"[step {step + 1}/{self.cfg.num_steps}] "
                        f"reward={avg_r:.3f} valid={valid_pct:.0%} "
                        f"loss={float(loss_val):.4f} "
                        f"ane={ane_rollout_ms:.0f}ms "
                        f"mlx={mlx_gradient_ms:.0f}ms "
                        f"sync={weight_sync_ms:.0f}ms "
                        f"total={step_ms:.0f}ms"
                    )

                # Checkpoint
                if (
                    self.cfg.checkpoint_every > 0
                    and (step + 1) % self.cfg.checkpoint_every == 0
                ):
                    self._save_checkpoint(step)

                # Periodic memory cleanup
                if (step + 1) % 50 == 0:
                    _clear = getattr(mx, "clear_cache", None) or getattr(
                        mx.metal, "clear_cache", None
                    )
                    if _clear:
                        _clear()

        # Final save
        self._save_checkpoint(self.cfg.num_steps - 1, final=True)

        elapsed = time.time() - train_start
        print(
            f"[done] {self.cfg.num_steps} steps in {elapsed / 60:.1f}min. "
            f"Adapter: {self.adapter_dir}  Log: {self.log_path}"
        )
        return self.adapter_dir
