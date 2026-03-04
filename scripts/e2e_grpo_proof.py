#!/usr/bin/env python3
"""
End-to-end proof that hybrid ANE+MLX GRPO training works.

Runs HybridGRPOTrainer for 5 steps using:
  - ANE (CoreML) for inference rollouts on Qwen2.5-0.5B
  - MLX for gradient computation on the same model

Verifies:
  1. ANE rollouts generate text
  2. Rewards compute correctly
  3. MLX gradients flow (loss is a real number)
  4. Loss changes over steps (training is happening)
  5. Timing breakdown is captured (ane_rollout_ms, mlx_gradient_ms, weight_sync_ms)

Usage:
  python scripts/e2e_grpo_proof.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    print("=" * 72)
    print("  Hybrid ANE+MLX GRPO -- End-to-End Proof")
    print("=" * 72)

    # Configure ANE for Qwen2.5-0.5B
    ane_meta_dir = str(Path("models/ane/qwen2.5-0.5b").resolve())
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    task_file = "tasks/clinc_en.jsonl"

    if not Path(ane_meta_dir).exists():
        print(f"ERROR: ANE model not found at {ane_meta_dir}")
        print("Run: python scripts/smoke_test_ane.py --model-dir models/ane/qwen2.5-0.5b")
        sys.exit(1)

    if not Path(task_file).exists():
        print(f"ERROR: Task file not found: {task_file}")
        sys.exit(1)

    print(f"\n  ANE model: {ane_meta_dir}")
    print(f"  Base model: {base_model}")
    print(f"  Task file: {task_file}")

    # Import trainer
    from agent_stable_slo.train.ane_grpo_adapter import (
        HybridGRPOConfig,
        HybridGRPOTrainer,
    )

    # Configure for minimal proof (5 steps, small group)
    cfg = HybridGRPOConfig(
        base_model=base_model,
        ane_meta_dir=ane_meta_dir,
        tasks=[task_file],
        num_steps=5,
        group_size=2,          # Minimal group
        max_tokens=32,         # Short generations
        lora_rank=4,           # Small LoRA
        lora_layers=4,         # Few layers
        learning_rate=1e-4,
        checkpoint_every=0,    # No checkpoints for proof
        batch_update_interval=5,  # Sync once at end
        seed=42,
    )

    print(f"\n  Steps: {cfg.num_steps}")
    print(f"  Group size: {cfg.group_size}")
    print(f"  Max tokens: {cfg.max_tokens}")
    print(f"  LoRA rank: {cfg.lora_rank}, layers: {cfg.lora_layers}")

    # Run training
    print(f"\n{'='*72}")
    print("  Starting training...")
    print(f"{'='*72}\n")

    t0 = time.time()
    try:
        adapter_dir = cfg_trainer_run(cfg)
    except Exception as e:
        print(f"\n  TRAINING ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0

    # Read training log
    log_path = adapter_dir.parent / "train_log.jsonl"
    if not log_path.exists():
        print(f"\n  ERROR: Training log not found at {log_path}")
        sys.exit(1)

    records = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("\n  ERROR: Training log is empty")
        sys.exit(1)

    # Verify results
    print(f"\n{'='*72}")
    print("  Verification")
    print(f"{'='*72}")

    checks = []

    # Check 1: All steps completed
    n_steps = len(records)
    ok = n_steps == cfg.num_steps
    checks.append(("Steps completed", ok, f"{n_steps}/{cfg.num_steps}"))

    # Check 2: ANE rollouts generated (ane_rollout_ms > 0)
    ane_times = [r.get("ane_rollout_ms", 0) for r in records]
    ok = all(t > 0 for t in ane_times)
    avg_ane = sum(ane_times) / len(ane_times)
    checks.append(("ANE rollouts", ok, f"avg {avg_ane:.0f}ms"))

    # Check 3: Rewards computed (mean_reward is a number)
    rewards = [r.get("mean_reward", None) for r in records]
    ok = all(r is not None for r in rewards)
    checks.append(("Rewards computed", ok, f"values: {rewards}"))

    # Check 4: MLX gradients flow (loss is a finite number)
    losses = [r.get("loss", None) for r in records]
    ok = all(l is not None and abs(l) < 1e6 for l in losses)
    mlx_times = [r.get("mlx_gradient_ms", 0) for r in records]
    avg_mlx = sum(mlx_times) / len(mlx_times)
    checks.append(("MLX gradients", ok, f"losses: {[round(l, 4) for l in losses]}, avg {avg_mlx:.0f}ms"))

    # Check 5: Loss changed (training is happening)
    if len(losses) >= 2:
        loss_changed = losses[0] != losses[-1]
    else:
        loss_changed = False
    checks.append(("Loss changes", loss_changed, f"first={losses[0]:.4f} last={losses[-1]:.4f}"))

    # Check 6: Timing breakdown captured
    has_timing = all(
        "ane_rollout_ms" in r and "mlx_gradient_ms" in r and "weight_sync_ms" in r
        for r in records
    )
    checks.append(("Timing breakdown", has_timing, "ane_rollout_ms, mlx_gradient_ms, weight_sync_ms"))

    # Print results
    all_pass = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}: {detail}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Adapter: {adapter_dir}")
    print(f"  Log: {log_path}")

    # Print training log summary
    print(f"\n  Training Log:")
    print(f"  {'Step':>5} {'Reward':>8} {'Loss':>10} {'ANE ms':>8} {'MLX ms':>8} {'Sync ms':>8}")
    for r in records:
        print(
            f"  {r['step']:>5} {r['mean_reward']:>8.4f} {r['loss']:>10.4f} "
            f"{r['ane_rollout_ms']:>8.0f} {r['mlx_gradient_ms']:>8.0f} "
            f"{r['weight_sync_ms']:>8.0f}"
        )

    overall = "PASS" if all_pass else "FAIL"
    print(f"\n  Overall: {overall}")

    if not all_pass:
        sys.exit(1)


def cfg_trainer_run(cfg):
    """Run the trainer and return the adapter directory."""
    from agent_stable_slo.train.ane_grpo_adapter import HybridGRPOTrainer
    trainer = HybridGRPOTrainer(cfg)
    return trainer.run()


if __name__ == "__main__":
    main()
