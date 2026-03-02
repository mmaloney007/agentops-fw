#!/usr/bin/env python3
"""Smoke-test: validate an MLX model loads, generates, and runs 3 GRPO steps.

Usage:
    python scripts/smoke_test_mlx_model.py --model qwen3.5-35b-a3b
    python scripts/smoke_test_mlx_model.py --model llama-3.2-1b  # quick baseline check
    python scripts/smoke_test_mlx_model.py --model qwen3.5-35b-a3b --steps 10

Exit codes:
    0 = all checks passed
    1 = load failed
    2 = generation failed
    3 = training step failed

Author: Mike Maloney <mike.maloney@unh.edu>
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# Import model registry from orchestrator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_p6_mlx_training import MODELS


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test MLX model for GRPO training.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help="Model key from MODELS registry.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of GRPO steps for training smoke test (default: 3).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="tasks/clinc_en.jsonl",
        help="Single task file for smoke test.",
    )
    args = parser.parse_args()

    model_info = MODELS[args.model]
    mlx_id = model_info["mlx_id"]
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {args.model}")
    print(f"  MLX ID: {mlx_id}")
    print(f"  Size:   {model_info['size']}")
    print(f"  Steps:  {args.steps}")
    print(f"{'='*60}\n")

    # ---- Phase 1: Load ----
    print("[1/4] Loading model...")
    t0 = time.time()
    try:
        import mlx.core as mx
        from mlx_lm import load as mlx_load

        model, tokenizer = mlx_load(mlx_id)
        load_time = time.time() - t0
        n_params = sum(p.size for _, p in model.parameters())
        print(f"  OK: {n_params:,} params loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1

    # ---- Phase 2: Memory check ----
    print("\n[2/4] Memory check...")
    try:
        active_mem = mx.metal.get_active_memory() / 1e9
        peak_mem = mx.metal.get_peak_memory() / 1e9
        print(f"  Active memory: {active_mem:.1f} GB")
        print(f"  Peak memory:   {peak_mem:.1f} GB")
        if active_mem > 50:
            print(f"  WARNING: Model uses {active_mem:.1f} GB — tight fit on 64GB")
    except Exception:
        print("  (memory tracking unavailable)")

    # ---- Phase 3: Generate ----
    print("\n[3/4] Generation test...")
    t0 = time.time()
    try:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=0.7, top_p=0.95)
        prompt = (
            'Output ONLY a valid JSON object. No explanations.\n\n'
            'Classify this utterance: "I need to cancel my flight"\n\n'
            'JSON Schema: {"type":"object","properties":{"intent":{"type":"string"},"confidence":{"type":"number"}}}\n\n'
            'JSON:'
        )
        response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=64, sampler=sampler)
        gen_time = time.time() - t0
        print(f"  Response ({gen_time:.1f}s): {response[:200]}")

        # Try parsing JSON
        try:
            parsed = json.loads(response.strip())
            print(f"  JSON valid: YES — {parsed}")
        except json.JSONDecodeError:
            print(f"  JSON valid: NO (expected for smoke test)")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 2

    # ---- Phase 4: Training steps ----
    print(f"\n[4/4] Training smoke test ({args.steps} steps)...")
    t0 = time.time()
    try:
        from agent_stable_slo.train.mlx_train_config import MLXTrainConfig
        from agent_stable_slo.train.mlx_grpo_adapter import MLXGRPOTrainer

        # Load config
        config_path = Path(f"configs/mlx_grpo/{model_info['config']}.yaml")
        if not config_path.exists():
            print(f"  Config not found: {config_path}, using defaults")
            cfg = MLXTrainConfig(
                base_model=mlx_id,
                num_steps=args.steps,
                group_size=2,
                lora_layers=8 if "35b" in args.model.lower() else 16,
                max_tokens=128,
                tasks=[args.task],
                checkpoint_every=0,
                eval_interval=1,
            )
        else:
            import yaml
            with open(config_path, "r") as f:
                cfg_dict = yaml.safe_load(f)
            cfg_dict["num_steps"] = args.steps
            cfg_dict["checkpoint_every"] = 0
            cfg_dict["eval_interval"] = 1
            cfg_dict["tasks"] = [args.task]
            cfg_dict["adapter_path"] = f"/tmp/smoke_test_{args.model}"
            cfg = MLXTrainConfig(**cfg_dict)

        # Delete the model we loaded in phase 1 to free memory for trainer
        del model, tokenizer
        gc.collect()
        try:
            mx.metal.clear_cache()
        except Exception:
            pass

        trainer = MLXGRPOTrainer(cfg)
        adapter_path = trainer.run()
        train_time = time.time() - t0

        print(f"\n  OK: {args.steps} steps completed in {train_time:.1f}s")
        print(f"  Adapter saved to: {adapter_path}")

        # Check memory after training
        try:
            peak_mem = mx.metal.get_peak_memory() / 1e9
            print(f"  Peak memory during training: {peak_mem:.1f} GB")
        except Exception:
            pass

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 3

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SMOKE TEST PASSED: {args.model}")
    print(f"  Load: {load_time:.1f}s")
    print(f"  Generate: {gen_time:.1f}s")
    print(f"  Train ({args.steps} steps): {train_time:.1f}s")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
