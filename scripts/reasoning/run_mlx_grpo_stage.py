#!/usr/bin/env python3
"""Run MLX GRPO optimization stage for reasoning-stack pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running via `python scripts/reasoning/run_mlx_grpo_stage.py ...`
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLX GRPO RL stage")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tasks", action="append", required=True, help="Task JSONL path (repeatable)")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=17)

    parser.add_argument("--lam-latency", type=float, default=0.1)
    parser.add_argument("--mu-cost", type=float, default=0.01)
    parser.add_argument("--gamma-stability", type=float, default=0.0)

    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "base_model": args.base_model,
        "tasks": [str(Path(t).resolve()) for t in args.tasks],
        "adapter_path": str((out_dir / "adapter").resolve()),
        "log_path": str((out_dir / "train_log.jsonl").resolve()),
        "group_size": args.group_size,
        "num_steps": args.steps,
        "lora_rank": args.lora_rank,
        "lora_layers": args.lora_layers,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "lam_latency": args.lam_latency,
        "mu_cost": args.mu_cost,
        "gamma_stability": args.gamma_stability,
        "checkpoint_every": args.checkpoint_every,
    }

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "config": payload}, indent=2))
        return

    from agent_stable_slo.train.mlx_grpo_adapter import MLXGRPOTrainer
    from agent_stable_slo.train.mlx_train_config import MLXTrainConfig

    cfg = MLXTrainConfig(**payload)
    trainer = MLXGRPOTrainer(cfg)
    adapter_path = trainer.run()

    print(
        json.dumps(
            {
                "status": "ok",
                "adapter_path": str(adapter_path),
                "log_path": payload["log_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
