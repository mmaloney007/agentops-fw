#!/usr/bin/env python3
"""P6 MLX GRPO training orchestrator.

Runs 9 models x 6 tasks x 3 seeds = 162 training runs on Apple Silicon
using the MLX backend with LoRA adapters.

Modes:
  --tasks-mode mixed   : All 6 tasks mixed into a single training run (default).
  --tasks-mode single  : One task per run (8 models x 6 tasks x 3 seeds = 144).

Usage:
    python scripts/run_p6_mlx_training.py --all
    python scripts/run_p6_mlx_training.py --models llama-3.2-1b qwen2.5-3b
    python scripts/run_p6_mlx_training.py --all --seeds 42,123,456 --resume
    python scripts/run_p6_mlx_training.py --all --tasks-mode single

Author: Mike Maloney <mike.maloney@unh.edu>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Model definitions (MLX-community 4-bit quantised variants)
# ---------------------------------------------------------------------------

MODELS = {
    "llama-3.2-1b": {
        "config": "llama-3.2-1b",
        "mlx_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "size": "1B",
        "vendor": "Meta",
    },
    "llama-3.2-3b": {
        "config": "llama-3.2-3b",
        "mlx_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "size": "3B",
        "vendor": "Meta",
    },
    "qwen2.5-3b": {
        "config": "qwen2.5-3b",
        "mlx_id": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "size": "3B",
        "vendor": "Alibaba",
    },
    "qwen3-4b": {
        "config": "qwen3-4b",
        "mlx_id": "mlx-community/Qwen3-4B-4bit",
        "hf_id": "Qwen/Qwen3-4B",
        "size": "4B",
        "vendor": "Alibaba",
    },
    "phi-3-mini": {
        "config": "phi-3-mini",
        "mlx_id": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "size": "3.8B",
        "vendor": "Microsoft",
    },
    "mistral-7b": {
        "config": "mistral-7b",
        "mlx_id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "size": "7B",
        "vendor": "Mistral",
    },
    "ministral-8b": {
        "config": "ministral-8b",
        "mlx_id": "mlx-community/Ministral-8B-Instruct-2410-4bit",
        "hf_id": "mistralai/Ministral-8B-Instruct-2410",
        "size": "8B",
        "vendor": "Mistral",
    },
    "gemma-2-9b": {
        "config": "gemma-2-9b",
        "mlx_id": "mlx-community/gemma-2-9b-it-4bit",
        "hf_id": "google/gemma-2-9b-it",
        "size": "9B",
        "vendor": "Google",
    },
    # ---------- P5: MoE capacity threshold models ----------
    "qwen3.5-35b-a3b": {
        "config": "qwen3.5-35b-a3b",
        "mlx_id": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "hf_id": "Qwen/Qwen3.5-35B-A3B",
        "size": "35B (3B active)",
        "vendor": "Alibaba",
        "arch": "MoE",
        "active_params": "3B",
        "notes": "P5 centerpiece: does 3B-active MoE hit same wall as 3B dense?",
    },
}

# T-Suite task files
ALL_TASKS = [
    "tasks/clinc_en.jsonl",       # T1: Intent classification
    "tasks/hotpot_dev.jsonl",     # T2: Grounded reasoning
    "tasks/t3_tools.jsonl",       # T3: Tool selection
    "tasks/t4_bfcl.jsonl",       # T4: Function calling
    "tasks/t5_swebench.jsonl",   # T5: SWE-bench
    "tasks/public_gsm8k.jsonl",  # T6: Math reasoning
]

TASK_NAMES = {
    "tasks/clinc_en.jsonl": "T1_clinc",
    "tasks/hotpot_dev.jsonl": "T2_hotpot",
    "tasks/t3_tools.jsonl": "T3_tools",
    "tasks/t4_bfcl.jsonl": "T4_bfcl",
    "tasks/t5_swebench.jsonl": "T5_swe",
    "tasks/public_gsm8k.jsonl": "T6_gsm8k",
}


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress(progress_path: Path) -> dict:
    """Load progress file, returning empty dict if missing."""
    if progress_path.exists():
        with open(progress_path, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "in_progress": None}


def save_progress(progress_path: Path, progress: dict) -> None:
    """Atomically save progress file."""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = progress_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    tmp.rename(progress_path)


def run_key(model_name: str, task_label: str, seed: int) -> str:
    """Unique key for a training run."""
    return f"{model_name}/{task_label}/seed{seed}"


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_single_training(
    model_name: str,
    model_info: dict,
    tasks: list[str],
    task_label: str,
    seed: int,
    out_root: Path,
    config_dir: Path,
) -> dict:
    """Execute one MLX GRPO training run. Returns result dict."""
    import yaml

    from agent_stable_slo.train.mlx_train_config import MLXTrainConfig
    from agent_stable_slo.train.mlx_grpo_adapter import MLXGRPOTrainer

    # Load base config
    config_path = config_dir / f"{model_info['config']}.yaml"
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Override for this run
    out_dir = out_root / model_name / task_label / f"seed{seed}"
    cfg_dict["tasks"] = tasks
    cfg_dict["seed"] = seed
    cfg_dict["adapter_path"] = str(out_dir / "adapter")
    cfg_dict["log_path"] = str(out_dir / "train_log.jsonl")

    # Validate config
    cfg = MLXTrainConfig(**cfg_dict)

    # Run training
    t0 = time.time()
    trainer = MLXGRPOTrainer(cfg)
    adapter_path = trainer.run()
    wallclock_s = time.time() - t0

    # Summarise from log
    log_path = out_dir / "train_log.jsonl"
    summary = _summarise_log(log_path)
    summary["wallclock_s"] = round(wallclock_s, 1)
    summary["model"] = model_name
    summary["task_label"] = task_label
    summary["seed"] = seed
    summary["adapter_path"] = str(adapter_path)

    # Save per-run summary
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _summarise_log(log_path: Path) -> dict:
    """Extract summary metrics from a training JSONL log."""
    if not log_path.exists():
        return {"error": "log not found"}

    rewards = []
    valid_count = 0
    total_count = 0

    with open(log_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rewards.append(rec.get("mean_reward", rec.get("reward", 0)))
            valid_count += rec.get("json_valid", 0)
            total_count += 1

    if total_count == 0:
        return {"total_steps": 0}

    last_50 = rewards[-50:] if len(rewards) >= 50 else rewards
    return {
        "total_steps": total_count,
        "avg_reward": round(sum(rewards) / len(rewards), 4),
        "final_50_reward": round(sum(last_50) / len(last_50), 4),
        "json_valid_pct": round(100 * valid_count / total_count, 1),
        "peak_reward": round(max(rewards), 4),
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="P6 MLX GRPO training orchestrator for Apple Silicon."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help="Select specific models to train.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all 9 models.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated seeds (default: 42,123,456).",
    )
    parser.add_argument(
        "--tasks-mode",
        choices=["mixed", "single"],
        default="mixed",
        help="mixed=all tasks together; single=one task per run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed runs (check progress.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/mlx_training",
        help="Output root directory.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/mlx_grpo",
        help="Directory containing MLX GRPO YAML configs.",
    )
    args = parser.parse_args()

    if not args.all and not args.models:
        parser.error("Specify --all or --models <model_name ...>")

    # Resolve models
    model_names = list(MODELS.keys()) if args.all else args.models
    seeds = [int(s) for s in args.seeds.split(",")]
    out_root = Path(args.out_dir)
    config_dir = Path(args.config_dir)
    progress_path = out_root / "progress.json"

    # Build run plan
    run_plan: list[tuple[str, dict, list[str], str, int]] = []

    for model_name in model_names:
        model_info = MODELS[model_name]
        for seed in seeds:
            if args.tasks_mode == "mixed":
                run_plan.append(
                    (model_name, model_info, ALL_TASKS, "Mixed", seed)
                )
            else:
                for task_path in ALL_TASKS:
                    task_label = TASK_NAMES.get(task_path, Path(task_path).stem)
                    run_plan.append(
                        (model_name, model_info, [task_path], task_label, seed)
                    )

    # Load progress for resume
    progress = load_progress(progress_path)

    total = len(run_plan)
    completed = 0
    failed = 0
    skipped = 0

    print(f"\n{'#' * 60}")
    print(f"# P6 MLX GRPO Training")
    print(f"# Models: {len(model_names)}  Seeds: {seeds}  Mode: {args.tasks_mode}")
    print(f"# Total runs planned: {total}")
    print(f"# Output: {out_root}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#' * 60}\n")

    global_start = time.time()

    for idx, (model_name, model_info, tasks, task_label, seed) in enumerate(run_plan):
        key = run_key(model_name, task_label, seed)

        # Skip if already completed
        if args.resume and key in progress.get("completed", []):
            skipped += 1
            print(f"[{idx+1}/{total}] SKIP (completed): {key}")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx+1}/{total}] {key}")
        print(f"  Model: {model_info['mlx_id']}")
        print(f"  Tasks: {tasks}")
        print(f"  Seed: {seed}")
        print(f"{'=' * 60}")

        progress["in_progress"] = key
        save_progress(progress_path, progress)

        try:
            summary = run_single_training(
                model_name=model_name,
                model_info=model_info,
                tasks=tasks,
                task_label=task_label,
                seed=seed,
                out_root=out_root,
                config_dir=config_dir,
            )
            completed += 1
            progress["completed"].append(key)
            progress["in_progress"] = None
            save_progress(progress_path, progress)

            print(f"  -> OK: reward={summary.get('avg_reward', '?')} "
                  f"valid={summary.get('json_valid_pct', '?')}% "
                  f"wall={summary.get('wallclock_s', '?')}s")

        except Exception as exc:
            failed += 1
            err_info = {
                "key": key,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
            progress.setdefault("failed", []).append(key)
            progress["in_progress"] = None
            save_progress(progress_path, progress)

            # Save error log
            err_dir = out_root / model_name / task_label / f"seed{seed}"
            err_dir.mkdir(parents=True, exist_ok=True)
            with open(err_dir / "error.json", "w") as f:
                json.dump(err_info, f, indent=2)

            print(f"  -> FAILED: {exc}")
            print(f"     (continuing to next run)")

    elapsed = time.time() - global_start

    print(f"\n{'#' * 60}")
    print(f"# P6 MLX Training Complete")
    print(f"# Completed: {completed}  Failed: {failed}  Skipped: {skipped}")
    print(f"# Duration: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"# Progress: {progress_path}")
    print(f"{'#' * 60}")

    # Final progress update
    progress["finished_at"] = datetime.now().isoformat()
    progress["total_wallclock_s"] = round(elapsed, 1)
    save_progress(progress_path, progress)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
