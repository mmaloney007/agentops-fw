#!/usr/bin/env python3
"""
Full P1/P2 experiment runner.
Runs baseline evals, training at 250/500 steps, and post-training evals for all 4 models.

Usage:
    python scripts/run_p1_p2_full.py --phase all
    python scripts/run_p1_p2_full.py --phase baseline  # Just baseline evals
    python scripts/run_p1_p2_full.py --phase train     # Just training
    python scripts/run_p1_p2_full.py --phase post      # Just post-training evals

Author: Mike Maloney <mike.maloney@unh.edu>
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Model configurations
# Note: Ministral-3-8B is VLM (Mistral3ForConditionalGeneration) - needs special handling
MODELS = [
    {
        "name": "qwen3-4b",
        "path": "./models/qwen3-4b-instruct",
        "config": "p2_qwen3_4b",
        "hf_id": "Qwen/Qwen3-4B-Instruct-2507",
        "skip_eval": False,
    },
    {
        "name": "ministral-8b",
        "path": "./models/ministral-8b-instruct",
        "config": "p2_ministral_8b",
        "hf_id": "mistralai/Ministral-3-8B-Instruct-2512",
        "skip_eval": True,  # VLM needs special processor handling
    },
    {
        "name": "llama-3.2-3b",
        "path": "./models/llama-3.2-3b-instruct",
        "config": "p2_llama_3b",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "skip_eval": False,
    },
    {
        "name": "gemma-3-12b",
        "path": "./models/gemma-3-12b-it",
        "config": "p2_gemma_12b",
        "hf_id": "google/gemma-3-12b-it",
        "skip_eval": False,
    },
]

# T-Suite task files
TASKS = [
    "tasks/clinc_en.jsonl",      # T1: Intent classification
    "tasks/hotpot_dev.jsonl",    # T2: Grounded reasoning
    "tasks/t3_tools.jsonl",      # T3: Tool selection
    "tasks/t4_bfcl.jsonl",       # T4: Function calling
    "tasks/t5_swebench.jsonl",   # T5: SWE-bench
]


def run_cmd(cmd: list, env: dict = None, cwd: str = None) -> int:
    """Run a command and return exit code."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    print(f"\n{'='*60}")
    print(f"[RUN] {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, env=full_env, cwd=cwd)
    return result.returncode


def run_baseline_evals(out_dir: Path, max_records: int = 0):
    """Run baseline evals for all models on T1-T5."""
    print("\n" + "#"*60)
    print("# PHASE 1: BASELINE EVALUATIONS")
    print("#"*60)

    results = {}
    for model in MODELS:
        model_name = model["name"]
        model_path = model["path"]

        if model.get("skip_eval"):
            print(f"\n>>> Skipping eval for {model_name} (VLM model)")
            continue

        print(f"\n>>> Evaluating: {model_name}")

        # Set up environment for hf_local provider
        env = {
            "AOFW_PROVIDER": "hf_local",
            "HF_LOCAL_MODEL": model_path,
            "HF_LOCAL_MAX_TOKENS": "256",
            "WANDB_MODE": "online",
        }

        run_name = f"baseline_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cmd = [
            "python", "scripts/eval_t_suite.py",
            "--models", f"hf_local:{model_name}",
            "--tasks", *TASKS,
            "--out-dir", str(out_dir / "baseline"),
            "--run-name", run_name,
        ]
        if max_records > 0:
            cmd.extend(["--max-records", str(max_records)])

        exit_code = run_cmd(cmd, env=env)
        results[model_name] = {"baseline": exit_code == 0}

    return results


def run_training(out_dir: Path, steps: list = [250, 500]):
    """Run P2 training for all models at specified step checkpoints."""
    print("\n" + "#"*60)
    print("# PHASE 2: SLO-AWARE GRPO TRAINING")
    print("#"*60)

    results = {}
    for model in MODELS:
        model_name = model["name"]
        config_preset = model["config"]

        for step_count in steps:
            print(f"\n>>> Training: {model_name} @ {step_count} steps")

            run_name = f"p2_{model_name}_{step_count}steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_path = out_dir / "training" / f"{model_name}_{step_count}steps"

            cmd = [
                "python", "-m", "agent_stable_slo.train.grpo_train_loop",
                "--config-preset", config_preset,
                "--steps", str(step_count),
                "--out", str(out_path),
                "--checkpoint-every", str(step_count),  # Save at end
            ]

            env = {"WANDB_MODE": "online"}
            exit_code = run_cmd(cmd, env=env)

            if model_name not in results:
                results[model_name] = {}
            results[model_name][f"train_{step_count}"] = exit_code == 0

    return results


def run_post_training_evals(out_dir: Path, max_records: int = 0):
    """Run post-training evals on trained checkpoints."""
    print("\n" + "#"*60)
    print("# PHASE 3: POST-TRAINING EVALUATIONS")
    print("#"*60)

    results = {}
    training_dir = out_dir / "training"

    for model in MODELS:
        model_name = model["name"]

        for step_count in [250, 500]:
            checkpoint_dir = training_dir / f"{model_name}_{step_count}steps" / "adapter"

            if not checkpoint_dir.exists():
                print(f">>> Skipping {model_name}@{step_count}: checkpoint not found")
                continue

            print(f"\n>>> Evaluating: {model_name}@{step_count} steps")

            env = {
                "AOFW_PROVIDER": "hf_local",
                "HF_LOCAL_MODEL": str(checkpoint_dir.parent.parent / "adapter"),
                "HF_LOCAL_MAX_TOKENS": "256",
                "WANDB_MODE": "online",
            }

            run_name = f"post_{model_name}_{step_count}steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cmd = [
                "python", "scripts/eval_t_suite.py",
                "--models", f"hf_local:{model_name}_trained_{step_count}",
                "--tasks", *TASKS,
                "--out-dir", str(out_dir / "post_training"),
                "--run-name", run_name,
            ]
            if max_records > 0:
                cmd.extend(["--max-records", str(max_records)])

            exit_code = run_cmd(cmd, env=env)

            if model_name not in results:
                results[model_name] = {}
            results[model_name][f"post_{step_count}"] = exit_code == 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Run full P1/P2 experiments")
    parser.add_argument("--phase", choices=["all", "baseline", "train", "post"], default="all")
    parser.add_argument("--out-dir", default="out/p1_p2_results", help="Output directory")
    parser.add_argument("--max-records", type=int, default=0, help="Limit eval records (0=all)")
    parser.add_argument("--steps", nargs="+", type=int, default=[250, 500], help="Training steps")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    start_time = datetime.now()

    print(f"\n{'#'*60}")
    print(f"# P1/P2 Full Experiment Run")
    print(f"# Started: {start_time.isoformat()}")
    print(f"# Output: {out_dir}")
    print(f"# Phase: {args.phase}")
    print(f"{'#'*60}")

    if args.phase in ["all", "baseline"]:
        all_results["baseline"] = run_baseline_evals(out_dir, args.max_records)

    if args.phase in ["all", "train"]:
        all_results["training"] = run_training(out_dir, args.steps)

    if args.phase in ["all", "post"]:
        all_results["post_training"] = run_post_training_evals(out_dir, args.max_records)

    # Write summary
    end_time = datetime.now()
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_minutes": (end_time - start_time).total_seconds() / 60,
        "phase": args.phase,
        "results": all_results,
    }

    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"# COMPLETE")
    print(f"# Duration: {summary['duration_minutes']:.1f} minutes")
    print(f"# Summary: {summary_path}")
    print(f"{'#'*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
