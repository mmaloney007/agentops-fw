#!/usr/bin/env python3
"""
Retry failed P2 training runs with more aggressive memory settings.

For OOM failures:
- Enable 4-bit quantization
- Increase gradient accumulation to 4
- Reduce max_new_tokens if needed

Usage:
    python scripts/retry_failed_runs.py --input out/p2_training_20260124
    python scripts/retry_failed_runs.py --input out/p2_training_20260124 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Memory-optimized settings for OOM retries
OOM_RETRY_SETTINGS = {
    "gradient_accumulation": 4,
    "load_in_4bit": True,
    "max_new_tokens": 128,  # Reduced from 192
    "max_prompt_len": 512,  # Truncate long prompts
    "gradient_checkpointing": True,  # Trade compute for memory
    "lora_rank": 8,  # Smaller LoRA = less memory
}

# Models that need cache disabled due to DynamicCache issues
MODELS_NO_CACHE = {"phi-3-mini"}

# Model HuggingFace IDs
MODEL_HF_IDS = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "yi-1.5-6b": "01-ai/Yi-1.5-6B-Chat",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "falcon-mamba-7b": "tiiuae/falcon-mamba-7b-instruct",
    "ministral-8b": "mistralai/Ministral-8B-Instruct-2410",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}

TASK_FILES = {
    "T1": "tasks/t1_expanded.jsonl",
    "T2": "tasks/t2_expanded.jsonl",
    "T3": "tasks/t3_tools.jsonl",
    "T4": "tasks/t4_bfcl.jsonl",
    "T5": "tasks/t5_swebench.jsonl",
    "T6": "tasks/public_gsm8k.jsonl",
    "Mixed": "tasks/t1t5_balanced.jsonl",
}


def parse_run_key(run_key: str) -> Dict[str, Any]:
    """Parse run key like 'llama-3.2-3b_T5_seed42' into components."""
    parts = run_key.rsplit("_", 2)
    if len(parts) == 3:
        model, task, seed_str = parts
        seed = int(seed_str.replace("seed", ""))
        return {"model": model, "task": task, "seed": seed}
    return {}


def load_progress_state(input_dir: Path) -> Dict[str, Any]:
    """Load progress state from JSON file."""
    state_file = input_dir / "progress_state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"Progress state not found: {state_file}")
    with open(state_file) as f:
        return json.load(f)


def save_progress_state(input_dir: Path, state: Dict[str, Any]):
    """Save progress state to JSON file."""
    state_file = input_dir / "progress_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def get_failed_runs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get list of failed runs with their details."""
    failed = []
    for run_key, run_data in state.get("runs", {}).items():
        status = run_data.get("status", "")
        if status in ("oom_failed", "error"):
            parsed = parse_run_key(run_key)
            if parsed:
                failed.append({
                    "run_key": run_key,
                    "status": status,
                    "error": run_data.get("error_message", ""),
                    **parsed,
                })
    return failed


def build_retry_command(
    run: Dict[str, Any],
    input_dir: Path,
    steps: int = 1000,
    checkpoint_every: int = 100,
) -> List[str]:
    """Build training command with OOM-safe settings or cache fix."""
    model = run["model"]
    task = run["task"]
    seed = run["seed"]
    status = run.get("status", "")

    run_dir = input_dir / model / task / f"seed_{seed}"

    cmd = [
        sys.executable, "-m", "agent_stable_slo.train.grpo_train_loop",
        "--base-model", MODEL_HF_IDS[model],
        "--tasks", TASK_FILES[task],
        "--out", str(run_dir),
        "--steps", str(steps),
        "--seed", str(seed),
        "--checkpoint-every", str(checkpoint_every),
        "--repro",
        "--cache-dataset",
        "--allow-dataset-drift",
    ]

    # For OOM failures, use memory-safe settings
    if status == "oom_failed":
        cmd.extend([
            "--gradient-accumulation", str(OOM_RETRY_SETTINGS["gradient_accumulation"]),
            "--load-in-4bit", "true",
            "--max-new-tokens", str(OOM_RETRY_SETTINGS["max_new_tokens"]),
            "--max-prompt-len", str(OOM_RETRY_SETTINGS["max_prompt_len"]),
            "--lora-rank", str(OOM_RETRY_SETTINGS["lora_rank"]),
            "--gradient-checkpointing",
        ])
    else:
        cmd.extend(["--gradient-accumulation", "1"])

    # For models with DynamicCache issues
    if model in MODELS_NO_CACHE:
        cmd.append("--no-use-cache")

    return cmd


def reset_run_status(state: Dict[str, Any], run_key: str):
    """Reset a failed run to pending status."""
    if run_key in state["runs"]:
        state["runs"][run_key] = {
            "status": "pending",
            "last_checkpoint": 0,
            "last_50_valid_pct": 0,
            "avg_reward": 0,
            "total_steps": 0,
            "duration_seconds": None,
            "error_message": None,
            "started_at": None,
            "completed_at": None,
            "retry_with_oom_settings": True,
        }


def run_training(cmd: List[str], run_key: str, verbose: bool = False) -> tuple[bool, str]:
    """Execute training command and return success status."""
    print(f"  Command: {' '.join(cmd[:8])}...")

    try:
        if verbose:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in process.stdout:
                print(f"    {line}", end="")
            process.wait()
            return_code = process.returncode
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            return_code = result.returncode
            if return_code != 0:
                # Check for OOM again
                output = result.stdout + result.stderr
                if "CUDA out of memory" in output or "OutOfMemoryError" in output:
                    return False, "oom_again"
                return False, f"error: {return_code}"

        return return_code == 0, "completed" if return_code == 0 else f"error: {return_code}"

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"exception: {e}"


def main():
    parser = argparse.ArgumentParser(description="Retry failed P2 training runs")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with progress_state.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be retried without executing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show training output",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Training steps (default: 1000)",
    )
    args = parser.parse_args()

    print(f"Loading progress from: {args.input}")
    state = load_progress_state(args.input)

    failed_runs = get_failed_runs(state)

    if not failed_runs:
        print("No failed runs to retry!")
        return

    print(f"\nFound {len(failed_runs)} failed runs:")
    for run in failed_runs:
        print(f"  {run['run_key']}: {run['status']} - {run['error']}")

    print(f"\nRetry settings:")
    for k, v in OOM_RETRY_SETTINGS.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for run in failed_runs:
            cmd = build_retry_command(run, args.input, args.steps)
            print(f"\n{run['run_key']}:")
            print(f"  {' '.join(cmd)}")
        print("\nDry run complete. Remove --dry-run to execute.")
        return

    print(f"\nStarting retries...")
    for i, run in enumerate(failed_runs, 1):
        print(f"\n[{i}/{len(failed_runs)}] Retrying: {run['run_key']}")
        print(f"  Model: {run['model']}")
        print(f"  Task: {run['task']}")
        print(f"  Seed: {run['seed']}")
        print(f"  Settings: 4-bit, grad_accum=4, max_tokens=128")

        # Reset status to in_progress
        state["runs"][run["run_key"]]["status"] = "in_progress"
        state["runs"][run["run_key"]]["started_at"] = datetime.now().isoformat()
        state["runs"][run["run_key"]]["retry_with_oom_settings"] = True
        save_progress_state(args.input, state)

        cmd = build_retry_command(run, args.input, args.steps)
        start_time = time.time()
        success, message = run_training(cmd, run["run_key"], args.verbose)
        duration = time.time() - start_time

        if success:
            print(f"  Result: SUCCESS ({duration/60:.1f} min)")
            state["runs"][run["run_key"]]["status"] = "completed"
            state["runs"][run["run_key"]]["duration_seconds"] = duration
            state["runs"][run["run_key"]]["completed_at"] = datetime.now().isoformat()
            state["timing"]["completed_runs"] = sum(
                1 for r in state["runs"].values() if r.get("status") == "completed"
            )
        else:
            print(f"  Result: FAILED - {message}")
            if message == "oom_again":
                state["runs"][run["run_key"]]["status"] = "oom_failed"
                state["runs"][run["run_key"]]["error_message"] = "OOM even with 4-bit + grad_accum=4"
            else:
                state["runs"][run["run_key"]]["status"] = "error"
                state["runs"][run["run_key"]]["error_message"] = message

        save_progress_state(args.input, state)

    print("\n=== Retry Complete ===")
    completed = sum(1 for r in state["runs"].values() if r.get("status") == "completed")
    failed = sum(1 for r in state["runs"].values() if r.get("status") in ("oom_failed", "error"))
    print(f"Completed: {completed}/{len(state['runs'])}")
    print(f"Still failed: {failed}")


if __name__ == "__main__":
    main()
