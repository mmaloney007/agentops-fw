#!/usr/bin/env python3
"""
P2 Training Matrix Orchestrator

Runs 13 models × 6 tasks × 3 seeds × 1000 steps with:
- Crash recovery and checkpoint resume
- Progress tracking (progress_state.json)
- Time estimation
- OOM detection with fallback strategies
- Paper-ready result capture

Usage:
    # Dry run (preview without training)
    python scripts/run_p2_training_matrix.py --out-dir out/p2_training --steps 1000 --dry-run

    # Full run
    python scripts/run_p2_training_matrix.py --out-dir out/p2_training --steps 1000

    # Resume after crash
    python scripts/run_p2_training_matrix.py --resume out/p2_training

    # Single model smoke test
    python scripts/run_p2_training_matrix.py \
        --out-dir out/p2_smoke_test \
        --models qwen2.5-3b \
        --tasks T1 T2 T3 T4 T5 Mixed \
        --seeds 42 \
        --steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# OOM detection patterns
OOM_PATTERNS = [
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "MPS backend out of memory",
    "RuntimeError: CUDA error: out of memory",
    "OutOfMemoryError",
    "CUDA_ERROR_OUT_OF_MEMORY",
]

# Model configurations (smallest to largest)
MODEL_CONFIGS = {
    "llama-3.2-1b": {
        "hf_model": "meta-llama/Llama-3.2-1B-Instruct",
        "preset": "p2_llama_1b",
        "params_b": 1.0,
        "gradient_accumulation": 1,
        "load_in_4bit": False,
    },
    "llama-3.2-3b": {
        "hf_model": "meta-llama/Llama-3.2-3B-Instruct",
        "preset": "p2_llama_3b",
        "params_b": 3.0,
        "gradient_accumulation": 1,
        "load_in_4bit": False,
    },
    "qwen2.5-3b": {
        "hf_model": "Qwen/Qwen2.5-3B-Instruct",
        "preset": "p2_qwen25_3b",
        "params_b": 3.0,
        "gradient_accumulation": 1,
        "load_in_4bit": False,
    },
    "phi-3-mini": {
        "hf_model": "microsoft/Phi-3-mini-4k-instruct",
        "preset": "p2_phi3_mini",
        "params_b": 3.8,
        "gradient_accumulation": 1,
        "load_in_4bit": False,
        "no_use_cache": True,  # Fix for DynamicCache issue
    },
    "qwen3-4b": {
        "hf_model": "Qwen/Qwen3-4B",
        "preset": "p2_qwen3_4b",
        "params_b": 4.0,
        "gradient_accumulation": 1,
        "load_in_4bit": False,
    },
    "yi-1.5-6b": {
        "hf_model": "01-ai/Yi-1.5-6B-Chat",
        "preset": "p2_yi_6b",
        "params_b": 6.0,
        "gradient_accumulation": 2,
        "load_in_4bit": False,
    },
    "mistral-7b-v0.3": {
        "hf_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "preset": "p2_mistral_7b",
        "params_b": 7.0,
        "gradient_accumulation": 2,
        "load_in_4bit": True,
    },
    "falcon-mamba-7b": {
        "hf_model": "tiiuae/falcon-mamba-7b-instruct",
        "preset": "p2_falcon_mamba",
        "params_b": 7.0,
        "gradient_accumulation": 2,
        "load_in_4bit": True,
    },
    "ministral-8b": {
        "hf_model": "mistralai/Ministral-8B-Instruct-2410",
        "preset": "p2_ministral_8b",
        "params_b": 8.0,
        "gradient_accumulation": 2,
        "load_in_4bit": True,
    },
    "llama-3.1-8b": {
        "hf_model": "meta-llama/Llama-3.1-8B-Instruct",
        "preset": "p2_llama_8b",
        "params_b": 8.0,
        "gradient_accumulation": 2,
        "load_in_4bit": True,
    },
    "gemma-2-9b": {
        "hf_model": "google/gemma-2-9b-it",
        "preset": "p2_gemma_9b",
        "params_b": 9.0,
        "gradient_accumulation": 2,
        "load_in_4bit": True,
    },
    "gemma-3-12b": {
        "hf_model": "google/gemma-3-12b-it",
        "preset": "p2_gemma_12b",
        "params_b": 12.0,
        "gradient_accumulation": 4,
        "load_in_4bit": True,
    },
    "gpt-oss-20b": {
        "hf_model": "openai/gpt-oss-20b",
        "preset": "p2_gpt_oss",
        "params_b": 20.0,
        "gradient_accumulation": 4,
        "load_in_4bit": True,
    },
}

# Task configurations
TASK_CONFIGS = {
    "T1": {
        "file": "tasks/t1_expanded.jsonl",
        "description": "Incident classification (100 samples)",
        "samples": 100,
    },
    "T2": {
        "file": "tasks/t2_expanded.jsonl",
        "description": "Grounded summarization (100 samples)",
        "samples": 100,
    },
    "T3": {
        "file": "tasks/t3_tools.jsonl",
        "description": "Tool selection (500 samples)",
        "samples": 500,
    },
    "T4": {
        "file": "tasks/t4_bfcl.jsonl",
        "description": "Function calling (500 samples)",
        "samples": 500,
    },
    "T5": {
        "file": "tasks/t5_swebench.jsonl",
        "description": "SWE-bench patches (300 samples)",
        "samples": 300,
    },
    "Mixed": {
        "file": "tasks/t1t5_balanced.jsonl",
        "description": "Balanced mix (500 samples)",
        "samples": 500,
    },
}

DEFAULT_SEEDS = [42, 123, 456]


@dataclass
class RunConfig:
    """Configuration for a single training run."""
    model: str
    task: str
    seed: int
    steps: int
    out_dir: Path
    checkpoint_every: int = 100

    @property
    def run_key(self) -> str:
        return f"{self.model}_{self.task}_seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        return self.out_dir / self.model / self.task / f"seed_{self.seed}"


@dataclass
class RunStatus:
    """Status of a single training run."""
    status: str = "pending"  # pending, in_progress, completed, oom_failed, error
    last_checkpoint: int = 0
    last_50_valid_pct: float = 0.0
    avg_reward: float = 0.0
    total_steps: int = 0
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class ProgressState:
    """Full progress state for the training matrix."""
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    runs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=lambda: {
        "completed_runs": 0,
        "total_runs": 0,
        "total_duration_seconds": 0,
        "estimated_remaining_seconds": 0,
    })


class TimeEstimator:
    """Estimates remaining time based on completed runs."""

    def __init__(self):
        self.completed_times: Dict[str, float] = {}  # model -> avg seconds

    def record_completion(self, model: str, duration_seconds: float):
        if model not in self.completed_times:
            self.completed_times[model] = duration_seconds
        else:
            # Running average
            self.completed_times[model] = (
                self.completed_times[model] + duration_seconds
            ) / 2

    def estimate_remaining(
        self,
        pending_runs: List[RunConfig],
        default_seconds_per_run: float = 3600,
    ) -> float:
        """Estimate total remaining time in seconds."""
        total = 0.0
        for run in pending_runs:
            if run.model in self.completed_times:
                total += self.completed_times[run.model]
            else:
                # Estimate based on model size
                model_cfg = MODEL_CONFIGS.get(run.model, {})
                params_b = model_cfg.get("params_b", 3.0)
                # Rough: 30 min per 1B params for 1000 steps
                total += params_b * 30 * 60
        return total


class ProgressTracker:
    """Manages progress_state.json with atomic updates."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.state_file = out_dir / "progress_state.json"
        self.state: ProgressState = ProgressState()
        self.time_estimator = TimeEstimator()

    def load(self) -> bool:
        """Load existing state. Returns True if loaded successfully."""
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            self.state = ProgressState(
                version=data.get("version", "1.0.0"),
                created_at=data.get("created_at", datetime.now().isoformat()),
                config=data.get("config", {}),
                runs=data.get("runs", {}),
                timing=data.get("timing", {}),
            )
            # Reconstruct time estimator from completed runs
            for run_key, run_data in self.state.runs.items():
                if run_data.get("status") == "completed" and run_data.get("duration_seconds"):
                    model = run_key.split("_")[0]
                    self.time_estimator.record_completion(
                        model, run_data["duration_seconds"]
                    )
            return True
        except Exception as e:
            print(f"Warning: Failed to load progress state: {e}")
            return False

    def save(self):
        """Atomically save state to file."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Write to temp file first
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.out_dir,
            suffix=".json",
            delete=False,
        ) as tmp:
            json.dump(asdict(self.state), tmp, indent=2)
            tmp_path = tmp.name
        # Atomic rename
        shutil.move(tmp_path, self.state_file)

    def init_runs(self, runs: List[RunConfig]):
        """Initialize run tracking for all planned runs."""
        self.state.config = {
            "steps": runs[0].steps if runs else 1000,
            "seeds": sorted(set(r.seed for r in runs)),
            "checkpoint_every": runs[0].checkpoint_every if runs else 100,
        }
        for run in runs:
            if run.run_key not in self.state.runs:
                self.state.runs[run.run_key] = asdict(RunStatus())
        self.state.timing["total_runs"] = len(runs)
        self.save()

    def get_pending_runs(self, all_runs: List[RunConfig]) -> List[RunConfig]:
        """Get runs that haven't completed yet."""
        pending = []
        for run in all_runs:
            status = self.state.runs.get(run.run_key, {}).get("status", "pending")
            if status not in ("completed",):
                pending.append(run)
        return pending

    def start_run(self, run: RunConfig):
        """Mark a run as in progress."""
        self.state.runs[run.run_key] = {
            **self.state.runs.get(run.run_key, asdict(RunStatus())),
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
        }
        self.save()

    def update_checkpoint(self, run: RunConfig, checkpoint: int):
        """Update last checkpoint for a run."""
        if run.run_key in self.state.runs:
            self.state.runs[run.run_key]["last_checkpoint"] = checkpoint
            self.save()

    def complete_run(
        self,
        run: RunConfig,
        metrics: Dict[str, Any],
        duration_seconds: float,
    ):
        """Mark a run as completed with metrics."""
        self.state.runs[run.run_key] = {
            **self.state.runs.get(run.run_key, {}),
            "status": "completed",
            "last_50_valid_pct": metrics.get("last_50_valid_pct", 0),
            "avg_reward": metrics.get("avg_reward", 0),
            "total_steps": metrics.get("total_steps", run.steps),
            "duration_seconds": duration_seconds,
            "completed_at": datetime.now().isoformat(),
        }
        self.state.timing["completed_runs"] += 1
        self.state.timing["total_duration_seconds"] += duration_seconds
        self.time_estimator.record_completion(run.model, duration_seconds)

        # Update ETA
        pending = [r for r, d in self.state.runs.items() if d.get("status") != "completed"]
        self.state.timing["estimated_remaining_seconds"] = len(pending) * (
            self.state.timing["total_duration_seconds"] /
            max(1, self.state.timing["completed_runs"])
        )
        self.save()

    def fail_run(self, run: RunConfig, status: str, error_message: str):
        """Mark a run as failed."""
        self.state.runs[run.run_key] = {
            **self.state.runs.get(run.run_key, {}),
            "status": status,
            "error_message": error_message,
            "completed_at": datetime.now().isoformat(),
        }
        self.save()


def detect_oom(output: str) -> bool:
    """Check if output contains OOM error patterns."""
    return any(pattern in output for pattern in OOM_PATTERNS)


def extract_last_checkpoint(run_dir: Path) -> int:
    """Find the last checkpoint in a run directory."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return 0
    checkpoint_dirs = sorted(checkpoints_dir.glob("step_*"))
    if not checkpoint_dirs:
        return 0
    # Extract step number from directory name
    match = re.search(r"step_(\d+)", checkpoint_dirs[-1].name)
    return int(match.group(1)) if match else 0


def extract_metrics_from_log(run_dir: Path) -> Dict[str, Any]:
    """Extract metrics from train_log.jsonl."""
    log_path = run_dir / "train_log.jsonl"
    if not log_path.exists():
        return {}

    steps = []
    rewards = []
    json_valids = []

    with open(log_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                steps.append(entry.get("step", 0))
                rewards.append(entry.get("reward", 0))
                json_valids.append(entry.get("json_valid", 0))
            except json.JSONDecodeError:
                continue

    if not steps:
        return {}

    # Last 50 steps validity
    last_50_valid = json_valids[-50:] if len(json_valids) >= 50 else json_valids
    last_50_valid_pct = sum(last_50_valid) / len(last_50_valid) * 100 if last_50_valid else 0

    return {
        "total_steps": len(steps),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
        "last_50_valid_pct": last_50_valid_pct,
        "total_valid_pct": sum(json_valids) / len(json_valids) * 100 if json_valids else 0,
    }


def build_train_command(run: RunConfig, resume_from: Optional[int] = None, oom_safe: bool = False) -> List[str]:
    """Build the training command for a run.

    Args:
        run: Run configuration
        resume_from: Checkpoint step to resume from
        oom_safe: Use memory-safe settings (4-bit, higher grad accum, lower tokens)
    """
    model_cfg = MODEL_CONFIGS[run.model]
    task_cfg = TASK_CONFIGS[run.task]

    # Use OOM-safe settings for T5/Mixed with models >= 3B (long prompts cause OOM)
    use_oom_safe = oom_safe or (run.task in ("T5", "Mixed") and model_cfg["params_b"] >= 3.0)

    grad_accum = 4 if use_oom_safe else model_cfg["gradient_accumulation"]
    load_4bit = True if use_oom_safe else model_cfg["load_in_4bit"]

    cmd = [
        sys.executable, "-m", "agent_stable_slo.train.grpo_train_loop",
        "--base-model", model_cfg["hf_model"],
        "--tasks", task_cfg["file"],
        "--out", str(run.run_dir),
        "--steps", str(run.steps),
        "--seed", str(run.seed),
        "--checkpoint-every", str(run.checkpoint_every),
        "--gradient-accumulation", str(grad_accum),
        "--repro",
        "--cache-dataset",
        "--allow-dataset-drift",
    ]

    if use_oom_safe:
        # Full memory-saving suite for long-prompt tasks
        cmd.extend([
            "--max-new-tokens", "128",
            "--max-prompt-len", "512",
            "--lora-rank", "8",
            "--gradient-checkpointing",
        ])

    if load_4bit:
        cmd.extend(["--load-in-4bit", "true"])

    # Handle models with cache issues (e.g., Phi-3)
    if model_cfg.get("no_use_cache", False):
        cmd.append("--no-use-cache")

    if resume_from and resume_from > 0:
        checkpoint_path = run.run_dir / "checkpoints" / f"step_{resume_from:06d}"
        if checkpoint_path.exists():
            cmd.extend(["--resume-from", str(checkpoint_path)])

    return cmd


def run_single_training(
    run: RunConfig,
    tracker: ProgressTracker,
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """
    Execute a single training run.
    Returns (success, message).
    """
    # Check for existing checkpoint to resume from
    resume_from = extract_last_checkpoint(run.run_dir)
    if resume_from > 0:
        print(f"  Resuming from checkpoint step {resume_from}")

    cmd = build_train_command(run, resume_from)

    if dry_run:
        print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
        return True, "dry_run"

    # Create output directory
    run.run_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest
    manifest = {
        "model": run.model,
        "task": run.task,
        "seed": run.seed,
        "steps": run.steps,
        "started_at": datetime.now().isoformat(),
        "command": cmd,
        "model_config": MODEL_CONFIGS[run.model],
        "task_config": TASK_CONFIGS[run.task],
    }
    with open(run.run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Start training
    tracker.start_run(run)
    start_time = time.time()

    try:
        if verbose:
            # Stream output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_lines = []
            for line in process.stdout:
                print(line, end="")
                output_lines.append(line)
                # Check for checkpoint saves
                if "Saved checkpoint" in line:
                    match = re.search(r"step_(\d+)", line)
                    if match:
                        tracker.update_checkpoint(run, int(match.group(1)))
            process.wait()
            output = "".join(output_lines)
            return_code = process.returncode
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )
            output = result.stdout + result.stderr
            return_code = result.returncode

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"exception: {e}"

    duration = time.time() - start_time

    # Check for OOM
    if detect_oom(output):
        tracker.fail_run(run, "oom_failed", "CUDA OOM detected")
        return False, "oom"

    # Check return code
    if return_code != 0:
        tracker.fail_run(run, "error", f"Exit code {return_code}")
        # Save error log
        with open(run.run_dir / "error.log", "w") as f:
            f.write(output)
        return False, f"error: exit code {return_code}"

    # Extract metrics and complete
    metrics = extract_metrics_from_log(run.run_dir)
    tracker.complete_run(run, metrics, duration)

    return True, f"completed in {duration:.0f}s"


def print_progress_summary(tracker: ProgressTracker):
    """Print a summary of current progress."""
    completed = sum(1 for r in tracker.state.runs.values() if r.get("status") == "completed")
    in_progress = sum(1 for r in tracker.state.runs.values() if r.get("status") == "in_progress")
    failed = sum(1 for r in tracker.state.runs.values() if r.get("status") in ("oom_failed", "error"))
    pending = sum(1 for r in tracker.state.runs.values() if r.get("status") == "pending")
    total = len(tracker.state.runs)

    print(f"\n{'='*60}")
    print(f"Progress: {completed}/{total} completed ({completed/total*100:.1f}%)")
    print(f"  In progress: {in_progress}")
    print(f"  Failed: {failed}")
    print(f"  Pending: {pending}")

    if tracker.state.timing.get("estimated_remaining_seconds"):
        eta = timedelta(seconds=int(tracker.state.timing["estimated_remaining_seconds"]))
        print(f"  ETA: {eta}")

    print(f"{'='*60}\n")


def generate_all_runs(
    models: List[str],
    tasks: List[str],
    seeds: List[int],
    steps: int,
    out_dir: Path,
    checkpoint_every: int,
) -> List[RunConfig]:
    """Generate all run configurations."""
    runs = []
    for model in models:
        for task in tasks:
            for seed in seeds:
                runs.append(RunConfig(
                    model=model,
                    task=task,
                    seed=seed,
                    steps=steps,
                    out_dir=out_dir,
                    checkpoint_every=checkpoint_every,
                ))
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="P2 Training Matrix Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run
  python scripts/run_p2_training_matrix.py --out-dir out/p2 --dry-run

  # Full run
  python scripts/run_p2_training_matrix.py --out-dir out/p2 --steps 1000

  # Resume
  python scripts/run_p2_training_matrix.py --resume out/p2

  # Smoke test
  python scripts/run_p2_training_matrix.py --out-dir out/smoke --models qwen2.5-3b --steps 50
        """,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for training runs",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from existing output directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to train (default: all 13)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_CONFIGS.keys()),
        choices=list(TASK_CONFIGS.keys()),
        help="Tasks to train on (default: all 6)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds (default: 42 123 456)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Training steps per run (default: 1000)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Checkpoint interval (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without executing training",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream training output",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip task file validation",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.resume:
        out_dir = args.resume
        if not out_dir.exists():
            print(f"Error: Resume directory does not exist: {out_dir}")
            sys.exit(1)
    elif args.out_dir:
        out_dir = args.out_dir
    else:
        print("Error: Must specify --out-dir or --resume")
        sys.exit(1)

    # Validate task files exist
    if not args.skip_validation and not args.dry_run:
        print("Validating task files...")
        missing = []
        for task_name, task_cfg in TASK_CONFIGS.items():
            if task_name in args.tasks:
                task_path = Path(task_cfg["file"])
                if not task_path.exists():
                    missing.append(f"  {task_name}: {task_path}")
        if missing:
            print("Missing task files:")
            for m in missing:
                print(m)
            print("\nRun: python scripts/expand_t1_t2_samples.py to generate T1")
            sys.exit(1)
        print("All task files found.")

    # Initialize tracker
    tracker = ProgressTracker(out_dir)

    # Load existing state or initialize
    if args.resume or tracker.state_file.exists():
        if tracker.load():
            print(f"Loaded existing progress from {tracker.state_file}")
        else:
            print("Starting fresh (no valid state found)")

    # Generate all runs
    all_runs = generate_all_runs(
        models=args.models,
        tasks=args.tasks,
        seeds=args.seeds,
        steps=args.steps,
        out_dir=out_dir,
        checkpoint_every=args.checkpoint_every,
    )

    # Initialize tracking if new
    if not args.resume:
        tracker.init_runs(all_runs)

    # Get pending runs
    pending_runs = tracker.get_pending_runs(all_runs)

    print(f"\n{'='*60}")
    print(f"P2 Training Matrix")
    print(f"{'='*60}")
    print(f"Models: {len(args.models)}")
    print(f"Tasks: {len(args.tasks)}")
    print(f"Seeds: {args.seeds}")
    print(f"Steps: {args.steps}")
    print(f"Total runs: {len(all_runs)}")
    print(f"Pending runs: {len(pending_runs)}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("=== DRY RUN MODE ===\n")
        for i, run in enumerate(pending_runs[:10], 1):
            print(f"{i}. {run.run_key}")
            cmd = build_train_command(run)
            print(f"   Command: {' '.join(cmd[:8])}...")
        if len(pending_runs) > 10:
            print(f"   ... and {len(pending_runs) - 10} more runs")
        print("\nDry run complete. Remove --dry-run to execute.")
        return

    # Execute pending runs
    for i, run in enumerate(pending_runs, 1):
        print(f"\n[{i}/{len(pending_runs)}] Starting: {run.run_key}")
        print(f"  Model: {run.model} ({MODEL_CONFIGS[run.model]['params_b']}B)")
        print(f"  Task: {run.task}")
        print(f"  Seed: {run.seed}")
        print(f"  Output: {run.run_dir}")

        success, message = run_single_training(
            run, tracker, dry_run=False, verbose=args.verbose
        )

        if success:
            print(f"  Result: SUCCESS - {message}")
        else:
            print(f"  Result: FAILED - {message}")
            if message == "oom":
                print("  Suggestion: Try with --gradient-accumulation 4 or --load-in-4bit")

        print_progress_summary(tracker)

    # Check for failed runs and offer to retry
    failed_runs = [
        (k, v) for k, v in tracker.state.runs.items()
        if v.get("status") in ("oom_failed", "error")
    ]

    print("\n=== Training Matrix Complete ===")
    print(f"Results saved to: {out_dir}")
    print(f"Progress state: {tracker.state_file}")

    if failed_runs:
        print(f"\n{len(failed_runs)} runs failed. To retry with OOM-safe settings:")
        print(f"  python scripts/retry_failed_runs.py --input {out_dir}")

    print(f"\nTo aggregate results:")
    print(f"  python scripts/aggregate_p2_results.py --input {out_dir}")


if __name__ == "__main__":
    main()
