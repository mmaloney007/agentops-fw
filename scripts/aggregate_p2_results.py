#!/usr/bin/env python3
"""
Aggregate P2 Training Results

Extracts metrics from completed training runs and generates:
- aggregated_results.json - Full metrics for all runs
- latex_tables/table_training.tex - Paper-ready LaTeX table
- training_curves.json - Step-by-step data for plots
- summary_by_model.csv - Per-model summary

Usage:
    python scripts/aggregate_p2_results.py --input out/p2_training
    python scripts/aggregate_p2_results.py --input out/p2_training --latex --csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_train_log(log_path: Path) -> List[Dict[str, Any]]:
    """Load and parse train_log.jsonl."""
    if not log_path.exists():
        return []

    entries = []
    with open(log_path, "r") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def extract_run_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract all metrics from a single training run."""
    log_path = run_dir / "train_log.jsonl"
    manifest_path = run_dir / "manifest.json"

    entries = load_train_log(log_path)
    if not entries:
        return None

    # Load manifest for metadata
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    # Extract per-step metrics
    steps = [e.get("step", 0) for e in entries]
    rewards = [e.get("reward", 0) for e in entries]
    json_valids = [e.get("json_valid", 0) for e in entries]
    schema_valids = [e.get("schema_valid", 0) for e in entries]
    latencies = [e.get("latency_ms", 0) for e in entries if e.get("latency_ms")]
    tokens_out = [e.get("tokens_out", 0) for e in entries if e.get("tokens_out")]

    # Compute summary metrics
    total_steps = len(steps)
    last_50 = json_valids[-50:] if len(json_valids) >= 50 else json_valids
    last_100 = json_valids[-100:] if len(json_valids) >= 100 else json_valids

    metrics = {
        # Run metadata
        "model": manifest.get("model", run_dir.parent.parent.name),
        "task": manifest.get("task", run_dir.parent.name),
        "seed": manifest.get("seed", int(run_dir.name.split("_")[-1])),

        # Core metrics
        "total_steps": total_steps,
        "json_valid_count": sum(json_valids),
        "json_valid_pct": sum(json_valids) / len(json_valids) * 100 if json_valids else 0,
        "last_50_valid_pct": sum(last_50) / len(last_50) * 100 if last_50 else 0,
        "last_100_valid_pct": sum(last_100) / len(last_100) * 100 if last_100 else 0,

        # Reward metrics
        "avg_reward": statistics.mean(rewards) if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
        "reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0,

        # Learning trajectory
        "first_10_avg_reward": statistics.mean(rewards[:10]) if len(rewards) >= 10 else 0,
        "last_10_avg_reward": statistics.mean(rewards[-10:]) if len(rewards) >= 10 else 0,
        "reward_improvement": (
            statistics.mean(rewards[-10:]) - statistics.mean(rewards[:10])
            if len(rewards) >= 20 else 0
        ),

        # Latency metrics
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0,
        "p95_latency_ms": (
            sorted(latencies)[int(len(latencies) * 0.95)]
            if len(latencies) >= 20 else (max(latencies) if latencies else 0)
        ),

        # Token metrics
        "avg_tokens_out": statistics.mean(tokens_out) if tokens_out else 0,
        "total_tokens_out": sum(tokens_out),

        # Learning detection
        "learning_detected": (
            statistics.mean(rewards[-10:]) > statistics.mean(rewards[:10]) + 0.1
            if len(rewards) >= 20 else False
        ),

        # Run info
        "run_dir": str(run_dir),
        "started_at": manifest.get("started_at"),
    }

    return metrics


def extract_training_curve(run_dir: Path) -> List[Dict[str, Any]]:
    """Extract step-by-step data for training curve plotting."""
    entries = load_train_log(run_dir / "train_log.jsonl")

    curve = []
    for e in entries:
        curve.append({
            "step": e.get("step", 0),
            "reward": e.get("reward", 0),
            "json_valid": e.get("json_valid", 0),
            "latency_ms": e.get("latency_ms", 0),
            "tokens_out": e.get("tokens_out", 0),
        })
    return curve


def aggregate_all_runs(input_dir: Path) -> Dict[str, Any]:
    """Aggregate metrics from all runs in the directory."""
    results = {
        "metadata": {
            "input_dir": str(input_dir),
            "aggregated_at": None,
        },
        "runs": [],
        "training_curves": {},
        "summary_by_model": {},
        "summary_by_task": {},
        "summary_overall": {},
    }

    # Find all run directories (model/task/seed_N pattern)
    run_dirs = []
    for model_dir in input_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        for task_dir in model_dir.iterdir():
            if not task_dir.is_dir():
                continue
            for seed_dir in task_dir.iterdir():
                if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                    run_dirs.append(seed_dir)

    print(f"Found {len(run_dirs)} run directories")

    # Extract metrics from each run
    for run_dir in sorted(run_dirs):
        metrics = extract_run_metrics(run_dir)
        if metrics:
            results["runs"].append(metrics)

            # Extract training curve
            curve_key = f"{metrics['model']}_{metrics['task']}_seed{metrics['seed']}"
            curve = extract_training_curve(run_dir)
            if curve:
                results["training_curves"][curve_key] = curve

    # Compute summaries
    results["summary_by_model"] = compute_model_summary(results["runs"])
    results["summary_by_task"] = compute_task_summary(results["runs"])
    results["summary_overall"] = compute_overall_summary(results["runs"])

    from datetime import datetime
    results["metadata"]["aggregated_at"] = datetime.now().isoformat()
    results["metadata"]["total_runs"] = len(results["runs"])

    return results


def compute_model_summary(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute per-model summary statistics."""
    by_model = defaultdict(list)
    for run in runs:
        by_model[run["model"]].append(run)

    summary = {}
    for model, model_runs in by_model.items():
        valid_pcts = [r["last_50_valid_pct"] for r in model_runs]
        rewards = [r["avg_reward"] for r in model_runs]

        summary[model] = {
            "num_runs": len(model_runs),
            "num_tasks": len(set(r["task"] for r in model_runs)),
            "num_seeds": len(set(r["seed"] for r in model_runs)),

            # Validity
            "mean_last_50_valid_pct": statistics.mean(valid_pcts) if valid_pcts else 0,
            "std_last_50_valid_pct": statistics.stdev(valid_pcts) if len(valid_pcts) > 1 else 0,
            "min_last_50_valid_pct": min(valid_pcts) if valid_pcts else 0,
            "max_last_50_valid_pct": max(valid_pcts) if valid_pcts else 0,

            # Reward
            "mean_avg_reward": statistics.mean(rewards) if rewards else 0,
            "std_avg_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0,

            # Learning
            "learning_rate": sum(1 for r in model_runs if r["learning_detected"]) / len(model_runs) if model_runs else 0,
        }

    return dict(summary)


def compute_task_summary(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute per-task summary statistics."""
    by_task = defaultdict(list)
    for run in runs:
        by_task[run["task"]].append(run)

    summary = {}
    for task, task_runs in by_task.items():
        valid_pcts = [r["last_50_valid_pct"] for r in task_runs]
        rewards = [r["avg_reward"] for r in task_runs]

        summary[task] = {
            "num_runs": len(task_runs),
            "mean_last_50_valid_pct": statistics.mean(valid_pcts) if valid_pcts else 0,
            "std_last_50_valid_pct": statistics.stdev(valid_pcts) if len(valid_pcts) > 1 else 0,
            "mean_avg_reward": statistics.mean(rewards) if rewards else 0,
            "learning_rate": sum(1 for r in task_runs if r["learning_detected"]) / len(task_runs) if task_runs else 0,
        }

    return dict(summary)


def compute_overall_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall summary statistics."""
    if not runs:
        return {}

    valid_pcts = [r["last_50_valid_pct"] for r in runs]
    rewards = [r["avg_reward"] for r in runs]

    return {
        "total_runs": len(runs),
        "unique_models": len(set(r["model"] for r in runs)),
        "unique_tasks": len(set(r["task"] for r in runs)),
        "unique_seeds": len(set(r["seed"] for r in runs)),

        "mean_last_50_valid_pct": statistics.mean(valid_pcts),
        "std_last_50_valid_pct": statistics.stdev(valid_pcts) if len(valid_pcts) > 1 else 0,

        "mean_avg_reward": statistics.mean(rewards),
        "std_avg_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0,

        "learning_rate": sum(1 for r in runs if r["learning_detected"]) / len(runs),

        "total_steps_completed": sum(r["total_steps"] for r in runs),
    }


def generate_latex_table(results: Dict[str, Any], output_path: Path):
    """Generate LaTeX table for the paper."""
    model_summary = results.get("summary_by_model", {})

    # Model order (smallest to largest)
    model_order = [
        "llama-3.2-1b", "llama-3.2-3b", "qwen2.5-3b", "phi-3-mini",
        "qwen3-4b", "yi-1.5-6b", "mistral-7b-v0.3", "falcon-mamba-7b",
        "ministral-8b", "llama-3.1-8b", "gemma-2-9b", "gemma-3-12b", "gpt-oss-20b",
    ]

    latex = []
    latex.append(r"% Auto-generated P2 training results table")
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{P2 Training Results: JSON Validity and Learning Detection}")
    latex.append(r"\label{tab:p2_training}")
    latex.append(r"\begin{tabular}{lccccc}")
    latex.append(r"\toprule")
    latex.append(r"Model & Runs & Last-50 Valid\% & Avg Reward & Learning\% \\")
    latex.append(r"\midrule")

    for model in model_order:
        if model not in model_summary:
            continue
        s = model_summary[model]
        model_display = model.replace("-", " ").replace(".", " ").title()
        latex.append(
            f"{model_display} & "
            f"{s['num_runs']} & "
            f"{s['mean_last_50_valid_pct']:.1f} $\\pm$ {s['std_last_50_valid_pct']:.1f} & "
            f"{s['mean_avg_reward']:.2f} & "
            f"{s['learning_rate']*100:.0f}\\% \\\\"
        )

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(latex))

    print(f"Wrote LaTeX table to {output_path}")


def generate_csv_summary(results: Dict[str, Any], output_path: Path):
    """Generate CSV summary for easy analysis."""
    runs = results.get("runs", [])

    if not runs:
        print("No runs to export to CSV")
        return

    # Determine columns from first run
    columns = list(runs[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            writer.writerow(run)

    print(f"Wrote CSV summary to {output_path}")


def generate_model_summary_csv(results: Dict[str, Any], output_path: Path):
    """Generate per-model summary CSV."""
    model_summary = results.get("summary_by_model", {})

    if not model_summary:
        return

    columns = [
        "model", "num_runs", "num_tasks", "num_seeds",
        "mean_last_50_valid_pct", "std_last_50_valid_pct",
        "min_last_50_valid_pct", "max_last_50_valid_pct",
        "mean_avg_reward", "std_avg_reward", "learning_rate",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for model, stats in model_summary.items():
            row = {"model": model, **stats}
            writer.writerow(row)

    print(f"Wrote model summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate P2 training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing training runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for aggregated results (default: input/aggregated_results.json)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX table",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Generate CSV summaries",
    )
    parser.add_argument(
        "--curves",
        action="store_true",
        help="Include full training curves in output (large file)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        return

    print(f"Aggregating results from: {args.input}")

    # Aggregate all results
    results = aggregate_all_runs(args.input)

    # Optionally exclude training curves to reduce file size
    if not args.curves:
        results["training_curves"] = {
            "_note": "Training curves excluded. Use --curves to include."
        }

    # Write main results file
    output_path = args.output or (args.input / "aggregated_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote aggregated results to {output_path}")

    # Print summary
    overall = results.get("summary_overall", {})
    print(f"\n{'='*60}")
    print("Overall Summary")
    print(f"{'='*60}")
    print(f"Total runs: {overall.get('total_runs', 0)}")
    print(f"Models: {overall.get('unique_models', 0)}")
    print(f"Tasks: {overall.get('unique_tasks', 0)}")
    print(f"Mean Last-50 Valid%: {overall.get('mean_last_50_valid_pct', 0):.1f}")
    print(f"Mean Avg Reward: {overall.get('mean_avg_reward', 0):.3f}")
    print(f"Learning Detection Rate: {overall.get('learning_rate', 0)*100:.1f}%")
    print(f"{'='*60}")

    # Generate optional outputs
    if args.latex:
        latex_path = args.input / "latex_tables" / "table_training.tex"
        generate_latex_table(results, latex_path)

    if args.csv:
        csv_path = args.input / "summary_all_runs.csv"
        generate_csv_summary(results, csv_path)

        model_csv_path = args.input / "summary_by_model.csv"
        generate_model_summary_csv(results, model_csv_path)

    # Write training curves separately if requested
    if args.curves and results.get("training_curves"):
        curves_path = args.input / "training_curves.json"
        with open(curves_path, "w") as f:
            json.dump(results["training_curves"], f)
        print(f"Wrote training curves to {curves_path}")


if __name__ == "__main__":
    main()
