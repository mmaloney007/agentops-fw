#!/usr/bin/env python3
"""P6 Analysis: Cross-Platform Training Comparison (MLX vs CUDA).

Compares GRPO training outcomes between Apple Silicon (MLX) and NVIDIA (CUDA)
backends across matching models, tasks, and seeds.

Analysis outputs:
  - Final validity % by model x task (MLX vs CUDA)
  - Capacity threshold: at what model size does learning sustain on each platform?
  - Wall-clock cost comparison
  - Reward trajectory overlay data

Input:
  results/p2_all_runs.csv         -- CUDA training results (P2)
  results/mlx_training/           -- MLX training results (P6)

Output:
  results/p6_analysis/platform_comparison.json
  results/curves/pgfplots/p6_validity_overlay.dat
  results/curves/pgfplots/p6_capacity_threshold.dat
  results/curves/pgfplots/p6_wallclock_comparison.dat

Author: Mike Maloney <mike.maloney@unh.edu>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
CUDA_CSV = ROOT / "results" / "p2_all_runs.csv"
MLX_ROOT = ROOT / "results" / "mlx_training"
OUT_JSON = ROOT / "results" / "p6_analysis" / "platform_comparison.json"
DAT_DIR = ROOT / "results" / "curves" / "pgfplots"

# Sustained learning threshold (matching P4)
SUSTAINED_VALIDITY_MIN = 60.0  # percent


def load_cuda_results(csv_path: Path) -> List[Dict[str, Any]]:
    """Load P2 CUDA training results from CSV."""
    if not csv_path.exists():
        print(f"[warn] CUDA results not found: {csv_path}")
        return []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_mlx_results(mlx_root: Path) -> List[Dict[str, Any]]:
    """Load MLX training results from per-run summary files."""
    results = []
    if not mlx_root.exists():
        print(f"[warn] MLX results not found: {mlx_root}")
        return results

    for summary_path in mlx_root.rglob("run_summary.json"):
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
            if "error" in data:
                continue
            results.append(data)
        except Exception as exc:
            print(f"[warn] Failed to load {summary_path}: {exc}")
    return results


def load_mlx_trajectories(mlx_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load step-by-step reward trajectories from MLX training logs."""
    trajectories: Dict[str, List[Dict[str, Any]]] = {}
    if not mlx_root.exists():
        return trajectories

    for log_path in mlx_root.rglob("train_log.jsonl"):
        # Derive key from path: model/task/seed
        parts = log_path.relative_to(mlx_root).parts
        if len(parts) >= 3:
            key = "/".join(parts[:3])
        else:
            key = str(log_path.relative_to(mlx_root))

        steps = []
        with open(log_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                steps.append({
                    "step": rec.get("step", 0),
                    "reward": rec.get("mean_reward", rec.get("reward", 0)),
                    "json_valid": rec.get("json_valid", 0),
                })
        trajectories[key] = steps
    return trajectories


def compare_validity(
    cuda_runs: List[Dict[str, Any]],
    mlx_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare final validity % by model x task across platforms."""
    cuda_by_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    mlx_by_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for run in cuda_runs:
        model = run.get("model", "")
        task = run.get("task", "")
        try:
            valid_pct = float(run.get("json_valid_pct", 0))
        except (ValueError, TypeError):
            continue
        cuda_by_model[model][task].append(valid_pct)

    for run in mlx_runs:
        model = run.get("model", "")
        task = run.get("task_label", "")
        valid_pct = run.get("json_valid_pct", 0)
        mlx_by_model[model][task].append(valid_pct)

    # Build comparison table
    all_models = sorted(set(list(cuda_by_model.keys()) + list(mlx_by_model.keys())))
    comparison = {}
    for model in all_models:
        cuda_tasks = cuda_by_model.get(model, {})
        mlx_tasks = mlx_by_model.get(model, {})
        all_tasks = sorted(set(list(cuda_tasks.keys()) + list(mlx_tasks.keys())))

        model_comp = {}
        for task in all_tasks:
            cuda_vals = cuda_tasks.get(task, [])
            mlx_vals = mlx_tasks.get(task, [])
            model_comp[task] = {
                "cuda_mean": round(sum(cuda_vals) / len(cuda_vals), 1) if cuda_vals else None,
                "mlx_mean": round(sum(mlx_vals) / len(mlx_vals), 1) if mlx_vals else None,
                "cuda_n": len(cuda_vals),
                "mlx_n": len(mlx_vals),
                "delta": None,
            }
            if cuda_vals and mlx_vals:
                model_comp[task]["delta"] = round(
                    model_comp[task]["mlx_mean"] - model_comp[task]["cuda_mean"], 1
                )
        comparison[model] = model_comp

    return comparison


def compute_capacity_thresholds(
    cuda_runs: List[Dict[str, Any]],
    mlx_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Determine at what model size learning sustains on each platform."""
    # Model size mapping
    SIZE_MAP = {
        "llama-3.2-1b": 1.0, "llama-3.2-3b": 3.0, "qwen2.5-3b": 3.0,
        "phi-3-mini": 3.8, "qwen3-4b": 4.0, "yi-1.5-6b": 6.0,
        "mistral-7b": 7.0, "mistral-7b-v0.3": 7.0,
        "falcon-mamba-7b": 7.0, "ministral-8b": 8.0, "llama-3.1-8b": 8.0,
        "gemma-2-9b": 9.0, "gemma-3-12b": 12.0,
    }

    def _aggregate(runs: List[Dict[str, Any]], platform: str) -> Dict[str, Dict[str, Any]]:
        by_model: Dict[str, List[float]] = defaultdict(list)
        for run in runs:
            model = run.get("model", "")
            try:
                valid = float(run.get("json_valid_pct", 0))
            except (ValueError, TypeError):
                continue
            by_model[model].append(valid)

        result = {}
        for model, vals in sorted(by_model.items()):
            mean_valid = sum(vals) / len(vals)
            sustained = mean_valid >= SUSTAINED_VALIDITY_MIN
            size_b = SIZE_MAP.get(model, 0)
            result[model] = {
                "size_b": size_b,
                "mean_validity_pct": round(mean_valid, 1),
                "sustained": sustained,
                "n_runs": len(vals),
                "platform": platform,
            }
        return result

    cuda_thresholds = _aggregate(cuda_runs, "cuda")
    mlx_thresholds = _aggregate(mlx_runs, "mlx")

    # Find threshold (smallest model that sustains)
    def _find_threshold(data: Dict[str, Dict[str, Any]]) -> Optional[float]:
        sustained = [v["size_b"] for v in data.values() if v["sustained"] and v["size_b"] > 0]
        return min(sustained) if sustained else None

    return {
        "cuda": cuda_thresholds,
        "mlx": mlx_thresholds,
        "cuda_threshold_b": _find_threshold(cuda_thresholds),
        "mlx_threshold_b": _find_threshold(mlx_thresholds),
        "sustained_min_pct": SUSTAINED_VALIDITY_MIN,
    }


def compare_wallclock(
    cuda_runs: List[Dict[str, Any]],
    mlx_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare wall-clock training cost per model."""
    cuda_times: Dict[str, List[float]] = defaultdict(list)
    mlx_times: Dict[str, List[float]] = defaultdict(list)

    for run in cuda_runs:
        model = run.get("model", "")
        try:
            wall = float(run.get("wallclock_s", 0))
        except (ValueError, TypeError):
            continue
        if wall > 0:
            cuda_times[model].append(wall)

    for run in mlx_runs:
        model = run.get("model", "")
        wall = run.get("wallclock_s", 0)
        if wall and wall > 0:
            mlx_times[model].append(wall)

    all_models = sorted(set(list(cuda_times.keys()) + list(mlx_times.keys())))
    comparison = {}
    for model in all_models:
        ct = cuda_times.get(model, [])
        mt = mlx_times.get(model, [])
        comparison[model] = {
            "cuda_mean_s": round(sum(ct) / len(ct), 1) if ct else None,
            "mlx_mean_s": round(sum(mt) / len(mt), 1) if mt else None,
            "speedup": None,
        }
        if ct and mt:
            cuda_mean = sum(ct) / len(ct)
            mlx_mean = sum(mt) / len(mt)
            if mlx_mean > 0:
                comparison[model]["speedup"] = round(cuda_mean / mlx_mean, 2)

    return comparison


def write_pgfplots(
    validity_comp: Dict[str, Any],
    thresholds: Dict[str, Any],
    wallclock: Dict[str, Any],
) -> None:
    """Write pgfplots .dat files for LaTeX figures."""
    DAT_DIR.mkdir(parents=True, exist_ok=True)

    # Validity overlay
    with open(DAT_DIR / "p6_validity_overlay.dat", "w") as f:
        f.write("model cuda_validity mlx_validity delta\n")
        for model, tasks in sorted(validity_comp.items()):
            # Aggregate across tasks
            cuda_vals = [t["cuda_mean"] for t in tasks.values() if t["cuda_mean"] is not None]
            mlx_vals = [t["mlx_mean"] for t in tasks.values() if t["mlx_mean"] is not None]
            cuda_avg = sum(cuda_vals) / len(cuda_vals) if cuda_vals else 0
            mlx_avg = sum(mlx_vals) / len(mlx_vals) if mlx_vals else 0
            delta = mlx_avg - cuda_avg if cuda_vals and mlx_vals else 0
            f.write(f"{model} {cuda_avg:.1f} {mlx_avg:.1f} {delta:.1f}\n")
    print(f"Wrote {DAT_DIR / 'p6_validity_overlay.dat'}")

    # Capacity threshold
    with open(DAT_DIR / "p6_capacity_threshold.dat", "w") as f:
        f.write("model size_b cuda_validity mlx_validity cuda_sustained mlx_sustained\n")
        all_models = sorted(
            set(list(thresholds["cuda"].keys()) + list(thresholds["mlx"].keys()))
        )
        for model in all_models:
            cuda = thresholds["cuda"].get(model, {})
            mlx = thresholds["mlx"].get(model, {})
            size_b = cuda.get("size_b", mlx.get("size_b", 0))
            cv = cuda.get("mean_validity_pct", 0)
            mv = mlx.get("mean_validity_pct", 0)
            cs = 1 if cuda.get("sustained", False) else 0
            ms = 1 if mlx.get("sustained", False) else 0
            f.write(f"{model} {size_b:.1f} {cv:.1f} {mv:.1f} {cs} {ms}\n")
    print(f"Wrote {DAT_DIR / 'p6_capacity_threshold.dat'}")

    # Wall-clock comparison
    with open(DAT_DIR / "p6_wallclock_comparison.dat", "w") as f:
        f.write("model cuda_s mlx_s speedup\n")
        for model, data in sorted(wallclock.items()):
            cuda_s = data["cuda_mean_s"] or 0
            mlx_s = data["mlx_mean_s"] or 0
            speedup = data["speedup"] or 0
            f.write(f"{model} {cuda_s:.1f} {mlx_s:.1f} {speedup:.2f}\n")
    print(f"Wrote {DAT_DIR / 'p6_wallclock_comparison.dat'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P6: Compare MLX vs CUDA GRPO training results."
    )
    parser.add_argument(
        "--cuda-csv",
        type=str,
        default=str(CUDA_CSV),
        help="Path to P2 CUDA results CSV.",
    )
    parser.add_argument(
        "--mlx-dir",
        type=str,
        default=str(MLX_ROOT),
        help="Path to MLX training results directory.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT_JSON),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    cuda_csv = Path(args.cuda_csv)
    mlx_dir = Path(args.mlx_dir)

    print(f"[load] CUDA results: {cuda_csv}")
    cuda_runs = load_cuda_results(cuda_csv)
    print(f"  -> {len(cuda_runs)} CUDA runs")

    print(f"[load] MLX results: {mlx_dir}")
    mlx_runs = load_mlx_results(mlx_dir)
    print(f"  -> {len(mlx_runs)} MLX runs")

    if not cuda_runs and not mlx_runs:
        print("[warn] No training results found on either platform. Nothing to compare.")
        # Write empty output for pipeline compatibility
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"error": "no_data", "cuda_n": 0, "mlx_n": 0}, f, indent=2)
        return 0

    print("\n[analysis] Comparing validity ...")
    validity = compare_validity(cuda_runs, mlx_runs)

    print("[analysis] Computing capacity thresholds ...")
    thresholds = compute_capacity_thresholds(cuda_runs, mlx_runs)
    print(f"  CUDA threshold: {thresholds['cuda_threshold_b']}B")
    print(f"  MLX  threshold: {thresholds['mlx_threshold_b']}B")

    print("[analysis] Comparing wall-clock costs ...")
    wallclock = compare_wallclock(cuda_runs, mlx_runs)

    # Write main output
    output = {
        "validity_comparison": validity,
        "capacity_thresholds": thresholds,
        "wallclock_comparison": wallclock,
        "metadata": {
            "cuda_n_runs": len(cuda_runs),
            "mlx_n_runs": len(mlx_runs),
            "sustained_threshold_pct": SUSTAINED_VALIDITY_MIN,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")

    # pgfplots
    write_pgfplots(validity, thresholds, wallclock)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
