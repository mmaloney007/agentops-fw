#!/usr/bin/env python3
"""
P2 Learning Curve Analysis
===========================

Scans P2 training runs and classifies each as sustained / transient / flat
based on rolling-window validity metrics.  Produces:

  results/p2_analysis/learning_curves_summary.json   -- per-run metrics
  results/p2_analysis/learning_curves/*.dat           -- pgfplots curves
  stdout summary table

Usage:
    python scripts/p2_analysis/learning_curves.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "out" / "p2_training_20260124"
OUT_DIR = BASE_DIR / "results" / "p2_analysis"
CURVES_DIR = OUT_DIR / "learning_curves"

WINDOW = 50  # rolling-window size for smoothing

# Model-name -> approximate parameter count (billions)
MODEL_SIZES: dict[str, float] = {
    "llama-3.2-1b": 1.0,
    "llama-3.2-3b": 3.0,
    "qwen2.5-3b": 3.0,
    "phi-3-mini": 3.8,
    "qwen3-4b": 4.0,
    "yi-1.5-6b": 6.0,
    "mistral-7b-v0.3": 7.0,
    "falcon-mamba-7b": 7.0,
    "ministral-8b": 8.0,
    "llama-3.1-8b": 8.0,
    "gemma-2-9b": 9.0,
    "gemma-3-12b": 12.0,
    "gpt-oss-20b": 20.0,
}

# Size-group boundaries
def size_group(model: str) -> str:
    b = MODEL_SIZES.get(model, 0)
    if b < 4:
        return "small"
    if b <= 8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rolling_mean(values: list[float], window: int) -> list[float]:
    """Simple rolling average; pads the first (window-1) values with
    expanding-window means so output length == input length."""
    out: list[float] = []
    cumsum = 0.0
    for i, v in enumerate(values):
        cumsum += v
        w = min(i + 1, window)
        if i >= window:
            cumsum -= values[i - window]
        out.append(cumsum / w)
    return out


def load_run(log_path: Path) -> list[dict[str, Any]]:
    """Read train_log.jsonl, returning list of dicts with the fields we need."""
    rows: list[dict[str, Any]] = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows.append({
                    "step": int(rec["step"]),
                    "reward": float(rec["reward"]),
                    "json_valid": int(rec["json_valid"]),
                    "latency_ms": float(rec["latency_ms"]),
                })
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"  WARNING: malformed line in {log_path}: {exc}", file=sys.stderr)
    return rows


def classify_run(
    smoothed_validity: list[float],
) -> str:
    """
    sustained  -- last-50 mean validity >= 0.70
    transient  -- peak smoothed validity >= 0.50 but last-50 < 0.70
    flat       -- never reaches 0.50
    """
    if not smoothed_validity:
        return "flat"
    last_50 = smoothed_validity[-min(50, len(smoothed_validity)):]
    last_50_mean = sum(last_50) / len(last_50)
    peak = max(smoothed_validity)
    if last_50_mean >= 0.70:
        return "sustained"
    if peak >= 0.50:
        return "transient"
    return "flat"


def steps_to_threshold(smoothed_validity: list[float], threshold: float) -> int | None:
    """Return first step index where smoothed validity >= threshold, or None."""
    for i, v in enumerate(smoothed_validity):
        if v >= threshold:
            return i
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.is_dir():
        print(f"ERROR: data directory not found: {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    # Discover runs: model/task/seed/train_log.jsonl  (skip checkpoints/)
    run_results: list[dict[str, Any]] = []

    model_dirs = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir()],
        key=lambda d: MODEL_SIZES.get(d.name, 99),
    )

    for model_dir in model_dirs:
        model = model_dir.name
        if model not in MODEL_SIZES:
            continue  # skip non-model dirs (latex_tables, etc.)
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            for seed_dir in sorted(task_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                seed = seed_dir.name
                log_path = seed_dir / "train_log.jsonl"
                if not log_path.exists():
                    continue

                rows = load_run(log_path)
                n_steps = len(rows)

                if n_steps == 0:
                    run_results.append({
                        "model": model,
                        "task": task,
                        "seed": seed,
                        "size_group": size_group(model),
                        "params_b": MODEL_SIZES[model],
                        "n_steps": 0,
                        "classification": "flat",
                        "steps_to_50pct": None,
                        "peak_validity": 0.0,
                        "last_50_validity": 0.0,
                        "mean_reward": 0.0,
                        "mean_latency_ms": 0.0,
                    })
                    continue

                rewards = [r["reward"] for r in rows]
                valids = [float(r["json_valid"]) for r in rows]
                latencies = [r["latency_ms"] for r in rows]

                sm_reward = rolling_mean(rewards, WINDOW)
                sm_valid = rolling_mean(valids, WINDOW)

                last_n = min(50, n_steps)
                last_50_vals = valids[-last_n:]
                last_50_validity = sum(last_50_vals) / len(last_50_vals)

                peak_validity = max(sm_valid)
                cls = classify_run(sm_valid)
                s2_50 = steps_to_threshold(sm_valid, 0.50)

                run_results.append({
                    "model": model,
                    "task": task,
                    "seed": seed,
                    "size_group": size_group(model),
                    "params_b": MODEL_SIZES[model],
                    "n_steps": n_steps,
                    "classification": cls,
                    "steps_to_50pct": s2_50,
                    "peak_validity": round(peak_validity, 4),
                    "last_50_validity": round(last_50_validity, 4),
                    "mean_reward": round(sum(rewards) / len(rewards), 4),
                    "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
                })

    # ------------------------------------------------------------------
    # Write summary JSON
    # ------------------------------------------------------------------
    summary_path = OUT_DIR / "learning_curves_summary.json"
    with open(summary_path, "w") as f:
        json.dump(run_results, f, indent=2)
    print(f"Wrote {summary_path}  ({len(run_results)} runs)")

    # ------------------------------------------------------------------
    # Select representative curves for .dat files
    # ------------------------------------------------------------------
    # For each size group, pick one example of each classification type
    # (sustained, transient, flat).  Prefer seed_42 for reproducibility,
    # and pick the run with the most steps.
    # ------------------------------------------------------------------

    # Group runs by (size_group, classification)
    from collections import defaultdict
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in run_results:
        if r["n_steps"] > 0:
            grouped[(r["size_group"], r["classification"])].append(r)

    dat_files_written = 0
    for (sg, cls), candidates in sorted(grouped.items()):
        # prefer seed_42, then most steps
        candidates.sort(key=lambda r: (r["seed"] == "seed_42", r["n_steps"]), reverse=True)
        rep = candidates[0]

        # Re-load the data to write .dat
        log_path = (
            DATA_DIR / rep["model"] / rep["task"] / rep["seed"] / "train_log.jsonl"
        )
        rows = load_run(log_path)
        if not rows:
            continue

        rewards = [r["reward"] for r in rows]
        valids = [float(r["json_valid"]) for r in rows]
        sm_reward = rolling_mean(rewards, WINDOW)
        sm_valid = rolling_mean(valids, WINDOW)

        fname = f"{sg}_{cls}_{rep['model']}_{rep['task']}_{rep['seed']}.dat"
        dat_path = CURVES_DIR / fname
        with open(dat_path, "w") as f:
            f.write("# step  reward  validity\n")
            f.write(f"# model={rep['model']}  task={rep['task']}  seed={rep['seed']}\n")
            f.write(f"# classification={cls}  size_group={sg}\n")
            for i in range(len(rows)):
                f.write(f"{rows[i]['step']}  {sm_reward[i]:.4f}  {sm_valid[i]:.4f}\n")
        dat_files_written += 1

    print(f"Wrote {dat_files_written} .dat files to {CURVES_DIR}/")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 110)
    print(f"{'Model':<20s} {'Task':<7s} {'Seed':<10s} {'Group':<8s} "
          f"{'Steps':>6s} {'Class':<11s} {'Peak':>6s} {'Last50':>7s} "
          f"{'StepsTo50':>10s} {'MeanRwd':>8s} {'MeanLat':>9s}")
    print("-" * 110)

    # Counters
    total = 0
    cls_counts: dict[str, int] = {"sustained": 0, "transient": 0, "flat": 0}
    group_cls: dict[str, dict[str, int]] = defaultdict(lambda: {"sustained": 0, "transient": 0, "flat": 0})

    for r in run_results:
        total += 1
        cls_counts[r["classification"]] += 1
        group_cls[r["size_group"]][r["classification"]] += 1

        s2 = str(r["steps_to_50pct"]) if r["steps_to_50pct"] is not None else "-"
        print(
            f"{r['model']:<20s} {r['task']:<7s} {r['seed']:<10s} {r['size_group']:<8s} "
            f"{r['n_steps']:>6d} {r['classification']:<11s} "
            f"{r['peak_validity']:>6.2%} {r['last_50_validity']:>6.2%} "
            f"{s2:>10s} {r['mean_reward']:>8.3f} {r['mean_latency_ms']:>8.1f}ms"
        )

    print("-" * 110)
    print()

    # Overall summary
    print("CLASSIFICATION SUMMARY")
    print(f"  Total runs:  {total}")
    for cls_name in ["sustained", "transient", "flat"]:
        pct = cls_counts[cls_name] / total * 100 if total else 0
        print(f"  {cls_name:<11s}:  {cls_counts[cls_name]:>4d}  ({pct:5.1f}%)")
    print()

    # Per size-group breakdown
    print("BY SIZE GROUP")
    for sg in ["small", "medium", "large"]:
        if sg not in group_cls:
            continue
        gc = group_cls[sg]
        sg_total = sum(gc.values())
        print(f"  {sg:<8s} (n={sg_total:>3d}):  "
              f"sustained={gc['sustained']:>3d}  "
              f"transient={gc['transient']:>3d}  "
              f"flat={gc['flat']:>3d}")
    print()


if __name__ == "__main__":
    main()
