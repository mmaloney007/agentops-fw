#!/usr/bin/env python3
"""P6 Analysis: MLX Training Dynamics.

Replicates the P4 training dynamics analyses on MLX training data to
enable direct comparison of learning behaviour across platforms.

Analyses:
  1. Reward decomposition: component breakdown over training
  2. Forgetting matrix: which tasks help/hurt others
  3. Curve taxonomy: monotonic/transient/plateau/fail classification
  4. Early prediction: do first-50-step features predict final performance?

Input:  results/mlx_training/
Output: results/p6_analysis/training_dynamics.json
        results/curves/pgfplots/p6_forgetting_heatmap.dat
        results/curves/pgfplots/p6_curve_taxonomy.dat
        results/curves/pgfplots/p6_reward_decomposition.dat

Author: Mike Maloney <mike.maloney@unh.edu>
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
MLX_ROOT = ROOT / "results" / "mlx_training"
OUT_JSON = ROOT / "results" / "p6_analysis" / "training_dynamics.json"
DAT_DIR = ROOT / "results" / "curves" / "pgfplots"

# Reward weights (matching MLX configs)
LAM_LATENCY = 0.1
MU_COST = 0.01

# Classification thresholds (matching P4)
SUSTAINED_FINAL_MIN = 0.60
TRANSIENT_PEAK_MIN = 0.30
ROBUST_FORGETTING = -0.05
SELECTIVE_FORGETTING = -0.30


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_logs(mlx_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all training JSONL logs keyed by model/task/seed."""
    logs: Dict[str, List[Dict[str, Any]]] = {}
    if not mlx_root.exists():
        print(f"[warn] MLX results directory not found: {mlx_root}")
        return logs

    for log_path in mlx_root.rglob("train_log.jsonl"):
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
                try:
                    rec = json.loads(line)
                    steps.append(rec)
                except json.JSONDecodeError:
                    continue
        if steps:
            logs[key] = steps

    return logs


def load_run_summaries(mlx_root: Path) -> List[Dict[str, Any]]:
    """Load all run_summary.json files."""
    summaries = []
    if not mlx_root.exists():
        return summaries

    for path in mlx_root.rglob("run_summary.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if "error" not in data:
                summaries.append(data)
        except Exception:
            continue
    return summaries


# ---------------------------------------------------------------------------
# 1. Reward Decomposition
# ---------------------------------------------------------------------------

def reward_decomposition(
    summaries: List[Dict[str, Any]],
    logs: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Decompose composite reward into schema/latency/cost components per model."""
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for summary in summaries:
        model = summary.get("model", "unknown")
        task = summary.get("task_label", "")
        seed = summary.get("seed", 0)

        avg_reward = summary.get("avg_reward", 0)
        json_valid_pct = summary.get("json_valid_pct", 0) / 100.0

        # Estimate components
        r_schema = json_valid_pct  # 1.0 * fraction valid

        # Get average latency/tokens from logs if available
        key = f"{model}/{task}/seed{seed}"
        log_steps = logs.get(key, [])
        if log_steps:
            avg_lat = sum(s.get("latency_ms", 0) for s in log_steps) / len(log_steps)
            avg_tok = sum(s.get("tokens_out", 0) for s in log_steps) / len(log_steps)
        else:
            avg_lat = 0
            avg_tok = 0

        r_latency = -LAM_LATENCY * avg_lat / 1000.0
        r_cost = -MU_COST * avg_tok / 100.0
        r_residual = avg_reward - r_schema - r_latency - r_cost

        by_model[model].append({
            "task": task,
            "seed": seed,
            "avg_reward": round(avg_reward, 4),
            "r_schema": round(r_schema, 4),
            "r_latency": round(r_latency, 4),
            "r_cost": round(r_cost, 4),
            "r_residual": round(max(0, r_residual), 4),
        })

    # Aggregate per model
    model_summary = {}
    for model, runs in sorted(by_model.items()):
        n = len(runs)
        model_summary[model] = {
            "n_runs": n,
            "mean_reward": round(sum(r["avg_reward"] for r in runs) / n, 4),
            "mean_r_schema": round(sum(r["r_schema"] for r in runs) / n, 4),
            "mean_r_latency": round(sum(r["r_latency"] for r in runs) / n, 4),
            "mean_r_cost": round(sum(r["r_cost"] for r in runs) / n, 4),
            "mean_r_residual": round(sum(r["r_residual"] for r in runs) / n, 4),
            "runs": runs,
        }

    return {
        "models": model_summary,
        "weights": {"lam_latency": LAM_LATENCY, "mu_cost": MU_COST},
    }


# ---------------------------------------------------------------------------
# 2. Forgetting Matrix
# ---------------------------------------------------------------------------

def forgetting_matrix(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute forgetting matrix: mixed vs single-task training outcomes.

    Requires both "Mixed" and single-task runs for comparison.
    """
    by_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for summary in summaries:
        model = summary.get("model", "unknown")
        task = summary.get("task_label", "")
        valid_pct = summary.get("json_valid_pct", 0)
        by_model[model][task].append(valid_pct)

    matrix = []
    for model, tasks in sorted(by_model.items()):
        mixed_vals = tasks.get("Mixed", [])
        mixed_avg = sum(mixed_vals) / len(mixed_vals) if mixed_vals else None

        single_tasks = {t: vals for t, vals in tasks.items() if t != "Mixed"}
        single_avgs = {}
        for task, vals in single_tasks.items():
            single_avgs[task] = round(sum(vals) / len(vals), 1)

        if mixed_avg is not None and single_avgs:
            single_mean = sum(single_avgs.values()) / len(single_avgs)
            delta = (mixed_avg - single_mean) / 100.0  # normalise to 0-1 scale

            if delta > ROBUST_FORGETTING:
                profile = "robust"
            elif delta > SELECTIVE_FORGETTING:
                profile = "selective"
            else:
                profile = "catastrophic"
        else:
            delta = 0
            profile = "insufficient_data"

        matrix.append({
            "model": model,
            "mixed_validity": round(mixed_avg, 1) if mixed_avg is not None else None,
            "single_task_validity": single_avgs,
            "single_avg": round(sum(single_avgs.values()) / max(1, len(single_avgs)), 1) if single_avgs else None,
            "interference_delta": round(delta, 4),
            "profile": profile,
        })

    profiles = [m["profile"] for m in matrix if m["profile"] != "insufficient_data"]
    summary = {
        "n_models": len(matrix),
        "robust": sum(1 for p in profiles if p == "robust"),
        "selective": sum(1 for p in profiles if p == "selective"),
        "catastrophic": sum(1 for p in profiles if p == "catastrophic"),
    }

    return {
        "matrix": matrix,
        "summary": summary,
        "thresholds": {
            "robust": ROBUST_FORGETTING,
            "selective": SELECTIVE_FORGETTING,
        },
    }


# ---------------------------------------------------------------------------
# 3. Curve Taxonomy
# ---------------------------------------------------------------------------

def _extract_validity_series(steps: List[Dict[str, Any]]) -> List[float]:
    """Extract per-step validity (1 or 0) from training log entries."""
    return [float(s.get("json_valid", 0)) for s in steps]


def _rolling_mean(values: List[float], window: int = 50) -> List[float]:
    """Compute rolling mean over a window."""
    if len(values) < window:
        return [sum(values) / max(1, len(values))] * len(values)
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        segment = values[start : i + 1]
        result.append(sum(segment) / len(segment))
    return result


def classify_curve(peak: float, final: float) -> str:
    """Classify a learning curve based on peak and final validity."""
    if final >= SUSTAINED_FINAL_MIN:
        return "sustained"
    elif peak >= TRANSIENT_PEAK_MIN:
        return "transient"
    else:
        return "flat"


def curve_taxonomy(logs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Classify each training run's learning curve."""
    taxonomy_counts: Dict[str, int] = defaultdict(int)
    runs = []

    for key, steps in sorted(logs.items()):
        parts = key.split("/")
        model = parts[0] if len(parts) >= 1 else "unknown"
        task = parts[1] if len(parts) >= 2 else "unknown"
        seed = parts[2] if len(parts) >= 3 else "unknown"

        validity = _extract_validity_series(steps)
        if len(validity) < 10:
            continue

        smoothed = _rolling_mean(validity, window=50)
        peak = max(smoothed)
        final = sum(smoothed[-50:]) / min(50, len(smoothed[-50:]))

        # Decay rate
        peak_idx = smoothed.index(peak)
        if peak_idx < len(smoothed) - 1 and peak > 0:
            decay = (peak - final) / peak
        else:
            decay = 0

        # Early signal
        early_mean = sum(validity[:50]) / min(50, len(validity[:50]))

        category = classify_curve(peak, final)
        taxonomy_counts[category] += 1

        # Reward trajectory
        rewards = [s.get("mean_reward", s.get("reward", 0)) for s in steps]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0

        runs.append({
            "model": model,
            "task": task,
            "seed": seed,
            "category": category,
            "peak_validity": round(peak, 4),
            "final_validity": round(final, 4),
            "decay_rate": round(decay, 4),
            "early_mean_50": round(early_mean, 4),
            "peak_step": peak_idx,
            "avg_reward": round(avg_reward, 4),
            "n_steps": len(steps),
        })

    return {
        "runs": runs,
        "summary": {
            "total_runs": len(runs),
            "taxonomy_counts": dict(taxonomy_counts),
        },
        "thresholds": {
            "sustained_final_min": SUSTAINED_FINAL_MIN,
            "transient_peak_min": TRANSIENT_PEAK_MIN,
        },
    }


# ---------------------------------------------------------------------------
# 4. Early Prediction
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def _simple_logistic_regression(
    X: List[List[float]], y: List[int], lr: float = 0.1, epochs: int = 1000
) -> tuple:
    """Minimal logistic regression without sklearn dependency."""
    if not X or not y:
        return [], 0.0
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = sum(w * x for w, x in zip(weights, xi)) + bias
            pred = _sigmoid(z)
            error = pred - yi
            for j in range(n_features):
                weights[j] -= lr * error * xi[j]
            bias -= lr * error

    return weights, bias


def early_prediction(taxonomy_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test whether first-50-step features predict final learning outcome."""
    runs = taxonomy_data["runs"]

    # Model size mapping
    SIZE_MAP = {
        "llama-3.2-1b": 1.0, "llama-3.2-3b": 3.0, "qwen2.5-3b": 3.0,
        "phi-3-mini": 3.8, "qwen3-4b": 4.0,
        "mistral-7b": 7.0, "ministral-8b": 8.0, "gemma-2-9b": 9.0,
    }

    labeled = []
    for r in runs:
        label = 1 if r["category"] == "sustained" else 0
        size_b = SIZE_MAP.get(r["model"], 0)
        features = {
            "early_mean_50": r.get("early_mean_50", 0),
            "peak_step_norm": r.get("peak_step", 0) / 1000.0,
            "size_b_norm": size_b / 12.0,
        }
        labeled.append({
            "features": features,
            "label": label,
            "model": r["model"],
            "task": r["task"],
        })

    if len(labeled) < 5:
        return {"error": "insufficient data", "n_runs": len(labeled)}

    feature_names = ["early_mean_50", "peak_step_norm", "size_b_norm"]
    X = [[d["features"][f] for f in feature_names] for d in labeled]
    y = [d["label"] for d in labeled]

    weights, bias = _simple_logistic_regression(X, y)

    def _predict(w, b, x):
        z = sum(wi * xi for wi, xi in zip(w, x)) + b
        return 1 if _sigmoid(z) >= 0.5 else 0

    preds = [_predict(weights, bias, x) for x in X]
    accuracy = sum(1 for p, yi in zip(preds, y) if p == yi) / len(y)

    # Leave-one-out CV
    loo_correct = 0
    for i in range(len(labeled)):
        X_train = X[:i] + X[i + 1 :]
        y_train = y[:i] + y[i + 1 :]
        w, b = _simple_logistic_regression(X_train, y_train)
        if _predict(w, b, X[i]) == y[i]:
            loo_correct += 1

    return {
        "full_accuracy": round(accuracy, 4),
        "loo_cv_accuracy": round(loo_correct / len(labeled), 4),
        "n_samples": len(labeled),
        "n_sustained": sum(y),
        "n_other": len(y) - sum(y),
        "weights": {name: round(w, 4) for name, w in zip(feature_names, weights)},
        "bias": round(bias, 4),
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# pgfplots output
# ---------------------------------------------------------------------------

def write_pgfplots(
    decomp: Dict[str, Any],
    forgetting: Dict[str, Any],
    taxonomy: Dict[str, Any],
) -> None:
    """Write pgfplots .dat files for LaTeX figures."""
    DAT_DIR.mkdir(parents=True, exist_ok=True)

    # Reward decomposition
    with open(DAT_DIR / "p6_reward_decomposition.dat", "w") as f:
        f.write("model r_schema r_residual r_latency r_cost mean_reward\n")
        for model, data in sorted(decomp.get("models", {}).items()):
            f.write(
                f"{model} {data['mean_r_schema']:.4f} {data['mean_r_residual']:.4f} "
                f"{data['mean_r_latency']:.4f} {data['mean_r_cost']:.4f} "
                f"{data['mean_reward']:.4f}\n"
            )
    print(f"Wrote {DAT_DIR / 'p6_reward_decomposition.dat'}")

    # Forgetting heatmap
    with open(DAT_DIR / "p6_forgetting_heatmap.dat", "w") as f:
        f.write("model mixed_validity single_avg interference_delta profile\n")
        for m in forgetting.get("matrix", []):
            mv = m["mixed_validity"] if m["mixed_validity"] is not None else 0
            sa = m["single_avg"] if m["single_avg"] is not None else 0
            f.write(
                f"{m['model']} {mv:.1f} {sa:.1f} "
                f"{m['interference_delta']:.4f} {m['profile']}\n"
            )
    print(f"Wrote {DAT_DIR / 'p6_forgetting_heatmap.dat'}")

    # Curve taxonomy
    with open(DAT_DIR / "p6_curve_taxonomy.dat", "w") as f:
        f.write("model task category peak_validity final_validity decay_rate avg_reward\n")
        for r in sorted(taxonomy.get("runs", []), key=lambda x: (x["model"], x["task"])):
            f.write(
                f"{r['model']} {r['task']} {r['category']} "
                f"{r['peak_validity']:.4f} {r['final_validity']:.4f} "
                f"{r['decay_rate']:.4f} {r['avg_reward']:.4f}\n"
            )
    print(f"Wrote {DAT_DIR / 'p6_curve_taxonomy.dat'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="P6: Replicate P4 training dynamics analyses on MLX results."
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

    mlx_root = Path(args.mlx_dir)

    print(f"[load] MLX training logs: {mlx_root}")
    logs = load_training_logs(mlx_root)
    print(f"  -> {len(logs)} training runs")

    summaries = load_run_summaries(mlx_root)
    print(f"  -> {len(summaries)} run summaries")

    if not logs and not summaries:
        print("[warn] No MLX training data found. Writing empty output.")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"error": "no_data"}, f, indent=2)
        return 0

    print("\n[analysis] Reward decomposition ...")
    decomp = reward_decomposition(summaries, logs)

    print("[analysis] Forgetting matrix ...")
    forgetting = forgetting_matrix(summaries)

    print("[analysis] Curve taxonomy ...")
    taxonomy = curve_taxonomy(logs)
    counts = taxonomy["summary"]["taxonomy_counts"]
    print(f"  -> {counts}")

    print("[analysis] Early prediction ...")
    prediction = early_prediction(taxonomy)
    if "error" not in prediction:
        print(f"  -> Full accuracy: {prediction['full_accuracy']:.1%}")
        print(f"  -> LOO-CV accuracy: {prediction['loo_cv_accuracy']:.1%}")

    # Assemble output
    output = {
        "reward_decomposition": decomp,
        "forgetting_matrix": forgetting,
        "curve_taxonomy": taxonomy,
        "early_prediction": prediction,
        "metadata": {
            "n_logs": len(logs),
            "n_summaries": len(summaries),
            "mlx_root": str(mlx_root),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")

    # pgfplots
    write_pgfplots(decomp, forgetting, taxonomy)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
