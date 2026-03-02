#!/usr/bin/env python3
"""P5 Analysis: MoE vs Dense model comparison on Apple MLX.

Compares the Qwen3.5-35B-A3B (MoE, 3B active / 35B total) against
dense baselines at matched active-parameter counts and nearby dense sizes:
  - Qwen2.5-3B (dense, 3B) -- same active params
  - Qwen3-4B (dense, 4B)   -- closest dense
  - Gemma-2-9B (dense, 9B)  -- midrange dense

Computes:
  - Accuracy metrics (json_valid, task-specific scores)
  - Latency distributions and percentile comparisons
  - Spearman correlations: active_params vs latency, total_params vs latency,
    active_params vs accuracy, total_params vs accuracy
  - S@SLO at Interactive (2s), Standard (5s), Batch (30s) tiers
  - Whether adding MoE changes the accuracy-vs-S@SLO Spearman rho

Inputs:
  --results-dir  Path to MLX eval directory (default: results/mlx_eval/)

Outputs:
  results/p5_analysis/moe_comparison.json
  results/curves/pgfplots/p5_accuracy_vs_slo.dat
  results/curves/pgfplots/p5_latency_cdf_moe.dat
  results/curves/pgfplots/p5_memory_comparison.dat
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_DIR = ROOT / "results" / "mlx_eval"
OUT_DIR = ROOT / "results" / "p5_analysis"
DAT_DIR = ROOT / "results" / "curves" / "pgfplots"

TIERS = {"interactive": 2000, "standard": 5000, "batch": 30000}

# Focus comparison group
MOE_MODEL = "qwen3-30b-a3b"
COMPARISON_DENSE = ["qwen2.5-3b", "qwen3-4b", "gemma-2-9b"]
FOCUS_MODELS = [MOE_MODEL] + COMPARISON_DENSE

# Map from MLX eval directory slug -> short name used throughout analysis.
SLUG_TO_SHORT = {
    "mlx_local_mlx-community-Llama-3.2-1B-Instruct-4bit": "llama-3.2-1b",
    "mlx_local_mlx-community-Llama-3.2-3B-Instruct-4bit": "llama-3.2-3b",
    "mlx_local_mlx-community-Qwen2.5-3B-Instruct-4bit":   "qwen2.5-3b",
    "mlx_local_mlx-community-Qwen3-4B-4bit":              "qwen3-4b",
    "mlx_local_mlx-community-Phi-3-mini-4k-instruct-4bit": "phi-3-mini",
    "mlx_local_mlx-community-Mistral-7B-Instruct-v0.3-4bit": "mistral-7b",
    "mlx_local_mlx-community-Ministral-8B-Instruct-2410-4bit": "ministral-8b",
    "mlx_local_mlx-community-gemma-2-9b-it-4bit":         "gemma-2-9b",
    "mlx_local_mlx-community-Qwen3-30B-A3B-4bit":          "qwen3-30b-a3b",
}

# Total parameter counts (billions)
MODEL_PARAMS_TOTAL = {
    "llama-3.2-1b":       1.24,
    "llama-3.2-3b":       3.21,
    "qwen2.5-3b":         3.09,
    "phi-3-mini":          3.82,
    "qwen3-4b":            4.02,
    "mistral-7b":          7.25,
    "ministral-8b":        8.02,
    "gemma-2-9b":          9.24,
    "qwen3-30b-a3b":   30.0,
}

# Active parameter counts (billions)
MODEL_PARAMS_ACTIVE = {
    "llama-3.2-1b":       1.24,
    "llama-3.2-3b":       3.21,
    "qwen2.5-3b":         3.09,
    "phi-3-mini":          3.82,
    "qwen3-4b":            4.02,
    "mistral-7b":          7.25,
    "ministral-8b":        8.02,
    "gemma-2-9b":          9.24,
    "qwen3-30b-a3b":    3.0,
}

IS_MOE = {
    "qwen3-30b-a3b": True,
}

# Approximate peak memory in GB for MLX inference (from model card / empirical)
# These will be filled from summary.json if available, otherwise use estimates.
MODEL_MEMORY_EST_GB = {
    "llama-3.2-1b":       0.7,
    "llama-3.2-3b":       1.8,
    "qwen2.5-3b":         1.8,
    "phi-3-mini":          2.3,
    "qwen3-4b":            2.4,
    "mistral-7b":          4.3,
    "ministral-8b":        4.8,
    "gemma-2-9b":          5.4,
    "qwen3-30b-a3b":   21.0,   # Q4 quantized: ~21GB for 35B params
}

# Map schema_path to task ID (plus T6 for GSM8K)
SCHEMA_TO_TASK = {
    "tasks/schemas/clinc_nlu_schema.json": "T1",
    "tasks/schemas/t1_incident_schema.json": "T1v",
    "tasks/schemas/t2_summary_schema.json": "T2",
    "tasks/schemas/hotpot_explainer_schema.json": "T2v",
    "tasks/schemas/t3_tool_call_schema.json": "T3",
    "tasks/schemas/t4_function_call_schema.json": "T4",
    "tasks/schemas/t5_patch_schema.json": "T5",
    "tasks/schemas/qa_schema.json": "T6",
}


# ---------------------------------------------------------------------------
# Task-correctness check (mirrors P3 compute_real_tiers.py)
# ---------------------------------------------------------------------------
def is_task_correct(pred: dict) -> bool:
    """Determine if a prediction is task-correct.

    Works with both detailed and non-detailed evaluation output.
    """
    metrics = pred.get("metrics", {})
    json_valid = metrics.get("json_valid", 0) == 1.0
    if not json_valid:
        return False

    schema = pred.get("schema_path", "")

    # T4 (BFCL): function name must match
    if schema == "tasks/schemas/t4_function_call_schema.json":
        gold = pred.get("gold", {})
        output = pred.get("output_json", {})
        if not gold or not output:
            return False
        gold_func = list(gold.keys())[0] if gold else None
        pred_func = output.get("name", output.get("function_name", ""))
        return gold_func is not None and pred_func == gold_func

    # If detailed capture is available, use overall_field_accuracy
    ofa = pred.get("detailed", {}).get("overall_field_accuracy", None)
    if ofa is not None:
        return ofa > 0

    # Fallback: use task-specific success metrics from eval_t_suite scoring
    for key in ("t1_field_acc", "t2_summary_f1", "t3_success", "t4_success",
                "t5_valid_diff", "t6_success"):
        val = metrics.get(key)
        if val is not None:
            return float(val) > 0

    # Last resort: json_valid is True but no task-specific metric found
    return bool(pred.get("output_json"))


# ---------------------------------------------------------------------------
# Latency / S@SLO helpers
# ---------------------------------------------------------------------------
def percentile(sorted_vals: list[float], pct: float) -> float:
    """Return the pct-th percentile from a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = int(pct / 100.0 * len(sorted_vals))
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def compute_latency_stats(latencies: list[float]) -> dict:
    """Compute standard latency percentiles."""
    s = sorted(latencies)
    n = len(s)
    if n == 0:
        return {"count": 0, "mean_ms": 0, "p50_ms": 0, "p75_ms": 0,
                "p90_ms": 0, "p95_ms": 0, "p99_ms": 0}
    return {
        "count": n,
        "mean_ms": round(statistics.mean(s), 1),
        "p50_ms": round(percentile(s, 50), 1),
        "p75_ms": round(percentile(s, 75), 1),
        "p90_ms": round(percentile(s, 90), 1),
        "p95_ms": round(percentile(s, 95), 1),
        "p99_ms": round(percentile(s, 99), 1),
    }


def compute_slo_rates(preds: list[dict]) -> dict:
    """Compute S@SLO rates at each tier."""
    n = len(preds)
    if n == 0:
        return {f"s_at_slo_{t}": 0.0 for t in TIERS}
    result = {}
    for tier_name, deadline_ms in TIERS.items():
        s_at_slo = sum(
            1 for p in preds
            if is_task_correct(p) and p.get("latency_ms", float("inf")) <= deadline_ms
        )
        result[f"s_at_slo_{tier_name}"] = round(s_at_slo / n, 4)
    total_correct = sum(1 for p in preds if is_task_correct(p))
    result["accuracy"] = round(total_correct / n, 4)
    total_json_valid = sum(
        1 for p in preds if p.get("metrics", {}).get("json_valid", 0) == 1.0
    )
    result["json_valid"] = round(total_json_valid / n, 4)
    return result


def compute_slo_per_task(preds: list[dict]) -> dict:
    """Compute per-task and aggregate S@SLO."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        task = SCHEMA_TO_TASK.get(p.get("schema_path", ""), "unknown")
        by_task[task].append(p)

    per_task = {}
    for task_id, task_preds in sorted(by_task.items()):
        per_task[task_id] = compute_slo_rates(task_preds)
        per_task[task_id]["count"] = len(task_preds)

    aggregate = compute_slo_rates(preds)
    aggregate["total"] = len(preds)

    latencies = sorted(p.get("latency_ms", 0) for p in preds)
    aggregate.update(compute_latency_stats(latencies))

    return {"per_task": per_task, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------
def rank_values(values: list[float]) -> list[float]:
    """Assign ranks (1=best/highest). Handles ties with average rank."""
    indexed = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = rank_values(x)
    ry = rank_values(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def spearman_p_value(rho: float, n: int) -> float:
    """Approximate two-tailed p-value."""
    if n < 3 or abs(rho) >= 1.0:
        return 0.0 if abs(rho) >= 1.0 else 1.0
    t_stat = rho * math.sqrt((n - 2) / (1 - rho * rho))
    df = n - 2
    z = abs(t_stat) * math.sqrt(1 - 1 / (4 * df) - 7 / (120 * df * df)) if df > 2 else abs(t_stat)
    p_one_tail = 0.5 * math.erfc(z / math.sqrt(2))
    return 2 * p_one_tail


def bootstrap_spearman(
    x: list[float], y: list[float],
    n_bootstrap: int = 10000, seed: int = 42,
) -> dict:
    """Bootstrap Spearman rho with 95% CI and p-value."""
    n = len(x)
    if n < 3:
        return {"rho": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "n": n}
    rng = random.Random(seed)
    observed = spearman_rho(x, y)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        bx = [x[i] for i in idx]
        by = [y[i] for i in idx]
        boot_rhos.append(spearman_rho(bx, by))
    boot_rhos.sort()
    return {
        "rho": round(observed, 4),
        "ci_lower": round(boot_rhos[int(0.025 * n_bootstrap)], 4),
        "ci_upper": round(boot_rhos[int(0.975 * n_bootstrap)], 4),
        "p_value": round(spearman_p_value(observed, n), 4),
        "n": n,
    }


# Try scipy for additional Spearman
try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def scipy_spearman(x: list[float], y: list[float]) -> dict | None:
    """Compute Spearman via scipy if available."""
    if not HAS_SCIPY or len(x) < 3:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _scipy_spearmanr(x, y)
    rho = float(res.statistic) if not math.isnan(res.statistic) else 0.0
    pval = float(res.pvalue) if not math.isnan(res.pvalue) else 1.0
    return {"rho": round(rho, 4), "p_value": round(pval, 4)}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_mlx_model(model_dir: Path) -> tuple[dict | None, list[dict]]:
    """Load summary.json and predictions.jsonl from a model directory."""
    summary = None
    preds = []

    summary_path = model_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    pred_path = model_dir / "predictions.jsonl"
    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    preds.append(json.loads(line))

    return summary, preds


def _resolve_short_name(dirname: str) -> str:
    """Map an eval directory slug to a short model name."""
    if dirname in SLUG_TO_SHORT:
        return SLUG_TO_SHORT[dirname]
    # Fallback heuristic: lowercase, strip common prefixes
    name = dirname.lower()
    for prefix in ("mlx_local_mlx-community-", "mlx_local_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ("-4bit", "-8bit", "-instruct"):
        name = name.replace(suffix, "")
    return name


def load_all_mlx(results_dir: Path) -> dict[str, dict]:
    """Load all MLX model results.

    Scans the run directory (or the latest run inside it) for per-model
    subdirectories.  Directory slugs are normalised to short names via
    SLUG_TO_SHORT.

    Returns {model_name: {"summary": ..., "preds": [...], "slo": ..., "latency_stats": ...}}.
    """
    results = {}
    if not results_dir.exists():
        print(f"  [WARN] Results directory not found: {results_dir}")
        return results

    # If results_dir itself contains a run_manifest.json, scan its children.
    # Otherwise look for the most recent run subdirectory.
    scan_dir = results_dir
    if not (scan_dir / "run_manifest.json").exists():
        run_dirs = [
            d for d in sorted(results_dir.iterdir())
            if d.is_dir() and (d / "run_manifest.json").exists()
        ]
        if run_dirs:
            scan_dir = run_dirs[-1]
            print(f"  Using MLX run directory: {scan_dir.name}")
        else:
            print(f"  [WARN] No run_manifest.json found in {results_dir}")

    for model_dir in sorted(scan_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        slug = model_dir.name
        short_name = _resolve_short_name(slug)
        summary, preds = load_mlx_model(model_dir)
        if not preds:
            print(f"  [WARN] No predictions for model: {slug} ({short_name})")
            continue
        slo_data = compute_slo_per_task(preds)
        latencies = [p.get("latency_ms", 0) for p in preds]
        results[short_name] = {
            "summary": summary,
            "preds": preds,
            "slo": slo_data,
            "latency_stats": compute_latency_stats(latencies),
        }
        print(f"  Loaded {len(preds):,} predictions for {short_name} (from {slug})")

    return results


# ---------------------------------------------------------------------------
# CDF computation
# ---------------------------------------------------------------------------
def compute_cdf(latencies: list[float], n_points: int = 100) -> list[tuple[float, float]]:
    """Compute empirical CDF as (latency_ms, fraction) pairs.

    Returns n_points evenly spaced percentile points from 0 to 100.
    """
    if not latencies:
        return []
    s = sorted(latencies)
    n = len(s)
    points = []
    for i in range(n_points + 1):
        pct = i / n_points * 100
        idx = min(int(pct / 100.0 * n), n - 1)
        points.append((round(s[idx], 1), round(pct, 1)))
    return points


# ---------------------------------------------------------------------------
# Focus comparison
# ---------------------------------------------------------------------------
def focus_comparison(all_data: dict) -> dict:
    """Run the MoE vs dense focus comparison.

    Compares Qwen3.5-35B-A3B against Qwen2.5-3B, Qwen3-4B, and Gemma-2-9B.
    """
    available_focus = [m for m in FOCUS_MODELS if m in all_data]
    missing = [m for m in FOCUS_MODELS if m not in all_data]
    if missing:
        print(f"  [WARN] Missing focus models: {missing}")

    if len(available_focus) < 2:
        print("  [WARN] Need at least 2 focus models for comparison")
        return {"available": available_focus, "missing": missing}

    # Per-model summary
    per_model = {}
    for model_name in available_focus:
        agg = all_data[model_name]["slo"]["aggregate"]
        lstats = all_data[model_name]["latency_stats"]
        per_model[model_name] = {
            "total_params_b": MODEL_PARAMS_TOTAL.get(model_name, 0),
            "active_params_b": MODEL_PARAMS_ACTIVE.get(model_name, 0),
            "is_moe": IS_MOE.get(model_name, False),
            "accuracy": agg["accuracy"],
            "json_valid": agg.get("json_valid", 0),
            "s_at_slo_interactive": agg["s_at_slo_interactive"],
            "s_at_slo_standard": agg["s_at_slo_standard"],
            "s_at_slo_batch": agg["s_at_slo_batch"],
            "latency_p50_ms": lstats["p50_ms"],
            "latency_p95_ms": lstats["p95_ms"],
            "latency_p99_ms": lstats["p99_ms"],
            "latency_mean_ms": lstats["mean_ms"],
            "memory_est_gb": MODEL_MEMORY_EST_GB.get(model_name, 0),
        }

    # Pairwise comparison: MoE vs each dense baseline
    pairwise = {}
    if MOE_MODEL in all_data:
        moe_agg = all_data[MOE_MODEL]["slo"]["aggregate"]
        moe_lat = all_data[MOE_MODEL]["latency_stats"]
        for dense_name in COMPARISON_DENSE:
            if dense_name not in all_data:
                continue
            d_agg = all_data[dense_name]["slo"]["aggregate"]
            d_lat = all_data[dense_name]["latency_stats"]
            pairwise[f"{MOE_MODEL}_vs_{dense_name}"] = {
                "accuracy_delta": round(moe_agg["accuracy"] - d_agg["accuracy"], 4),
                "latency_p50_ratio": round(
                    moe_lat["p50_ms"] / d_lat["p50_ms"], 4
                ) if d_lat["p50_ms"] > 0 else None,
                "slo_batch_delta": round(
                    moe_agg["s_at_slo_batch"] - d_agg["s_at_slo_batch"], 4
                ),
                "slo_standard_delta": round(
                    moe_agg["s_at_slo_standard"] - d_agg["s_at_slo_standard"], 4
                ),
                "memory_ratio": round(
                    MODEL_MEMORY_EST_GB.get(MOE_MODEL, 0) /
                    max(MODEL_MEMORY_EST_GB.get(dense_name, 1), 0.1), 2
                ),
            }

    return {
        "available": available_focus,
        "missing": missing,
        "per_model": per_model,
        "pairwise": pairwise,
    }


# ---------------------------------------------------------------------------
# Spearman correlation analysis
# ---------------------------------------------------------------------------
def correlation_analysis(all_data: dict) -> dict:
    """Compute Spearman correlations across all MLX models.

    Correlations:
      - active_params vs median_latency
      - total_params vs median_latency
      - active_params vs accuracy
      - total_params vs accuracy
    Also: does adding MoE change accuracy-vs-S@SLO rho?
    """
    models = sorted(all_data.keys())
    n = len(models)
    if n < 3:
        print(f"  [WARN] Only {n} models, need >= 3 for Spearman")
        return {"n_models": n}

    active_params = [MODEL_PARAMS_ACTIVE.get(m, 0) for m in models]
    total_params = [MODEL_PARAMS_TOTAL.get(m, 0) for m in models]
    median_latencies = [all_data[m]["latency_stats"]["p50_ms"] for m in models]
    accuracies = [all_data[m]["slo"]["aggregate"]["accuracy"] for m in models]

    correlations = {
        "active_params_vs_latency": bootstrap_spearman(active_params, median_latencies),
        "total_params_vs_latency": bootstrap_spearman(total_params, median_latencies),
        "active_params_vs_accuracy": bootstrap_spearman(active_params, accuracies),
        "total_params_vs_accuracy": bootstrap_spearman(total_params, accuracies),
    }

    # Add scipy results if available
    for key, (x, y) in {
        "active_params_vs_latency": (active_params, median_latencies),
        "total_params_vs_latency": (total_params, median_latencies),
        "active_params_vs_accuracy": (active_params, accuracies),
        "total_params_vs_accuracy": (total_params, accuracies),
    }.items():
        sci = scipy_spearman(x, y)
        if sci:
            correlations[key]["scipy"] = sci

    # Accuracy vs S@SLO: with and without MoE
    moe_effect = {}
    for tier_name in TIERS:
        slo_key = f"s_at_slo_{tier_name}"
        # All models
        slo_all = [all_data[m]["slo"]["aggregate"][slo_key] for m in models]
        rho_all = bootstrap_spearman(accuracies, slo_all)

        # Without MoE
        dense_models = [m for m in models if not IS_MOE.get(m, False)]
        if len(dense_models) >= 3:
            dense_acc = [all_data[m]["slo"]["aggregate"]["accuracy"] for m in dense_models]
            dense_slo = [all_data[m]["slo"]["aggregate"][slo_key] for m in dense_models]
            rho_dense = bootstrap_spearman(dense_acc, dense_slo)
        else:
            rho_dense = {"rho": 0.0, "n": len(dense_models)}

        moe_effect[tier_name] = {
            "with_moe": rho_all,
            "dense_only": rho_dense,
            "rho_delta": round(rho_all["rho"] - rho_dense.get("rho", 0), 4),
        }

    correlations["accuracy_vs_slo_moe_effect"] = moe_effect
    correlations["n_models"] = n
    correlations["models"] = models

    return correlations


# ---------------------------------------------------------------------------
# pgfplots data output
# ---------------------------------------------------------------------------
def write_accuracy_vs_slo(all_data: dict, path: Path):
    """Write accuracy vs S@SLO for all MLX models (for scatter plot)."""
    with open(path, "w") as f:
        f.write("# model accuracy slo_interactive slo_standard slo_batch is_moe active_params total_params\n")
        for model_name in sorted(all_data.keys()):
            agg = all_data[model_name]["slo"]["aggregate"]
            acc = agg["accuracy"] * 100
            si = agg["s_at_slo_interactive"] * 100
            ss = agg["s_at_slo_standard"] * 100
            sb = agg["s_at_slo_batch"] * 100
            moe = 1 if IS_MOE.get(model_name, False) else 0
            ap = MODEL_PARAMS_ACTIVE.get(model_name, 0)
            tp = MODEL_PARAMS_TOTAL.get(model_name, 0)
            f.write(f"{model_name} {acc:.1f} {si:.1f} {ss:.1f} {sb:.1f} {moe} {ap:.1f} {tp:.1f}\n")
    print(f"  Wrote {path}")


def write_latency_cdf(all_data: dict, path: Path):
    """Write CDF data for MoE and comparison dense models."""
    # Determine which focus models are available
    available = [m for m in FOCUS_MODELS if m in all_data]
    if not available:
        print(f"  [WARN] No focus models available for CDF, skipping {path}")
        return

    n_points = 100
    cdfs = {}
    for model_name in available:
        latencies = [p.get("latency_ms", 0) for p in all_data[model_name]["preds"]]
        cdfs[model_name] = compute_cdf(latencies, n_points)

    with open(path, "w") as f:
        # Header: percentile then one column per available model
        header = "percentile " + " ".join(m.replace("-", "_").replace(".", "_") for m in available)
        f.write(f"# {header}\n")
        for i in range(n_points + 1):
            pct = i / n_points * 100
            vals = []
            for model_name in available:
                if i < len(cdfs[model_name]):
                    vals.append(f"{cdfs[model_name][i][0]:.1f}")
                else:
                    vals.append("NaN")
            f.write(f"{pct:.1f} {' '.join(vals)}\n")
    print(f"  Wrote {path}")


def write_memory_comparison(all_data: dict, path: Path):
    """Write memory usage comparison for all MLX models."""
    with open(path, "w") as f:
        f.write("# model peak_memory_gb active_params total_params is_moe\n")
        for model_name in sorted(all_data.keys()):
            mem = MODEL_MEMORY_EST_GB.get(model_name, 0)
            ap = MODEL_PARAMS_ACTIVE.get(model_name, 0)
            tp = MODEL_PARAMS_TOTAL.get(model_name, 0)
            moe = 1 if IS_MOE.get(model_name, False) else 0
            f.write(f"{model_name} {mem:.1f} {ap:.1f} {tp:.1f} {moe}\n")
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="P5 MoE vs dense comparison on Apple MLX"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Path to MLX eval results directory (default: results/mlx_eval/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("P5 MoE vs Dense Comparison (MLX)")
    print("=" * 60)

    # --- Load all MLX results ---
    all_data = load_all_mlx(args.results_dir)
    if not all_data:
        print("[ERROR] No MLX results loaded. Expected per-model directories under:")
        print(f"  {args.results_dir}/")
        print("  Each containing predictions.jsonl and optionally summary.json")
        return
    print(f"\nLoaded {len(all_data)} models: {sorted(all_data.keys())}")

    # --- Focus comparison ---
    print("\n--- Focus Comparison: MoE vs Dense ---")
    focus = focus_comparison(all_data)
    if "per_model" in focus:
        for m, data in sorted(focus["per_model"].items()):
            moe_tag = " [MoE]" if data["is_moe"] else ""
            print(f"  {m:20s}{moe_tag}: "
                  f"acc={data['accuracy']:.3f}  "
                  f"p50={data['latency_p50_ms']:.0f}ms  "
                  f"S@5s={data['s_at_slo_standard']:.3f}  "
                  f"S@30s={data['s_at_slo_batch']:.3f}  "
                  f"mem~{data['memory_est_gb']:.1f}GB")
    if "pairwise" in focus:
        print("\n  Pairwise (MoE - Dense):")
        for pair, data in sorted(focus["pairwise"].items()):
            print(f"    {pair}:")
            print(f"      accuracy delta:  {data['accuracy_delta']:+.4f}")
            if data['latency_p50_ratio'] is not None:
                print(f"      latency p50 ratio: {data['latency_p50_ratio']:.2f}x")
            print(f"      S@SLO batch delta: {data['slo_batch_delta']:+.4f}")
            print(f"      memory ratio: {data['memory_ratio']:.1f}x")

    # --- Spearman correlations ---
    print("\n--- Spearman Correlations (all MLX models) ---")
    corr = correlation_analysis(all_data)
    for key in ["active_params_vs_latency", "total_params_vs_latency",
                "active_params_vs_accuracy", "total_params_vs_accuracy"]:
        if key in corr:
            r = corr[key]
            print(f"  {key:35s}: rho={r['rho']:+.4f}  "
                  f"CI=[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]  "
                  f"p={r['p_value']:.4f}")

    if "accuracy_vs_slo_moe_effect" in corr:
        print("\n  MoE effect on accuracy-vs-S@SLO rho:")
        for tier_name, data in corr["accuracy_vs_slo_moe_effect"].items():
            w = data["with_moe"]
            d = data["dense_only"]
            print(f"    {tier_name:12s}: with_moe={w['rho']:+.4f}  "
                  f"dense_only={d.get('rho', 0):+.4f}  "
                  f"delta={data['rho_delta']:+.4f}")

    # --- S@SLO table ---
    print("\n--- S@SLO Table (all models) ---")
    slo_table = {}
    for model_name in sorted(all_data.keys()):
        agg = all_data[model_name]["slo"]["aggregate"]
        slo_table[model_name] = {
            "accuracy": agg["accuracy"],
            "json_valid": agg.get("json_valid", 0),
            "s_at_slo_interactive": agg["s_at_slo_interactive"],
            "s_at_slo_standard": agg["s_at_slo_standard"],
            "s_at_slo_batch": agg["s_at_slo_batch"],
            "total_params_b": MODEL_PARAMS_TOTAL.get(model_name, 0),
            "active_params_b": MODEL_PARAMS_ACTIVE.get(model_name, 0),
            "is_moe": IS_MOE.get(model_name, False),
            "p50_ms": all_data[model_name]["latency_stats"].get("p50_ms", 0),
            "p95_ms": all_data[model_name]["latency_stats"].get("p95_ms", 0),
        }
        moe_tag = "*" if IS_MOE.get(model_name, False) else " "
        print(f"  {moe_tag}{model_name:20s}: acc={agg['accuracy']:.3f}  "
              f"jv={agg.get('json_valid', 0):.3f}  "
              f"S@2s={agg['s_at_slo_interactive']:.3f}  "
              f"S@5s={agg['s_at_slo_standard']:.3f}  "
              f"S@30s={agg['s_at_slo_batch']:.3f}")

    # --- Write output JSON ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "focus_comparison": focus,
        "correlations": corr,
        "slo_table": slo_table,
        "tiers": dict(TIERS),
        "all_models": sorted(all_data.keys()),
        "moe_model": MOE_MODEL,
        "comparison_dense": COMPARISON_DENSE,
    }
    out_json = OUT_DIR / "moe_comparison.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_json}")

    # --- Write pgfplots .dat files ---
    print("\n--- Generating pgfplots .dat files ---")
    DAT_DIR.mkdir(parents=True, exist_ok=True)
    write_accuracy_vs_slo(all_data, DAT_DIR / "p5_accuracy_vs_slo.dat")
    write_latency_cdf(all_data, DAT_DIR / "p5_latency_cdf_moe.dat")
    write_memory_comparison(all_data, DAT_DIR / "p5_memory_comparison.dat")

    print("\nDone!")


if __name__ == "__main__":
    main()
