#!/usr/bin/env python3
"""P5 Analysis: Cross-hardware comparison (CUDA via LM Studio vs Apple MLX).

Loads P1 CUDA evaluation results and MLX evaluation results for the 8
overlapping dense models, computes hardware normalization factors
(MLX/CUDA latency ratios), and produces hardware-normalized MoE latency
estimates (what would the MoE model look like on CUDA?).

Computes S@SLO at three tiers for MLX results and compares rank orderings
across hardware backends via Spearman correlations.

Inputs:
  --cuda-results  Path to CUDA S@SLO JSON (default: results/p3_analysis/real_slo_tiers.json)
  --mlx-results   Path to MLX eval directory (default: results/mlx_eval/)

Outputs:
  results/p5_analysis/hardware_comparison.json
  results/curves/pgfplots/p5_latency_scatter_total.dat
  results/curves/pgfplots/p5_latency_scatter_active.dat
  results/curves/pgfplots/p5_slo_comparison.dat
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
DEFAULT_CUDA_RESULTS = ROOT / "results" / "p3_analysis" / "real_slo_tiers.json"
DEFAULT_MLX_RESULTS = ROOT / "results" / "mlx_eval"
OUT_DIR = ROOT / "results" / "p5_analysis"
DAT_DIR = ROOT / "results" / "curves" / "pgfplots"

TIERS = {"interactive": 2000, "standard": 5000, "batch": 30000}

# Map from MLX eval directory slug -> short name used throughout analysis.
# eval_t_suite.py produces slugs like "mlx_local_mlx-community-Llama-3.2-1B-Instruct-4bit".
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

# The dense models that overlap between CUDA P1 evals and MLX evals.
# Maps short name -> CUDA short name used in real_slo_tiers.json.
OVERLAP_MODELS = {
    "llama-3.2-1b":   "llama-3.2-1b",
    "llama-3.2-3b":   "llama-3.2-3b",
    "qwen2.5-3b":     "qwen2.5-3b",
    "phi-3-mini":      "phi-3-mini",
    "qwen3-4b":        "qwen3-4b",
    "mistral-7b":      "mistral-7b",
    "ministral-8b":    "ministral-8b",
    "gemma-2-9b":      "gemma-2-9b",
}

# Total parameter counts (billions) for all 9 MLX-evaluated models.
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

# Active parameter counts (billions). For dense models, active == total.
MODEL_PARAMS_ACTIVE = {
    "llama-3.2-1b":       1.24,
    "llama-3.2-3b":       3.21,
    "qwen2.5-3b":         3.09,
    "phi-3-mini":          3.82,
    "qwen3-4b":            4.02,
    "mistral-7b":          7.25,
    "ministral-8b":        8.02,
    "gemma-2-9b":          9.24,
    "qwen3-30b-a3b":    3.0,   # MoE: 3B active of 30B total
}

IS_MOE = {
    "qwen3-30b-a3b": True,
}

# Map schema_path to task ID (same as P1/P3, plus T6 for GSM8K)
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
    # T1: t1_exact_match or t1_field_acc > 0
    # T2: t2_summary_f1 > 0
    # T3: t3_success
    # T5: t5_valid_diff
    # T6: t6_success
    for key in ("t1_field_acc", "t2_summary_f1", "t3_success", "t4_success",
                "t5_valid_diff", "t6_success"):
        val = metrics.get(key)
        if val is not None:
            return float(val) > 0

    # Last resort: json_valid is True but no task-specific metric found
    # Count as correct if we have a non-empty output
    return bool(pred.get("output_json"))


# ---------------------------------------------------------------------------
# Latency and S@SLO helpers
# ---------------------------------------------------------------------------
def percentile(sorted_vals: list[float], pct: float) -> float:
    """Return the pct-th percentile from a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = int(pct / 100.0 * len(sorted_vals))
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def compute_latency_stats(latencies: list[float]) -> dict:
    """Compute standard latency percentiles from a list of latencies in ms."""
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
    """Compute S@SLO rates at each tier for a list of predictions."""
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
    return result


def compute_slo_per_task(preds: list[dict]) -> dict:
    """Compute per-task and aggregate S@SLO for a list of predictions."""
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
# Spearman correlation (mirrors P3, plus scipy fallback)
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
    """Approximate two-tailed p-value for Spearman rho."""
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


# Try scipy for additional robust Spearman when available
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
def load_cuda_results(path: Path) -> dict:
    """Load CUDA S@SLO results from the P3 analysis JSON."""
    with open(path) as f:
        return json.load(f)


def load_mlx_model(model_dir: Path) -> tuple[dict | None, list[dict]]:
    """Load summary.json and predictions.jsonl from an MLX model directory.

    Returns (summary_dict_or_None, list_of_predictions).
    """
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
    # Strip quantization suffix
    for suffix in ("-4bit", "-8bit", "-instruct"):
        name = name.replace(suffix, "")
    return name


def load_all_mlx(mlx_dir: Path) -> dict[str, dict]:
    """Load all MLX model results from the results directory.

    Scans the run directory (or the latest run inside it) for per-model
    subdirectories.  Directory slugs are normalised to short names via
    SLUG_TO_SHORT.

    Returns {model_short_name: {"summary": ..., "preds": [...], "slo": ...}}.
    """
    results = {}
    if not mlx_dir.exists():
        print(f"  [WARN] MLX results directory not found: {mlx_dir}")
        return results

    # If mlx_dir itself contains a run_manifest.json, scan its children.
    # Otherwise look for the most recent run subdirectory.
    scan_dir = mlx_dir
    if not (scan_dir / "run_manifest.json").exists():
        # Find most recent run subdirectory
        run_dirs = [
            d for d in sorted(mlx_dir.iterdir())
            if d.is_dir() and (d / "run_manifest.json").exists()
        ]
        if run_dirs:
            scan_dir = run_dirs[-1]
            print(f"  Using MLX run directory: {scan_dir.name}")
        else:
            print(f"  [WARN] No run_manifest.json found in {mlx_dir}")

    for model_dir in sorted(scan_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        slug = model_dir.name
        short_name = _resolve_short_name(slug)
        summary, preds = load_mlx_model(model_dir)
        if not preds:
            print(f"  [WARN] No predictions for MLX model: {slug} ({short_name})")
            continue
        slo_data = compute_slo_per_task(preds)
        latencies = [p.get("latency_ms", 0) for p in preds]
        results[short_name] = {
            "summary": summary,
            "preds": preds,
            "slo": slo_data,
            "latency_stats": compute_latency_stats(latencies),
        }
        print(f"  Loaded {len(preds):,} MLX predictions for {short_name} (from {slug})")

    return results


# ---------------------------------------------------------------------------
# Hardware normalization
# ---------------------------------------------------------------------------
def compute_hardware_ratios(
    cuda_data: dict,
    mlx_data: dict,
) -> dict:
    """Compute MLX/CUDA latency ratios for the overlapping dense models.

    For each model in OVERLAP_MODELS that appears in both datasets, compute
    the ratio of MLX p50 latency to CUDA p50 latency. Also compute per-task
    ratios where both datasets have matching task data.
    """
    per_model = {}
    all_ratios = []

    cuda_models = cuda_data.get("models", {})

    for mlx_name, cuda_name in OVERLAP_MODELS.items():
        if mlx_name not in mlx_data:
            print(f"  [WARN] Overlap model {mlx_name} not found in MLX results, skipping")
            continue
        if cuda_name not in cuda_models:
            print(f"  [WARN] Overlap model {cuda_name} not found in CUDA results, skipping")
            continue

        cuda_agg = cuda_models[cuda_name]["aggregate"]
        mlx_stats = mlx_data[mlx_name]["latency_stats"]

        cuda_p50 = cuda_agg.get("p50_latency_ms", 0)
        mlx_p50 = mlx_stats.get("p50_ms", 0)

        if cuda_p50 > 0 and mlx_p50 > 0:
            ratio = mlx_p50 / cuda_p50
            all_ratios.append(ratio)

            # Per-task ratios
            per_task_ratios = {}
            mlx_slo = mlx_data[mlx_name]["slo"]
            cuda_per_task = cuda_models[cuda_name].get("per_task", {})
            mlx_per_task = mlx_slo.get("per_task", {})

            for task_id in cuda_per_task:
                if task_id in mlx_per_task:
                    # We need per-task latency from predictions
                    mlx_task_lats = [
                        p["latency_ms"]
                        for p in mlx_data[mlx_name]["preds"]
                        if SCHEMA_TO_TASK.get(p.get("schema_path", ""), "") == task_id
                    ]
                    if mlx_task_lats:
                        mlx_task_p50 = sorted(mlx_task_lats)[len(mlx_task_lats) // 2]
                        per_task_ratios[task_id] = {
                            "mlx_p50_ms": round(mlx_task_p50, 1),
                            "note": "CUDA per-task p50 not stored separately",
                        }

            per_model[mlx_name] = {
                "cuda_p50_ms": round(cuda_p50, 1),
                "mlx_p50_ms": round(mlx_p50, 1),
                "ratio_mlx_over_cuda": round(ratio, 4),
                "total_params_b": MODEL_PARAMS_TOTAL.get(mlx_name, 0),
                "per_task_ratios": per_task_ratios,
            }
        else:
            print(f"  [WARN] Zero p50 for {mlx_name}: cuda={cuda_p50}, mlx={mlx_p50}")

    # Aggregate normalization factors
    normalization = {}
    if all_ratios:
        normalization = {
            "mean_ratio": round(statistics.mean(all_ratios), 4),
            "median_ratio": round(statistics.median(all_ratios), 4),
            "std_ratio": round(statistics.stdev(all_ratios), 4) if len(all_ratios) > 1 else 0.0,
            "min_ratio": round(min(all_ratios), 4),
            "max_ratio": round(max(all_ratios), 4),
            "n_models": len(all_ratios),
        }

    return {
        "per_model": per_model,
        "normalization_factors": normalization,
    }


def estimate_moe_on_cuda(
    mlx_data: dict,
    normalization: dict,
) -> dict:
    """Estimate what the MoE model's latency would be on CUDA hardware.

    Uses the median normalization factor to project MLX latencies to CUDA.
    """
    median_ratio = normalization.get("median_ratio", 1.0)
    if median_ratio <= 0:
        print("  [WARN] Invalid median ratio, using 1.0")
        median_ratio = 1.0

    estimates = {}
    moe_model = "qwen3-30b-a3b"

    if moe_model not in mlx_data:
        print(f"  [WARN] MoE model {moe_model} not found in MLX results")
        return estimates

    mlx_stats = mlx_data[moe_model]["latency_stats"]

    # Normalize each percentile: CUDA_est = MLX / ratio
    for key in ["mean_ms", "p50_ms", "p75_ms", "p90_ms", "p95_ms", "p99_ms"]:
        mlx_val = mlx_stats.get(key, 0)
        cuda_est = mlx_val / median_ratio if median_ratio > 0 else mlx_val
        estimates[f"cuda_est_{key}"] = round(cuda_est, 1)
        estimates[f"mlx_{key}"] = round(mlx_val, 1)

    estimates["normalization_ratio_used"] = median_ratio
    estimates["model"] = moe_model
    estimates["total_params_b"] = MODEL_PARAMS_TOTAL[moe_model]
    estimates["active_params_b"] = MODEL_PARAMS_ACTIVE[moe_model]

    # Estimated S@SLO on CUDA at each tier
    preds = mlx_data[moe_model]["preds"]
    for tier_name, deadline_ms in TIERS.items():
        # Scale the deadline to MLX-equivalent: what MLX deadline gives CUDA deadline?
        mlx_equiv_deadline = deadline_ms * median_ratio
        s_at_slo = sum(
            1 for p in preds
            if is_task_correct(p) and p.get("latency_ms", float("inf")) <= mlx_equiv_deadline
        )
        n = len(preds)
        estimates[f"cuda_est_s_at_slo_{tier_name}"] = round(s_at_slo / max(1, n), 4)

    return estimates


# ---------------------------------------------------------------------------
# Rank correlation comparisons
# ---------------------------------------------------------------------------
def compare_rank_orderings(
    cuda_data: dict,
    mlx_data: dict,
) -> dict:
    """Compare accuracy and S@SLO rank orderings between CUDA and MLX.

    For the overlapping models, compute Spearman rho between:
      - CUDA accuracy ranking vs MLX accuracy ranking
      - CUDA S@SLO ranking vs MLX S@SLO ranking (per tier)
    """
    cuda_models = cuda_data.get("models", {})
    overlap = []
    for mlx_name, cuda_name in OVERLAP_MODELS.items():
        if mlx_name in mlx_data and cuda_name in cuda_models:
            overlap.append((mlx_name, cuda_name))

    if len(overlap) < 3:
        print(f"  [WARN] Only {len(overlap)} overlapping models, need >= 3 for Spearman")
        return {"n_overlap": len(overlap)}

    # Accuracy comparison
    cuda_acc = [cuda_models[cn]["aggregate"]["accuracy"] for _, cn in overlap]
    mlx_acc = [mlx_data[mn]["slo"]["aggregate"]["accuracy"] for mn, _ in overlap]
    acc_spearman = bootstrap_spearman(cuda_acc, mlx_acc)

    # S@SLO comparison per tier
    slo_spearman = {}
    for tier_name in TIERS:
        key = f"s_at_slo_{tier_name}"
        cuda_slo = [cuda_models[cn]["aggregate"][key] for _, cn in overlap]
        mlx_slo = [mlx_data[mn]["slo"]["aggregate"][key] for mn, _ in overlap]
        slo_spearman[tier_name] = bootstrap_spearman(cuda_slo, mlx_slo)

    # Latency rank comparison (p50)
    cuda_lat = [cuda_models[cn]["aggregate"]["p50_latency_ms"] for _, cn in overlap]
    mlx_lat = [mlx_data[mn]["latency_stats"]["p50_ms"] for mn, _ in overlap]
    lat_spearman = bootstrap_spearman(cuda_lat, mlx_lat)

    return {
        "n_overlap": len(overlap),
        "models": [mn for mn, _ in overlap],
        "accuracy_rank_correlation": acc_spearman,
        "latency_rank_correlation": lat_spearman,
        "slo_rank_correlations": slo_spearman,
    }


# ---------------------------------------------------------------------------
# pgfplots data output
# ---------------------------------------------------------------------------
def write_latency_scatter_total(mlx_data: dict, cuda_data: dict, path: Path):
    """Write model, total_params, mlx_p50_ms, cuda_p50_ms for overlapping models."""
    cuda_models = cuda_data.get("models", {})
    with open(path, "w") as f:
        f.write("# model total_params mlx_p50_ms cuda_p50_ms\n")
        for mlx_name, cuda_name in sorted(OVERLAP_MODELS.items()):
            if mlx_name not in mlx_data or cuda_name not in cuda_models:
                continue
            total_p = MODEL_PARAMS_TOTAL.get(mlx_name, 0)
            mlx_p50 = mlx_data[mlx_name]["latency_stats"].get("p50_ms", 0)
            cuda_p50 = cuda_models[cuda_name]["aggregate"].get("p50_latency_ms", 0)
            f.write(f"{mlx_name} {total_p:.1f} {mlx_p50:.1f} {cuda_p50:.1f}\n")
    print(f"  Wrote {path}")


def write_latency_scatter_active(mlx_data: dict, path: Path):
    """Write model, active_params, mlx_p50_ms for all MLX models (including MoE)."""
    with open(path, "w") as f:
        f.write("# model active_params mlx_p50_ms is_moe\n")
        for model_name in sorted(mlx_data.keys()):
            active_p = MODEL_PARAMS_ACTIVE.get(model_name, 0)
            mlx_p50 = mlx_data[model_name]["latency_stats"].get("p50_ms", 0)
            moe = 1 if IS_MOE.get(model_name, False) else 0
            f.write(f"{model_name} {active_p:.1f} {mlx_p50:.1f} {moe}\n")
    print(f"  Wrote {path}")


def write_slo_comparison(mlx_data: dict, path: Path):
    """Write S@SLO at each tier for all MLX models."""
    with open(path, "w") as f:
        f.write("# model slo_interactive slo_standard slo_batch accuracy is_moe\n")
        for model_name in sorted(mlx_data.keys()):
            agg = mlx_data[model_name]["slo"]["aggregate"]
            si = agg.get("s_at_slo_interactive", 0) * 100
            ss = agg.get("s_at_slo_standard", 0) * 100
            sb = agg.get("s_at_slo_batch", 0) * 100
            acc = agg.get("accuracy", 0) * 100
            moe = 1 if IS_MOE.get(model_name, False) else 0
            f.write(f"{model_name} {si:.1f} {ss:.1f} {sb:.1f} {acc:.1f} {moe}\n")
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="P5 cross-hardware comparison: CUDA vs MLX"
    )
    parser.add_argument(
        "--cuda-results", type=Path, default=DEFAULT_CUDA_RESULTS,
        help="Path to CUDA S@SLO JSON (default: results/p3_analysis/real_slo_tiers.json)",
    )
    parser.add_argument(
        "--mlx-results", type=Path, default=DEFAULT_MLX_RESULTS,
        help="Path to MLX eval results directory (default: results/mlx_eval/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("P5 Hardware Comparison: CUDA vs MLX")
    print("=" * 60)

    # --- Load CUDA results ---
    if not args.cuda_results.exists():
        print(f"[ERROR] CUDA results not found: {args.cuda_results}")
        return
    cuda_data = load_cuda_results(args.cuda_results)
    n_cuda = len(cuda_data.get("models", {}))
    print(f"\nLoaded CUDA results: {n_cuda} models from {args.cuda_results}")

    # --- Load MLX results ---
    mlx_data = load_all_mlx(args.mlx_results)
    if not mlx_data:
        print("[ERROR] No MLX results loaded. Expected per-model directories under:")
        print(f"  {args.mlx_results}/")
        print("  Each containing predictions.jsonl and optionally summary.json")
        return
    print(f"\nLoaded MLX results: {len(mlx_data)} models")

    # --- Compute hardware ratios ---
    print("\n--- Hardware Normalization ---")
    hw_ratios = compute_hardware_ratios(cuda_data, mlx_data)
    nf = hw_ratios["normalization_factors"]
    if nf:
        print(f"  Mean MLX/CUDA ratio:   {nf['mean_ratio']:.4f}")
        print(f"  Median MLX/CUDA ratio: {nf['median_ratio']:.4f}")
        print(f"  Std:                   {nf['std_ratio']:.4f}")
        print(f"  Range:                 [{nf['min_ratio']:.4f}, {nf['max_ratio']:.4f}]")
        print(f"  N models:              {nf['n_models']}")
    else:
        print("  [WARN] No normalization factors computed (no overlapping models)")

    for m, data in sorted(hw_ratios["per_model"].items()):
        print(f"    {m:20s}: CUDA={data['cuda_p50_ms']:8.1f}ms  "
              f"MLX={data['mlx_p50_ms']:8.1f}ms  ratio={data['ratio_mlx_over_cuda']:.3f}")

    # --- MoE normalized estimates ---
    print("\n--- MoE CUDA Estimates (via normalization) ---")
    moe_estimates = estimate_moe_on_cuda(mlx_data, nf)
    if moe_estimates:
        print(f"  Model: {moe_estimates.get('model', 'N/A')}")
        print(f"  Total params: {moe_estimates.get('total_params_b', 0):.0f}B, "
              f"Active: {moe_estimates.get('active_params_b', 0):.0f}B")
        print(f"  MLX p50:  {moe_estimates.get('mlx_p50_ms', 0):.1f}ms")
        print(f"  CUDA est: {moe_estimates.get('cuda_est_p50_ms', 0):.1f}ms")
        for tier_name in TIERS:
            key = f"cuda_est_s_at_slo_{tier_name}"
            val = moe_estimates.get(key, 0)
            print(f"  CUDA est S@SLO {tier_name}: {val:.4f}")

    # --- MLX S@SLO table ---
    print("\n--- MLX S@SLO Results ---")
    mlx_slo_table = {}
    for model_name in sorted(mlx_data.keys()):
        agg = mlx_data[model_name]["slo"]["aggregate"]
        mlx_slo_table[model_name] = {
            "accuracy": agg["accuracy"],
            "s_at_slo_interactive": agg["s_at_slo_interactive"],
            "s_at_slo_standard": agg["s_at_slo_standard"],
            "s_at_slo_batch": agg["s_at_slo_batch"],
            "total_params_b": MODEL_PARAMS_TOTAL.get(model_name, 0),
            "active_params_b": MODEL_PARAMS_ACTIVE.get(model_name, 0),
            "is_moe": IS_MOE.get(model_name, False),
            "p50_ms": mlx_data[model_name]["latency_stats"].get("p50_ms", 0),
        }
        print(f"  {model_name:20s}: acc={agg['accuracy']:.3f}  "
              f"S@2s={agg['s_at_slo_interactive']:.3f}  "
              f"S@5s={agg['s_at_slo_standard']:.3f}  "
              f"S@30s={agg['s_at_slo_batch']:.3f}")

    # --- Rank correlations ---
    print("\n--- Cross-Hardware Rank Correlations ---")
    rank_comp = compare_rank_orderings(cuda_data, mlx_data)
    if "accuracy_rank_correlation" in rank_comp:
        acc_r = rank_comp["accuracy_rank_correlation"]
        print(f"  Accuracy rank rho: {acc_r['rho']:+.4f}  "
              f"CI=[{acc_r['ci_lower']:+.4f}, {acc_r['ci_upper']:+.4f}]  "
              f"p={acc_r['p_value']:.4f}")
        lat_r = rank_comp["latency_rank_correlation"]
        print(f"  Latency rank rho:  {lat_r['rho']:+.4f}  "
              f"CI=[{lat_r['ci_lower']:+.4f}, {lat_r['ci_upper']:+.4f}]  "
              f"p={lat_r['p_value']:.4f}")
        for tier_name, sr in rank_comp["slo_rank_correlations"].items():
            print(f"  S@SLO {tier_name:12s} rho: {sr['rho']:+.4f}  "
                  f"CI=[{sr['ci_lower']:+.4f}, {sr['ci_upper']:+.4f}]  "
                  f"p={sr['p_value']:.4f}")

    # --- Write output JSON ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "hardware_ratios": hw_ratios,
        "moe_cuda_estimates": moe_estimates,
        "mlx_slo_table": mlx_slo_table,
        "rank_correlations": rank_comp,
        "tiers": dict(TIERS),
        "overlap_models": list(OVERLAP_MODELS.keys()),
        "all_mlx_models": sorted(mlx_data.keys()),
    }
    out_json = OUT_DIR / "hardware_comparison.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_json}")

    # --- Write pgfplots .dat files ---
    print("\n--- Generating pgfplots .dat files ---")
    DAT_DIR.mkdir(parents=True, exist_ok=True)
    write_latency_scatter_total(mlx_data, cuda_data, DAT_DIR / "p5_latency_scatter_total.dat")
    write_latency_scatter_active(mlx_data, DAT_DIR / "p5_latency_scatter_active.dat")
    write_slo_comparison(mlx_data, DAT_DIR / "p5_slo_comparison.dat")

    print("\nDone!")


if __name__ == "__main__":
    main()
