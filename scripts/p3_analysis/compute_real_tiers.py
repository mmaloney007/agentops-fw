#!/usr/bin/env python3
"""Compute real Success@SLO at three tiers from per-request prediction data.

Reads the p1_full_eval predictions.jsonl files (42,900 total: 13 models x 3,300 each)
and computes Success@SLO directly from per-request latencies — no extrapolation.

Also computes bootstrapped Spearman rho with 95% CI and p-values.

Output:
  - results/p3_analysis/real_slo_tiers.json
  - results/p3_analysis/p3_scatter_2s.dat, p3_scatter_5s.dat, p3_scatter_30s.dat
  - results/p3_analysis/p3_heatmap.dat
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = BASE_DIR / "out" / "p1_full_eval" / "p1_13models_5tasks_20260124_221550"
OUT_DIR = BASE_DIR / "results" / "p3_analysis"

TIERS = {"interactive": 2000, "standard": 5000, "batch": 30000}

# Map LM Studio directory names to short model names
DIR_TO_MODEL = {
    "lmstudio_llama-3.2-1b-instruct": "llama-3.2-1b",
    "lmstudio_meta-llama_-_llama-3.2-3b-instruct": "llama-3.2-3b",
    "lmstudio_qwen2.5-3b-instruct": "qwen2.5-3b",
    "lmstudio_phi-3-mini-4k-instruct": "phi-3-mini",
    "lmstudio_qwen3-4b": "qwen3-4b",
    "lmstudio_01-ai_-_yi-1.5-6b-chat": "yi-1.5-6b",
    "lmstudio_mistralai_-_mistral-7b-instruct-v0.3": "mistral-7b",
    "lmstudio_falcon-mamba-7b-instruct": "falcon-mamba-7b",
    "lmstudio_openai-gpt-oss-20b": "gpt-oss-20b",
    "lmstudio_ministral-8b-instruct-2410": "ministral-8b",
    "lmstudio_meta-llama-llama-3.1-8b-instruct": "llama-3.1-8b",
    "lmstudio_google-gemma-2-9b": "gemma-2-9b",
    "lmstudio_google-gemma-3-12b": "gemma-3-12b",
}

# Map schema_path to task ID
SCHEMA_TO_TASK = {
    "tasks/schemas/clinc_nlu_schema.json": "T1",
    "tasks/schemas/t1_incident_schema.json": "T1v",
    "tasks/schemas/t2_summary_schema.json": "T2",
    "tasks/schemas/hotpot_explainer_schema.json": "T2v",
    "tasks/schemas/t3_tool_call_schema.json": "T3",
    "tasks/schemas/t4_function_call_schema.json": "T4",
    "tasks/schemas/t5_patch_schema.json": "T5",
}

# Model sizes in billions for scatter plot coloring
MODEL_SIZES = {
    "llama-3.2-1b": 1.0, "llama-3.2-3b": 3.0, "qwen2.5-3b": 3.0,
    "phi-3-mini": 3.8, "qwen3-4b": 4.0, "yi-1.5-6b": 6.0,
    "mistral-7b": 7.0, "falcon-mamba-7b": 7.0, "gpt-oss-20b": 20.0,
    "ministral-8b": 8.0, "llama-3.1-8b": 8.0, "gemma-2-9b": 9.0,
    "gemma-3-12b": 12.0,
}

# Size range for scatter coloring
def size_range(size_b: float) -> str:
    if size_b <= 3:
        return "small"
    elif size_b <= 7:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------
def load_all_predictions() -> dict[str, list[dict]]:
    """Load all predictions, keyed by model short name."""
    all_preds: dict[str, list[dict]] = {}
    for model_dir in sorted(EVAL_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = DIR_TO_MODEL.get(model_dir.name)
        if model_name is None:
            print(f"  [WARN] Unknown dir: {model_dir.name}")
            continue
        pred_file = model_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue
        preds = []
        with open(pred_file) as f:
            for line in f:
                preds.append(json.loads(line))
        all_preds[model_name] = preds
        print(f"  Loaded {len(preds):,} predictions for {model_name}")
    return all_preds


# ---------------------------------------------------------------------------
# Compute S@SLO per request
# ---------------------------------------------------------------------------
def is_task_correct(pred: dict) -> bool:
    """Determine if a prediction is task-correct.

    For most tasks: uses overall_field_accuracy > 0 as the accuracy signal.
    For T4 (function routing): the gold format uses {func_name: {args}} keys,
    which doesn't match the prediction schema's {name, arguments} format.
    We check if the predicted function name matches the gold's key instead.
    Also requires json_valid == 1.0 for all tasks.
    """
    json_valid = pred.get("metrics", {}).get("json_valid", 0) == 1.0
    if not json_valid:
        return False

    schema = pred.get("schema_path", "")

    # T4 special handling: gold = {"func_name": {"arg": [val]}}
    if schema == "tasks/schemas/t4_function_call_schema.json":
        gold = pred.get("gold", {})
        output = pred.get("output_json", {})
        if not gold or not output:
            return False
        # Gold key is the function name
        gold_func = list(gold.keys())[0] if gold else None
        pred_func = output.get("name", output.get("function_name", ""))
        return gold_func is not None and pred_func == gold_func

    ofa = pred.get("detailed", {}).get("overall_field_accuracy", 0)
    return ofa > 0


def compute_slo_metrics(preds: list[dict]) -> dict:
    """Compute per-task and aggregate S@SLO metrics for one model."""
    # Group by task
    by_task: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        task = SCHEMA_TO_TASK.get(p.get("schema_path", ""), "unknown")
        by_task[task].append(p)

    per_task = {}
    for task_id, task_preds in sorted(by_task.items()):
        task_metrics = {"count": len(task_preds)}
        correct_count = sum(1 for p in task_preds if is_task_correct(p))
        task_metrics["accuracy"] = round(correct_count / max(1, len(task_preds)), 4)

        for tier_name, deadline_ms in TIERS.items():
            on_time = sum(1 for p in task_preds if p["latency_ms"] <= deadline_ms)
            s_at_slo = sum(
                1 for p in task_preds
                if is_task_correct(p) and p["latency_ms"] <= deadline_ms
            )
            task_metrics[f"on_time_{tier_name}"] = round(on_time / max(1, len(task_preds)), 4)
            task_metrics[f"s_at_slo_{tier_name}"] = round(s_at_slo / max(1, len(task_preds)), 4)

        per_task[task_id] = task_metrics

    # Aggregate across all tasks
    total_preds = len(preds)
    total_correct = sum(1 for p in preds if is_task_correct(p))
    agg = {
        "total": total_preds,
        "accuracy": round(total_correct / max(1, total_preds), 4),
    }
    for tier_name, deadline_ms in TIERS.items():
        on_time = sum(1 for p in preds if p["latency_ms"] <= deadline_ms)
        s_at_slo = sum(
            1 for p in preds
            if is_task_correct(p) and p["latency_ms"] <= deadline_ms
        )
        agg[f"on_time_{tier_name}"] = round(on_time / max(1, total_preds), 4)
        agg[f"s_at_slo_{tier_name}"] = round(s_at_slo / max(1, total_preds), 4)

    # Latency stats
    latencies = [p["latency_ms"] for p in preds]
    latencies.sort()
    agg["avg_latency_ms"] = round(sum(latencies) / max(1, len(latencies)), 1)
    agg["p50_latency_ms"] = round(latencies[len(latencies) // 2], 1) if latencies else 0
    agg["p95_latency_ms"] = round(latencies[int(0.95 * len(latencies))], 1) if latencies else 0
    agg["p99_latency_ms"] = round(latencies[int(0.99 * len(latencies))], 1) if latencies else 0

    return {"per_task": per_task, "aggregate": agg}


# ---------------------------------------------------------------------------
# Spearman with bootstrap CI
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
    """Compute Spearman rank correlation from already-ranked data."""
    n = len(x)
    if n < 3:
        return 0.0
    d_sq = sum((a - b) ** 2 for a, b in zip(x, y))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def spearman_from_values(acc: list[float], slo: list[float]) -> float:
    """Compute Spearman rho from raw metric values."""
    return spearman_rho(rank_values(acc), rank_values(slo))


def spearman_p_value(rho: float, n: int) -> float:
    """Approximate two-tailed p-value for Spearman rho using t-distribution approximation."""
    if n < 3 or abs(rho) >= 1.0:
        return 0.0 if abs(rho) >= 1.0 else 1.0
    t_stat = rho * math.sqrt((n - 2) / (1 - rho * rho))
    # Use normal approximation for p-value (adequate for n >= 10)
    # Two-tailed p-value from t-statistic
    df = n - 2
    # Approximation: use the incomplete beta function relationship
    df / (df + t_stat * t_stat)
    # Simple approximation using normal CDF for large enough df
    z = abs(t_stat) * math.sqrt(1 - 1 / (4 * df) - 7 / (120 * df * df)) if df > 2 else abs(t_stat)
    # Standard normal CDF approximation (Abramowitz & Stegun)
    p_one_tail = 0.5 * math.erfc(z / math.sqrt(2))
    return 2 * p_one_tail


def bootstrap_spearman(
    accuracies: list[float],
    slo_scores: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap Spearman rho with 95% CI and p-value."""
    n = len(accuracies)
    rng = random.Random(seed)

    observed_rho = spearman_from_values(accuracies, slo_scores)

    # Bootstrap resampling
    boot_rhos = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        boot_acc = [accuracies[i] for i in indices]
        boot_slo = [slo_scores[i] for i in indices]
        boot_rhos.append(spearman_from_values(boot_acc, boot_slo))

    boot_rhos.sort()
    ci_lower = boot_rhos[int(0.025 * n_bootstrap)]
    ci_upper = boot_rhos[int(0.975 * n_bootstrap)]

    # Analytical p-value
    p_val = spearman_p_value(observed_rho, n)

    return {
        "rho": round(observed_rho, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_val, 4),
        "n_bootstrap": n_bootstrap,
        "n_models": n,
    }


# ---------------------------------------------------------------------------
# Generate pgfplots .dat files
# ---------------------------------------------------------------------------
def write_scatter_dat(models: dict, tier_name: str, deadline_key: str, out_path: Path):
    """Write accuracy-rank vs S@SLO-rank scatter data for pgfplots."""
    model_names = sorted(models.keys())
    accuracies = [models[m]["aggregate"]["accuracy"] * 100 for m in model_names]
    slo_scores = [models[m]["aggregate"][deadline_key] * 100 for m in model_names]

    acc_ranks = rank_values(accuracies)
    slo_ranks = rank_values(slo_scores)

    with open(out_path, "w") as f:
        f.write("# Accuracy_rank S_at_SLO_rank Accuracy S_at_SLO Size_B Size_Range Model\n")
        for i, m in enumerate(model_names):
            sr = size_range(MODEL_SIZES[m])
            f.write(
                f"{acc_ranks[i]:.1f} {slo_ranks[i]:.1f} "
                f"{accuracies[i]:.1f} {slo_scores[i]:.1f} "
                f"{MODEL_SIZES[m]:.1f} {sr} {m}\n"
            )
    print(f"  Wrote {out_path}")


def write_heatmap_dat(models: dict, out_path: Path):
    """Write per-task S@SLO heatmap data for TikZ."""
    # Use the primary task variants (T1, T2v=QA, T3, T4, T5)
    tasks = ["T1", "T2v", "T3", "T4", "T5"]
    task_labels = ["T1:Intent", "T2:QA", "T3:Tools", "T4:FuncRoute", "T5:Code"]
    model_order = [
        "llama-3.2-1b", "llama-3.2-3b", "qwen2.5-3b", "phi-3-mini",
        "qwen3-4b", "yi-1.5-6b", "mistral-7b", "falcon-mamba-7b",
        "gpt-oss-20b", "ministral-8b", "llama-3.1-8b", "gemma-2-9b",
        "gemma-3-12b",
    ]

    with open(out_path, "w") as f:
        f.write("# Model Task Tier S_at_SLO\n")
        for model in model_order:
            mdata = models.get(model, {}).get("per_task", {})
            for task, label in zip(tasks, task_labels):
                tdata = mdata.get(task, {})
                for tier_name in ["interactive", "standard", "batch"]:
                    val = tdata.get(f"s_at_slo_{tier_name}", 0) * 100
                    f.write(f"{model} {label} {tier_name} {val:.1f}\n")
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Computing real S@SLO from 42,900 per-request predictions")
    print("=" * 60)

    # Load all predictions
    all_preds = load_all_predictions()
    total = sum(len(v) for v in all_preds.values())
    print(f"\nTotal predictions loaded: {total:,}")

    # Compute per-model metrics
    models = {}
    for model_name, preds in sorted(all_preds.items()):
        models[model_name] = compute_slo_metrics(preds)
        agg = models[model_name]["aggregate"]
        print(
            f"  {model_name:20s}: acc={agg['accuracy']:.3f}  "
            f"S@SLO_2s={agg['s_at_slo_interactive']:.3f}  "
            f"S@SLO_5s={agg['s_at_slo_standard']:.3f}  "
            f"S@SLO_30s={agg['s_at_slo_batch']:.3f}"
        )

    # Compute Spearman correlations with bootstrap CI
    print("\n--- Spearman Correlations ---")
    model_names = sorted(models.keys())
    accuracies = [models[m]["aggregate"]["accuracy"] * 100 for m in model_names]

    spearman_results = {}
    for tier_name in TIERS:
        key = f"s_at_slo_{tier_name}"
        slo_scores = [models[m]["aggregate"][key] * 100 for m in model_names]
        result = bootstrap_spearman(accuracies, slo_scores)
        spearman_results[tier_name] = result
        print(
            f"  {tier_name:12s}: rho={result['rho']:+.4f}  "
            f"CI=[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]  "
            f"p={result['p_value']:.4f}"
        )

    # Build output JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": "p1_full_eval_42900",
        "source": str(EVAL_DIR),
        "total_predictions": total,
        "tiers": {k: v for k, v in TIERS.items()},
        "models": models,
        "spearman": spearman_results,
    }

    out_json = OUT_DIR / "real_slo_tiers.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_json}")

    # Generate pgfplots data files
    print("\n--- Generating pgfplots .dat files ---")
    write_scatter_dat(models, "interactive", "s_at_slo_interactive", OUT_DIR / "p3_scatter_2s.dat")
    write_scatter_dat(models, "standard", "s_at_slo_standard", OUT_DIR / "p3_scatter_5s.dat")
    write_scatter_dat(models, "batch", "s_at_slo_batch", OUT_DIR / "p3_scatter_30s.dat")
    write_heatmap_dat(models, OUT_DIR / "p3_heatmap.dat")

    print("\nDone!")


if __name__ == "__main__":
    main()
