"""AgentSLO-Bench leaderboard generation.

Computes dual rankings (accuracy vs Success@SLO) and Spearman correlations
across SLO tiers, with optional bootstrapped confidence intervals.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from .slo_tiers import TIERS, TIER_MAP
from .benchmark_runner import BenchmarkResult, compute_from_p1_data

# Path to built-in P1 baseline data
P1_DATA_PATH = Path(__file__).parent.parent.parent / "out" / "p1_comprehensive_20260118" / "all_results.json"


def load_p1_baseline() -> dict:
    """Load the built-in 13-model P1 baseline results."""
    with open(P1_DATA_PATH, "r") as f:
        return json.load(f)


def compute_spearman_rho(ranks_a: list[float], ranks_b: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(ranks_a)
    if n < 3:
        return 0.0
    d_sq = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def rank_values(values: list[float], descending: bool = True) -> list[float]:
    """Assign ranks to values (1 = best). Handles ties with average rank."""
    indexed = sorted(enumerate(values), key=lambda x: x[1], reverse=descending)
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


def bootstrap_spearman(
    values_a: list[float],
    values_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Compute Spearman rho with bootstrapped 95% CI and analytical p-value.

    Args:
        values_a: First set of values (e.g., accuracy scores).
        values_b: Second set of values (e.g., S@SLO scores).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with rho, ci_lower, ci_upper, p_value, n_models.
    """
    n = len(values_a)
    rng = random.Random(seed)

    # Observed rho from ranks
    ranks_a = rank_values(values_a)
    ranks_b = rank_values(values_b)
    observed_rho = compute_spearman_rho(ranks_a, ranks_b)

    # Bootstrap
    boot_rhos = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        boot_a = [values_a[i] for i in indices]
        boot_b = [values_b[i] for i in indices]
        ra = rank_values(boot_a)
        rb = rank_values(boot_b)
        boot_rhos.append(compute_spearman_rho(ra, rb))

    boot_rhos.sort()
    ci_lower = boot_rhos[int(0.025 * n_bootstrap)]
    ci_upper = boot_rhos[int(0.975 * n_bootstrap)]

    # Analytical p-value (t-distribution approximation)
    if n >= 3 and abs(observed_rho) < 1.0:
        t_stat = observed_rho * math.sqrt((n - 2) / (1 - observed_rho * observed_rho))
        z = abs(t_stat)
        p_value = 2 * 0.5 * math.erfc(z / math.sqrt(2))
    else:
        p_value = 0.0 if abs(observed_rho) >= 1.0 else 1.0

    return {
        "rho": round(observed_rho, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "n_bootstrap": n_bootstrap,
        "n_models": n,
    }


def generate_leaderboard(
    p1_data: dict | None = None,
) -> dict[str, Any]:
    """Generate full leaderboard from P1 baseline data.

    Returns dict with:
    - models: list of model names
    - tiers: {tier_name: {model: {accuracy_rank, slo_rank, accuracy, success_at_slo}}}
    - spearman: {tier_name: rho}
    - rank_inversions: {tier_name: count}
    """
    if p1_data is None:
        p1_data = load_p1_baseline()

    models = list(p1_data["models"].keys())

    # Compute per-model aggregated metrics across all tasks for each tier
    tier_data = {}
    for tier in TIERS:
        model_metrics = []
        for model_name in models:
            model_info = p1_data["models"][model_name]
            results = compute_from_p1_data(model_info, model_name)

            # Average across tasks
            total_correct = 0
            total_slo = 0
            total_count = 0
            for r in results:
                tr = r.tier_results.get(tier.name)
                if tr:
                    total_correct += tr.correct
                    total_slo += tr.success_at_slo
                    total_count += tr.total

            avg_accuracy = 100.0 * total_correct / max(1, total_count)
            avg_slo = 100.0 * total_slo / max(1, total_count)
            model_metrics.append({
                "model": model_name,
                "accuracy": round(avg_accuracy, 1),
                "success_at_slo": round(avg_slo, 1),
            })

        accuracies = [m["accuracy"] for m in model_metrics]
        slo_scores = [m["success_at_slo"] for m in model_metrics]

        acc_ranks = rank_values(accuracies)
        slo_ranks = rank_values(slo_scores)

        rho = compute_spearman_rho(acc_ranks, slo_ranks)
        inversions = sum(1 for a, s in zip(acc_ranks, slo_ranks) if abs(a - s) > 2)

        tier_data[tier.name] = {
            "metrics": model_metrics,
            "accuracy_ranks": acc_ranks,
            "slo_ranks": slo_ranks,
            "spearman_rho": round(rho, 3),
            "rank_inversions": inversions,
        }

    return {
        "models": models,
        "tiers": tier_data,
    }


def format_markdown(leaderboard: dict) -> str:
    """Format leaderboard as markdown table."""
    lines = ["# AgentSLO-Bench Leaderboard", ""]

    for tier_name, data in leaderboard["tiers"].items():
        tier = TIER_MAP[tier_name]
        lines.append(f"## {tier.name.title()} Tier ({tier.deadline_ms/1000:.0f}s SLO)")
        lines.append(f"Spearman rho = {data['spearman_rho']:.3f} | Rank inversions: {data['rank_inversions']}")
        lines.append("")
        lines.append("| Rank | Model | Accuracy (%) | S@SLO (%) | Acc Rank | SLO Rank |")
        lines.append("|------|-------|-------------|-----------|----------|----------|")

        metrics = data["metrics"]
        acc_ranks = data["accuracy_ranks"]
        slo_ranks = data["slo_ranks"]
        sorted_idx = sorted(range(len(metrics)), key=lambda i: slo_ranks[i])

        for rank, idx in enumerate(sorted_idx, 1):
            m = metrics[idx]
            lines.append(
                f"| {rank} | {m['model']} | {m['accuracy']:.1f} | "
                f"{m['success_at_slo']:.1f} | {acc_ranks[idx]:.0f} | {slo_ranks[idx]:.0f} |"
            )
        lines.append("")

    return "\n".join(lines)


def format_latex(leaderboard: dict) -> str:
    """Format leaderboard as LaTeX table."""
    lines = []
    for tier_name, data in leaderboard["tiers"].items():
        tier = TIER_MAP[tier_name]
        lines.append(f"% {tier.name.title()} Tier ({tier.deadline_ms/1000:.0f}s)")
        lines.append(f"% Spearman rho = {data['spearman_rho']:.3f}")
        lines.append("\\begin{tabular}{lrrrr}")
        lines.append("\\toprule")
        lines.append("\\textbf{Model} & \\textbf{Acc (\\%)} & \\textbf{S@SLO (\\%)} & \\textbf{Acc Rank} & \\textbf{SLO Rank} \\\\")
        lines.append("\\midrule")

        metrics = data["metrics"]
        slo_ranks = data["slo_ranks"]
        acc_ranks = data["accuracy_ranks"]
        sorted_idx = sorted(range(len(metrics)), key=lambda i: slo_ranks[i])

        for idx in sorted_idx:
            m = metrics[idx]
            name = m["model"].replace("_", "\\_").replace("-", "{-}")
            lines.append(
                f"{name} & {m['accuracy']:.1f} & {m['success_at_slo']:.1f} & "
                f"{acc_ranks[idx]:.0f} & {slo_ranks[idx]:.0f} \\\\"
            )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("")

    return "\n".join(lines)
