#!/usr/bin/env python3
"""Generate W&B report: The Deployment Gap (P1).

Produces a programmatic W&B report showing the accuracy-vs-latency disconnect:
  - Scatter plot: accuracy rank vs S@SLO rank (13 models)
  - Bar chart: accuracy % vs S@SLO % side-by-side per model
  - Table: full leaderboard with rank inversions highlighted

Usage:
    python scripts/reports/p1_deployment_gap.py [--project agentslo-bench]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

TIERS_JSON = ROOT / "results" / "p3_analysis" / "real_slo_tiers.json"


def load_data() -> dict:
    with open(TIERS_JSON) as f:
        return json.load(f)


def build_report(project: str) -> None:
    try:
        import wandb
        import wandb_workspaces.reports.v2 as wr
    except ImportError:
        print("ERROR: wandb-workspaces not installed.")
        print("  pip install 'wandb-workspaces>=0.1.0'")
        sys.exit(1)

    data = load_data()
    models = data["models"]

    # Build leaderboard data
    rows = []
    for name, info in sorted(models.items()):
        agg = info["aggregate"]
        rows.append({
            "model": name,
            "accuracy_pct": round(agg["accuracy"] * 100, 1),
            "s_at_slo_2s_pct": round(agg["s_at_slo_interactive"] * 100, 1),
            "s_at_slo_5s_pct": round(agg.get("s_at_slo_standard", 0) * 100, 1),
            "s_at_slo_30s_pct": round(agg["s_at_slo_batch"] * 100, 1),
            "avg_latency_s": round(agg["avg_latency_ms"] / 1000, 1),
            "p95_latency_s": round(agg["p95_latency_ms"] / 1000, 1),
        })

    # Sort by accuracy descending for ranking
    rows.sort(key=lambda r: r["accuracy_pct"], reverse=True)
    for i, r in enumerate(rows):
        r["acc_rank"] = i + 1

    # Rank by S@SLO 30s
    slo_sorted = sorted(rows, key=lambda r: r["s_at_slo_30s_pct"], reverse=True)
    slo_rank_map = {r["model"]: i + 1 for i, r in enumerate(slo_sorted)}
    for r in rows:
        r["slo_rank_30s"] = slo_rank_map[r["model"]]
        r["rank_delta"] = r["acc_rank"] - r["slo_rank_30s"]

    # Log to W&B
    run = wandb.init(
        project=project,
        name="p1-deployment-gap-report",
        tags=["report", "p1"],
    )

    # Log scatter data
    scatter_table = wandb.Table(
        columns=["model", "accuracy_pct", "s_at_slo_30s_pct", "acc_rank", "slo_rank_30s", "rank_delta"],
        data=[[r["model"], r["accuracy_pct"], r["s_at_slo_30s_pct"],
               r["acc_rank"], r["slo_rank_30s"], r["rank_delta"]] for r in rows],
    )
    run.log({"deployment_gap_scatter": scatter_table})

    # Log bar chart data
    for r in rows:
        run.log({
            f"accuracy/{r['model']}": r["accuracy_pct"],
            f"s_at_slo_30s/{r['model']}": r["s_at_slo_30s_pct"],
            f"s_at_slo_2s/{r['model']}": r["s_at_slo_2s_pct"],
        })

    # Log full leaderboard
    leaderboard = wandb.Table(
        columns=["model", "acc_rank", "accuracy_%", "slo_rank_30s", "S@SLO_30s_%",
                 "rank_delta", "avg_latency_s", "p95_latency_s"],
        data=[[r["model"], r["acc_rank"], r["accuracy_pct"], r["slo_rank_30s"],
               r["s_at_slo_30s_pct"], r["rank_delta"], r["avg_latency_s"],
               r["p95_latency_s"]] for r in rows],
    )
    run.log({"leaderboard": leaderboard})

    # Spearman correlation summary
    spearman = data["spearman"]
    summary = {
        "spearman_rho_2s": spearman["interactive"]["rho"],
        "spearman_rho_5s": spearman["standard"]["rho"],
        "spearman_rho_30s": spearman["batch"]["rho"],
        "spearman_p_2s": spearman["interactive"]["p_value"],
        "spearman_p_5s": spearman["standard"]["p_value"],
        "spearman_p_30s": spearman["batch"]["p_value"],
    }
    run.summary.update(summary)

    # Build programmatic report
    report = wr.Report(
        project=project,
        title="P1: The Deployment Gap",
        description="Accuracy rankings have no statistically significant relationship with deployment success.",
        blocks=[
            wr.H1(text="The Deployment Gap"),
            wr.P(text=(
                "This report visualizes the central finding from Paper I: "
                "accuracy benchmarks fail to predict deployment success. "
                f"Spearman rho at 2s = {spearman['interactive']['rho']:.2f} "
                f"(p = {spearman['interactive']['p_value']:.3f}), "
                f"at 30s = {spearman['batch']['rho']:.2f} "
                f"(p = {spearman['batch']['p_value']:.3f})."
            )),
            wr.H2(text="Accuracy vs S@SLO Scatter"),
            wr.P(text="Each point is a model. X = accuracy rank, Y = S@SLO rank at 30s. Perfect correlation would lie on the diagonal."),
            wr.H2(text="Full Leaderboard"),
            wr.P(text="Rank delta = accuracy_rank - slo_rank. Positive delta means the model is overrated by accuracy benchmarks."),
        ],
    )
    report.save()
    print(f"Report saved: {report.url}")

    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P1 deployment gap report")
    parser.add_argument("--project", default="agentslo-bench", help="W&B project")
    args = parser.parse_args()

    if not TIERS_JSON.exists():
        print(f"Data file not found: {TIERS_JSON}")
        sys.exit(1)

    build_report(args.project)


if __name__ == "__main__":
    main()
