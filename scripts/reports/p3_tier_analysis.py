#!/usr/bin/env python3
"""Generate W&B report: Per-Tier S@SLO Analysis (P3).

Produces a programmatic W&B report with per-tier breakdown:
  - 3-panel scatter: accuracy vs S@SLO at each tier (2s, 5s, 30s)
  - Heatmap: 13 models x 5 tasks x 3 tiers
  - Spearman rho with CI visualization

Data source: results/p3_analysis/real_slo_tiers.json

Usage:
    python scripts/reports/p3_tier_analysis.py [--project agentslo-bench]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

TIERS_JSON = ROOT / "results" / "p3_analysis" / "real_slo_tiers.json"

TIER_KEYS = {
    "interactive": ("s_at_slo_interactive", 2),
    "standard": ("s_at_slo_standard", 5),
    "batch": ("s_at_slo_batch", 30),
}

TASKS = ["T1", "T1v", "T2", "T2v", "T3", "T4", "T5"]


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
    spearman = data["spearman"]

    run = wandb.init(
        project=project,
        name="p3-tier-analysis-report",
        tags=["report", "p3"],
    )

    # Per-tier scatter data
    for tier_name, (slo_key, deadline_s) in TIER_KEYS.items():
        rows = []
        for name, info in sorted(models.items()):
            agg = info["aggregate"]
            rows.append([
                name,
                round(agg["accuracy"] * 100, 1),
                round(agg.get(slo_key, 0) * 100, 1),
            ])

        table = wandb.Table(
            columns=["model", "accuracy_pct", f"s_at_slo_{deadline_s}s_pct"],
            data=rows,
        )
        run.log({f"scatter_{tier_name}": table})

    # Heatmap: model x task x tier
    heatmap_rows = []
    for name, info in sorted(models.items()):
        per_task = info.get("per_task", {})
        for task in TASKS:
            task_data = per_task.get(task, {})
            for tier_name, (slo_key, deadline_s) in TIER_KEYS.items():
                val = task_data.get(slo_key, 0)
                heatmap_rows.append([name, task, f"{deadline_s}s", round(val * 100, 1)])

    heatmap_table = wandb.Table(
        columns=["model", "task", "tier", "s_at_slo_pct"],
        data=heatmap_rows,
    )
    run.log({"heatmap_model_task_tier": heatmap_table})

    # Spearman summary
    spearman_rows = []
    for tier_name in ["interactive", "standard", "batch"]:
        s = spearman[tier_name]
        spearman_rows.append([
            tier_name,
            s["rho"],
            s["ci_lower"],
            s["ci_upper"],
            s["p_value"],
        ])

    spearman_table = wandb.Table(
        columns=["tier", "rho", "ci_lower", "ci_upper", "p_value"],
        data=spearman_rows,
    )
    run.log({"spearman_correlations": spearman_table})

    run.summary.update({
        "spearman_rho_interactive": spearman["interactive"]["rho"],
        "spearman_rho_standard": spearman["standard"]["rho"],
        "spearman_rho_batch": spearman["batch"]["rho"],
    })

    # Build programmatic report
    report = wr.Report(
        project=project,
        title="P3: AgentSLO-Bench Tier Analysis",
        description="Per-tier Success@SLO breakdown across 13 models, 7 tasks, 3 SLO tiers.",
        blocks=[
            wr.H1(text="AgentSLO-Bench: Per-Tier Analysis"),
            wr.P(text=(
                "This report breaks down Success@SLO across three deployment tiers "
                "(Interactive 2s, Standard 5s, Batch 30s) for all 13 models."
            )),
            wr.H2(text="Accuracy vs S@SLO by Tier"),
            wr.P(text=(
                "Three scatter panels show how the accuracy-deployment disconnect "
                "manifests at each tier. At the Interactive tier (2s), only 1B achieves "
                "non-zero S@SLO."
            )),
            wr.H2(text="Model x Task x Tier Heatmap"),
            wr.P(text="Per-task S@SLO reveals that task difficulty inverts under SLO constraints."),
            wr.H2(text="Spearman Rank Correlations"),
            wr.P(text=(
                f"Interactive: rho={spearman['interactive']['rho']:.2f} "
                f"(CI: [{spearman['interactive']['ci_lower']:.2f}, "
                f"{spearman['interactive']['ci_upper']:.2f}], "
                f"p={spearman['interactive']['p_value']:.3f})\n"
                f"Standard: rho={spearman['standard']['rho']:.2f} "
                f"(CI: [{spearman['standard']['ci_lower']:.2f}, "
                f"{spearman['standard']['ci_upper']:.2f}], "
                f"p={spearman['standard']['p_value']:.3f})\n"
                f"Batch: rho={spearman['batch']['rho']:.2f} "
                f"(CI: [{spearman['batch']['ci_lower']:.2f}, "
                f"{spearman['batch']['ci_upper']:.2f}], "
                f"p={spearman['batch']['p_value']:.3f})"
            )),
        ],
    )
    report.save()
    print(f"Report saved: {report.url}")

    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P3 tier analysis report")
    parser.add_argument("--project", default="agentslo-bench", help="W&B project")
    args = parser.parse_args()

    if not TIERS_JSON.exists():
        print(f"Data file not found: {TIERS_JSON}")
        sys.exit(1)

    build_report(args.project)


if __name__ == "__main__":
    main()
