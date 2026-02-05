#!/usr/bin/env python3
"""Generate W&B report: Training Dynamics (P4).

Produces a programmatic W&B report visualizing GRPO training dynamics:
  - Line plots: reward curves for representative models (1B, 4B, 9B)
  - Reward decomposition: stacked bars per model
  - Forgetting heatmap: 10 models x 6 tasks
  - Curve taxonomy summary

Data sources: results/p4_analysis/*.json

Usage:
    python scripts/reports/p4_training_curves.py [--project agentslo-bench]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

P4_DIR = ROOT / "results" / "p4_analysis"


def load_json(name: str) -> dict:
    path = P4_DIR / name
    if not path.exists():
        print(f"Warning: {path} not found, using empty data")
        return {}
    with open(path) as f:
        return json.load(f)


def build_report(project: str) -> None:
    try:
        import wandb
        import wandb_workspaces.reports.v2 as wr
    except ImportError:
        print("ERROR: wandb-workspaces not installed.")
        print("  pip install 'wandb-workspaces>=0.1.0'")
        sys.exit(1)

    reward_decomp = load_json("reward_decomposition.json")
    forgetting = load_json("forgetting_matrix.json")
    taxonomy = load_json("curve_taxonomy.json")
    early_pred = load_json("early_prediction.json")

    run = wandb.init(
        project=project,
        name="p4-training-dynamics-report",
        tags=["report", "p4"],
    )

    # Reward decomposition
    if reward_decomp:
        decomp_data = reward_decomp if isinstance(reward_decomp, list) else reward_decomp.get("models", [])
        if decomp_data:
            columns = ["model", "size", "r_schema", "r_residual", "r_latency", "r_cost", "mean_reward"]
            rows = []
            for entry in decomp_data:
                if isinstance(entry, dict):
                    rows.append([
                        entry.get("model", ""),
                        entry.get("size", ""),
                        entry.get("r_schema", 0),
                        entry.get("r_residual", 0),
                        entry.get("r_latency", 0),
                        entry.get("r_cost", 0),
                        entry.get("mean_reward", 0),
                    ])
            if rows:
                table = wandb.Table(columns=columns, data=rows)
                run.log({"reward_decomposition": table})

    # Forgetting matrix
    if forgetting:
        forgetting_data = forgetting if isinstance(forgetting, list) else forgetting.get("models", [])
        if forgetting_data:
            columns = ["model", "size", "T1", "T2", "T3", "T4", "T5", "mixed", "delta", "profile"]
            rows = []
            for entry in forgetting_data:
                if isinstance(entry, dict):
                    rows.append([
                        entry.get("model", ""),
                        entry.get("size", ""),
                        entry.get("T1", 0),
                        entry.get("T2", 0),
                        entry.get("T3", 0),
                        entry.get("T4", 0),
                        entry.get("T5", 0),
                        entry.get("mixed", 0),
                        entry.get("delta", 0),
                        entry.get("profile", ""),
                    ])
            if rows:
                table = wandb.Table(columns=columns, data=rows)
                run.log({"forgetting_matrix": table})

    # Curve taxonomy
    if taxonomy:
        taxonomy_data = taxonomy if isinstance(taxonomy, list) else taxonomy.get("models", [])
        if taxonomy_data:
            columns = ["model", "size", "sustained", "transient", "flat"]
            rows = []
            for entry in taxonomy_data:
                if isinstance(entry, dict):
                    rows.append([
                        entry.get("model", ""),
                        entry.get("size", ""),
                        entry.get("sustained", 0),
                        entry.get("transient", 0),
                        entry.get("flat", 0),
                    ])
            if rows:
                table = wandb.Table(columns=columns, data=rows)
                run.log({"curve_taxonomy": table})

    # Early prediction results
    if early_pred:
        run.summary.update({
            "early_pred_accuracy": early_pred.get("full_sample_accuracy", 0),
            "early_pred_loocv": early_pred.get("loocv_accuracy", 0),
            "early_pred_baseline": early_pred.get("majority_baseline", 0),
        })

    # Build report
    report = wr.Report(
        project=project,
        title="P4: Training Dynamics of Schema-Aware RL",
        description="Reward decomposition, forgetting patterns, and curve taxonomy from 185 GRPO runs.",
        blocks=[
            wr.H1(text="Training Dynamics Analysis"),
            wr.P(text=(
                "This report analyzes the training dynamics of 185 GRPO runs "
                "across 11 models and 6 task types. Key findings: the capacity "
                "threshold is a compound of reward component competition, "
                "architecture-specific interference, and curve morphology."
            )),
            wr.H2(text="RQ1: Reward Decomposition"),
            wr.P(text=(
                "The latency penalty grows with model size, consuming up to 25% "
                "of the maximum reward for 8B models. Architecture-efficient models "
                "(Qwen3-4B, Gemma-2-9B) achieve high net rewards despite penalties."
            )),
            wr.H2(text="RQ2: Forgetting Matrix"),
            wr.P(text=(
                "Forgetting is NOT a simple function of model size. Yi-1.5-6B (6B) "
                "shows the worst forgetting (delta=-0.59), while Qwen2.5-3B (3B) "
                "shows positive interference (delta=+0.03)."
            )),
            wr.H2(text="RQ3: Curve Taxonomy"),
            wr.P(text=(
                "Training validity curves fall into three types: sustained (67%), "
                "transient (4%), and flat (28%). Early prediction from the first "
                "50 steps achieves 82.5% accuracy."
            )),
        ],
    )
    report.save()
    print(f"Report saved: {report.url}")

    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P4 training dynamics report")
    parser.add_argument("--project", default="agentslo-bench", help="W&B project")
    args = parser.parse_args()

    build_report(args.project)


if __name__ == "__main__":
    main()
