#!/usr/bin/env python3
"""Generate Weave-native model comparison dashboard.

Creates Weave Evaluations for all 13 models from P1 predictions,
enabling side-by-side comparison via the Weave UI with:
  - Per-model SLO compliance at 3 tiers
  - Per-task drill-down with individual prediction inspection
  - Latency distribution visualization

Usage:
    python scripts/reports/weave_model_comparison.py [--project agentslo-bench] [--all]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent_stable_slo.logging.weave_integration import (
    AccuracyScorer,
    JSONValidityScorer,
    SLOScorer,
    init_weave,
    _weave_available,
)

TIERS_JSON = ROOT / "results" / "p3_analysis" / "real_slo_tiers.json"
DEFAULT_PRED_DIR = ROOT / "out" / "p1_full_eval"

TIER_DEADLINES = {
    "interactive": 2_000,
    "standard": 5_000,
    "batch": 30_000,
}


def load_tier_data() -> dict:
    with open(TIERS_JSON) as f:
        return json.load(f)


def find_prediction_files(pred_dir: Path) -> dict[str, Path]:
    """Find all model prediction files."""
    models = {}
    for subdir in sorted(pred_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for jsonl in subdir.rglob("predictions.jsonl"):
            model_name = jsonl.parent.name
            models[model_name] = jsonl
    return models


def create_comparison(project: str, pred_dir: Path) -> None:
    if not _weave_available():
        print("ERROR: weave not installed. pip install 'weave>=0.51'")
        sys.exit(1)

    import weave

    print(f"Initializing Weave project: {project}")
    if not init_weave(project):
        print("ERROR: Failed to initialize Weave. Check WANDB_API_KEY.")
        sys.exit(1)

    # Load pre-computed summary data for models without prediction files
    tier_data = load_tier_data()

    # Try to find raw prediction files
    pred_files = find_prediction_files(pred_dir)

    if pred_files:
        print(f"Found {len(pred_files)} model prediction files")
        _run_with_predictions(pred_files)
    else:
        print("No prediction files found, creating comparison from summary data")
        _run_from_summary(tier_data)

    print(f"\nView comparison at: https://wandb.ai/weave/{project}")


def _run_with_predictions(pred_files: dict[str, Path]) -> None:
    """Create Weave evaluations from raw prediction files."""
    import weave

    scorers = [
        SLOScorer(tier_ms=TIER_DEADLINES["interactive"]),
        SLOScorer(tier_ms=TIER_DEADLINES["standard"]),
        SLOScorer(tier_ms=TIER_DEADLINES["batch"]),
        JSONValidityScorer(),
        AccuracyScorer(),
    ]

    for model_name, pred_path in sorted(pred_files.items()):
        print(f"\nProcessing: {model_name}")
        rows = []
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if not rows:
            print(f"  Skipping {model_name}: no predictions")
            continue

        dataset = weave.Dataset(name=model_name, rows=rows)
        print(f"  Dataset: {len(rows)} predictions")

        @weave.op()
        def passthrough(output: dict, **kwargs) -> dict:
            return output

        evaluation = weave.Evaluation(
            dataset=dataset,
            scorers=scorers,
        )

        import asyncio
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(evaluation.evaluate(passthrough))
            print(f"  Evaluation complete: {model_name}")
        except Exception as e:
            print(f"  ERROR evaluating {model_name}: {e}")
        finally:
            loop.close()


def _run_from_summary(tier_data: dict) -> None:
    """Create Weave datasets from pre-computed summary statistics.

    When raw predictions aren't available, we create synthetic datasets
    from the aggregated data in real_slo_tiers.json for visualization.
    """
    import weave

    models = tier_data["models"]
    rows = []
    for model_name, info in sorted(models.items()):
        agg = info["aggregate"]
        rows.append({
            "model": model_name,
            "accuracy": round(agg["accuracy"] * 100, 1),
            "s_at_slo_2s": round(agg.get("s_at_slo_interactive", 0) * 100, 1),
            "s_at_slo_5s": round(agg.get("s_at_slo_standard", 0) * 100, 1),
            "s_at_slo_30s": round(agg.get("s_at_slo_batch", 0) * 100, 1),
            "avg_latency_ms": round(agg["avg_latency_ms"]),
            "p95_latency_ms": round(agg["p95_latency_ms"]),
        })

    dataset = weave.Dataset(name="agentslo-bench-13models", rows=rows)
    print(f"Created summary dataset with {len(rows)} models")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Weave model comparison")
    parser.add_argument("--project", default="agentslo-bench", help="Weave project")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help="Predictions directory",
    )
    args = parser.parse_args()

    create_comparison(args.project, args.pred_dir)


if __name__ == "__main__":
    main()
