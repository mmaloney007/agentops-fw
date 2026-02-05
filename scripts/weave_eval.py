#!/usr/bin/env python3
"""Retroactive evaluation of P1 predictions through Weave.

Loads predictions from out/p1_full_eval/*/predictions.jsonl,
creates Weave Datasets per model, and runs SLO scorers at
three tiers (2s, 5s, 30s) via weave.Evaluation.

Results appear in the Weave UI for interactive exploration.

Usage:
    python scripts/weave_eval.py --project agentslo-bench [--model MODEL] [--all]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_stable_slo.logging.weave_integration import (
    AccuracyScorer,
    JSONValidityScorer,
    SLOScorer,
    create_dataset_from_predictions,
    init_weave,
    run_retroactive_eval,
)

# Default predictions directory
DEFAULT_PRED_DIR = ROOT / "out" / "p1_full_eval"

# SLO tier deadlines (ms)
TIERS = {
    "interactive_2s": 2_000,
    "standard_5s": 5_000,
    "batch_30s": 30_000,
}


def find_prediction_files(pred_dir: Path) -> dict[str, Path]:
    """Find all model prediction files under the prediction directory."""
    models = {}
    for subdir in sorted(pred_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Look for predictions.jsonl in model subdirectories
        for jsonl in subdir.rglob("predictions.jsonl"):
            model_name = jsonl.parent.name
            if model_name == subdir.name:
                model_name = subdir.name
            models[model_name] = jsonl
    return models


def evaluate_model(model_name: str, predictions_path: Path) -> dict:
    """Evaluate a single model's predictions through Weave."""
    print(f"  Loading predictions from {predictions_path}")
    dataset = create_dataset_from_predictions(predictions_path, name=model_name)
    print(f"  Created dataset with {len(dataset.rows)} rows")

    scorers = [
        SLOScorer(tier_ms=TIERS["interactive_2s"]),
        SLOScorer(tier_ms=TIERS["standard_5s"]),
        SLOScorer(tier_ms=TIERS["batch_30s"]),
        JSONValidityScorer(),
        AccuracyScorer(),
    ]

    print(f"  Running evaluation with {len(scorers)} scorers...")
    results = run_retroactive_eval(
        dataset=dataset,
        scorers=scorers,
        model_name=model_name,
    )

    print(f"  Done: {model_name}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Retroactive Weave evaluation")
    parser.add_argument(
        "--project",
        default="agentslo-bench",
        help="Weave project name (default: agentslo-bench)",
    )
    parser.add_argument(
        "--model",
        help="Evaluate a single model (by directory name)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all models found in the predictions directory",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help=f"Predictions directory (default: {DEFAULT_PRED_DIR})",
    )
    args = parser.parse_args()

    # Initialize Weave
    print(f"Initializing Weave project: {args.project}")
    if not init_weave(args.project):
        print("ERROR: Failed to initialize Weave. Check:")
        print("  - weave is installed: pip install 'weave>=0.51'")
        print("  - WANDB_API_KEY is set in environment")
        sys.exit(1)

    # Find predictions
    pred_files = find_prediction_files(args.pred_dir)
    if not pred_files:
        print(f"No prediction files found in {args.pred_dir}")
        print("Expected structure: {pred_dir}/<run_dir>/<model>/predictions.jsonl")
        sys.exit(1)

    print(f"Found {len(pred_files)} model(s): {', '.join(sorted(pred_files.keys()))}")

    # Select models
    if args.model:
        if args.model not in pred_files:
            # Try partial match
            matches = [k for k in pred_files if args.model.lower() in k.lower()]
            if len(matches) == 1:
                target = {matches[0]: pred_files[matches[0]]}
            else:
                print(f"Model '{args.model}' not found. Available: {sorted(pred_files.keys())}")
                sys.exit(1)
        else:
            target = {args.model: pred_files[args.model]}
    elif args.all:
        target = pred_files
    else:
        print("Specify --model MODEL or --all")
        sys.exit(1)

    # Evaluate
    all_results = {}
    for name, path in sorted(target.items()):
        print(f"\nEvaluating: {name}")
        try:
            results = evaluate_model(name, path)
            all_results[name] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Completed {len(all_results)}/{len(target)} model evaluations")
    print(f"View results at: https://wandb.ai/weave/{args.project}")


if __name__ == "__main__":
    main()
