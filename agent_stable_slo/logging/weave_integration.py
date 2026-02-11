"""Weave integration for AgentSLO-Bench.

Provides Weave Scorers for SLO-aware evaluation and helpers for
retroactive evaluation of existing predictions through the Weave UI.

Pattern: mirrors wandb_utils.py — lazy import, env-var gated, graceful fallback.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _weave_available() -> bool:
    """Check if weave is importable."""
    try:
        import weave  # noqa: F401
        return True
    except ImportError:
        return False


def init_weave(project: str) -> bool:
    """Initialize Weave for a project.

    Returns True if weave is available and initialized, False otherwise.
    Requires WANDB_API_KEY in the environment (Weave uses W&B auth).
    """
    if not _weave_available():
        return False
    if not os.getenv("WANDB_API_KEY"):
        return False
    try:
        import weave
        weave.init(project)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Scorers — only defined if weave is available
# ---------------------------------------------------------------------------

if _weave_available():
    import weave

    class SLOScorer(weave.Scorer):
        """Gates accuracy behind latency compliance.

        A prediction is successful only if it is correct AND arrived
        within the tier's latency deadline.

        For retroactive evaluation, the passthrough model returns the full
        prediction row, so we read values from the output dict.
        """

        tier_ms: float

        @weave.op()
        def score(self, output: Any) -> dict:
            # output is the full prediction row from passthrough_model
            if isinstance(output, dict):
                on_time = output.get("latency_ms", float("inf")) <= self.tier_ms
                correct = output.get("task_correct", False)
                json_valid = output.get("json_valid", False)
            else:
                on_time, correct, json_valid = False, False, False
            success = on_time and correct and json_valid
            return {
                "success_at_slo": success,
                "on_time": on_time,
                "correct": correct,
                "json_valid": json_valid,
            }

    class JSONValidityScorer(weave.Scorer):
        """Checks JSON structural validity."""

        @weave.op()
        def score(self, output: Any) -> dict:
            if isinstance(output, dict):
                return {"json_valid": output.get("json_valid", False)}
            return {"json_valid": False}

    class AccuracyScorer(weave.Scorer):
        """Task-specific accuracy (field accuracy, function match, etc.)."""

        @weave.op()
        def score(self, output: Any) -> dict:
            if isinstance(output, dict):
                return {"task_correct": output.get("task_correct", False)}
            return {"task_correct": False}

else:
    # Stub classes when weave is not installed
    class SLOScorer:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            raise ImportError("weave is not installed. Install with: pip install 'weave>=0.51'")

    class JSONValidityScorer:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            raise ImportError("weave is not installed. Install with: pip install 'weave>=0.51'")

    class AccuracyScorer:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            raise ImportError("weave is not installed. Install with: pip install 'weave>=0.51'")


# ---------------------------------------------------------------------------
# Dataset + evaluation helpers
# ---------------------------------------------------------------------------

def _normalize_prediction_row(row: dict) -> dict:
    """Normalize a prediction row for Weave scoring.

    Extracts nested metrics to top-level keys expected by scorers:
      - json_valid: from metrics.json_valid
      - task_correct: from task-specific success metric (t3_success, t4_success, etc.)
    """
    normalized = dict(row)  # shallow copy
    metrics = row.get("metrics", {})

    # Extract json_valid from nested metrics
    if "json_valid" not in normalized and "json_valid" in metrics:
        normalized["json_valid"] = bool(metrics["json_valid"])

    # Extract task_correct from task-specific success metrics
    if "task_correct" not in normalized:
        # Check for task-specific success keys
        success_keys = ["t3_success", "t4_success", "t5_success", "t2_success"]
        for key in success_keys:
            if key in metrics:
                normalized["task_correct"] = bool(metrics[key])
                break
        else:
            # For t1 (clinc) and t2 (hotpot), check field matches
            detailed = row.get("detailed", {})
            fb = detailed.get("field_breakdown", {})
            if fb:
                # Consider correct if the primary field matches
                # For clinc: intent match; for hotpot: answer match
                matches = [v.get("match", False) for v in fb.values() if isinstance(v, dict)]
                # Use first field (typically the main one) as correctness signal
                normalized["task_correct"] = matches[0] if matches else False
            else:
                normalized["task_correct"] = False

    return normalized


def create_dataset_from_predictions(
    predictions_path: Path,
    name: str,
) -> Any:
    """Create a Weave Dataset from a predictions.jsonl file.

    Normalizes rows and wraps them in a 'row' column so that
    passthrough_model can receive the full dict via a single argument.

    The normalized row includes:
      - latency_ms: float (passthrough)
      - json_valid: bool (extracted from metrics.json_valid)
      - task_correct: bool (extracted from task-specific metrics)

    Returns a weave.Dataset, or raises ImportError if weave is unavailable.
    """
    if not _weave_available():
        raise ImportError("weave is not installed. Install with: pip install 'weave>=0.51'")

    import weave

    rows = []
    with open(predictions_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_row = json.loads(line)
            normalized = _normalize_prediction_row(raw_row)
            # Wrap in 'row' key so model can receive it as single arg
            rows.append({"row": normalized})

    return weave.Dataset(name=name, rows=rows)


def run_retroactive_eval(
    dataset: Any,
    scorers: list,
    model_name: str,
) -> dict:
    """Run retroactive evaluation on a dataset with given scorers.

    This creates a Weave Evaluation that scores existing predictions
    (no model call needed — the predictions are already in the dataset).

    Args:
        dataset: A weave.Dataset containing prediction rows.
        scorers: List of weave.Scorer instances.
        model_name: Name for the model in the evaluation.

    Returns:
        Evaluation summary dict.
    """
    if not _weave_available():
        raise ImportError("weave is not installed. Install with: pip install 'weave>=0.51'")

    import asyncio
    import weave

    @weave.op()
    def passthrough_model(row: dict) -> dict:
        """Identity model — predictions are already computed.

        Receives the full prediction row (wrapped in 'row' column by dataset)
        and returns it directly for scorers to process.
        """
        return row

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=scorers,
    )

    # Run synchronously
    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(evaluation.evaluate(passthrough_model))
    except Exception as e:
        # Weave 0.52 has a numpy aggregation bug with large datasets
        # Return partial results if available
        error_msg = str(e)
        if "ufunc 'add'" in error_msg:
            print("  Note: Weave aggregation completed but summary failed (numpy bug). Check Weave UI for results.")
            results = {"error": "aggregation_failed", "message": error_msg}
        else:
            raise
    finally:
        loop.close()

    return results
