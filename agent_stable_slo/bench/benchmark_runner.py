"""AgentSLO-Bench benchmark runner.

Evaluates a model endpoint against benchmark tasks and computes
per-tier Success@SLO metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .slo_tiers import TIERS, SLOTier


@dataclass
class TaskResult:
    task_id: str
    latency_ms: float
    json_valid: bool
    task_correct: bool
    output: Any = None


@dataclass
class TierResult:
    tier_name: str
    deadline_ms: float
    total: int = 0
    on_time: int = 0
    correct: int = 0
    success_at_slo: int = 0  # correct AND on_time

    @property
    def success_at_slo_pct(self) -> float:
        return 100.0 * self.success_at_slo / max(1, self.total)

    @property
    def accuracy_pct(self) -> float:
        return 100.0 * self.correct / max(1, self.total)

    @property
    def on_time_pct(self) -> float:
        return 100.0 * self.on_time / max(1, self.total)


@dataclass
class BenchmarkResult:
    model_name: str
    task_name: str
    tier_results: dict[str, TierResult] = field(default_factory=dict)
    task_results: list[TaskResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "task": self.task_name,
            "tiers": {k: asdict(v) for k, v in self.tier_results.items()},
            "n_samples": len(self.task_results),
            "metadata": self.metadata,
        }


def compute_tier_results(
    task_results: list[TaskResult],
    tiers: list[SLOTier] | None = None,
) -> dict[str, TierResult]:
    """Compute Success@SLO for each tier from raw task results."""
    if tiers is None:
        tiers = TIERS

    tier_results = {}
    for tier in tiers:
        tr = TierResult(tier_name=tier.name, deadline_ms=tier.deadline_ms)
        for r in task_results:
            tr.total += 1
            on_time = r.latency_ms <= tier.deadline_ms
            if on_time:
                tr.on_time += 1
            if r.task_correct:
                tr.correct += 1
            if on_time and r.task_correct and r.json_valid:
                tr.success_at_slo += 1
        tier_results[tier.name] = tr
    return tier_results


def compute_from_p1_data(
    model_data: dict,
    model_name: str,
) -> list[BenchmarkResult]:
    """Compute benchmark results from existing P1 evaluation data.

    Uses latency distributions to estimate Success@SLO at different tiers.
    P1 data was collected at 2s SLO; we extrapolate to 5s and 30s using
    p95/p99 latency percentiles.
    """
    results = []
    for task_name, task_data in model_data.get("tasks", {}).items():
        avg_lat = task_data.get("avg_latency_ms", 0)
        p95_lat = task_data.get("p95_latency_ms", 0)
        p99_lat = task_data.get("p99_latency_ms", 0)
        count = task_data.get("count", 0)
        slo_pct_2s = task_data.get("success_at_slo_pct", 0)

        # Extract accuracy metric based on task type
        accuracy = 0.0
        for key in ["t1_field_acc", "t2_summary_f1", "t3_success", "t4_func_match", "t5_has_patch"]:
            if key in task_data:
                accuracy = task_data[key]
                break

        tier_results = {}

        # Interactive tier (2s) - use actual P1 data
        tier_results["interactive"] = TierResult(
            tier_name="interactive",
            deadline_ms=2000.0,
            total=count,
            on_time=int(count * task_data.get("latency_slo_pct", 0) / 100.0),
            correct=int(count * accuracy),
            success_at_slo=int(count * slo_pct_2s / 100.0),
        )

        # Standard tier (5s) - estimate from latency distribution
        # If p95 < 5s, ~95% of requests are on time
        if p95_lat <= 5000:
            on_time_pct = 95.0 + 4.0 * max(0, 1 - p99_lat / 5000)
        elif avg_lat <= 5000:
            on_time_pct = 100.0 * min(1.0, 5000 / max(1, p95_lat)) * 0.95
        else:
            on_time_pct = max(0, 100.0 * (1 - (avg_lat - 5000) / max(1, p99_lat - 5000))) * 0.5

        on_time_pct = max(0, min(100, on_time_pct))
        s_at_slo_5s = on_time_pct * accuracy
        tier_results["standard"] = TierResult(
            tier_name="standard",
            deadline_ms=5000.0,
            total=count,
            on_time=int(count * on_time_pct / 100.0),
            correct=int(count * accuracy),
            success_at_slo=int(count * s_at_slo_5s / 100.0),
        )

        # Batch tier (30s) - almost everything finishes in 30s
        if p99_lat <= 30000:
            on_time_pct_30 = 99.5
        elif p95_lat <= 30000:
            on_time_pct_30 = 95.0 + 4.5 * max(0, 1 - p99_lat / 30000)
        else:
            on_time_pct_30 = max(0, 100.0 * min(1.0, 30000 / max(1, p99_lat))) * 0.95

        on_time_pct_30 = max(0, min(100, on_time_pct_30))
        s_at_slo_30 = on_time_pct_30 * accuracy
        tier_results["batch"] = TierResult(
            tier_name="batch",
            deadline_ms=30000.0,
            total=count,
            on_time=int(count * on_time_pct_30 / 100.0),
            correct=int(count * accuracy),
            success_at_slo=int(count * s_at_slo_30 / 100.0),
        )

        br = BenchmarkResult(
            model_name=model_name,
            task_name=task_name,
            tier_results=tier_results,
            metadata={
                "source": "p1_baseline",
                "avg_latency_ms": avg_lat,
                "p95_latency_ms": p95_lat,
                "p99_latency_ms": p99_lat,
                "accuracy": accuracy,
            },
        )
        results.append(br)
    return results


def compute_from_predictions(
    predictions_path: Path,
    model_name: str,
    tiers: list[SLOTier] | None = None,
) -> list[BenchmarkResult]:
    """Compute benchmark results from per-request predictions.jsonl data.

    Each line in predictions_path is a JSON object with:
      - schema_path: str (maps to task)
      - latency_ms: float
      - metrics.json_valid: float (1.0 or 0.0)
      - detailed.overall_field_accuracy: float
      - gold: dict (for T4 function name matching)
      - output_json: dict (for T4 function name matching)

    Unlike compute_from_p1_data, this computes exact per-request S@SLO
    with no extrapolation from aggregate percentiles.
    """
    if tiers is None:
        tiers = TIERS

    # Schema path to task mapping
    schema_to_task = {
        "tasks/schemas/clinc_nlu_schema.json": "T1",
        "tasks/schemas/t1_incident_schema.json": "T1v",
        "tasks/schemas/t2_summary_schema.json": "T2",
        "tasks/schemas/hotpot_explainer_schema.json": "T2v",
        "tasks/schemas/t3_tool_call_schema.json": "T3",
        "tasks/schemas/t4_function_call_schema.json": "T4",
        "tasks/schemas/t5_patch_schema.json": "T5",
    }

    # Load predictions
    predictions: dict[str, list] = {}
    with open(predictions_path) as f:
        for line in f:
            pred = json.loads(line)
            schema = pred.get("schema_path", "")
            task = schema_to_task.get(schema, "unknown")
            predictions.setdefault(task, []).append(pred)

    results = []
    for task_name, preds in sorted(predictions.items()):
        task_results = []
        for pred in preds:
            json_valid = pred.get("metrics", {}).get("json_valid", 0) == 1.0

            # T4 uses function-name matching; others use field accuracy
            schema = pred.get("schema_path", "")
            if schema == "tasks/schemas/t4_function_call_schema.json":
                gold = pred.get("gold", {})
                output = pred.get("output_json", {})
                gold_func = list(gold.keys())[0] if gold else None
                pred_func = output.get("name", output.get("function_name", "")) if output else ""
                task_correct = json_valid and gold_func is not None and pred_func == gold_func
            else:
                ofa = pred.get("detailed", {}).get("overall_field_accuracy", 0)
                task_correct = json_valid and ofa > 0

            task_results.append(TaskResult(
                task_id=pred.get("id", "unknown"),
                latency_ms=pred.get("latency_ms", 0),
                json_valid=json_valid,
                task_correct=task_correct,
            ))

        tier_results = compute_tier_results(task_results, tiers)
        br = BenchmarkResult(
            model_name=model_name,
            task_name=task_name,
            tier_results=tier_results,
            task_results=task_results,
            metadata={
                "source": "predictions_jsonl",
                "n_predictions": len(preds),
            },
        )
        results.append(br)

    return results


def save_results(results: list[BenchmarkResult], out_path: Path) -> None:
    """Save benchmark results to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
