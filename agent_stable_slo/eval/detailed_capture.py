"""
Detailed result capture module for P1 evaluation.

Provides comprehensive dataclasses, error classification, field breakdown analysis,
and gold comparison utilities for capturing detailed evaluation results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ErrorCategory(str, Enum):
    """Categories of errors that can occur during structured generation."""

    NONE = "none"
    PARSE_ERROR = "parse_error"
    SCHEMA_VIOLATION = "schema_violation"
    SEMANTIC_ERROR = "semantic_error"
    HALLUCINATION = "hallucination"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"


@dataclass
class ErrorClassification:
    """Classification of an error with category, confidence, and details."""

    category: ErrorCategory
    confidence: float
    raw_error: Optional[str] = None
    schema_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "raw_error": self.raw_error,
            "schema_path": self.schema_path,
            "details": self.details,
        }


@dataclass
class FieldComparison:
    """Comparison between gold and predicted values for a single field."""

    field_name: str
    gold_value: Any
    pred_value: Any
    match: bool
    similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "gold": self.gold_value,
            "pred": self.pred_value,
            "match": self.match,
            "similarity": self.similarity,
        }


@dataclass
class GoldComparison:
    """Complete comparison between predicted and gold outputs."""

    exact_match: bool
    field_comparisons: List[FieldComparison]
    field_accuracy: Dict[str, float]
    overall_accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "field_comparisons": [fc.to_dict() for fc in self.field_comparisons],
            "field_accuracy": self.field_accuracy,
            "overall_accuracy": self.overall_accuracy,
        }


@dataclass
class LatencyDecomposition:
    """Breakdown of latency into component parts."""

    queue_ms: float = 0.0
    inference_ms: float = 0.0
    validation_ms: float = 0.0
    total_ms: float = 0.0
    timestamps: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_ms": self.queue_ms,
            "inference_ms": self.inference_ms,
            "validation_ms": self.validation_ms,
            "total_ms": self.total_ms,
            "timestamps": self.timestamps,
        }


@dataclass
class LogprobsData:
    """Logprobs information from model output."""

    available: bool = False
    mean_logprob: Optional[float] = None
    entropy: Optional[float] = None
    finish_reason: Optional[str] = None
    token_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "mean_logprob": self.mean_logprob,
            "entropy": self.entropy,
            "finish_reason": self.finish_reason,
            "token_count": self.token_count,
        }


@dataclass
class AttemptDetail:
    """Detailed information about a single generation attempt."""

    attempt_idx: int
    raw_text: str
    parsed_json: Optional[Dict[str, Any]]
    error: Optional[ErrorClassification]
    latency: LatencyDecomposition
    logprobs: Optional[LogprobsData] = None
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_idx": self.attempt_idx,
            "raw_text": self.raw_text,
            "parsed_json": self.parsed_json,
            "error": self.error.to_dict() if self.error else None,
            "latency": self.latency.to_dict(),
            "logprobs": self.logprobs.to_dict() if self.logprobs else None,
            "success": self.success,
        }


@dataclass
class DetailedResult:
    """Complete detailed result for a single evaluation task."""

    task_id: str
    task_type: str
    prompt_hash: str
    final_output: Optional[Dict[str, Any]]
    raw_output: str
    attempts: List[AttemptDetail]
    gold_comparison: Optional[GoldComparison]
    latency: LatencyDecomposition
    final_error: Optional[ErrorClassification] = None
    logprobs: Optional[LogprobsData] = None
    stability_canonical: Optional[str] = None
    model: str = ""
    provider: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "prompt_hash": self.prompt_hash,
            "final_output": self.final_output,
            "raw_output": self.raw_output,
            "attempts": [a.to_dict() for a in self.attempts],
            "gold_comparison": self.gold_comparison.to_dict()
            if self.gold_comparison
            else None,
            "latency": self.latency.to_dict(),
            "final_error": self.final_error.to_dict() if self.final_error else None,
            "logprobs": self.logprobs.to_dict() if self.logprobs else None,
            "stability_canonical": self.stability_canonical,
            "model": self.model,
            "provider": self.provider,
        }


@dataclass
class CaptureConfig:
    """Configuration for what detailed information to capture."""

    capture_field_breakdown: bool = True
    capture_error_taxonomy: bool = True
    capture_raw_output: bool = True
    capture_attempt_history: bool = True
    capture_logprobs: bool = False
    capture_gold_comparison: bool = True
    capture_latency_decomposition: bool = True
    stability_runs: int = 1
    max_attempts_stored: int = 10

    @classmethod
    def minimal(cls) -> "CaptureConfig":
        """Minimal capture config for fast evaluation."""
        return cls(
            capture_field_breakdown=False,
            capture_error_taxonomy=False,
            capture_raw_output=False,
            capture_attempt_history=False,
            capture_logprobs=False,
            capture_gold_comparison=False,
            capture_latency_decomposition=False,
        )

    @classmethod
    def standard(cls) -> "CaptureConfig":
        """Standard capture config for typical evaluation."""
        return cls()

    @classmethod
    def full(cls) -> "CaptureConfig":
        """Full capture config for detailed analysis."""
        return cls(capture_logprobs=True, stability_runs=3)


class ErrorClassifier:
    """Classifies errors into categories based on error signals."""

    @staticmethod
    def classify(
        raw_text: str,
        parsed_json: Optional[Dict],
        parse_error: Optional[str],
        schema_error: Optional[str],
        gold: Optional[Dict] = None,
        latency_ms: float = 0.0,
        slo_budget_ms: float = 2000.0,
    ) -> ErrorClassification:
        """
        Classify an error based on available signals.

        Args:
            raw_text: The raw text output from the model
            parsed_json: The parsed JSON object (if successful)
            parse_error: Parse error message (if any)
            schema_error: Schema validation error message (if any)
            gold: Gold/expected output for semantic comparison
            latency_ms: Actual latency in milliseconds
            slo_budget_ms: SLO budget in milliseconds

        Returns:
            ErrorClassification with category and details
        """
        # Check for timeout first (highest priority)
        if latency_ms > slo_budget_ms:
            return ErrorClassification(
                ErrorCategory.TIMEOUT,
                1.0,
                details={"latency_ms": latency_ms, "budget_ms": slo_budget_ms},
            )

        # Check for parse errors
        if parse_error:
            return ErrorClassification(
                ErrorCategory.PARSE_ERROR,
                1.0,
                raw_error=parse_error,
                details={"preview": raw_text[:200] if raw_text else ""},
            )

        # Check for schema violations
        if schema_error:
            path = (
                schema_error.split(":")[0].strip() if ":" in schema_error else None
            )
            return ErrorClassification(
                ErrorCategory.SCHEMA_VIOLATION,
                1.0,
                raw_error=schema_error,
                schema_path=path,
            )

        # Check for semantic errors (when we have gold comparison)
        if parsed_json and gold:
            return ErrorClassifier._classify_semantic(parsed_json, gold)

        # No error detected
        return ErrorClassification(ErrorCategory.NONE, 1.0)

    @staticmethod
    def _classify_semantic(pred: Dict, gold: Dict) -> ErrorClassification:
        """Classify semantic errors by comparing pred to gold."""
        mismatched = [k for k in gold if gold.get(k) != pred.get(k)]
        if mismatched:
            return ErrorClassification(
                ErrorCategory.SEMANTIC_ERROR,
                0.9,
                details={"mismatched_fields": mismatched},
            )
        return ErrorClassification(ErrorCategory.NONE, 1.0)

    @staticmethod
    def from_provider_error(error_msg: str) -> ErrorClassification:
        """Create error classification from provider error."""
        return ErrorClassification(
            ErrorCategory.PROVIDER_ERROR,
            1.0,
            raw_error=error_msg,
        )


class FieldBreakdownAnalyzer:
    """Analyzes field-level accuracy between predictions and gold."""

    @staticmethod
    def analyze(
        pred: Optional[Dict], gold: Dict, schema: Dict
    ) -> GoldComparison:
        """
        Analyze field-level accuracy.

        Args:
            pred: Predicted output dictionary
            gold: Gold/expected output dictionary
            schema: JSON schema for the output

        Returns:
            GoldComparison with per-field analysis
        """
        fields = list(schema.get("properties", {}).keys())
        comparisons: List[FieldComparison] = []
        accuracy: Dict[str, float] = {}

        for field_name in fields:
            gold_val = gold.get(field_name)
            pred_val = pred.get(field_name) if pred else None
            match, similarity = FieldBreakdownAnalyzer._compare(gold_val, pred_val)
            comparisons.append(
                FieldComparison(field_name, gold_val, pred_val, match, similarity)
            )
            accuracy[field_name] = 1.0 if match else 0.0

        overall = sum(accuracy.values()) / len(accuracy) if accuracy else 0.0
        exact = all(c.match for c in comparisons) if comparisons else False

        return GoldComparison(exact, comparisons, accuracy, overall)

    @staticmethod
    def _compare(gold: Any, pred: Any) -> Tuple[bool, float]:
        """Compare two values, returning (match, similarity)."""
        if gold == pred:
            return True, 1.0
        if gold is None or pred is None:
            return False, 0.0

        # String comparison with normalization
        if isinstance(gold, str) and isinstance(pred, str):
            norm_g = " ".join(str(gold).lower().split())
            norm_p = " ".join(str(pred).lower().split())
            if norm_g == norm_p:
                return True, 1.0
            return False, FieldBreakdownAnalyzer._token_f1(norm_g, norm_p)

        # List comparison
        if isinstance(gold, list) and isinstance(pred, list):
            if gold == pred:
                return True, 1.0
            if not gold and not pred:
                return True, 1.0
            if not gold or not pred:
                return False, 0.0
            # Set overlap for lists
            gold_set = set(str(x) for x in gold)
            pred_set = set(str(x) for x in pred)
            inter = len(gold_set & pred_set)
            union = len(gold_set | pred_set)
            return False, inter / union if union else 0.0

        # Dict comparison
        if isinstance(gold, dict) and isinstance(pred, dict):
            if gold == pred:
                return True, 1.0
            gold_keys = set(gold.keys())
            pred_keys = set(pred.keys())
            common_keys = gold_keys & pred_keys
            if not common_keys:
                return False, 0.0
            matches = sum(1 for k in common_keys if gold[k] == pred[k])
            return False, matches / len(gold_keys) if gold_keys else 0.0

        return False, 0.0

    @staticmethod
    def _token_f1(s1: str, s2: str) -> float:
        """Compute token-level F1 between two strings."""
        t1, t2 = set(s1.split()), set(s2.split())
        if not t1 or not t2:
            return 0.0
        inter = len(t1 & t2)
        prec = inter / len(t2)
        rec = inter / len(t1)
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def compute_prompt_hash(prompt: str) -> str:
    """Compute a hash of the prompt for stability tracking."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def canonical_json(obj: Optional[Dict[str, Any]]) -> str:
    """Create a canonical JSON string for stability comparison."""
    if not obj:
        return ""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_detailed_result(
    task_id: str,
    task_type: str,
    prompt: str,
    raw_output: str,
    parsed_output: Optional[Dict[str, Any]],
    gold: Optional[Dict[str, Any]],
    schema: Dict[str, Any],
    parse_error: Optional[str],
    schema_error: Optional[str],
    latency_ms: float,
    slo_budget_ms: float = 2000.0,
    model: str = "",
    provider: str = "",
    timings: Optional[Dict[str, float]] = None,
    logprobs_data: Optional[Dict[str, Any]] = None,
    config: Optional[CaptureConfig] = None,
) -> Dict[str, Any]:
    """
    Build a detailed result dictionary from evaluation outputs.

    This is the main entry point for adding detailed capture to eval results.

    Args:
        task_id: Task identifier
        task_type: Task type (t1, t2, t3, etc.)
        prompt: Input prompt
        raw_output: Raw text output from model
        parsed_output: Parsed JSON output
        gold: Gold/expected output
        schema: JSON schema for output
        parse_error: Parse error if any
        schema_error: Schema validation error if any
        latency_ms: Total latency in ms
        slo_budget_ms: SLO budget in ms
        model: Model identifier
        provider: Provider identifier
        timings: Timing breakdown dict
        logprobs_data: Logprobs data from provider
        config: CaptureConfig controlling what to capture

    Returns:
        Dictionary with detailed capture fields
    """
    config = config or CaptureConfig.standard()
    result: Dict[str, Any] = {}

    # Always capture gold for reference
    if gold is not None:
        result["gold"] = gold

    # Raw output capture
    if config.capture_raw_output:
        result["raw_output"] = raw_output

    # Field breakdown
    if config.capture_field_breakdown and gold:
        gold_comp = FieldBreakdownAnalyzer.analyze(parsed_output, gold, schema)
        result["field_breakdown"] = {
            fc.field_name: fc.to_dict() for fc in gold_comp.field_comparisons
        }
        result["field_accuracy"] = gold_comp.field_accuracy
        result["overall_field_accuracy"] = gold_comp.overall_accuracy
        result["exact_match"] = gold_comp.exact_match

    # Error taxonomy
    if config.capture_error_taxonomy:
        error = ErrorClassifier.classify(
            raw_output,
            parsed_output,
            parse_error,
            schema_error,
            gold,
            latency_ms,
            slo_budget_ms,
        )
        if error.category != ErrorCategory.NONE:
            result["error_classification"] = error.to_dict()

    # Latency decomposition
    if config.capture_latency_decomposition and timings:
        decomp = LatencyDecomposition(
            queue_ms=0.0,  # Not tracked at this level
            inference_ms=latency_ms,
            validation_ms=timings.get("validation_end", 0)
            - timings.get("validation_start", 0)
            if "validation_end" in timings
            else 0.0,
            total_ms=latency_ms,
            timestamps=timings,
        )
        result["latency_decomposition"] = decomp.to_dict()

    # Logprobs
    if config.capture_logprobs and logprobs_data:
        lp = LogprobsData(
            available=logprobs_data.get("available", False),
            mean_logprob=logprobs_data.get("mean_logprob"),
            entropy=logprobs_data.get("entropy"),
            finish_reason=logprobs_data.get("finish_reason"),
            token_count=logprobs_data.get("token_count"),
        )
        result["logprobs"] = lp.to_dict()

    # Stability canonical form
    result["stability_canonical"] = canonical_json(parsed_output)

    return result


def aggregate_stability_metrics(
    results: List[Dict[str, Any]], prompt_key: str = "prompt_hash"
) -> Dict[str, Any]:
    """
    Aggregate stability metrics across multiple runs of the same prompts.

    Args:
        results: List of result dictionaries
        prompt_key: Key to group by (typically prompt_hash)

    Returns:
        Dictionary with stability metrics
    """
    groups: Dict[str, List[str]] = {}
    for r in results:
        key = r.get(prompt_key, "")
        if not key:
            continue
        canonical = r.get("detailed", {}).get("stability_canonical") or r.get(
            "stability_canonical", ""
        )
        groups.setdefault(key, []).append(canonical)

    if not groups:
        return {"agreement_rate": None, "num_groups": 0}

    agreements = []
    for _key, canonicals in groups.items():
        if len(canonicals) < 2:
            continue
        mode_val = max(set(canonicals), key=canonicals.count)
        agreements.append(canonicals.count(mode_val) / len(canonicals))

    return {
        "agreement_rate": sum(agreements) / len(agreements) if agreements else None,
        "num_groups": len(groups),
        "num_groups_with_multiple_runs": len(agreements),
    }
