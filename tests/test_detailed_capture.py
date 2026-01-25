"""
Unit tests for the detailed_capture module.
"""

import pytest
from agent_stable_slo.eval.detailed_capture import (
    ErrorCategory,
    ErrorClassification,
    ErrorClassifier,
    FieldBreakdownAnalyzer,
    FieldComparison,
    GoldComparison,
    LatencyDecomposition,
    LogprobsData,
    CaptureConfig,
    build_detailed_result,
    compute_prompt_hash,
    canonical_json,
    aggregate_stability_metrics,
)


class TestErrorCategory:
    def test_enum_values(self):
        assert ErrorCategory.NONE.value == "none"
        assert ErrorCategory.PARSE_ERROR.value == "parse_error"
        assert ErrorCategory.SCHEMA_VIOLATION.value == "schema_violation"
        assert ErrorCategory.SEMANTIC_ERROR.value == "semantic_error"
        assert ErrorCategory.TIMEOUT.value == "timeout"


class TestErrorClassifier:
    def test_no_error(self):
        result = ErrorClassifier.classify(
            raw_text='{"intent": "greeting"}',
            parsed_json={"intent": "greeting"},
            parse_error=None,
            schema_error=None,
        )
        assert result.category == ErrorCategory.NONE
        assert result.confidence == 1.0

    def test_parse_error(self):
        result = ErrorClassifier.classify(
            raw_text="not json",
            parsed_json=None,
            parse_error="Expecting value: line 1 column 1",
            schema_error=None,
        )
        assert result.category == ErrorCategory.PARSE_ERROR
        assert result.confidence == 1.0
        assert "Expecting value" in result.raw_error

    def test_schema_violation(self):
        result = ErrorClassifier.classify(
            raw_text='{"wrong_field": "value"}',
            parsed_json={"wrong_field": "value"},
            parse_error=None,
            schema_error="intent: 'intent' is a required property",
        )
        assert result.category == ErrorCategory.SCHEMA_VIOLATION
        assert result.confidence == 1.0
        assert "intent" in result.schema_path

    def test_timeout(self):
        result = ErrorClassifier.classify(
            raw_text='{"intent": "greeting"}',
            parsed_json={"intent": "greeting"},
            parse_error=None,
            schema_error=None,
            latency_ms=3000.0,
            slo_budget_ms=2000.0,
        )
        assert result.category == ErrorCategory.TIMEOUT
        assert result.details["latency_ms"] == 3000.0
        assert result.details["budget_ms"] == 2000.0

    def test_semantic_error(self):
        result = ErrorClassifier.classify(
            raw_text='{"intent": "booking"}',
            parsed_json={"intent": "booking"},
            parse_error=None,
            schema_error=None,
            gold={"intent": "greeting"},
        )
        assert result.category == ErrorCategory.SEMANTIC_ERROR
        assert "intent" in result.details["mismatched_fields"]

    def test_from_provider_error(self):
        result = ErrorClassifier.from_provider_error("Connection timeout")
        assert result.category == ErrorCategory.PROVIDER_ERROR
        assert "Connection timeout" in result.raw_error


class TestFieldBreakdownAnalyzer:
    @pytest.fixture
    def t1_schema(self):
        return {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "domain": {"type": "string"},
                "is_oos": {"type": "boolean"},
            },
            "required": ["intent", "domain", "is_oos"],
        }

    def test_exact_match(self, t1_schema):
        gold = {"intent": "greeting", "domain": "general", "is_oos": False}
        pred = {"intent": "greeting", "domain": "general", "is_oos": False}

        result = FieldBreakdownAnalyzer.analyze(pred, gold, t1_schema)

        assert result.exact_match is True
        assert result.overall_accuracy == 1.0
        assert all(fc.match for fc in result.field_comparisons)

    def test_partial_match(self, t1_schema):
        gold = {"intent": "greeting", "domain": "general", "is_oos": False}
        pred = {"intent": "greeting", "domain": "banking", "is_oos": False}

        result = FieldBreakdownAnalyzer.analyze(pred, gold, t1_schema)

        assert result.exact_match is False
        assert result.field_accuracy["intent"] == 1.0
        assert result.field_accuracy["domain"] == 0.0
        assert result.field_accuracy["is_oos"] == 1.0
        assert result.overall_accuracy == pytest.approx(2 / 3)

    def test_no_match(self, t1_schema):
        gold = {"intent": "greeting", "domain": "general", "is_oos": False}
        pred = {"intent": "booking", "domain": "travel", "is_oos": True}

        result = FieldBreakdownAnalyzer.analyze(pred, gold, t1_schema)

        assert result.exact_match is False
        assert result.overall_accuracy == 0.0

    def test_string_normalization(self, t1_schema):
        gold = {"intent": "GREETING", "domain": "General", "is_oos": False}
        pred = {"intent": "greeting", "domain": "general", "is_oos": False}

        result = FieldBreakdownAnalyzer.analyze(pred, gold, t1_schema)

        assert result.field_accuracy["intent"] == 1.0
        assert result.field_accuracy["domain"] == 1.0

    def test_null_pred(self, t1_schema):
        gold = {"intent": "greeting", "domain": "general", "is_oos": False}

        result = FieldBreakdownAnalyzer.analyze(None, gold, t1_schema)

        assert result.exact_match is False
        assert result.overall_accuracy == 0.0

    def test_token_f1_similarity(self):
        s1 = "the quick brown fox"
        s2 = "the slow brown fox"

        similarity = FieldBreakdownAnalyzer._token_f1(s1, s2)

        # 3 common tokens out of 4 each
        assert 0.5 < similarity < 1.0


class TestGoldComparison:
    def test_to_dict(self):
        fc = FieldComparison("intent", "greeting", "greeting", True, 1.0)
        gc = GoldComparison(
            exact_match=True,
            field_comparisons=[fc],
            field_accuracy={"intent": 1.0},
            overall_accuracy=1.0,
        )

        d = gc.to_dict()

        assert d["exact_match"] is True
        assert d["overall_accuracy"] == 1.0
        assert len(d["field_comparisons"]) == 1
        assert d["field_comparisons"][0]["field_name"] == "intent"


class TestLatencyDecomposition:
    def test_to_dict(self):
        ld = LatencyDecomposition(
            queue_ms=10.0,
            inference_ms=500.0,
            validation_ms=5.0,
            total_ms=515.0,
            timestamps={"start": 1000.0, "end": 1515.0},
        )

        d = ld.to_dict()

        assert d["queue_ms"] == 10.0
        assert d["inference_ms"] == 500.0
        assert d["total_ms"] == 515.0


class TestLogprobsData:
    def test_to_dict(self):
        lp = LogprobsData(
            available=True,
            mean_logprob=-0.5,
            entropy=0.3,
            finish_reason="stop",
            token_count=50,
        )

        d = lp.to_dict()

        assert d["available"] is True
        assert d["mean_logprob"] == -0.5
        assert d["finish_reason"] == "stop"


class TestCaptureConfig:
    def test_minimal(self):
        config = CaptureConfig.minimal()

        assert config.capture_field_breakdown is False
        assert config.capture_error_taxonomy is False
        assert config.capture_logprobs is False

    def test_standard(self):
        config = CaptureConfig.standard()

        assert config.capture_field_breakdown is True
        assert config.capture_error_taxonomy is True
        assert config.capture_logprobs is False

    def test_full(self):
        config = CaptureConfig.full()

        assert config.capture_logprobs is True
        assert config.stability_runs == 3


class TestBuildDetailedResult:
    def test_basic_capture(self):
        result = build_detailed_result(
            task_id="test_1",
            task_type="t1",
            prompt="Classify this intent",
            raw_output='{"intent": "greeting"}',
            parsed_output={"intent": "greeting"},
            gold={"intent": "greeting"},
            schema={
                "type": "object",
                "properties": {"intent": {"type": "string"}},
            },
            parse_error=None,
            schema_error=None,
            latency_ms=500.0,
        )

        assert "gold" in result
        assert "field_breakdown" in result
        assert "stability_canonical" in result
        assert result["exact_match"] is True

    def test_minimal_config(self):
        config = CaptureConfig.minimal()

        result = build_detailed_result(
            task_id="test_1",
            task_type="t1",
            prompt="Test",
            raw_output="{}",
            parsed_output={},
            gold={"intent": "greeting"},
            schema={"type": "object", "properties": {"intent": {"type": "string"}}},
            parse_error=None,
            schema_error=None,
            latency_ms=500.0,
            config=config,
        )

        assert "field_breakdown" not in result
        assert "error_classification" not in result


class TestUtilityFunctions:
    def test_compute_prompt_hash(self):
        hash1 = compute_prompt_hash("test prompt")
        hash2 = compute_prompt_hash("test prompt")
        hash3 = compute_prompt_hash("different prompt")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA-256 hex

    def test_canonical_json(self):
        obj1 = {"b": 2, "a": 1}
        obj2 = {"a": 1, "b": 2}

        assert canonical_json(obj1) == canonical_json(obj2)
        assert canonical_json(obj1) == '{"a":1,"b":2}'

    def test_canonical_json_none(self):
        assert canonical_json(None) == ""


class TestAggregateStabilityMetrics:
    def test_perfect_agreement(self):
        results = [
            {"prompt_hash": "hash1", "detailed": {"stability_canonical": '{"a":1}'}},
            {"prompt_hash": "hash1", "detailed": {"stability_canonical": '{"a":1}'}},
            {"prompt_hash": "hash1", "detailed": {"stability_canonical": '{"a":1}'}},
        ]

        metrics = aggregate_stability_metrics(results)

        assert metrics["agreement_rate"] == 1.0
        assert metrics["num_groups"] == 1

    def test_partial_agreement(self):
        results = [
            {"prompt_hash": "hash1", "detailed": {"stability_canonical": '{"a":1}'}},
            {"prompt_hash": "hash1", "detailed": {"stability_canonical": '{"a":1}'}},
            {"prompt_hash": "hash1", "detailed": {"stability_canonical": '{"a":2}'}},
        ]

        metrics = aggregate_stability_metrics(results)

        assert metrics["agreement_rate"] == pytest.approx(2 / 3)

    def test_multiple_groups(self):
        results = [
            {"prompt_hash": "hash1", "detailed": {"stability_canonical": '{"a":1}'}},
            {"prompt_hash": "hash1", "detailed": {"stability_canonical": '{"a":1}'}},
            {"prompt_hash": "hash2", "detailed": {"stability_canonical": '{"b":1}'}},
            {"prompt_hash": "hash2", "detailed": {"stability_canonical": '{"b":2}'}},
        ]

        metrics = aggregate_stability_metrics(results)

        # hash1: 100% agreement, hash2: 50% agreement, average = 75%
        assert metrics["agreement_rate"] == pytest.approx(0.75)
        assert metrics["num_groups"] == 2

    def test_single_run_groups(self):
        results = [
            {"prompt_hash": "hash1", "detailed": {"stability_canonical": '{"a":1}'}},
            {"prompt_hash": "hash2", "detailed": {"stability_canonical": '{"b":1}'}},
        ]

        metrics = aggregate_stability_metrics(results)

        # No groups with multiple runs
        assert metrics["agreement_rate"] is None
        assert metrics["num_groups_with_multiple_runs"] == 0


class TestErrorClassificationSerialization:
    def test_to_dict(self):
        ec = ErrorClassification(
            category=ErrorCategory.PARSE_ERROR,
            confidence=0.95,
            raw_error="Invalid JSON",
            schema_path=None,
            details={"line": 1},
        )

        d = ec.to_dict()

        assert d["category"] == "parse_error"
        assert d["confidence"] == 0.95
        assert d["raw_error"] == "Invalid JSON"
        assert d["details"]["line"] == 1
