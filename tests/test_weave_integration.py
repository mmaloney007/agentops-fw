"""Tests for Weave integration.

Uses WANDB_MODE=offline to avoid network calls during CI.
"""
import os
import pytest

# Set offline mode before imports
os.environ["WANDB_MODE"] = "offline"


def test_weave_available():
    """Check weave is importable."""
    import weave
    assert weave.__version__


def test_weave_integration_module():
    """Check our integration module loads."""
    from agent_stable_slo.logging.weave_integration import (
        _weave_available,
        SLOScorer,
        JSONValidityScorer,
        AccuracyScorer,
    )
    assert _weave_available()
    assert SLOScorer is not None
    assert JSONValidityScorer is not None
    assert AccuracyScorer is not None


def test_slo_scorer_scoring():
    """Test SLOScorer.score() logic.

    Scorers return floats (1.0/0.0) for numpy aggregation compatibility.
    """
    from agent_stable_slo.logging.weave_integration import SLOScorer

    scorer = SLOScorer(tier_ms=2000)  # 2s deadline

    # Under deadline, correct, valid JSON -> success
    result = scorer.score(output={"latency_ms": 1500, "task_correct": True, "json_valid": True})
    assert result["success_at_slo"] == 1.0
    assert result["on_time"] == 1.0
    assert result["correct"] == 1.0

    # Over deadline -> failure
    result = scorer.score(output={"latency_ms": 2500, "task_correct": True, "json_valid": True})
    assert result["success_at_slo"] == 0.0
    assert result["on_time"] == 0.0

    # Under deadline but incorrect -> failure
    result = scorer.score(output={"latency_ms": 1500, "task_correct": False, "json_valid": True})
    assert result["success_at_slo"] == 0.0
    assert result["correct"] == 0.0


def test_json_validity_scorer():
    """Test JSONValidityScorer.score() logic."""
    from agent_stable_slo.logging.weave_integration import JSONValidityScorer

    scorer = JSONValidityScorer()

    # Valid JSON field
    result = scorer.score(output={"json_valid": True})
    assert result["json_valid"] == 1.0

    # Invalid JSON field
    result = scorer.score(output={"json_valid": False})
    assert result["json_valid"] == 0.0

    # Missing field -> default 0.0
    result = scorer.score(output={})
    assert result["json_valid"] == 0.0


def test_accuracy_scorer():
    """Test AccuracyScorer.score() logic."""
    from agent_stable_slo.logging.weave_integration import AccuracyScorer

    scorer = AccuracyScorer()

    # Correct
    result = scorer.score(output={"task_correct": True})
    assert result["task_correct"] == 1.0

    # Incorrect
    result = scorer.score(output={"task_correct": False})
    assert result["task_correct"] == 0.0

    # Missing field -> default 0.0
    result = scorer.score(output={})
    assert result["task_correct"] == 0.0


@pytest.mark.skipif(
    os.environ.get("WANDB_MODE") == "offline",
    reason="Requires online mode for init_weave"
)
def test_init_weave():
    """Test init_weave requires API key."""
    from agent_stable_slo.logging.weave_integration import init_weave

    # Without API key, should fail
    old_key = os.environ.pop("WANDB_API_KEY", None)
    try:
        result = init_weave("test-project")
        assert result is False
    finally:
        if old_key:
            os.environ["WANDB_API_KEY"] = old_key
