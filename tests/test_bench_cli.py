"""Tests for agent_stable_slo.bench.cli helper logic."""

from agent_stable_slo.bench.cli import (
    _compute_task_correct,
    _extract_gold_function_name,
    _load_task_samples,
)


def test_t4_loads_bfcl_samples():
    samples = _load_task_samples("T4", limit=1)
    assert samples
    assert str(samples[0].get("id", "")).startswith("t4_")


def test_extract_gold_function_name_bfcl_shape():
    gold = {"math.factorial": {"number": [5]}}
    assert _extract_gold_function_name(gold) == "math.factorial"


def test_compute_task_correct_t4_bfcl_gold():
    gold = {"math.factorial": {"number": [5]}}
    output = {"name": "math.factorial", "arguments": {"number": 5}}
    assert _compute_task_correct("T4", gold, output, json_valid=True)


def test_compute_task_correct_does_not_overcredit_without_gold():
    assert not _compute_task_correct("T3", {}, {"tool": "fetch_metric"}, json_valid=True)

