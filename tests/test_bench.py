"""Tests for agent_stable_slo.bench modules.

Covers: slo_tiers, benchmark_runner (compute_tier_results, TierResult properties),
and leaderboard (Spearman rho, rank_values, bootstrap_spearman).
"""
import pytest

from agent_stable_slo.bench.slo_tiers import (
    SLOTier,
    INTERACTIVE,
    STANDARD,
    BATCH,
    TIERS,
    TIER_MAP,
)
from agent_stable_slo.bench.benchmark_runner import (
    TaskResult,
    TierResult,
    BenchmarkResult,
    compute_tier_results,
)
from agent_stable_slo.bench.leaderboard import (
    compute_spearman_rho,
    rank_values,
    bootstrap_spearman,
)


# ---- slo_tiers ----

class TestSLOTiers:
    def test_three_tiers_defined(self):
        assert len(TIERS) == 3

    def test_tier_deadlines_ascending(self):
        deadlines = [t.deadline_ms for t in TIERS]
        assert deadlines == sorted(deadlines)
        assert deadlines == [2000.0, 5000.0, 30000.0]

    def test_interactive_values(self):
        assert INTERACTIVE.name == "interactive"
        assert INTERACTIVE.deadline_ms == 2000.0

    def test_standard_values(self):
        assert STANDARD.name == "standard"
        assert STANDARD.deadline_ms == 5000.0

    def test_batch_values(self):
        assert BATCH.name == "batch"
        assert BATCH.deadline_ms == 30000.0

    def test_tier_map_keys(self):
        assert set(TIER_MAP.keys()) == {"interactive", "standard", "batch"}
        assert TIER_MAP["interactive"] is INTERACTIVE

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            SLOTier(name="test", deadline_ms=1000, description="d", typical_use="u", bogus="x")


# ---- benchmark_runner ----

class TestTierResult:
    def test_success_at_slo_pct(self):
        tr = TierResult(tier_name="test", deadline_ms=2000.0, total=100, success_at_slo=75)
        assert tr.success_at_slo_pct == 75.0

    def test_accuracy_pct(self):
        tr = TierResult(tier_name="test", deadline_ms=2000.0, total=200, correct=150)
        assert tr.accuracy_pct == 75.0

    def test_on_time_pct(self):
        tr = TierResult(tier_name="test", deadline_ms=2000.0, total=50, on_time=40)
        assert tr.on_time_pct == 80.0

    def test_zero_total_no_division_error(self):
        tr = TierResult(tier_name="test", deadline_ms=2000.0)
        assert tr.success_at_slo_pct == 0.0
        assert tr.accuracy_pct == 0.0
        assert tr.on_time_pct == 0.0


class TestComputeTierResults:
    def test_all_pass_all_tiers(self):
        tasks = [
            TaskResult("t1", latency_ms=500.0, json_valid=True, task_correct=True),
            TaskResult("t2", latency_ms=1000.0, json_valid=True, task_correct=True),
        ]
        results = compute_tier_results(tasks)

        for tier_name in ["interactive", "standard", "batch"]:
            tr = results[tier_name]
            assert tr.total == 2
            assert tr.correct == 2
            assert tr.on_time == 2
            assert tr.success_at_slo == 2

    def test_latency_filters_by_tier(self):
        tasks = [
            TaskResult("t1", latency_ms=3000.0, json_valid=True, task_correct=True),
        ]
        results = compute_tier_results(tasks)

        # 3s > 2s interactive deadline
        assert results["interactive"].on_time == 0
        assert results["interactive"].success_at_slo == 0

        # 3s < 5s standard deadline
        assert results["standard"].on_time == 1
        assert results["standard"].success_at_slo == 1

        # 3s < 30s batch deadline
        assert results["batch"].on_time == 1
        assert results["batch"].success_at_slo == 1

    def test_invalid_json_blocks_success(self):
        tasks = [
            TaskResult("t1", latency_ms=500.0, json_valid=False, task_correct=True),
        ]
        results = compute_tier_results(tasks)

        # Correct and on-time but invalid JSON → no S@SLO credit
        assert results["interactive"].correct == 1
        assert results["interactive"].on_time == 1
        assert results["interactive"].success_at_slo == 0

    def test_incorrect_blocks_success(self):
        tasks = [
            TaskResult("t1", latency_ms=500.0, json_valid=True, task_correct=False),
        ]
        results = compute_tier_results(tasks)

        assert results["interactive"].on_time == 1
        assert results["interactive"].correct == 0
        assert results["interactive"].success_at_slo == 0

    def test_custom_tiers(self):
        custom = [SLOTier(name="fast", deadline_ms=100.0, description="d", typical_use="u")]
        tasks = [
            TaskResult("t1", latency_ms=50.0, json_valid=True, task_correct=True),
            TaskResult("t2", latency_ms=200.0, json_valid=True, task_correct=True),
        ]
        results = compute_tier_results(tasks, tiers=custom)

        assert "fast" in results
        assert results["fast"].on_time == 1
        assert results["fast"].success_at_slo == 1

    def test_empty_tasks(self):
        results = compute_tier_results([])
        for tier_name in ["interactive", "standard", "batch"]:
            assert results[tier_name].total == 0


class TestBenchmarkResult:
    def test_to_dict(self):
        br = BenchmarkResult(
            model_name="test-model",
            task_name="T1",
            tier_results={
                "interactive": TierResult("interactive", 2000.0, total=10, success_at_slo=8)
            },
            task_results=[TaskResult("t1", 500.0, True, True)],
            metadata={"source": "test"},
        )
        d = br.to_dict()
        assert d["model"] == "test-model"
        assert d["task"] == "T1"
        assert d["n_samples"] == 1
        assert "interactive" in d["tiers"]
        assert d["metadata"]["source"] == "test"


# ---- leaderboard ----

class TestRankValues:
    def test_descending_ranking(self):
        values = [90.0, 70.0, 80.0]
        ranks = rank_values(values, descending=True)
        # 90 → rank 1, 80 → rank 2, 70 → rank 3
        assert ranks == [1.0, 3.0, 2.0]

    def test_ascending_ranking(self):
        values = [90.0, 70.0, 80.0]
        ranks = rank_values(values, descending=False)
        # 70 → rank 1, 80 → rank 2, 90 → rank 3
        assert ranks == [3.0, 1.0, 2.0]

    def test_tied_values(self):
        values = [80.0, 80.0, 90.0]
        ranks = rank_values(values, descending=True)
        # 90 → rank 1, two 80s share ranks 2 and 3 → average 2.5
        assert ranks == [2.5, 2.5, 1.0]

    def test_all_same(self):
        values = [50.0, 50.0, 50.0]
        ranks = rank_values(values)
        assert ranks == [2.0, 2.0, 2.0]

    def test_single_value(self):
        ranks = rank_values([42.0])
        assert ranks == [1.0]


class TestSpearmanRho:
    def test_perfect_correlation(self):
        ranks_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        ranks_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert compute_spearman_rho(ranks_a, ranks_b) == pytest.approx(1.0)

    def test_perfect_inverse(self):
        ranks_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        ranks_b = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert compute_spearman_rho(ranks_a, ranks_b) == pytest.approx(-1.0)

    def test_no_correlation(self):
        # Specific permutation that gives rho ≈ 0
        ranks_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        ranks_b = [3.0, 5.0, 1.0, 2.0, 4.0]
        rho = compute_spearman_rho(ranks_a, ranks_b)
        assert -0.5 < rho < 0.5

    def test_too_few_returns_zero(self):
        assert compute_spearman_rho([1.0], [1.0]) == 0.0
        assert compute_spearman_rho([1.0, 2.0], [2.0, 1.0]) == 0.0


class TestBootstrapSpearman:
    def test_perfect_correlation(self):
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = bootstrap_spearman(a, b, n_bootstrap=1000)

        assert result["rho"] == pytest.approx(1.0)
        assert result["n_models"] == 5
        assert result["ci_lower"] >= 0.5
        assert "p_value" in result

    def test_inverse_correlation(self):
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [500.0, 400.0, 300.0, 200.0, 100.0]
        result = bootstrap_spearman(a, b, n_bootstrap=1000)

        assert result["rho"] == pytest.approx(-1.0)

    def test_reproducible_with_seed(self):
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [50.0, 10.0, 40.0, 20.0, 30.0]
        r1 = bootstrap_spearman(a, b, seed=42)
        r2 = bootstrap_spearman(a, b, seed=42)
        assert r1 == r2

    def test_different_seeds_different_ci(self):
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [50.0, 10.0, 40.0, 20.0, 30.0]
        r1 = bootstrap_spearman(a, b, seed=42, n_bootstrap=500)
        r2 = bootstrap_spearman(a, b, seed=99, n_bootstrap=500)
        # Rho should be the same (it's the observed value) but CIs may differ
        assert r1["rho"] == r2["rho"]
