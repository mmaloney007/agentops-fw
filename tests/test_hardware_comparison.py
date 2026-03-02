"""Tests for P5 hardware-comparison analysis logic.

Covers normalization factor computation with synthetic latency data,
SLO tier classification at Interactive/Standard/Batch, Spearman
correlation computation with known data, and MoE vs dense grouping.
"""

import math

import pytest

from agent_stable_slo.bench.slo_tiers import (
    SLOTier,
    INTERACTIVE,
    STANDARD,
    BATCH,
    TIERS,
)
from agent_stable_slo.bench.benchmark_runner import (
    TaskResult,
    TierResult,
    compute_tier_results,
)
from agent_stable_slo.bench.leaderboard import (
    compute_spearman_rho,
    rank_values,
    bootstrap_spearman,
)
from agent_stable_slo.eval.moe_profiler import is_moe, MoEProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cuda_latencies():
    """Synthetic per-task latency data for CUDA backend (fast)."""
    return [120.0, 350.0, 800.0, 1500.0, 2200.0, 4500.0]


@pytest.fixture
def mlx_latencies():
    """Synthetic per-task latency data for MLX/Metal backend (slower)."""
    return [250.0, 600.0, 1400.0, 2800.0, 4000.0, 8000.0]


@pytest.fixture
def model_configs():
    """A set of model configs for grouping tests."""
    return {
        "llama-3.2-1b": {"model_type": "llama"},
        "qwen2.5-3b": {"model_type": "qwen2"},
        "mistral-7b": {"model_type": "mistral"},
        "gemma-2-9b": {"model_type": "gemma2"},
        "qwen-moe-35b": {
            "model_type": "qwen2_moe",
            "num_experts": 64,
            "num_experts_per_tok": 4,
        },
        "mixtral-8x7b": {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        },
    }


# ---------------------------------------------------------------------------
# 1. Normalization factor computation
# ---------------------------------------------------------------------------


class TestNormalizationFactorComputation:
    """Test normalization factors for cross-platform latency comparison."""

    def _compute_normalization(self, base_latencies, target_latencies):
        """Compute geometric mean ratio as normalization factor."""
        assert len(base_latencies) == len(target_latencies)
        ratios = [t / b for b, t in zip(base_latencies, target_latencies)]
        log_sum = sum(math.log(r) for r in ratios)
        return math.exp(log_sum / len(ratios))

    def test_same_platform_factor_is_one(self, cuda_latencies):
        factor = self._compute_normalization(cuda_latencies, cuda_latencies)
        assert factor == pytest.approx(1.0)

    def test_slower_platform_factor_greater_than_one(
        self, cuda_latencies, mlx_latencies
    ):
        factor = self._compute_normalization(cuda_latencies, mlx_latencies)
        assert factor > 1.0

    def test_factor_is_reciprocal(self, cuda_latencies, mlx_latencies):
        factor_forward = self._compute_normalization(cuda_latencies, mlx_latencies)
        factor_reverse = self._compute_normalization(mlx_latencies, cuda_latencies)
        assert factor_forward * factor_reverse == pytest.approx(1.0, abs=1e-10)

    def test_uniform_scaling_factor(self):
        base = [100.0, 200.0, 300.0]
        target = [200.0, 400.0, 600.0]  # exactly 2x slower
        factor = self._compute_normalization(base, target)
        assert factor == pytest.approx(2.0)

    def test_non_uniform_scaling(self):
        base = [100.0, 100.0]
        target = [200.0, 400.0]  # 2x and 4x
        factor = self._compute_normalization(base, target)
        # Geometric mean of [2, 4] = sqrt(8) ~= 2.828
        expected = math.sqrt(8)
        assert factor == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 2. SLO tier classification
# ---------------------------------------------------------------------------


class TestSloTierClassification:
    def test_fast_response_passes_all_tiers(self):
        tasks = [TaskResult("t1", latency_ms=500.0, json_valid=True, task_correct=True)]
        results = compute_tier_results(tasks)
        assert results["interactive"].success_at_slo == 1
        assert results["standard"].success_at_slo == 1
        assert results["batch"].success_at_slo == 1

    def test_medium_response_fails_interactive(self):
        tasks = [
            TaskResult("t1", latency_ms=3000.0, json_valid=True, task_correct=True)
        ]
        results = compute_tier_results(tasks)
        assert results["interactive"].success_at_slo == 0
        assert results["standard"].success_at_slo == 1
        assert results["batch"].success_at_slo == 1

    def test_slow_response_only_passes_batch(self):
        tasks = [
            TaskResult("t1", latency_ms=10000.0, json_valid=True, task_correct=True)
        ]
        results = compute_tier_results(tasks)
        assert results["interactive"].success_at_slo == 0
        assert results["standard"].success_at_slo == 0
        assert results["batch"].success_at_slo == 1

    def test_very_slow_response_fails_all_tiers(self):
        tasks = [
            TaskResult("t1", latency_ms=60000.0, json_valid=True, task_correct=True)
        ]
        results = compute_tier_results(tasks)
        assert results["interactive"].success_at_slo == 0
        assert results["standard"].success_at_slo == 0
        assert results["batch"].success_at_slo == 0

    def test_boundary_exactly_at_deadline(self):
        """Response at exactly the deadline should be on_time."""
        tasks = [
            TaskResult("t1", latency_ms=2000.0, json_valid=True, task_correct=True)
        ]
        results = compute_tier_results(tasks)
        assert results["interactive"].on_time == 1

    def test_mixed_latencies(self):
        tasks = [
            TaskResult("fast", latency_ms=500.0, json_valid=True, task_correct=True),
            TaskResult("medium", latency_ms=3000.0, json_valid=True, task_correct=True),
            TaskResult("slow", latency_ms=10000.0, json_valid=True, task_correct=True),
        ]
        results = compute_tier_results(tasks)
        assert results["interactive"].success_at_slo == 1  # only fast
        assert results["standard"].success_at_slo == 2  # fast + medium
        assert results["batch"].success_at_slo == 3  # all three

    def test_invalid_json_blocks_slo_success(self):
        tasks = [
            TaskResult("t1", latency_ms=500.0, json_valid=False, task_correct=True)
        ]
        results = compute_tier_results(tasks)
        assert results["interactive"].on_time == 1
        assert results["interactive"].correct == 1
        assert results["interactive"].success_at_slo == 0

    def test_slo_pct_computation(self):
        tasks = [
            TaskResult("t1", latency_ms=500.0, json_valid=True, task_correct=True),
            TaskResult("t2", latency_ms=3000.0, json_valid=True, task_correct=True),
            TaskResult("t3", latency_ms=500.0, json_valid=True, task_correct=False),
            TaskResult("t4", latency_ms=500.0, json_valid=True, task_correct=True),
        ]
        results = compute_tier_results(tasks)
        # Interactive: t1 and t4 succeed (correct + on_time + valid), t3 is on_time+valid but wrong
        assert results["interactive"].success_at_slo == 2
        assert results["interactive"].total == 4
        assert results["interactive"].success_at_slo_pct == 50.0


# ---------------------------------------------------------------------------
# 3. Spearman correlation
# ---------------------------------------------------------------------------


class TestSpearmanCorrelation:
    def test_perfect_positive(self):
        ranks_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        ranks_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        rho = compute_spearman_rho(ranks_a, ranks_b)
        assert rho == pytest.approx(1.0)

    def test_perfect_negative(self):
        ranks_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        ranks_b = [5.0, 4.0, 3.0, 2.0, 1.0]
        rho = compute_spearman_rho(ranks_a, ranks_b)
        assert rho == pytest.approx(-1.0)

    def test_known_mild_correlation(self):
        """A specific permutation with known rho."""
        # accuracy ranks
        acc = [90.0, 85.0, 80.0, 75.0, 70.0]
        # S@SLO ranks (partially reordered)
        slo = [88.0, 75.0, 82.0, 72.0, 69.0]
        ranks_a = rank_values(acc)
        ranks_b = rank_values(slo)
        rho = compute_spearman_rho(ranks_a, ranks_b)
        assert -1.0 <= rho <= 1.0
        # These should be positively correlated
        assert rho > 0.5

    def test_too_few_values_returns_zero(self):
        assert compute_spearman_rho([1.0], [1.0]) == 0.0
        assert compute_spearman_rho([1.0, 2.0], [2.0, 1.0]) == 0.0

    def test_bootstrap_produces_valid_ci(self):
        a = [90.0, 80.0, 70.0, 60.0, 50.0]
        b = [88.0, 78.0, 68.0, 58.0, 48.0]
        result = bootstrap_spearman(a, b, n_bootstrap=500, seed=42)
        assert result["rho"] == pytest.approx(1.0)
        assert result["ci_lower"] <= result["ci_upper"]
        assert result["n_models"] == 5
        assert "p_value" in result

    def test_rank_values_descending(self):
        values = [90.0, 70.0, 80.0]
        ranks = rank_values(values, descending=True)
        # 90 -> rank 1, 80 -> rank 2, 70 -> rank 3
        assert ranks == [1.0, 3.0, 2.0]

    def test_rank_values_with_ties(self):
        values = [80.0, 80.0, 90.0]
        ranks = rank_values(values, descending=True)
        # 90 -> rank 1, two 80s -> average of ranks 2 and 3 = 2.5
        assert ranks == [2.5, 2.5, 1.0]


# ---------------------------------------------------------------------------
# 4. MoE vs dense grouping
# ---------------------------------------------------------------------------


class TestMoeVsDenseGrouping:
    def test_correct_grouping(self, model_configs):
        dense = []
        moe = []
        for name, config in model_configs.items():
            if is_moe(config):
                moe.append(name)
            else:
                dense.append(name)

        assert set(dense) == {"llama-3.2-1b", "qwen2.5-3b", "mistral-7b", "gemma-2-9b"}
        assert set(moe) == {"qwen-moe-35b", "mixtral-8x7b"}

    def test_dense_group_no_false_positives(self, model_configs):
        for name, config in model_configs.items():
            if name in ("llama-3.2-1b", "qwen2.5-3b", "mistral-7b", "gemma-2-9b"):
                assert is_moe(config) is False, f"{name} falsely detected as MoE"

    def test_moe_group_no_false_negatives(self, model_configs):
        for name, config in model_configs.items():
            if name in ("qwen-moe-35b", "mixtral-8x7b"):
                assert is_moe(config) is True, f"{name} not detected as MoE"

    def test_grouping_preserves_count(self, model_configs):
        total = len(model_configs)
        dense_count = sum(1 for c in model_configs.values() if not is_moe(c))
        moe_count = sum(1 for c in model_configs.values() if is_moe(c))
        assert dense_count + moe_count == total

    def test_profile_architecture_type_matches(self, model_configs):
        """MoEProfile.architecture_type should match is_moe detection."""
        for name, config in model_configs.items():
            profile = MoEProfile(model_name=name)
            profile.architecture_type = "moe" if is_moe(config) else "dense"
            if name in ("qwen-moe-35b", "mixtral-8x7b"):
                assert profile.architecture_type == "moe"
            else:
                assert profile.architecture_type == "dense"
