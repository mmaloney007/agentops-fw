"""Tests for agent_stable_slo.rewards modules.

Covers: schema_reward, slo_reward, stability_reward, composite.
"""
import pytest

from agent_stable_slo.rewards.schema_reward import schema_valid
from agent_stable_slo.rewards.slo_reward import latency_penalty, cost_penalty
from agent_stable_slo.rewards.stability_reward import stability_penalty
from agent_stable_slo.rewards.composite import composite_reward


# ---- schema_reward ----

class TestSchemaValid:
    def test_valid_object(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert schema_valid({"name": "Alice"}, schema) == 1

    def test_missing_required_field(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert schema_valid({}, schema) == 0

    def test_wrong_type(self):
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        assert schema_valid({"age": "not-an-int"}, schema) == 0

    def test_none_input(self):
        schema = {"type": "object"}
        assert schema_valid(None, schema) == 0

    def test_empty_schema(self):
        assert schema_valid({"anything": "goes"}, {}) == 1

    def test_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                }
            },
            "required": ["user"],
        }
        assert schema_valid({"user": {"id": 42}}, schema) == 1
        assert schema_valid({"user": {}}, schema) == 0


# ---- slo_reward ----

class TestLatencyPenalty:
    def test_zero_latency(self):
        assert latency_penalty(0.0, lam=0.1) == 0.0

    def test_positive_penalty(self):
        # 2000ms with lambda=0.1 → -0.1 * 2.0 = -0.2
        assert latency_penalty(2000.0, lam=0.1) == pytest.approx(-0.2)

    def test_zero_lambda(self):
        assert latency_penalty(5000.0, lam=0.0) == 0.0

    def test_scales_linearly(self):
        p1 = latency_penalty(1000.0, lam=0.1)
        p2 = latency_penalty(2000.0, lam=0.1)
        assert p2 == pytest.approx(2 * p1)


class TestCostPenalty:
    def test_zero_tokens(self):
        assert cost_penalty(0, mu=0.1) == 0.0

    def test_positive_penalty(self):
        # 1000 tokens with mu=0.1 → -0.1 * 1.0 = -0.1
        assert cost_penalty(1000, mu=0.1) == pytest.approx(-0.1)

    def test_zero_mu(self):
        assert cost_penalty(5000, mu=0.0) == 0.0


# ---- stability_reward ----

class TestStabilityPenalty:
    def test_zero_disagreement(self):
        assert stability_penalty(0.0, gamma=1.0) == 0.0

    def test_full_disagreement(self):
        assert stability_penalty(1.0, gamma=0.5) == pytest.approx(-0.5)

    def test_zero_gamma(self):
        assert stability_penalty(0.8, gamma=0.0) == 0.0


# ---- composite_reward ----

class TestCompositeReward:
    @pytest.fixture
    def valid_schema(self):
        return {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        }

    def test_perfect_case(self, valid_schema):
        r = composite_reward(
            output_json={"intent": "greeting"},
            schema=valid_schema,
            ok_success=1,
            latency_ms=0.0,
            tokens=0,
            lam_latency=0.0,
            mu_cost=0.0,
        )
        # schema_valid=1 + ok_success=1 = 2.0
        assert r == pytest.approx(2.0)

    def test_schema_invalid_reduces_reward(self, valid_schema):
        r = composite_reward(
            output_json={},  # missing required "intent"
            schema=valid_schema,
            ok_success=1,
            latency_ms=0.0,
            tokens=0,
            lam_latency=0.0,
            mu_cost=0.0,
        )
        # schema_valid=0 + ok_success=1 = 1.0
        assert r == pytest.approx(1.0)

    def test_latency_penalty_applied(self, valid_schema):
        r = composite_reward(
            output_json={"intent": "greeting"},
            schema=valid_schema,
            ok_success=1,
            latency_ms=2000.0,
            tokens=0,
            lam_latency=0.1,
            mu_cost=0.0,
        )
        # 2.0 + latency_penalty(2000, 0.1) = 2.0 - 0.2 = 1.8
        assert r == pytest.approx(1.8)

    def test_cost_penalty_applied(self, valid_schema):
        r = composite_reward(
            output_json={"intent": "greeting"},
            schema=valid_schema,
            ok_success=1,
            latency_ms=0.0,
            tokens=1000,
            lam_latency=0.0,
            mu_cost=0.5,
        )
        # 2.0 + cost_penalty(1000, 0.5) = 2.0 - 0.5 = 1.5
        assert r == pytest.approx(1.5)

    def test_stability_penalty_applied(self, valid_schema):
        r = composite_reward(
            output_json={"intent": "greeting"},
            schema=valid_schema,
            ok_success=1,
            latency_ms=0.0,
            tokens=0,
            lam_latency=0.0,
            mu_cost=0.0,
            disagreement_rate=0.5,
            gamma_stability=1.0,
        )
        # 2.0 - 0.5 = 1.5
        assert r == pytest.approx(1.5)

    def test_faithfulness_bonus(self, valid_schema):
        r = composite_reward(
            output_json={"intent": "greeting"},
            schema=valid_schema,
            ok_success=1,
            latency_ms=0.0,
            tokens=0,
            lam_latency=0.0,
            mu_cost=0.0,
            faithfulness=1.0,
            kappa_faithfulness=1.0,
        )
        # 2.0 + 1.0 * (1.0 - 0.5) = 2.5
        assert r == pytest.approx(2.5)

    def test_faithfulness_penalty(self, valid_schema):
        r = composite_reward(
            output_json={"intent": "greeting"},
            schema=valid_schema,
            ok_success=1,
            latency_ms=0.0,
            tokens=0,
            lam_latency=0.0,
            mu_cost=0.0,
            faithfulness=0.0,
            kappa_faithfulness=1.0,
        )
        # 2.0 + 1.0 * (0.0 - 0.5) = 1.5
        assert r == pytest.approx(1.5)

    def test_all_penalties_combined(self, valid_schema):
        r = composite_reward(
            output_json={"intent": "greeting"},
            schema=valid_schema,
            ok_success=1,
            latency_ms=1000.0,
            tokens=500,
            lam_latency=0.1,
            mu_cost=0.2,
            disagreement_rate=0.3,
            gamma_stability=0.5,
            faithfulness=0.8,
            kappa_faithfulness=0.5,
        )
        expected = (
            1.0  # schema_valid
            + 1.0  # ok_success
            + (-0.1 * 1.0)  # latency: -0.1
            + (-0.2 * 0.5)  # cost: -0.1
            + (-0.5 * 0.3)  # stability: -0.15
            + (0.5 * (0.8 - 0.5))  # faithfulness: 0.15
        )
        assert r == pytest.approx(expected)

    def test_returns_float(self, valid_schema):
        r = composite_reward(
            output_json={"intent": "greeting"},
            schema=valid_schema,
            ok_success=1,
            latency_ms=500.0,
            tokens=100,
            lam_latency=0.1,
            mu_cost=0.1,
        )
        assert isinstance(r, float)
