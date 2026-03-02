"""Tests for agent_stable_slo.eval.moe_profiler module.

Covers MoEProfile dataclass serialization, MoE architecture detection,
hardware info collection (with mocked subprocess), and dense vs. MoE
classification.  All MLX imports are mocked.
"""

import json
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from agent_stable_slo.eval.moe_profiler import (
    MoEProfile,
    is_moe,
    _extract_expert_counts,
    _count_params_from_config,
    get_hardware_info,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dense_config():
    """Model config for a standard dense transformer."""
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
    }


@pytest.fixture
def moe_config_qwen():
    """Model config resembling Qwen MoE."""
    return {
        "model_type": "qwen2_moe",
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "num_parameters": 57_000_000_000,
    }


@pytest.fixture
def moe_config_mixtral():
    """Model config resembling Mixtral."""
    return {
        "model_type": "mixtral",
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "num_parameters": 46_700_000_000,
    }


@pytest.fixture
def moe_config_deepseek():
    """Model config resembling DeepSeek MoE with n_routed_experts."""
    return {
        "model_type": "deepseek",
        "n_routed_experts": 16,
        "top_k": 2,
    }


# ---------------------------------------------------------------------------
# 1. MoEProfile dataclass
# ---------------------------------------------------------------------------


class TestMoEProfileDataclass:
    def test_default_values(self):
        p = MoEProfile(model_name="test-model")
        assert p.model_name == "test-model"
        assert p.total_params == 0
        assert p.active_params == 0
        assert p.num_experts == 0
        assert p.num_active == 0
        assert p.architecture_type == "dense"
        assert p.peak_memory_gb == 0.0
        assert p.tokens_per_second == 0.0
        assert p.memory_stats == {}

    def test_serialization_to_dict(self):
        p = MoEProfile(
            model_name="qwen-moe",
            total_params=57_000_000_000,
            active_params=3_000_000_000,
            num_experts=64,
            num_active=4,
            architecture_type="moe",
            peak_memory_gb=12.5,
            tokens_per_second=45.2,
            memory_stats={"active_memory_gb": 10.1, "peak_memory_gb": 12.5},
        )
        d = asdict(p)
        assert isinstance(d, dict)
        assert d["model_name"] == "qwen-moe"
        assert d["total_params"] == 57_000_000_000
        assert d["architecture_type"] == "moe"
        assert d["num_experts"] == 64

    def test_serialization_to_json(self):
        p = MoEProfile(
            model_name="test",
            total_params=1000,
            peak_memory_gb=1.5,
        )
        s = json.dumps(asdict(p))
        roundtripped = json.loads(s)
        assert roundtripped["model_name"] == "test"
        assert roundtripped["total_params"] == 1000
        assert roundtripped["peak_memory_gb"] == 1.5

    def test_memory_stats_default_factory(self):
        p1 = MoEProfile(model_name="a")
        p2 = MoEProfile(model_name="b")
        # Ensure default_factory creates separate dicts
        p1.memory_stats["key"] = "val"
        assert "key" not in p2.memory_stats


# ---------------------------------------------------------------------------
# 2. MoE detection
# ---------------------------------------------------------------------------


class TestIsMoeDetection:
    def test_dense_config_returns_false(self, dense_config):
        assert is_moe(dense_config) is False

    def test_num_experts_triggers_moe(self, moe_config_qwen):
        assert is_moe(moe_config_qwen) is True

    def test_num_local_experts_triggers_moe(self, moe_config_mixtral):
        assert is_moe(moe_config_mixtral) is True

    def test_n_routed_experts_triggers_moe(self, moe_config_deepseek):
        assert is_moe(moe_config_deepseek) is True

    def test_model_type_moe_keyword(self):
        config = {"model_type": "some_moe_variant"}
        assert is_moe(config) is True

    def test_model_type_mixture_keyword(self):
        config = {"model_type": "mixture_of_experts"}
        assert is_moe(config) is True

    def test_num_experts_one_is_dense(self):
        """A model with num_experts=1 is effectively dense."""
        config = {"num_experts": 1}
        assert is_moe(config) is False

    def test_empty_config(self):
        assert is_moe({}) is False

    def test_moe_num_experts_key(self):
        config = {"moe_num_experts": 8}
        assert is_moe(config) is True

    def test_num_experts_per_token_key(self):
        config = {"num_experts_per_token": 2}
        assert is_moe(config) is True


# ---------------------------------------------------------------------------
# 3. Expert count extraction
# ---------------------------------------------------------------------------


class TestExtractExpertCounts:
    def test_qwen_style(self, moe_config_qwen):
        total, active = _extract_expert_counts(moe_config_qwen)
        assert total == 64
        assert active == 4

    def test_mixtral_style(self, moe_config_mixtral):
        total, active = _extract_expert_counts(moe_config_mixtral)
        assert total == 8
        assert active == 2

    def test_deepseek_style(self, moe_config_deepseek):
        total, active = _extract_expert_counts(moe_config_deepseek)
        assert total == 16
        assert active == 2

    def test_dense_returns_zero(self, dense_config):
        total, active = _extract_expert_counts(dense_config)
        assert total == 0
        assert active == 0

    def test_total_without_active_defaults_to_top2(self):
        config = {"num_experts": 8}
        total, active = _extract_expert_counts(config)
        assert total == 8
        assert active == 2  # fallback min(2, total)

    def test_single_expert_active(self):
        config = {"num_experts": 1}
        total, active = _extract_expert_counts(config)
        assert total == 1
        assert active == 1  # min(2, 1) = 1


# ---------------------------------------------------------------------------
# 4. Parameter counting
# ---------------------------------------------------------------------------


class TestCountParamsFromConfig:
    def test_dense_model_active_equals_total(self, dense_config):
        dense_config["num_parameters"] = 7_000_000_000
        total, active = _count_params_from_config(dense_config)
        assert total == 7_000_000_000
        assert active == 7_000_000_000

    def test_moe_model_active_less_than_total(self, moe_config_qwen):
        total, active = _count_params_from_config(moe_config_qwen)
        assert total == 57_000_000_000
        assert active < total

    def test_no_params_in_config(self):
        total, active = _count_params_from_config({})
        assert total == 0
        assert active == 0


# ---------------------------------------------------------------------------
# 5. Hardware info with mocked subprocess
# ---------------------------------------------------------------------------


class TestGetHardwareInfo:
    @patch("agent_stable_slo.eval.moe_profiler._sysctl")
    def test_dict_structure(self, mock_sysctl):
        mock_sysctl.side_effect = lambda key: {
            "machdep.cpu.brand_string": "Apple M2 Max",
            "hw.ncpu": "12",
            "hw.perflevel0.logicalcpu_max": "8",
            "hw.perflevel1.logicalcpu_max": "4",
            "machdep.cpu.core_count": "38",
            "hw.perflevel0.physicalcpu_max": "8",
            "hw.memsize": str(64 * 1024**3),
        }.get(key, "")

        with (
            patch.dict("sys.modules", {"mlx": MagicMock(__version__="0.22.0")}),
            patch.dict("sys.modules", {"mlx_lm": MagicMock(__version__="0.21.0")}),
        ):
            info = get_hardware_info()

        assert isinstance(info, dict)
        assert info["chip"] == "Apple M2 Max"
        assert "core_count_total" in info
        assert "core_count_perf" in info
        assert "core_count_eff" in info
        assert "memory_gb" in info
        assert "macos_version" in info
        assert "macos_build" in info

    @patch("agent_stable_slo.eval.moe_profiler._sysctl", return_value="")
    def test_handles_missing_sysctl(self, mock_sysctl):
        info = get_hardware_info()
        assert isinstance(info, dict)
        assert info["chip"] == ""

    def test_mlx_not_installed(self):
        """get_hardware_info should report 'not installed' without error."""
        with (
            patch(
                "agent_stable_slo.eval.moe_profiler._sysctl",
                return_value="",
            ),
            patch.dict("sys.modules", {"mlx": None}),
            patch.dict("sys.modules", {"mlx_lm": None}),
        ):
            info = get_hardware_info()
            assert info["mlx_version"] == "not installed"
            assert info["mlx_lm_version"] == "not installed"


# ---------------------------------------------------------------------------
# 6. Dense vs MoE classification
# ---------------------------------------------------------------------------


class TestDenseVsMoeClassification:
    def test_dense_architecture_type(self, dense_config):
        profile = MoEProfile(model_name="llama-8b")
        profile.architecture_type = "moe" if is_moe(dense_config) else "dense"
        assert profile.architecture_type == "dense"

    def test_moe_architecture_type(self, moe_config_qwen):
        profile = MoEProfile(model_name="qwen-moe")
        profile.architecture_type = "moe" if is_moe(moe_config_qwen) else "dense"
        assert profile.architecture_type == "moe"

    def test_all_13_models_classified(self):
        """The 13 models in the study are all dense (no MoE in the Lucky 13)."""
        dense_models = [
            {"model_type": "llama"},       # Llama-3.2-1B
            {"model_type": "llama"},       # Llama-3.2-3B
            {"model_type": "qwen2"},       # Qwen2.5-3B
            {"model_type": "phi3"},        # Phi-3-mini
            {"model_type": "qwen2"},       # Qwen3-4B
            {"model_type": "yi"},          # Yi-1.5-6B
            {"model_type": "mistral"},     # Mistral-7B
            {"model_type": "falcon_mamba"},  # Falcon-Mamba-7B
            {"model_type": "gpt"},         # GPT-OSS-20B
            {"model_type": "mistral"},     # Ministral-8B
            {"model_type": "llama"},       # Llama-3.1-8B
            {"model_type": "gemma2"},      # Gemma-2-9B
            {"model_type": "gemma"},       # Gemma-3-12B
        ]
        for config in dense_models:
            assert is_moe(config) is False, f"Falsely detected MoE: {config}"

    def test_known_moe_models_detected(self):
        """Known MoE models should be detected."""
        moe_models = [
            {"model_type": "mixtral", "num_local_experts": 8, "num_experts_per_tok": 2},
            {"model_type": "qwen2_moe", "num_experts": 64, "num_experts_per_tok": 4},
            {"n_routed_experts": 160, "top_k": 6},
        ]
        for config in moe_models:
            assert is_moe(config) is True, f"Failed to detect MoE: {config}"
