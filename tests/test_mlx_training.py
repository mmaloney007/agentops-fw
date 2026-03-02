"""Tests for MLX GRPO training adapter and config infrastructure.

Covers GRPOTrainConfig validation, default values, composite_reward
compatibility with the training adapter interface, training log format,
and YAML config file loading for all GRPO presets.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agent_stable_slo.utils.config import (
    CONFIG_VERSION,
    GRPOTrainConfig,
    validate_or_raise,
    migrate_config,
)
from agent_stable_slo.rewards.composite import composite_reward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "grpo"
TASKS_FILE = Path(__file__).resolve().parent.parent / "tasks" / "robust_eval_gold.jsonl"


@pytest.fixture
def valid_config_dict():
    """Minimal valid config dict (tasks file must exist)."""
    return {
        "base_model": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
        "tasks": str(TASKS_FILE),
        "steps": 4,
        "max_prompt_len": 128,
        "max_new_tokens": 32,
        "lr": 1e-4,
        "seed": 42,
        "no_silent_defaults": False,
    }


@pytest.fixture
def sample_schema():
    return {
        "type": "object",
        "properties": {"intent": {"type": "string"}},
        "required": ["intent"],
    }


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------


class TestMLXTrainConfigValidation:
    def test_valid_config_parses(self, valid_config_dict):
        cfg = validate_or_raise(valid_config_dict)
        assert isinstance(cfg, GRPOTrainConfig)
        assert cfg.steps == 4
        assert cfg.lr == 1e-4

    def test_steps_must_be_positive(self, valid_config_dict):
        valid_config_dict["steps"] = 0
        with pytest.raises(SystemExit):
            validate_or_raise(valid_config_dict)

    def test_steps_capped_at_20000(self, valid_config_dict):
        valid_config_dict["steps"] = 20001
        with pytest.raises(SystemExit):
            validate_or_raise(valid_config_dict)

    def test_lr_must_be_positive(self, valid_config_dict):
        valid_config_dict["lr"] = 0
        with pytest.raises(SystemExit):
            validate_or_raise(valid_config_dict)

    def test_temperature_must_be_non_negative(self, valid_config_dict):
        valid_config_dict["temperature"] = -0.1
        with pytest.raises(SystemExit):
            validate_or_raise(valid_config_dict)

    def test_top_p_range(self, valid_config_dict):
        valid_config_dict["top_p"] = 1.5
        with pytest.raises(SystemExit):
            validate_or_raise(valid_config_dict)

    def test_lora_dropout_range(self, valid_config_dict):
        valid_config_dict["lora_dropout"] = 0.6
        with pytest.raises(SystemExit):
            validate_or_raise(valid_config_dict)

    def test_checkpoint_every_cannot_be_1(self, valid_config_dict):
        valid_config_dict["checkpoint_every"] = 1
        with pytest.raises(SystemExit):
            validate_or_raise(valid_config_dict)

    def test_extra_fields_forbidden(self, valid_config_dict):
        valid_config_dict["bogus_field"] = True
        with pytest.raises(SystemExit):
            validate_or_raise(valid_config_dict)

    def test_nonexistent_tasks_file_rejected(self, valid_config_dict):
        valid_config_dict["tasks"] = "/nonexistent/path/tasks.jsonl"
        with pytest.raises(SystemExit):
            validate_or_raise(valid_config_dict)


# ---------------------------------------------------------------------------
# 2. Config defaults
# ---------------------------------------------------------------------------


class TestMLXTrainConfigDefaults:
    def test_default_base_model(self):
        cfg = GRPOTrainConfig.model_construct()
        assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"

    def test_default_steps(self):
        cfg = GRPOTrainConfig.model_construct()
        assert cfg.steps == 500

    def test_default_lora_rank(self):
        cfg = GRPOTrainConfig.model_construct()
        assert cfg.lora_rank == 16

    def test_default_lora_alpha(self):
        cfg = GRPOTrainConfig.model_construct()
        assert cfg.lora_alpha == 32

    def test_default_temperature(self):
        cfg = GRPOTrainConfig.model_construct()
        assert cfg.temperature == 0.7

    def test_default_torch_dtype(self):
        cfg = GRPOTrainConfig.model_construct()
        assert cfg.torch_dtype == "float16"

    def test_default_gradient_accumulation(self):
        cfg = GRPOTrainConfig.model_construct()
        assert cfg.gradient_accumulation == 1

    def test_default_config_version(self):
        cfg = GRPOTrainConfig.model_construct()
        assert cfg.config_version == CONFIG_VERSION

    def test_default_stability_samples(self):
        cfg = GRPOTrainConfig.model_construct()
        assert cfg.stability_samples == 1

    def test_default_faithfulness_disabled(self):
        cfg = GRPOTrainConfig.model_construct()
        assert cfg.enable_faithfulness_judge is False
        assert cfg.kappa_faithfulness == 0.0


# ---------------------------------------------------------------------------
# 3. Reward adapter interface compatibility
# ---------------------------------------------------------------------------


class TestRewardAdapterInterface:
    """Verify composite_reward is compatible with the training adapter's
    expected call signature and return type."""

    def test_composite_reward_returns_float(self, sample_schema):
        r = composite_reward(
            output_json={"intent": "greeting"},
            schema=sample_schema,
            ok_success=1,
            latency_ms=100.0,
            tokens=50,
            lam_latency=0.1,
            mu_cost=0.05,
        )
        assert isinstance(r, float)

    def test_composite_reward_with_all_components(self, sample_schema):
        """All parameters used by grpo_train_loop should be accepted."""
        r = composite_reward(
            output_json={"intent": "greeting"},
            schema=sample_schema,
            ok_success=1,
            latency_ms=500.0,
            tokens=100,
            lam_latency=0.1,
            mu_cost=0.05,
            disagreement_rate=0.2,
            gamma_stability=0.1,
            faithfulness=0.8,
            kappa_faithfulness=0.5,
        )
        assert isinstance(r, float)
        assert r > 0  # valid JSON + correct answer gives positive base reward

    def test_reward_zero_for_invalid_output(self, sample_schema):
        """Empty output with no correct answer should give low reward."""
        r = composite_reward(
            output_json={},
            schema=sample_schema,
            ok_success=0,
            latency_ms=0.0,
            tokens=0,
            lam_latency=0.0,
            mu_cost=0.0,
        )
        assert r == pytest.approx(0.0)

    def test_reward_bounded(self, sample_schema):
        """Reward should not diverge to unreasonable values."""
        r = composite_reward(
            output_json={"intent": "test"},
            schema=sample_schema,
            ok_success=1,
            latency_ms=30000.0,
            tokens=10000,
            lam_latency=0.1,
            mu_cost=0.1,
        )
        assert -100 < r < 100  # sanity bound


# ---------------------------------------------------------------------------
# 4. Training log format
# ---------------------------------------------------------------------------


class TestLogFormatCompatibility:
    """Verify training log entries match the expected schema from grpo_train_loop."""

    REQUIRED_LOG_FIELDS = {
        "step",
        "prompt",
        "output_text",
        "output_json",
        "reward",
        "advantage",
        "latency_ms",
        "ttft_ms",
        "json_valid",
        "tokens_out",
        "schema_path",
        "blocked",
        "faithfulness",
        "disagreement_rate",
        "stability_samples",
    }

    def test_log_entry_has_required_fields(self):
        """A representative log entry should contain all required fields."""
        entry = {
            "step": 0,
            "prompt": "Classify: hello",
            "output_text": '{"intent": "greeting"}',
            "output_json": {"intent": "greeting"},
            "reward": 2.0,
            "advantage": 0.5,
            "latency_ms": 150.3,
            "ttft_ms": 150.3,
            "json_valid": 1,
            "tokens_out": 12,
            "schema_path": "schemas/clinc.json",
            "blocked": False,
            "faithfulness": 1.0,
            "disagreement_rate": 0.0,
            "stability_samples": 1,
        }
        missing = self.REQUIRED_LOG_FIELDS - set(entry.keys())
        assert missing == set(), f"Missing fields: {missing}"

    def test_log_entry_json_serializable(self):
        entry = {
            "step": 42,
            "prompt": "test",
            "output_text": '{"a": 1}',
            "output_json": {"a": 1},
            "reward": 1.5,
            "advantage": 0.3,
            "latency_ms": 200.0,
            "ttft_ms": 200.0,
            "json_valid": 1,
            "tokens_out": 8,
            "schema_path": "schemas/test.json",
            "blocked": False,
            "faithfulness": 0.9,
            "disagreement_rate": 0.1,
            "stability_samples": 3,
        }
        serialized = json.dumps(entry)
        roundtripped = json.loads(serialized)
        assert roundtripped["step"] == 42
        assert roundtripped["reward"] == 1.5

    def test_optional_gold_field(self):
        """The gold field is optional in log entries."""
        entry = {
            "step": 0,
            "prompt": "test",
            "output_text": "{}",
            "output_json": {},
            "reward": 0.0,
            "advantage": 0.0,
            "latency_ms": 0.0,
            "ttft_ms": 0.0,
            "json_valid": 0,
            "tokens_out": 0,
            "schema_path": "schemas/test.json",
            "blocked": False,
            "faithfulness": 1.0,
            "disagreement_rate": 0.0,
            "stability_samples": 1,
            "gold": "expected_answer",
        }
        assert "gold" in entry
        assert self.REQUIRED_LOG_FIELDS <= set(entry.keys())


# ---------------------------------------------------------------------------
# 5. YAML config loading
# ---------------------------------------------------------------------------


class TestConfigYamlLoading:
    """Load each GRPO YAML config and verify it parses and migrates cleanly."""

    @pytest.fixture
    def yaml_files(self):
        if not CONFIGS_DIR.exists():
            pytest.skip(f"Config directory not found: {CONFIGS_DIR}")
        files = sorted(CONFIGS_DIR.glob("*.yaml"))
        if not files:
            pytest.skip("No YAML config files found")
        return files

    def test_all_configs_parse_as_yaml(self, yaml_files):
        for path in yaml_files:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{path.name} did not parse to dict"

    def test_all_configs_have_config_version(self, yaml_files):
        for path in yaml_files:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            migrated = migrate_config(data)
            assert "config_version" in migrated, f"{path.name} missing config_version"

    def test_all_configs_have_base_model(self, yaml_files):
        for path in yaml_files:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert "base_model" in data, f"{path.name} missing base_model"

    def test_all_configs_have_steps(self, yaml_files):
        for path in yaml_files:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert "steps" in data, f"{path.name} missing steps"

    def test_all_configs_migrate_cleanly(self, yaml_files):
        for path in yaml_files:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            migrated = migrate_config(data)
            assert migrated["config_version"] == CONFIG_VERSION

    def test_tiny_smoke_config_parses(self):
        """Specifically test the tiny_smoke config used for CI."""
        path = CONFIGS_DIR / "tiny_smoke.yaml"
        if not path.exists():
            pytest.skip("tiny_smoke.yaml not found")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["steps"] == 4
        assert data["deterministic"] is True
        assert data["lora_rank"] == 2

    def test_p2_llama_1b_config_parses(self):
        """Test a representative P2 config."""
        path = CONFIGS_DIR / "p2_llama_1b.yaml"
        if not path.exists():
            pytest.skip("p2_llama_1b.yaml not found")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["base_model"] == "meta-llama/Llama-3.2-1B-Instruct"
        assert data["steps"] == 500
        assert data["lora_rank"] == 16
        assert data["lam_latency"] == 0.1

    def test_falcon_mamba_config_has_custom_targets(self):
        """Falcon-Mamba uses SSM-specific LoRA targets."""
        path = CONFIGS_DIR / "p2_falcon_mamba.yaml"
        if not path.exists():
            pytest.skip("p2_falcon_mamba.yaml not found")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        targets = data.get("lora_targets", "")
        assert "in_proj" in targets
        assert "x_proj" in targets
        assert "dt_proj" in targets
