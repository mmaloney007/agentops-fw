"""Tests for the Hybrid ANE+MLX GRPO Trainer.

Tests cover:
 - HybridGRPOConfig validation (ane_meta_dir required, tasks required, defaults)
 - _export_lora_weights returns dict from trainable_parameters
 - _ane_rollout calls generate_raw and returns expected dict shape
 - _parse_json 3-stage extraction (direct, code block, regex)

All MLX and ANE dependencies are mocked so tests run without Apple Silicon
or MLX installed.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_schema():
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["intent", "confidence"],
    }


@pytest.fixture
def tmp_ane_dir(tmp_path):
    """Create a temporary directory pretending to be an Anemll model dir."""
    ane_dir = tmp_path / "ane_model"
    ane_dir.mkdir()
    (ane_dir / "meta.yaml").write_text("model_info:\n  parameters:\n    context_length: 2048\n")
    return str(ane_dir)


@pytest.fixture
def tmp_task_file(tmp_path, simple_schema):
    """Create a temporary JSONL task file with an inline schema file."""
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(simple_schema))

    task_path = tmp_path / "tasks.jsonl"
    rec = {
        "prompt": "Classify: hello world",
        "schema_path": str(schema_path),
        "gold": {"intent": "greeting", "confidence": 0.9},
    }
    task_path.write_text(json.dumps(rec) + "\n")
    return str(task_path)


# ---------------------------------------------------------------------------
# 1. HybridGRPOConfig validation
# ---------------------------------------------------------------------------


class TestHybridGRPOConfig:
    """Validate HybridGRPOConfig dataclass constraints."""

    def test_missing_ane_meta_dir_raises(self, tmp_task_file):
        """ane_meta_dir is required and must be a real directory."""
        from agent_stable_slo.train.ane_grpo_adapter import HybridGRPOConfig

        with pytest.raises(ValueError, match="ane_meta_dir"):
            HybridGRPOConfig(
                base_model="test/model",
                ane_meta_dir="/nonexistent/path/that/does/not/exist",
                tasks=[tmp_task_file],
            )

    def test_empty_ane_meta_dir_raises(self, tmp_task_file):
        """Empty ane_meta_dir string should raise."""
        from agent_stable_slo.train.ane_grpo_adapter import HybridGRPOConfig

        with pytest.raises(ValueError, match="ane_meta_dir"):
            HybridGRPOConfig(
                base_model="test/model",
                ane_meta_dir="",
                tasks=[tmp_task_file],
            )

    def test_empty_tasks_raises(self, tmp_ane_dir):
        """tasks list must be non-empty."""
        from agent_stable_slo.train.ane_grpo_adapter import HybridGRPOConfig

        with pytest.raises(ValueError, match="tasks"):
            HybridGRPOConfig(
                base_model="test/model",
                ane_meta_dir=tmp_ane_dir,
                tasks=[],
            )

    def test_valid_config_works(self, tmp_ane_dir, tmp_task_file):
        """A fully valid config should instantiate without error."""
        from agent_stable_slo.train.ane_grpo_adapter import HybridGRPOConfig

        cfg = HybridGRPOConfig(
            base_model="test/model",
            ane_meta_dir=tmp_ane_dir,
            tasks=[tmp_task_file],
        )
        assert cfg.base_model == "test/model"
        assert cfg.ane_meta_dir == tmp_ane_dir
        assert cfg.tasks == [tmp_task_file]
        # Check defaults
        assert cfg.num_steps == 200
        assert cfg.group_size == 4
        assert cfg.beta == 0.1
        assert cfg.lora_rank == 8
        assert cfg.lora_layers == 16
        assert cfg.max_tokens == 256
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.95
        assert cfg.lam_latency == 0.1
        assert cfg.mu_cost == 0.01
        assert cfg.learning_rate == 1e-4
        assert cfg.seed == 42
        assert cfg.checkpoint_every == 50
        assert cfg.batch_update_interval == 1
        assert cfg.measure_power is False

    def test_custom_values_accepted(self, tmp_ane_dir, tmp_task_file):
        """Override defaults with custom values."""
        from agent_stable_slo.train.ane_grpo_adapter import HybridGRPOConfig

        cfg = HybridGRPOConfig(
            base_model="custom/model",
            ane_meta_dir=tmp_ane_dir,
            tasks=[tmp_task_file],
            num_steps=500,
            group_size=8,
            beta=0.05,
            learning_rate=5e-5,
            batch_update_interval=3,
            measure_power=True,
        )
        assert cfg.num_steps == 500
        assert cfg.group_size == 8
        assert cfg.beta == 0.05
        assert cfg.learning_rate == 5e-5
        assert cfg.batch_update_interval == 3
        assert cfg.measure_power is True


# ---------------------------------------------------------------------------
# 2. Weight sync / export
# ---------------------------------------------------------------------------


class TestWeightSync:
    """Test _export_lora_weights extracts trainable parameters."""

    def test_export_lora_weights_returns_dict(self):
        """_export_lora_weights should return a dict of LoRA params via tree_flatten."""
        from agent_stable_slo.train.ane_grpo_adapter import _export_lora_weights

        mock_model = MagicMock()
        mock_param_a = MagicMock()
        mock_param_b = MagicMock()
        mock_param_w = MagicMock()
        # trainable_parameters() returns nested dict that tree_flatten flattens
        mock_model.trainable_parameters.return_value = {
            "layers": {
                "0": {
                    "lora_a": mock_param_a,
                    "lora_b": mock_param_b,
                    "weight": mock_param_w,  # non-lora param, should be excluded
                }
            }
        }

        result = _export_lora_weights(mock_model)
        assert isinstance(result, dict)
        assert len(result) == 2  # only lora params
        lora_keys = list(result.keys())
        assert any("lora_a" in k for k in lora_keys)
        assert any("lora_b" in k for k in lora_keys)

    def test_export_lora_weights_empty_model(self):
        """Model with no trainable params returns empty dict."""
        from agent_stable_slo.train.ane_grpo_adapter import _export_lora_weights

        mock_model = MagicMock()
        mock_model.trainable_parameters.return_value = {}

        result = _export_lora_weights(mock_model)
        assert result == {}


# ---------------------------------------------------------------------------
# 3. Rollout phase
# ---------------------------------------------------------------------------


class TestRolloutPhase:
    """Test _ane_rollout calls generate_raw and returns expected dict shape."""

    @patch("agent_stable_slo.train.ane_grpo_adapter.ane_local")
    def test_ane_rollout_returns_expected_keys(self, mock_ane_module, simple_schema):
        """_ane_rollout should return dict with text, parsed, latency_ms, ttft_ms, tokens_in, tokens_out."""
        from agent_stable_slo.train.ane_grpo_adapter import _ane_rollout

        # ane_local.generate_raw returns 6-tuple
        mock_ane_module.generate_raw.return_value = (
            '{"intent": "greeting", "confidence": 0.9}',  # raw_text
            {"intent": "greeting", "confidence": 0.9},     # parsed
            42.5,                                           # latency_ms
            10.2,                                           # ttft_ms
            15,                                             # tokens_in
            8,                                              # tokens_out
        )

        result = _ane_rollout(
            "Classify: hello", simple_schema,
            temperature=0.7, max_tokens=256,
        )

        assert isinstance(result, dict)
        assert "text" in result
        assert "parsed" in result
        assert "latency_ms" in result
        assert "ttft_ms" in result
        assert "tokens_in" in result
        assert "tokens_out" in result

        assert result["text"] == '{"intent": "greeting", "confidence": 0.9}'
        assert result["parsed"]["intent"] == "greeting"
        assert result["latency_ms"] == 42.5
        assert result["ttft_ms"] == 10.2
        assert result["tokens_in"] == 15
        assert result["tokens_out"] == 8

    @patch("agent_stable_slo.train.ane_grpo_adapter.ane_local")
    def test_ane_rollout_passes_kwargs(self, mock_ane_module, simple_schema):
        """Verify kwargs (temperature, max_tokens) are forwarded to generate_raw."""
        from agent_stable_slo.train.ane_grpo_adapter import _ane_rollout

        mock_ane_module.generate_raw.return_value = ("text", {}, 10.0, 5.0, 5, 3)

        _ane_rollout(
            "test prompt", simple_schema,
            temperature=0.3, max_tokens=128,
        )

        mock_ane_module.generate_raw.assert_called_once_with(
            prompt="test prompt",
            schema=simple_schema,
            mode="structured",
            temperature=0.3,
            max_tokens=128,
        )

    @patch("agent_stable_slo.train.ane_grpo_adapter.ane_local")
    def test_ane_rollout_handles_empty_response(self, mock_ane_module, simple_schema):
        """Empty model response should still return valid dict structure."""
        from agent_stable_slo.train.ane_grpo_adapter import _ane_rollout

        mock_ane_module.generate_raw.return_value = ("", {}, 5.0, 5.0, 5, 0)

        result = _ane_rollout("test", simple_schema)

        assert result["text"] == ""
        assert result["parsed"] == {}
        assert result["tokens_out"] == 0


# ---------------------------------------------------------------------------
# 4. JSON parsing (3-stage)
# ---------------------------------------------------------------------------


class TestParseJson:
    """Test _parse_json 3-stage extraction matching mlx_grpo_adapter pattern."""

    def _parse(self, text, schema=None):
        from agent_stable_slo.train.ane_grpo_adapter import _parse_json
        return _parse_json(text, schema or {})

    def test_direct_parse(self):
        result = self._parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_code_block_json(self):
        result = self._parse('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_bare_code_block(self):
        result = self._parse('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_regex_extraction(self):
        result = self._parse('Some text {"key": "value"} more text')
        assert result == {"key": "value"}

    def test_fallback_for_answer_schema(self):
        schema = {"properties": {"answer": {"type": "string"}}}
        result = self._parse("just plain text", schema)
        assert result == {"answer": "just plain text"}

    def test_invalid_text_no_answer_schema(self):
        result = self._parse("not json at all", {})
        assert result == {}


# ---------------------------------------------------------------------------
# 5. Dataset loading
# ---------------------------------------------------------------------------


class TestLoadDataset:
    """Test _load_dataset loads JSONL with inline schema loading."""

    def test_loads_single_file(self, tmp_task_file, simple_schema):
        from agent_stable_slo.train.ane_grpo_adapter import _load_dataset

        rows = _load_dataset([tmp_task_file])
        assert len(rows) == 1
        assert rows[0]["prompt"] == "Classify: hello world"
        assert "schema" in rows[0]
        assert rows[0]["schema"] == simple_schema
        assert rows[0]["_source_task"] == tmp_task_file

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")

        from agent_stable_slo.train.ane_grpo_adapter import _load_dataset

        with pytest.raises(ValueError, match="No task records"):
            _load_dataset([str(empty)])

    def test_loads_multiple_files(self, tmp_path, simple_schema):
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(simple_schema))

        task1 = tmp_path / "t1.jsonl"
        task1.write_text(json.dumps({
            "prompt": "task 1",
            "schema_path": str(schema_path),
        }) + "\n")

        task2 = tmp_path / "t2.jsonl"
        task2.write_text(json.dumps({
            "prompt": "task 2",
            "schema_path": str(schema_path),
        }) + "\n")

        from agent_stable_slo.train.ane_grpo_adapter import _load_dataset

        rows = _load_dataset([str(task1), str(task2)])
        assert len(rows) == 2
        assert rows[0]["prompt"] == "task 1"
        assert rows[1]["prompt"] == "task 2"


# ---------------------------------------------------------------------------
# 6. Trainer initialization (with mocked MLX)
# ---------------------------------------------------------------------------


class TestHybridGRPOTrainerInit:
    """Test HybridGRPOTrainer.__init__ sets up paths and loads data."""

    def test_init_creates_adapter_dir(self, tmp_ane_dir, tmp_task_file, tmp_path):
        from agent_stable_slo.train.ane_grpo_adapter import (
            HybridGRPOConfig,
            HybridGRPOTrainer,
        )

        adapter_dir = tmp_path / "adapter_out"
        log_path = tmp_path / "train.jsonl"

        cfg = HybridGRPOConfig(
            base_model="test/model",
            ane_meta_dir=tmp_ane_dir,
            tasks=[tmp_task_file],
            adapter_path=str(adapter_dir),
            log_path=str(log_path),
        )

        trainer = HybridGRPOTrainer(cfg)

        assert trainer.adapter_dir.exists()
        assert trainer.log_path == log_path
        assert len(trainer.dataset) == 1

    def test_init_auto_generates_paths(self, tmp_ane_dir, tmp_task_file):
        from agent_stable_slo.train.ane_grpo_adapter import (
            HybridGRPOConfig,
            HybridGRPOTrainer,
        )

        cfg = HybridGRPOConfig(
            base_model="test/model",
            ane_meta_dir=tmp_ane_dir,
            tasks=[tmp_task_file],
        )

        trainer = HybridGRPOTrainer(cfg)

        assert trainer.adapter_dir.exists()
        assert "hybrid_train_" in str(trainer.adapter_dir)
        assert trainer.log_path.name == "train_log.jsonl"
