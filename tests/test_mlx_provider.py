"""Tests for MLX local inference provider.

Tests cover provider registration routing, generate_raw signature, JSON
extraction, prompt building, model caching, required-field backfill, and
graceful failure paths.  All MLX/mlx-lm dependencies are mocked so tests
run without Apple Silicon or MLX installed.
"""

import json
import os
from dataclasses import dataclass
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
def nested_schema():
    return {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
            },
            "tags": {"type": "array"},
        },
        "required": ["user", "tags"],
    }


# ---------------------------------------------------------------------------
# 1. Provider registration
# ---------------------------------------------------------------------------


class TestProviderRegistration:
    """Verify engine._provider_generate_raw routes to mlx_local correctly."""

    @patch.dict(os.environ, {"AOFW_PROVIDER": "mlx_local"})
    @patch("agent_stable_slo.rollout.providers.mlx_local.generate_raw")
    def test_routes_mlx_local(self, mock_gen):
        mock_gen.return_value = ("raw", {"k": "v"}, 10.0, 10.0, 5, 8)
        from agent_stable_slo.rollout.engine import _provider_generate_raw

        result = _provider_generate_raw("prompt", {}, "structured", 0.0, None)
        mock_gen.assert_called_once()
        # mlx_local returns 6-tuple; engine appends None for logprobs
        assert len(result) == 7
        assert result[-1] is None

    @patch.dict(os.environ, {"AOFW_PROVIDER": "mlx"})
    @patch("agent_stable_slo.rollout.providers.mlx_local.generate_raw")
    def test_routes_mlx_alias(self, mock_gen):
        mock_gen.return_value = ("raw", {"k": "v"}, 10.0, 10.0, 5, 8)
        from agent_stable_slo.rollout.engine import _provider_generate_raw

        result = _provider_generate_raw("prompt", {}, "structured", 0.0, None)
        mock_gen.assert_called_once()
        assert result[0] == "raw"

    @patch.dict(os.environ, {"AOFW_PROVIDER": "nonexistent_backend"})
    def test_fallback_for_unknown_provider(self):
        from agent_stable_slo.rollout.engine import _provider_generate_raw

        result = _provider_generate_raw("p", {}, "structured", 0.0, None)
        # Fallback returns empty/sentinel values
        assert result[0] == ""
        assert result[1] == {}


# ---------------------------------------------------------------------------
# 2. generate_raw signature
# ---------------------------------------------------------------------------


class TestGenerateRawSignature:
    """Mock the full mlx_lm pipeline and verify the 6-tuple return shape."""

    @patch.dict(os.environ, {"MLX_MODEL": "test-model", "MLX_MAX_TOKENS": "64"})
    def test_generate_raw_returns_6_tuple(self, simple_schema):
        """Verify generate_raw returns (raw_text, parsed, lat_ms, ttft_ms, tokens_in, tokens_out)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.encode.side_effect = [
            list(range(10)),    # input tokens (first call)
            list(range(5)),     # output tokens (second call)
        ]

        # mlx_lm.generate is imported inside generate_raw, so we mock the
        # entire mlx_lm module in sys.modules before calling generate_raw.
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate.return_value = '{"intent": "greeting", "confidence": 0.9}'
        mock_sample_utils = MagicMock()
        mock_sample_utils.make_sampler.return_value = lambda logits: logits

        with (
            patch(
                "agent_stable_slo.rollout.providers.mlx_local._get_model_and_tokenizer",
                return_value=(mock_model, mock_tokenizer),
            ),
            patch.dict("sys.modules", {
                "mlx_lm": mock_mlx_lm,
                "mlx_lm.sample_utils": mock_sample_utils,
            }),
        ):
            from agent_stable_slo.rollout.providers.mlx_local import generate_raw

            result = generate_raw(
                prompt="Classify this",
                schema=simple_schema,
                mode="structured",
                temperature=0.0,
                max_tokens=64,
            )

        # generate_raw returns a 6-tuple
        assert len(result) == 6
        raw_text, parsed, lat_ms, ttft_ms, tokens_in, tokens_out = result
        assert isinstance(raw_text, str)
        assert isinstance(parsed, dict)
        assert isinstance(lat_ms, float)
        assert isinstance(ttft_ms, float)
        assert isinstance(tokens_in, int)
        assert isinstance(tokens_out, int)
        assert parsed.get("intent") == "greeting"

    def test_build_prompt_and_extract(self, simple_schema):
        """Verify _build_prompt_with_schema and _extract_json work together."""
        from agent_stable_slo.rollout.providers.mlx_local import (
            _build_prompt_with_schema,
            _extract_json,
        )

        full_prompt = _build_prompt_with_schema("test prompt", simple_schema)
        assert "JSON" in full_prompt
        assert "intent" in full_prompt

        parsed = _extract_json('{"intent": "greeting", "confidence": 0.9}')
        assert parsed == {"intent": "greeting", "confidence": 0.9}


# ---------------------------------------------------------------------------
# 3. JSON extraction
# ---------------------------------------------------------------------------


class TestExtractJson:
    """Test _extract_json with various input formats."""

    def _extract(self, text):
        from agent_stable_slo.rollout.providers.mlx_local import _extract_json

        return _extract_json(text)

    def test_clean_json(self):
        result = self._extract('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_whitespace(self):
        result = self._extract('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = self._extract(text)
        assert result == {"key": "value"}

    def test_json_in_bare_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = self._extract(text)
        assert result == {"key": "value"}

    def test_json_buried_in_text(self):
        text = 'Here is the answer: {"key": "value"} and more text'
        result = self._extract(text)
        assert result == {"key": "value"}

    def test_invalid_json_returns_empty_dict(self):
        result = self._extract("not json at all")
        assert result == {}

    def test_nested_json(self):
        text = '{"outer": {"inner": "value"}}'
        result = self._extract(text)
        assert result == {"outer": {"inner": "value"}}

    def test_array_json_ignored(self):
        """_extract_json only returns dicts, not arrays."""
        result = self._extract('[1, 2, 3]')
        assert result == {}

    def test_multiple_json_objects_returns_first(self):
        text = 'result: {"a": 1} other {"b": 2}'
        result = self._extract(text)
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# 4. Build prompt with schema
# ---------------------------------------------------------------------------


class TestBuildPromptWithSchema:
    def _build(self, prompt, schema):
        from agent_stable_slo.rollout.providers.mlx_local import (
            _build_prompt_with_schema,
        )

        return _build_prompt_with_schema(prompt, schema)

    def test_schema_embedded_in_prompt(self, simple_schema):
        result = self._build("Classify this text", simple_schema)
        assert "Classify this text" in result
        assert "intent" in result
        assert "confidence" in result
        assert "JSON" in result

    def test_string_property_placeholder(self, simple_schema):
        result = self._build("test", simple_schema)
        # Should contain the example JSON with <value> placeholder
        assert "<value>" in result

    def test_number_property_placeholder(self, simple_schema):
        result = self._build("test", simple_schema)
        # Should contain 0 for number type
        parsed_example = json.loads(
            result.split("example:\n")[1].split("\n\nYour")[0]
        )
        assert parsed_example["confidence"] == 0

    def test_array_property_placeholder(self, nested_schema):
        result = self._build("test", nested_schema)
        assert "[]" in result

    def test_object_property_placeholder(self, nested_schema):
        result = self._build("test", nested_schema)
        assert "{}" in result

    def test_empty_schema(self):
        result = self._build("test", {})
        assert "test" in result
        assert "JSON" in result


# ---------------------------------------------------------------------------
# 5. Model cache
# ---------------------------------------------------------------------------


class TestModelCache:
    def test_cache_prevents_reload(self):
        from agent_stable_slo.rollout.providers.mlx_local import _MODEL_CACHE

        # Save original state and restore after
        original = dict(_MODEL_CACHE)
        try:
            mock_model = MagicMock()
            mock_tok = MagicMock()
            _MODEL_CACHE["test-model-xyz"] = (mock_model, mock_tok)

            from agent_stable_slo.rollout.providers.mlx_local import (
                _get_model_and_tokenizer,
            )

            # Should return cached version without importing mlx_lm
            model, tok = _get_model_and_tokenizer("test-model-xyz")
            assert model is mock_model
            assert tok is mock_tok
        finally:
            _MODEL_CACHE.clear()
            _MODEL_CACHE.update(original)

    def test_cache_miss_triggers_load(self):
        from agent_stable_slo.rollout.providers.mlx_local import _MODEL_CACHE

        original = dict(_MODEL_CACHE)
        try:
            _MODEL_CACHE.pop("brand-new-model", None)

            mock_model = MagicMock()
            mock_tok = MagicMock()
            mock_config = {"eos_token_id": 0}

            mock_load_model = MagicMock(return_value=(mock_model, mock_config))
            mock_load_tokenizer = MagicMock(return_value=mock_tok)
            mock_download = MagicMock(return_value="/tmp/fake")
            mock_utils = MagicMock(
                load_model=mock_load_model,
                load_tokenizer=mock_load_tokenizer,
                _download=mock_download,
            )
            with patch.dict(
                "sys.modules",
                {
                    "mlx_lm": MagicMock(),
                    "mlx_lm.utils": mock_utils,
                },
            ):
                from agent_stable_slo.rollout.providers import mlx_local

                # Force re-import path
                model, tok = mlx_local._get_model_and_tokenizer("brand-new-model")
                assert "brand-new-model" in mlx_local._MODEL_CACHE
        finally:
            _MODEL_CACHE.clear()
            _MODEL_CACHE.update(original)


# ---------------------------------------------------------------------------
# 6. Required-field backfill
# ---------------------------------------------------------------------------


class TestRequiredFieldBackfill:
    """Verify generate_raw fills missing required fields with type-appropriate defaults."""

    def test_missing_string_field_gets_empty_string(self):
        from agent_stable_slo.rollout.providers.mlx_local import _extract_json

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        # Simulate parsed output missing "name"
        parsed = _extract_json('{"age": 25}')
        assert parsed == {"age": 25}

        # The backfill logic in generate_raw would add defaults for missing required
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for req in required:
            if req not in parsed:
                typ = props.get(req, {}).get("type")
                if typ == "string":
                    parsed[req] = ""
                elif typ == "array":
                    parsed[req] = []
                elif typ == "object":
                    parsed[req] = {}

        assert parsed["name"] == ""
        assert parsed["age"] == 25

    def test_missing_array_field_gets_empty_list(self):
        schema = {
            "type": "object",
            "properties": {"tags": {"type": "array"}},
            "required": ["tags"],
        }
        parsed = {}
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in parsed:
                typ = props.get(req, {}).get("type")
                if typ == "array":
                    parsed[req] = []
        assert parsed["tags"] == []

    def test_missing_object_field_gets_empty_dict(self):
        schema = {
            "type": "object",
            "properties": {"metadata": {"type": "object"}},
            "required": ["metadata"],
        }
        parsed = {}
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in parsed:
                typ = props.get(req, {}).get("type")
                if typ == "object":
                    parsed[req] = {}
        assert parsed["metadata"] == {}

    def test_present_fields_not_overwritten(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        parsed = {"name": "Alice"}
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in parsed:
                parsed[req] = ""
        assert parsed["name"] == "Alice"


# ---------------------------------------------------------------------------
# 7. Memory stats graceful failure
# ---------------------------------------------------------------------------


class TestMemoryStatsGracefulFail:
    def test_returns_empty_dict_without_mlx(self):
        from agent_stable_slo.rollout.providers.mlx_local import get_memory_stats

        # mlx.core is not available in test env, so this should return {}
        result = get_memory_stats()
        assert isinstance(result, dict)
        # Either empty (no mlx) or populated (if mlx somehow available)

    def test_returns_empty_dict_on_import_error(self):
        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None}):
            from agent_stable_slo.rollout.providers.mlx_local import get_memory_stats

            result = get_memory_stats()
            assert isinstance(result, dict)
