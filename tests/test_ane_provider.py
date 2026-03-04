"""Tests for ANE (Apple Neural Engine) local inference provider.

Tests cover provider registration routing, generate_raw signature, JSON
extraction, prompt building, tokenizer/model caching, required-field backfill,
causal mask generation, and chunked vs monolithic dispatch.  All CoreML/coremltools
dependencies are mocked so tests run without Apple Silicon or CoreML installed.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch, mock_open

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
# Helpers – mock coremltools so imports succeed
# ---------------------------------------------------------------------------


def _mock_coremltools():
    """Return a mock coremltools module with ComputeUnit enum."""
    ct = MagicMock()
    ct.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"
    ct.models.CompiledMLModel = MagicMock
    return ct


@pytest.fixture(autouse=True)
def _patch_coremltools():
    """Ensure coremltools is mocked for every test in this module."""
    ct = _mock_coremltools()
    with patch.dict(sys.modules, {"coremltools": ct, "coremltools.models": ct.models}):
        yield


# ---------------------------------------------------------------------------
# 1. Provider registration
# ---------------------------------------------------------------------------


class TestProviderRegistration:
    """Verify engine._provider_generate_raw routes to ane_local correctly."""

    @patch.dict(os.environ, {"AOFW_PROVIDER": "ane_local"})
    @patch("agent_stable_slo.rollout.providers.ane_local.generate_raw")
    def test_routes_ane_local(self, mock_gen):
        mock_gen.return_value = ("raw", {"k": "v"}, 10.0, 10.0, 5, 8)
        from agent_stable_slo.rollout.engine import _provider_generate_raw

        result = _provider_generate_raw("prompt", {}, "structured", 0.0, None)
        mock_gen.assert_called_once()
        # ane_local returns 6-tuple; engine appends None for logprobs
        assert len(result) == 7
        assert result[-1] is None

    @patch.dict(os.environ, {"AOFW_PROVIDER": "ane"})
    @patch("agent_stable_slo.rollout.providers.ane_local.generate_raw")
    def test_routes_ane_alias(self, mock_gen):
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
    """Mock the full CoreML pipeline and verify the 6-tuple return shape."""

    @patch.dict(
        os.environ,
        {
            "ANE_META_DIR": "/tmp/fake_model",
            "ANE_TOKENIZER": "fake-tokenizer",
            "ANE_MAX_TOKENS": "64",
        },
    )
    def test_generate_raw_returns_6_tuple(self, simple_schema):
        """Verify generate_raw returns (raw_text, parsed, lat_ms, ttft_ms, tokens_in, tokens_out)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.encode.return_value = list(range(10))
        mock_tokenizer.decode.return_value = '{"intent": "greeting", "confidence": 0.9}'
        mock_tokenizer.eos_token_id = 2

        mock_model_components = {
            "model": MagicMock(),
            "metadata": {"context_length": 2048, "num_chunks": 0},
        }

        with (
            patch(
                "agent_stable_slo.rollout.providers.ane_local._get_tokenizer",
                return_value=mock_tokenizer,
            ),
            patch(
                "agent_stable_slo.rollout.providers.ane_local._get_ane_model",
                return_value=mock_model_components,
            ),
            patch(
                "agent_stable_slo.rollout.providers.ane_local._ane_generate",
                return_value=(
                    '{"intent": "greeting", "confidence": 0.9}',
                    42.0,
                    8.0,
                    5,
                ),
            ),
        ):
            from agent_stable_slo.rollout.providers.ane_local import generate_raw

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


# ---------------------------------------------------------------------------
# 3. JSON extraction
# ---------------------------------------------------------------------------


class TestJsonExtraction:
    """Test _extract_json with various input formats."""

    def _extract(self, text):
        from agent_stable_slo.rollout.providers.ane_local import _extract_json

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
        result = self._extract("[1, 2, 3]")
        assert result == {}

    def test_multiple_json_objects_returns_first(self):
        text = 'result: {"a": 1} other {"b": 2}'
        result = self._extract(text)
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# 4. Prompt building
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    """Verify _build_prompt_with_schema embeds schema info into the prompt."""

    def _build(self, prompt, schema):
        from agent_stable_slo.rollout.providers.ane_local import (
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
        assert "<value>" in result

    def test_number_property_placeholder(self, simple_schema):
        result = self._build("test", simple_schema)
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
# 5. Required-field backfill
# ---------------------------------------------------------------------------


class TestFieldBackfill:
    """Verify _backfill_required fills missing required fields with type-appropriate defaults."""

    def _backfill(self, parsed, schema):
        from agent_stable_slo.rollout.providers.ane_local import _backfill_required

        return _backfill_required(parsed, schema)

    def test_missing_string_field_gets_empty_string(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        parsed = {"age": 25}
        result = self._backfill(parsed, schema)
        assert result["name"] == ""
        assert result["age"] == 25

    def test_missing_array_field_gets_empty_list(self):
        schema = {
            "type": "object",
            "properties": {"tags": {"type": "array"}},
            "required": ["tags"],
        }
        result = self._backfill({}, schema)
        assert result["tags"] == []

    def test_missing_object_field_gets_empty_dict(self):
        schema = {
            "type": "object",
            "properties": {"metadata": {"type": "object"}},
            "required": ["metadata"],
        }
        result = self._backfill({}, schema)
        assert result["metadata"] == {}

    def test_missing_boolean_field_gets_false(self):
        schema = {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
            "required": ["flag"],
        }
        result = self._backfill({}, schema)
        assert result["flag"] is False

    def test_missing_number_field_gets_zero(self):
        schema = {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        }
        result = self._backfill({}, schema)
        assert result["score"] == 0

    def test_present_fields_not_overwritten(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = self._backfill({"name": "Alice"}, schema)
        assert result["name"] == "Alice"


# ---------------------------------------------------------------------------
# 6. Tokenizer cache
# ---------------------------------------------------------------------------


class TestTokenizerCache:
    """Verify _TOKENIZER_CACHE prevents redundant loads."""

    def test_cache_prevents_reload(self):
        from agent_stable_slo.rollout.providers.ane_local import _TOKENIZER_CACHE

        original = dict(_TOKENIZER_CACHE)
        try:
            mock_tok = MagicMock()
            _TOKENIZER_CACHE["test-tok-xyz"] = mock_tok

            from agent_stable_slo.rollout.providers.ane_local import _get_tokenizer

            tok = _get_tokenizer("test-tok-xyz")
            assert tok is mock_tok
        finally:
            _TOKENIZER_CACHE.clear()
            _TOKENIZER_CACHE.update(original)

    def test_cache_miss_triggers_load(self):
        from agent_stable_slo.rollout.providers.ane_local import _TOKENIZER_CACHE

        original = dict(_TOKENIZER_CACHE)
        try:
            _TOKENIZER_CACHE.pop("brand-new-tok", None)

            mock_tok = MagicMock()
            mock_auto_tok = MagicMock()
            mock_auto_tok.from_pretrained.return_value = mock_tok

            with patch.dict(
                sys.modules,
                {
                    "transformers": MagicMock(AutoTokenizer=mock_auto_tok),
                },
            ):
                from agent_stable_slo.rollout.providers.ane_local import _get_tokenizer

                tok = _get_tokenizer("brand-new-tok")
                assert tok is mock_tok
                assert "brand-new-tok" in _TOKENIZER_CACHE
        finally:
            _TOKENIZER_CACHE.clear()
            _TOKENIZER_CACHE.update(original)


# ---------------------------------------------------------------------------
# 7. Causal mask generation
# ---------------------------------------------------------------------------


class TestCausalMask:
    """Verify _make_causal_mask produces correct shapes and values."""

    def test_mask_shape(self):
        from agent_stable_slo.rollout.providers.ane_local import _make_causal_mask
        import numpy as np

        mask = _make_causal_mask(context_length=4)
        assert mask.shape == (1, 1, 4, 4)
        assert mask.dtype == np.float16

    def test_mask_causal_pattern(self):
        """Upper triangle should be -inf (large negative), lower+diag should be 0."""
        from agent_stable_slo.rollout.providers.ane_local import _make_causal_mask
        import numpy as np

        mask = _make_causal_mask(context_length=3)
        m = mask[0, 0]  # (3, 3)
        # Diagonal and below should be 0
        assert m[0, 0] == 0.0
        assert m[1, 0] == 0.0
        assert m[1, 1] == 0.0
        assert m[2, 0] == 0.0
        assert m[2, 2] == 0.0
        # Above diagonal should be large negative
        assert m[0, 1] < -1000
        assert m[0, 2] < -1000
        assert m[1, 2] < -1000

    def test_mask_slice_for_single_position(self):
        """Slicing mask[pos:pos+1, :] gives correct shape for decode."""
        from agent_stable_slo.rollout.providers.ane_local import _make_causal_mask
        import numpy as np

        mask = _make_causal_mask(context_length=8)
        # Slice for position 3 (single token decode)
        single = mask[:, :, 3:4, :]
        assert single.shape == (1, 1, 1, 8)


# ---------------------------------------------------------------------------
# 8. Model cache
# ---------------------------------------------------------------------------


class TestModelCache:
    """Verify _MODEL_CACHE prevents redundant CoreML model loads."""

    def test_cache_hit(self):
        from agent_stable_slo.rollout.providers.ane_local import _MODEL_CACHE

        original = dict(_MODEL_CACHE)
        try:
            mock_components = {"model": MagicMock(), "metadata": {}}
            _MODEL_CACHE["/tmp/fake_dir"] = mock_components

            from agent_stable_slo.rollout.providers.ane_local import _get_ane_model

            result = _get_ane_model("/tmp/fake_dir")
            assert result is mock_components
        finally:
            _MODEL_CACHE.clear()
            _MODEL_CACHE.update(original)
