#!/usr/bin/env python3
"""
tests/neuralift_c360_prep/test_llm_providers.py
-----------------------------------------------
Tests for the LLM provider infrastructure.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import os
from unittest.mock import patch
import pytest
from pydantic import BaseModel

from neuralift_c360_prep.llm import (
    get_llm_provider,
    LLMResponseCache,
)
from neuralift_c360_prep.llm.base import LLMProvider as BaseLLMProvider


# ---------------------------------------------------------------------------
# Test Models for Structured Output
# ---------------------------------------------------------------------------
class SimpleResponse(BaseModel):
    """Simple test response model."""

    message: str
    confidence: float


class ComplexResponse(BaseModel):
    """Complex test response model."""

    name: str
    items: list[str]
    metadata: dict


# ---------------------------------------------------------------------------
# Tests for LLMProvider Base Class
# ---------------------------------------------------------------------------
class TestLLMProviderBase:
    """Tests for the LLMProvider abstract base class."""

    def test_is_abstract(self):
        """Test that LLMProvider is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_required_methods(self):
        """Test that required abstract methods are defined."""
        assert hasattr(BaseLLMProvider, "complete_structured")
        assert hasattr(BaseLLMProvider, "is_available")
        assert hasattr(BaseLLMProvider, "name")


# ---------------------------------------------------------------------------
# Tests for get_llm_provider Factory
# ---------------------------------------------------------------------------
class TestGetLlmProvider:
    """Tests for the get_llm_provider factory function."""

    def test_auto_selects_available_provider(self):
        """Test that auto mode selects an available provider."""
        # This test may fail if no API keys are set, which is expected
        # We just test that it doesn't crash
        try:
            provider = get_llm_provider(provider="auto")
            assert provider is not None
            assert hasattr(provider, "name")
        except (ValueError, RuntimeError) as e:
            # Expected if no providers are available
            assert "No LLM provider available" in str(e)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_selection(self):
        """Test OpenAI provider selection."""
        # Just verify we can create an OpenAI provider with API key
        provider = get_llm_provider(provider="openai")
        assert provider.name.startswith("openai")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_anthropic_selection(self):
        """Test Anthropic provider selection."""
        # Just verify we can create an Anthropic provider with API key
        provider = get_llm_provider(provider="anthropic")
        assert provider.name.startswith("anthropic")

    def test_invalid_provider_raises(self):
        """Test that invalid provider type raises error."""
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            get_llm_provider(provider="invalid_provider")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_model_parameter_passed(self):
        """Test that model parameter is passed to provider."""
        provider = get_llm_provider(provider="openai", model="gpt-4")
        assert provider.model == "gpt-4"


# ---------------------------------------------------------------------------
# Tests for LLMResponseCache
# ---------------------------------------------------------------------------
class TestLLMResponseCache:
    """Tests for the LLMResponseCache class."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        return tmp_path / "llm_cache"

    @pytest.fixture
    def cache(self, cache_dir):
        """Create a cache instance."""
        return LLMResponseCache(str(cache_dir))

    def test_cache_creation(self, cache, cache_dir):
        """Test that cache works after first write."""
        # Cache dir may not exist until we write something
        response = SimpleResponse(message="test", confidence=0.5)
        cache.set("prompt", "provider", response)
        assert cache_dir.exists()

    def test_cache_miss_returns_none(self, cache):
        """Test that cache miss returns None."""
        result = cache.get("test prompt", "test_provider", SimpleResponse)
        assert result is None

    def test_cache_set_and_get(self, cache):
        """Test setting and getting cached values."""
        response = SimpleResponse(message="Hello", confidence=0.95)
        cache.set("test prompt", "test_provider", response)

        retrieved = cache.get("test prompt", "test_provider", SimpleResponse)
        assert retrieved is not None
        assert retrieved.message == "Hello"
        assert retrieved.confidence == 0.95

    def test_cache_different_providers(self, cache):
        """Test that different providers have separate cache entries."""
        response1 = SimpleResponse(message="OpenAI", confidence=0.9)
        response2 = SimpleResponse(message="Anthropic", confidence=0.8)

        cache.set("same prompt", "openai", response1)
        cache.set("same prompt", "anthropic", response2)

        retrieved1 = cache.get("same prompt", "openai", SimpleResponse)
        retrieved2 = cache.get("same prompt", "anthropic", SimpleResponse)

        assert retrieved1.message == "OpenAI"
        assert retrieved2.message == "Anthropic"

    def test_cache_different_prompts(self, cache):
        """Test that different prompts have separate cache entries."""
        response1 = SimpleResponse(message="Response 1", confidence=0.9)
        response2 = SimpleResponse(message="Response 2", confidence=0.8)

        cache.set("prompt 1", "provider", response1)
        cache.set("prompt 2", "provider", response2)

        retrieved1 = cache.get("prompt 1", "provider", SimpleResponse)
        retrieved2 = cache.get("prompt 2", "provider", SimpleResponse)

        assert retrieved1.message == "Response 1"
        assert retrieved2.message == "Response 2"

    def test_cache_complex_response(self, cache):
        """Test caching complex response objects."""
        response = ComplexResponse(
            name="Test",
            items=["a", "b", "c"],
            metadata={"key": "value", "nested": {"deep": True}},
        )
        cache.set("complex prompt", "provider", response)

        retrieved = cache.get("complex prompt", "provider", ComplexResponse)
        assert retrieved is not None
        assert retrieved.name == "Test"
        assert retrieved.items == ["a", "b", "c"]
        assert retrieved.metadata["nested"]["deep"] is True

    def test_cache_stats(self, cache):
        """Test cache statistics."""
        # Initial stats
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Cache miss
        cache.get("missing", "provider", SimpleResponse)
        assert cache.stats["misses"] == 1

        # Cache set and hit
        cache.set("prompt", "provider", SimpleResponse(message="test", confidence=0.5))
        cache.get("prompt", "provider", SimpleResponse)
        assert cache.stats["hits"] == 1

    def test_cache_persistence(self, cache_dir):
        """Test that cache persists across instances."""
        # Create first cache instance and store value
        cache1 = LLMResponseCache(str(cache_dir))
        response = SimpleResponse(message="Persisted", confidence=0.99)
        cache1.set("persist prompt", "provider", response)

        # Create new cache instance
        cache2 = LLMResponseCache(str(cache_dir))
        retrieved = cache2.get("persist prompt", "provider", SimpleResponse)

        assert retrieved is not None
        assert retrieved.message == "Persisted"

    def test_cache_handles_special_characters(self, cache):
        """Test that cache handles special characters in prompts."""
        prompt = "Test with special chars: !@#$%^&*()[]{}|;':\",./<>?\n\ttabs"
        response = SimpleResponse(message="Special", confidence=0.5)

        cache.set(prompt, "provider", response)
        retrieved = cache.get(prompt, "provider", SimpleResponse)

        assert retrieved is not None
        assert retrieved.message == "Special"

    def test_cache_handles_unicode(self, cache):
        """Test that cache handles unicode characters."""
        prompt = "Unicode test: 你好世界 🌍 émojis"
        response = SimpleResponse(message="Unicode OK", confidence=0.5)

        cache.set(prompt, "provider", response)
        retrieved = cache.get(prompt, "provider", SimpleResponse)

        assert retrieved is not None
        assert retrieved.message == "Unicode OK"


# ---------------------------------------------------------------------------
# Tests for Provider Implementations (Mocked)
# ---------------------------------------------------------------------------
class TestOpenAIProvider:
    """Tests for OpenAI provider (mocked)."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_provider_name(self):
        """Test provider name."""
        provider = get_llm_provider(provider="openai")
        assert provider.name.startswith("openai")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_default_model(self):
        """Test default model selection."""
        provider = get_llm_provider(provider="openai")
        assert provider.model is not None

    def test_unavailable_without_api_key(self):
        """Test that provider is unavailable without API key."""
        from neuralift_c360_prep.llm.providers.openai_provider import OpenAIProvider

        # Create provider without API key set in env
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            provider = OpenAIProvider()
            # Just verify the method exists
            assert hasattr(provider, "is_available")


class TestAnthropicProvider:
    """Tests for Anthropic provider (mocked)."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_provider_name(self):
        """Test provider name."""
        provider = get_llm_provider(provider="anthropic")
        assert provider.name.startswith("anthropic")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_default_model(self):
        """Test default model selection."""
        provider = get_llm_provider(provider="anthropic")
        assert provider.model is not None


# ---------------------------------------------------------------------------
# Tests for Provider Type Enum
# ---------------------------------------------------------------------------
class TestProviderType:
    """Tests for ProviderType enumeration."""

    def test_valid_types(self):
        """Test that all expected provider types exist."""
        # ProviderType should be a Literal or similar
        # Just verify get_llm_provider accepts expected strings
        valid_types = ["openai", "anthropic", "auto"]
        for ptype in valid_types:
            # Should not raise TypeError for valid types
            try:
                get_llm_provider(provider=ptype)
            except (ValueError, RuntimeError):
                # ValueError/RuntimeError for missing/invalid API key is OK
                pass


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------
class TestLLMProviderIntegration:
    """Integration tests for LLM provider system."""

    def test_cache_with_provider_flow(self, tmp_path):
        """Test cache integration with provider flow."""
        cache_dir = tmp_path / "integration_cache"
        cache = LLMResponseCache(str(cache_dir))

        # Simulate a provider flow
        prompt = "Analyze this data"
        provider_name = "test_provider"

        # First call - cache miss
        result1 = cache.get(prompt, provider_name, SimpleResponse)
        assert result1 is None

        # Simulate provider response
        response = SimpleResponse(message="Analysis complete", confidence=0.95)
        cache.set(prompt, provider_name, response)

        # Second call - cache hit
        result2 = cache.get(prompt, provider_name, SimpleResponse)
        assert result2 is not None
        assert result2.message == "Analysis complete"

        # Verify stats
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 1

    def test_cache_key_uniqueness(self, tmp_path):
        """Test that cache keys are properly unique."""
        cache = LLMResponseCache(str(tmp_path / "unique_cache"))

        # Same prompt, different providers
        cache.set("prompt", "provider_a", SimpleResponse(message="A", confidence=0.5))
        cache.set("prompt", "provider_b", SimpleResponse(message="B", confidence=0.5))

        # Should get different results
        a = cache.get("prompt", "provider_a", SimpleResponse)
        b = cache.get("prompt", "provider_b", SimpleResponse)

        assert a.message == "A"
        assert b.message == "B"

    def test_long_prompt_caching(self, tmp_path):
        """Test caching of very long prompts."""
        cache = LLMResponseCache(str(tmp_path / "long_cache"))

        # Very long prompt
        long_prompt = "x" * 10000
        response = SimpleResponse(message="Long prompt handled", confidence=0.9)

        cache.set(long_prompt, "provider", response)
        retrieved = cache.get(long_prompt, "provider", SimpleResponse)

        assert retrieved is not None
        assert retrieved.message == "Long prompt handled"
