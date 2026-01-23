"""
Unified LLM client factory.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
from typing import Literal

from .base import LLMProvider

logger = logging.getLogger(__name__)

ProviderType = Literal["openai", "anthropic", "auto"]


def get_llm_provider(
    provider: ProviderType = "auto",
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> LLMProvider:
    """Factory function to get an LLM provider.

    Args:
        provider: Which provider to use. "auto" tries providers in order.
        model: Specific model to use (provider-dependent).
        api_key: Optional API key override.
        timeout: Request timeout in seconds.

    Returns:
        Configured LLMProvider instance.

    Raises:
        RuntimeError: If no provider is available.
        ValueError: If unknown provider is specified.
    """
    if provider == "auto":
        # Try providers in order of preference
        for p in ["openai", "anthropic"]:
            try:
                instance = get_llm_provider(p, model, api_key, timeout)
                if instance.is_available():
                    logger.info("Auto-selected LLM provider: %s", instance.name)
                    return instance
            except Exception as e:
                logger.debug("Provider %s not available: %s", p, e)
                continue
        raise RuntimeError(
            "No LLM provider available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )

    if provider == "openai":
        from .providers.openai_provider import OpenAIProvider

        return OpenAIProvider(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            timeout=timeout,
        )

    if provider == "anthropic":
        from .providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model=model or "claude-3-5-haiku-latest",
            api_key=api_key,
            timeout=timeout,
        )

    raise ValueError(f"Unknown provider: {provider}")


__all__ = ["get_llm_provider", "ProviderType"]
