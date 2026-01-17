"""
Abstract base class for LLM providers.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/config (e.g., 'openai/gpt-4o-mini')."""
        pass

    @abstractmethod
    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> T:
        """Generate a structured response matching the Pydantic model.

        Args:
            prompt: The user prompt to send.
            response_model: Pydantic model class for structured output.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.

        Returns:
            Instance of response_model with parsed response.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and available.

        Returns:
            True if the provider can be used (API key set, etc.).
        """
        pass

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string.

        Simple heuristic: ~4 characters per token.
        Override in subclasses for more accurate counting.
        """
        return len(text) // 4

    def get_cost_estimate(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Estimate cost in USD for a request.

        Override in subclasses with actual pricing.
        """
        return 0.0


__all__ = ["LLMProvider"]
