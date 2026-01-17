"""
Anthropic Claude LLM provider implementation.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
import os
from typing import Type, TypeVar

from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

from ..base import LLMProvider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is retryable (rate limit or transient)."""
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status in (429, 500, 502, 503, 529):
        return True
    text = str(exc).lower()
    return "rate" in text and "limit" in text or "overloaded" in text


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using Instructor for structured outputs."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        """Initialize Anthropic provider.

        Args:
            model: Model name (e.g., 'claude-3-5-haiku-latest', 'claude-3-5-sonnet-latest').
            api_key: API key (defaults to ANTHROPIC_API_KEY env var).
            timeout: Request timeout in seconds.
        """
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._timeout = timeout
        self._client = None

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def is_available(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self._api_key)

    def _get_client(self):
        """Lazily initialize the Instructor-wrapped client."""
        if self._client is None:
            try:
                import anthropic
                from instructor import from_anthropic

                base = anthropic.Anthropic(
                    api_key=self._api_key,
                    timeout=self._timeout,
                )
                self._client = from_anthropic(base)
            except ImportError as e:
                raise ImportError(
                    "anthropic and instructor packages required for Anthropic provider"
                ) from e
        return self._client

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(exp_base=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> T:
        """Generate structured response using Anthropic with Instructor."""
        client = self._get_client()

        try:
            result = client.messages.create(
                model=self.model,
                response_model=response_model,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result
        except Exception as e:
            if _is_retryable_error(e):
                logger.warning("Retryable error from Anthropic: %s", e)
            raise

    def get_cost_estimate(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Estimate cost based on Anthropic pricing (approximate)."""
        # Pricing as of late 2024 (USD per 1M tokens)
        pricing = {
            "claude-3-5-haiku-latest": (0.80, 4.00),  # input, output
            "claude-3-5-sonnet-latest": (3.00, 15.00),
            "claude-3-opus-latest": (15.00, 75.00),
        }
        input_rate, output_rate = pricing.get(self.model, (1.0, 5.0))
        return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


__all__ = ["AnthropicProvider"]
