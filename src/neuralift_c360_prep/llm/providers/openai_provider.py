"""
OpenAI LLM provider implementation.

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
    if status in (429, 500, 502, 503, 504):
        return True
    text = str(exc).lower()
    return "rate" in text and "limit" in text or "too many requests" in text


class OpenAIProvider(LLMProvider):
    """OpenAI provider using Instructor for structured outputs."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        """Initialize OpenAI provider.

        Args:
            model: Model name (e.g., 'gpt-4o-mini', 'gpt-4o').
            api_key: API key (defaults to OPENAI_API_KEY env var).
            base_url: Optional custom base URL.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url
        self._timeout = timeout
        self._client = None

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self._api_key) and not self._api_key.startswith("sk-test")

    def _get_client(self):
        """Lazily initialize the Instructor-wrapped client."""
        if self._client is None:
            try:
                import openai
                from instructor import Mode, from_openai

                base = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                )
                self._client = from_openai(base, mode=Mode.JSON)
            except ImportError as e:
                raise ImportError(
                    "openai and instructor packages required for OpenAI provider"
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
        """Generate structured response using OpenAI with Instructor."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            result = client.chat.completions.create(
                model=self.model,
                response_model=response_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result
        except Exception as e:
            if _is_retryable_error(e):
                logger.warning("Retryable error from OpenAI: %s", e)
            raise

    def get_cost_estimate(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on OpenAI pricing (approximate)."""
        # Pricing as of late 2024 (USD per 1M tokens)
        pricing = {
            "gpt-4o-mini": (0.15, 0.60),  # input, output
            "gpt-4o": (2.50, 10.00),
            "gpt-4-turbo": (10.00, 30.00),
            "gpt-3.5-turbo": (0.50, 1.50),
        }
        input_rate, output_rate = pricing.get(self.model, (1.0, 3.0))
        return (
            prompt_tokens * input_rate + completion_tokens * output_rate
        ) / 1_000_000


__all__ = ["OpenAIProvider"]
