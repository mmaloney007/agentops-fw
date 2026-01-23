"""
LLM response caching to avoid redundant API calls.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import json
import logging
from hashlib import blake2b
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMResponseCache:
    """File-based cache for LLM responses to avoid redundant API calls."""

    def __init__(self, cache_dir: str | Path = ".nl_id_cache"):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cached responses.
        """
        self.cache_dir = Path(cache_dir)
        self._hits = 0
        self._misses = 0

    def _ensure_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, prompt: str, model: str, response_model_name: str) -> str:
        """Generate cache key from prompt, model, and response type."""
        h = blake2b(digest_size=16)
        h.update(prompt.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update(response_model_name.encode("utf-8"))
        return h.hexdigest()

    def get(
        self,
        prompt: str,
        model: str,
        response_model: Type[T],
    ) -> T | None:
        """Retrieve cached response if available.

        Args:
            prompt: The prompt that was sent.
            model: The model name used.
            response_model: Pydantic model class to parse response.

        Returns:
            Parsed response if cached, None otherwise.
        """
        key = self._hash_key(prompt, model, response_model.__name__)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            result = response_model.model_validate(data)
            self._hits += 1
            logger.debug("Cache hit for %s", key[:8])
            return result
        except Exception as e:
            logger.debug("Cache parse error for %s: %s", key[:8], e)
            self._misses += 1
            return None

    def set(
        self,
        prompt: str,
        model: str,
        response: BaseModel,
    ) -> None:
        """Cache a response.

        Args:
            prompt: The prompt that was sent.
            model: The model name used.
            response: The response to cache.
        """
        self._ensure_dir()
        key = self._hash_key(prompt, model, type(response).__name__)
        cache_file = self.cache_dir / f"{key}.json"

        try:
            cache_file.write_text(
                response.model_dump_json(indent=2),
                encoding="utf-8",
            )
            logger.debug("Cached response for %s", key[:8])
        except Exception as e:
            logger.debug("Cache write error for %s: %s", key[:8], e)

    def clear(self) -> int:
        """Clear all cached responses.

        Returns:
            Number of cache entries cleared.
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        return count

    @property
    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


__all__ = ["LLMResponseCache"]
