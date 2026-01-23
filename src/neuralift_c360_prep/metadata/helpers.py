#!/usr/bin/env python3
"""
metadata/helpers.py
-------------------
Shared utilities for the metadata package.

Contains:
    - ASCII normalization helpers
    - Rate-limit helpers for OpenAI calls
    - Cache helpers for LLM responses
    - Column profiling utilities
    - Instructor client wrapping

Author: Mike Maloney - Neuralift, Inc.
Updated: 2026-01-13
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import threading
import unicodedata
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from instructor import Mode, from_openai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ASCII helpers
# ---------------------------------------------------------------------------
ASCII = r"^[\x20-\x7E]+$"  # printable ASCII
_ASCII_STRIP = re.compile(r"[^\x20-\x7E]")


def _ascii7(s: str) -> str:
    """Convert string to strict 7-bit ASCII."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return _ASCII_STRIP.sub("?", s)


def _ascii_deep(obj: Any) -> Any:
    """Recursively force every str (key or value) to strict 7-bit ASCII."""
    if isinstance(obj, dict):
        return {_ascii7(str(k)): _ascii_deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ascii_deep(v) for v in obj]
    if isinstance(obj, str):
        return _ascii7(obj)
    return obj


# ---------------------------------------------------------------------------
# Rate-limit helpers (for OpenAI calls)
# ---------------------------------------------------------------------------
_rate_limit_event = threading.Event()
_rate_limit_lock = threading.Lock()
_rate_limit_backoff = 0.0


def _is_rate_limit_error(exc: Exception | None) -> bool:
    """Check if an exception is a rate limit error."""
    if exc is None:
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "rate" in text and "limit" in text or "too many requests" in text


def _note_rate_limit() -> float:
    """Record a rate limit hit and return the backoff delay."""
    global _rate_limit_backoff
    with _rate_limit_lock:
        _rate_limit_backoff = (
            5.0 if _rate_limit_backoff == 0 else min(_rate_limit_backoff * 2.0, 60.0)
        )
        delay = _rate_limit_backoff
    _rate_limit_event.set()
    return delay


def _current_rate_limit_delay() -> float:
    """Get the current rate limit delay."""
    if not _rate_limit_event.is_set():
        return 0.0
    with _rate_limit_lock:
        return _rate_limit_backoff


def _relax_rate_limit() -> None:
    """Relax rate limit backoff after successful call."""
    global _rate_limit_backoff
    with _rate_limit_lock:
        if _rate_limit_backoff:
            _rate_limit_backoff = max(0.0, _rate_limit_backoff * 0.5)
            if _rate_limit_backoff < 1.0:
                _rate_limit_backoff = 0.0
                _rate_limit_event.clear()


def _rate_limit_before_sleep(retry_state) -> None:
    """Callback for tenacity before_sleep to handle rate limits."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if _is_rate_limit_error(exc):
        delay = _note_rate_limit()
        logging.getLogger("data-dict").warning(
            "Rate limit encountered; backing off for %.1fs", delay
        )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _hash_key(*parts: str) -> str:
    """Generate a hash key from string parts."""
    h = blake2b(digest_size=16)
    for p in parts:
        h.update(p.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_load(cache_dir: Path, key: str) -> str | None:
    """Load a cached definition."""
    p = cache_dir / f"{key}.json"
    try:
        if p.exists():
            return json.loads(p.read_text()).get("definition")
    except Exception:
        pass
    return None


def _cache_save(cache_dir: Path, key: str, definition: str) -> None:
    """Save a definition to cache."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.json").write_text(json.dumps({"definition": definition}))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Instructor client helpers
# ---------------------------------------------------------------------------
def _wrap_instructor(force_client) -> Any:
    """Wrap an OpenAI client with instructor for structured outputs."""
    if hasattr(force_client, "mode") and hasattr(force_client, "client"):
        force_client = getattr(force_client, "client", force_client)
    if not hasattr(force_client, "chat"):
        raise ValueError("Client must be openai.OpenAI() compatible (has .chat).")
    return from_openai(force_client, mode=Mode.JSON)


def _ensure_instructor_client(client):
    """Ensure a client is instructor-wrapped."""
    return client if hasattr(client, "mode") else from_openai(client, mode=Mode.JSON)


# ---------------------------------------------------------------------------
# Column profiling for data dictionary
# ---------------------------------------------------------------------------
def _profile_single_column(
    col_series: pd.Series,
    name: str,
    *,
    row_count: int,
    null_count: int,
    unique_count: int,
    top_n_values: int,
) -> Dict[str, Any]:
    """Profile ONE column from a pandas Series."""
    raw_dtype = str(col_series.dtype)
    pct_nulls = round((null_count / row_count) * 100, 2) if row_count else 0.0

    non_null = col_series[col_series.notna()]

    mode_val = "N/A"
    if not non_null.empty:
        try:
            m = non_null.mode()
            if not m.empty:
                mode_val = str(m.iat[0])
        except Exception:
            pass

    first_vals = non_null.head(20).tolist()

    describe = {}
    try:
        describe = non_null.describe(include="all").to_dict()
    except Exception:
        pass

    value_counts = {}
    try:
        vc = non_null.value_counts().head(top_n_values).to_dict()
        value_counts = {str(k): int(v) for k, v in vc.items()}
    except Exception:
        pass

    return {
        "name": name,
        "dtype": raw_dtype,
        "nulls": null_count,
        "pct_nulls": pct_nulls,
        "unique_count": unique_count,
        "mode": str(mode_val),
        "first_vals": first_vals,
        "describe": describe,
        "value_counts": value_counts,
        "cache_key_sample": non_null.head(10).tolist(),
    }


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------
def _gc_collect() -> None:
    """Force garbage collection."""
    gc.collect()


__all__ = [
    "ASCII",
    "_ascii7",
    "_ascii_deep",
    "_is_rate_limit_error",
    "_note_rate_limit",
    "_current_rate_limit_delay",
    "_relax_rate_limit",
    "_rate_limit_before_sleep",
    "_hash_key",
    "_cache_load",
    "_cache_save",
    "_wrap_instructor",
    "_ensure_instructor_client",
    "_profile_single_column",
    "_gc_collect",
]
