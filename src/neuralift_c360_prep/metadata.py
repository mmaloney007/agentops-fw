#!/usr/bin/env python3
"""
metadata.py (Dask)
------------------
Dask-first metadata utilities for Neuralift data prep.

Purpose:
    - Profile Dask DataFrames (or CSV/Parquet paths) with sampled pandas frames, optional approximate uniques, and ASCII normalization.
    - Generate LLM-backed column definitions and table comments with caching, retries, and rate-limit backoff; fall back to deterministic text when LLMs are skipped.
    - Build column tags YAML (id/kpi/categorical/continuous plus lift flags), flat JSON data dictionaries, and schema-alignment diagnostics for UC/BI pipelines.

Usage:
    from neuralift_c360_prep.metadata import (
        create_intelligent_data_dictionary,
        create_table_comment,
        build_column_tags_yaml_dask,
        build_data_dictionary_json,
        build_metadata,
    )

Dependencies:
    - dask[dataframe]
    - pandas
    - openai + instructor
    - pydantic
    - PyYAML
    - tenacity

Author: Mike Maloney - Neuralift, Inc.
Updated: 2026-01-01
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import gc
import math
import json
import logging
import os
import re
import statistics
import textwrap
import threading
import time
import tempfile
import unicodedata
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import dask.dataframe as dd
from dask import compute as dask_compute
import openai
import pandas as pd
import yaml
from instructor import Mode, from_openai  # type: ignore
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from pydantic import BaseModel, ConfigDict, Field, constr, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)
data_dict_logger = logging.getLogger("data-dict")
llm_logger = logging.getLogger("neuralift_c360_prep.llm")

# ---------------------------------------------------------------------------
# ASCII helpers
# ---------------------------------------------------------------------------
ASCII = r"^[\x20-\x7E]+$"  # printable ASCII
_ASCII_STRIP = re.compile(r"[^\x20-\x7E]")


def _ascii7(s: str) -> str:
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
    if exc is None:
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "rate" in text and "limit" in text or "too many requests" in text


def _note_rate_limit() -> float:
    global _rate_limit_backoff
    with _rate_limit_lock:
        _rate_limit_backoff = (
            5.0 if _rate_limit_backoff == 0 else min(_rate_limit_backoff * 2.0, 60.0)
        )
        delay = _rate_limit_backoff
    _rate_limit_event.set()
    return delay


def _current_rate_limit_delay() -> float:
    if not _rate_limit_event.is_set():
        return 0.0
    with _rate_limit_lock:
        return _rate_limit_backoff


def _relax_rate_limit() -> None:
    global _rate_limit_backoff
    with _rate_limit_lock:
        if _rate_limit_backoff:
            _rate_limit_backoff = max(0.0, _rate_limit_backoff * 0.5)
            if _rate_limit_backoff < 1.0:
                _rate_limit_backoff = 0.0
                _rate_limit_event.clear()


def _rate_limit_before_sleep(retry_state) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if _is_rate_limit_error(exc):
        delay = _note_rate_limit()
        data_dict_logger.warning("Rate limit encountered; backing off for %.1fs", delay)


# ---------------------------------------------------------------------------
# LLM column description schema + prompt
# ---------------------------------------------------------------------------
class ColumnDescription(BaseModel):
    column_name: constr(pattern=ASCII) = Field(..., alias="Column Name")
    definition: constr(max_length=500, pattern=ASCII) = Field(..., alias="Definition")
    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def __get_validators__(cls):
        yield cls._coerce

    @classmethod
    def _coerce(cls, value):
        if isinstance(value, dict):
            if "Column Name" in value:
                value["column_name"] = value.pop("Column Name")
            if "Definition" in value:
                value["definition"] = value.pop("Definition")
        obj = ColumnDescription.construct(**value)  # type: ignore
        obj.column_name = obj.column_name.lower()
        obj.definition = _ascii7(obj.definition.strip())
        if obj.definition:
            obj.definition = obj.definition[0].upper() + obj.definition[1:]
        return obj


# Prompt template for column definitions
_PROMPT = textwrap.dedent(
    """\
You are documenting a column in a marketing data dictionary. Avoid words like KPI, metric, or dimension.

Write ONE precise, plain-English sentence (<=350 ASCII characters, no line breaks, no double quotes) that defines the column for analysts and BI tools.
If the column's meaning or values are unclear, add a clarifying clause after a semicolon.
Do not mention SQL, data types, or implementation details.

Return exactly this single-line JSON (no markdown, no extra keys):

{{"Column Name":"{name}","Definition":"<your sentence>"}}

Context:
Organization summary : {context}
Column name  : {name}
dtype        : {dtype}
Nulls (#)    : {nulls}
Nulls (%)    : {pct_nulls}
Unique (#)   : {unique_count}
Mode value   : {mode}
Sample (<=20): {first_vals}
describe()   : {describe}
Top counts   : {val_counts}
"""
)


def _prompt_for_column(profile: Dict[str, Any], context: str) -> str:
    return _PROMPT.format(
        **{
            "context": context,
            "name": profile["name"],
            "dtype": profile["dtype"],
            "nulls": profile["nulls"],
            "pct_nulls": profile["pct_nulls"],
            "unique_count": profile["unique_count"],
            "mode": profile["mode"],
            "first_vals": profile["first_vals"],
            "describe": profile["describe"],
            "val_counts": profile["value_counts"],
        }
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _hash_key(*parts: str) -> str:
    h = blake2b(digest_size=16)
    for p in parts:
        h.update(p.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_load(cache_dir: Path, key: str) -> str | None:
    p = cache_dir / f"{key}.json"
    try:
        if p.exists():
            return json.loads(p.read_text()).get("definition")
    except Exception:
        pass
    return None


def _cache_save(cache_dir: Path, key: str, definition: str) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.json").write_text(json.dumps({"definition": definition}))
    except Exception:
        pass


def _wrap_instructor(force_client) -> Any:
    if hasattr(force_client, "mode") and hasattr(force_client, "client"):
        force_client = getattr(force_client, "client", force_client)
    if not hasattr(force_client, "chat"):
        raise ValueError("Client must be openai.OpenAI() compatible (has .chat).")
    return from_openai(force_client, mode=Mode.JSON)


# ---------------------------------------------------------------------------
# Column stats helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnStats:
    columns: list[str]
    row_count: int
    counts: pd.Series
    null_counts: pd.Series
    dtypes: pd.Series
    unique_counts: pd.Series | None
    unique_is_approx: bool

    def subset(self, columns: Iterable[str]) -> "ColumnStats":
        cols = list(columns)
        counts = self.counts.reindex(cols).fillna(0).astype(int)
        null_counts = self.null_counts.reindex(cols).fillna(0).astype(int)
        dtypes = self.dtypes.reindex(cols)
        if self.unique_counts is None:
            unique_counts = None
        else:
            unique_counts = self.unique_counts.reindex(cols).fillna(0).astype(int)
        return ColumnStats(
            columns=cols,
            row_count=self.row_count,
            counts=counts,
            null_counts=null_counts,
            dtypes=dtypes,
            unique_counts=unique_counts,
            unique_is_approx=self.unique_is_approx,
        )


def _empty_int_series(columns: Sequence[str]) -> pd.Series:
    return pd.Series(index=list(columns), dtype="int64")


def _compute_approx_unique_counts(ddf: dd.DataFrame, cols: Sequence[str]) -> pd.Series:
    if not cols:
        return _empty_int_series(cols)
    delayed_vals = [ddf[c].nunique_approx() for c in cols]
    approx_vals = dask_compute(*delayed_vals)
    return pd.Series(approx_vals, index=cols).astype(int)


def _compute_exact_unique_counts(ddf: dd.DataFrame, cols: Sequence[str]) -> pd.Series:
    if not cols:
        return _empty_int_series(cols)
    unique_counts = ddf[cols].nunique(dropna=True).compute()
    if isinstance(unique_counts, pd.DataFrame):
        if unique_counts.shape[0] == 1:
            unique_counts = unique_counts.iloc[0]
        else:
            raise TypeError(
                "Expected unique counts as Series/1-row DF; got "
                f"DF shape={unique_counts.shape}"
            )
    if not isinstance(unique_counts, pd.Series):
        raise TypeError(
            f"Unique counts must be a pandas Series; got {type(unique_counts)}. "
            "Refusing to broadcast a scalar across columns."
        )
    return unique_counts.reindex(cols).fillna(0).astype(int)


def compute_column_stats(
    ddf: dd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    use_approx_unique: bool = False,
    approx_row_threshold: int | None = 2_000_000,
    compute_unique_counts: bool = True,
) -> ColumnStats:
    """
    Compute column statistics with optimized batching.

    OPTIMIZATION: Batches count + approx uniques in a single dask_compute() call
    when use_approx_unique=True, reducing scheduler round-trips from 2 to 1.
    """
    cols = list(columns) if columns is not None else list(ddf.columns)
    dtypes = ddf.dtypes.reindex(cols)

    if not cols:
        empty = _empty_int_series(cols)
        return ColumnStats(
            columns=cols,
            row_count=0,
            counts=empty,
            null_counts=empty,
            dtypes=dtypes,
            unique_counts=empty if compute_unique_counts else None,
            unique_is_approx=False,
        )

    # OPTIMIZATION: Batch count + approx uniques in single compute call when possible
    # This reduces scheduler round-trips and graph traversals
    counts_delayed = ddf[cols].count()

    if compute_unique_counts and use_approx_unique:
        # Batch both operations together - approx uniques are cheap
        approx_unique_delayed = [ddf[c].nunique_approx() for c in cols]
        all_results = dask_compute(counts_delayed, *approx_unique_delayed)
        counts = all_results[0]
        approx_vals = all_results[1:]

        if not isinstance(counts, pd.Series):
            counts = pd.Series(counts, index=cols)
        counts = counts.reindex(cols).fillna(0).astype(int)
        row_count = int(counts.max()) if len(counts) else 0
        null_counts = (row_count - counts).astype(int)

        # Decide if we need exact uniques based on row count
        use_approx = bool(
            approx_row_threshold is None or row_count > approx_row_threshold
        )

        if use_approx:
            unique_counts = pd.Series(approx_vals, index=cols).astype(int)
        else:
            # Need exact uniques - compute them (approx was "free" in the batch)
            logger.info(
                "[stats] row_count=%s < threshold=%s; computing exact uniques...",
                f"{row_count:,}",
                f"{approx_row_threshold:,}",
            )
            unique_counts = _compute_exact_unique_counts(ddf, cols)
    else:
        # Non-batched path for exact uniques or no uniques needed
        counts = counts_delayed.compute()
        if not isinstance(counts, pd.Series):
            counts = pd.Series(counts, index=cols)
        counts = counts.reindex(cols).fillna(0).astype(int)
        row_count = int(counts.max()) if len(counts) else 0
        null_counts = (row_count - counts).astype(int)

        if not compute_unique_counts:
            return ColumnStats(
                columns=cols,
                row_count=row_count,
                counts=counts,
                null_counts=null_counts,
                dtypes=dtypes,
                unique_counts=None,
                unique_is_approx=False,
            )

        # Compute exact uniques
        use_approx = False
        unique_counts = _compute_exact_unique_counts(ddf, cols)

    return ColumnStats(
        columns=cols,
        row_count=row_count,
        counts=counts,
        null_counts=null_counts,
        dtypes=dtypes,
        unique_counts=unique_counts,
        unique_is_approx=use_approx,
    )


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
    """
    Profile ONE column from a pandas Series.
    """
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


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(exp_base=2, max=60),
    reraise=True,
    before_sleep=_rate_limit_before_sleep,
)
def _describe_column(
    client, prompt: str, model: str, col_name: str = ""
) -> ColumnDescription:
    llm_logger.debug("[llm] Calling model=%s for column='%s'", model, col_name)
    # Only show first 500 chars of prompt to avoid log spam
    prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
    llm_logger.debug("[llm] Prompt preview:\n%s", prompt_preview)

    response = client.chat.completions.create(
        model=model,
        response_model=ColumnDescription,
        messages=[
            {"role": "system", "content": "Respond only with valid json."},
            {"role": "user", "content": prompt},
        ],
        top_p=1.0,
    )

    # Log token usage if available
    if hasattr(response, "_raw_response") and hasattr(response._raw_response, "usage"):
        usage = response._raw_response.usage
        llm_logger.debug(
            "[llm] Response for '%s' | tokens: prompt=%d, completion=%d, total=%d",
            col_name,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )
    else:
        llm_logger.debug(
            "[llm] Response for '%s': %s", col_name, response.definition[:100]
        )

    return response


# ---------------------------------------------------------------------------
# Data dictionary main entrypoint
# ---------------------------------------------------------------------------
def create_intelligent_data_dictionary(
    source: str | dd.DataFrame,
    *,
    openai_client=None,
    model: str = "gpt-5-nano",
    context: str | None = None,
    max_cols: int | None = None,
    sample_rows: int = 5_000,
    top_n_values: int = 10,
    max_concurrency: int = 8,
    use_cache: bool = False,
    cache_dir: str | Path = ".nl_dd_cache",
    stats: ColumnStats | None = None,
    debug: bool = False,
    use_approx_unique: bool = True,
    use_sample_for_uniques: bool = False,
) -> Tuple[Dict[str, str], Dict[str, Any], str]:
    """
    MEMORY-OPTIMIZED Dask data dictionary generator.

    Dask-side optimizations:
      - Single stats pass for non-null counts and uniques (approx by default).
      - Sampling via ddf.sample(...), no head().
      - All per-column profiling only uses the in-memory pandas sample.
      - Optionally accepts precomputed stats to avoid recomputation.

    Returns:
      - defs:   {column_name -> definition string}
      - dtypes: {column_name -> *raw* pandas/dask dtype}
      - yaml_text: YAML of Column Name + Definition for table_comment helper
    """
    base_client = openai_client
    client_source = "passed client"
    if base_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set and no client was provided.")
        base_client = openai.OpenAI(api_key=api_key)
        client_source = "env OPENAI_API_KEY"

    client = _wrap_instructor(base_client)
    client_kind = "Instructor-wrapped (forced)"

    logger.info("=== data-dict init ===")
    logger.info("- model: %s", model)
    logger.info("- client_source: %s", client_source)
    logger.info("- client_kind: %s", client_kind)
    logger.info("- use_approx_unique: %s", use_approx_unique)
    logger.info("- use_sample_for_uniques: %s", use_sample_for_uniques)

    if isinstance(source, str):
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        ext = source.lower()
        if ext.endswith(".csv"):
            ddf = dd.read_csv(source, blocksize="128MB")
            context = context or f"CSV file {os.path.basename(source)}"
        elif ext.endswith(".parquet") or ext.endswith(".pq"):
            ddf = dd.read_parquet(source)
            context = context or f"Parquet file {os.path.basename(source)}"
        else:
            raise ValueError("Only CSV/Parquet supported")
    else:
        ddf = source
        context = context or "Dask DataFrame"

    col_names = list(ddf.columns[:max_cols]) if max_cols else list(ddf.columns)

    logger.info("🚀 Starting data dictionary for %s column(s)...", len(col_names))

    if stats is None:
        logger.info("📊 Computing Dask stats (counts, uniques)...")

        counts_delayed = ddf[col_names].count()

        if use_sample_for_uniques:
            stats_to_compute = [counts_delayed]
            has_unique = False
        else:
            unique_delayed = (
                ddf[col_names].nunique_approx()
                if use_approx_unique
                else ddf[col_names].nunique(dropna=True)
            )
            stats_to_compute = [counts_delayed, unique_delayed]
            has_unique = True

        computed_stats = dask_compute(*stats_to_compute)

        if has_unique:
            counts, unique_counts = computed_stats
        else:
            (counts,) = computed_stats
            unique_counts = None

        if not isinstance(counts, pd.Series):
            counts = pd.Series(counts, index=col_names)

        row_count = int(counts.max())
        null_counts = row_count - counts

        if unique_counts is not None and not isinstance(unique_counts, pd.Series):
            unique_counts = pd.Series(unique_counts, index=col_names)
        if unique_counts is not None:
            unique_counts = unique_counts.astype(int)
    else:
        stats = stats.subset(col_names)
        counts = stats.counts
        row_count = stats.row_count
        null_counts = stats.null_counts
        unique_counts = stats.unique_counts
        stats_unique_is_approx = stats.unique_is_approx

        if use_sample_for_uniques:
            unique_counts = None
        elif unique_counts is None:
            if use_approx_unique:
                unique_counts = _compute_approx_unique_counts(ddf, col_names)
                stats_unique_is_approx = True
            else:
                unique_counts = _compute_exact_unique_counts(ddf, col_names)
                stats_unique_is_approx = False
        elif not use_approx_unique and stats_unique_is_approx:
            unique_counts = _compute_exact_unique_counts(ddf, col_names)
            stats_unique_is_approx = False

        if unique_counts is not None and not isinstance(unique_counts, pd.Series):
            unique_counts = pd.Series(unique_counts, index=col_names)
        if unique_counts is not None:
            unique_counts = unique_counts.astype(int)

    logger.info(
        "✅ Stats done: ~%s rows over %s columns",
        f"{row_count:,}",
        len(col_names),
    )
    logger.info("🎯 Building sample via ddf.sample(...) (no head())...")

    sample_frac = min(1.0, sample_rows / max(row_count, 1))
    sample_ddf = ddf[col_names].sample(frac=sample_frac, random_state=42)
    sample_pdf = sample_ddf.compute()

    if use_sample_for_uniques or unique_counts is None:
        unique_counts = sample_pdf.nunique(dropna=True)
        if not isinstance(unique_counts, pd.Series):
            unique_counts = pd.Series(unique_counts, index=col_names)
        unique_counts = unique_counts.astype(int)

    logger.info(
        "📏 Sample ready: %s rows, %s columns (used for profiling)",
        len(sample_pdf),
        len(col_names),
    )
    logger.info("🧮 Profiling columns...")

    # OPTIMIZATION: Profile columns in parallel for wide tables (>50 columns)
    # For narrow tables, sequential is faster due to thread overhead
    def _profile_col(name: str) -> Dict[str, Any]:
        return _profile_single_column(
            sample_pdf[name],  # noqa: F821 - defined in enclosing scope before call
            name,
            row_count=row_count,
            null_count=int(null_counts.get(name, 0)),  # noqa: F821
            unique_count=int(unique_counts.get(name, 0)),  # noqa: F821
            top_n_values=top_n_values,
        )

    if len(col_names) > 50:
        # Parallel profiling for wide tables
        logger.info("[profile] using parallel profiling for %s columns", len(col_names))
        with ThreadPoolExecutor(max_workers=min(8, len(col_names))) as ex:
            profiles = list(ex.map(_profile_col, col_names))
    else:
        # Sequential for narrow tables
        profiles = [_profile_col(name) for name in col_names]

    logger.info("[profile] profiled %s columns", len(profiles))
    llm_logger.info("[llm] 🧮 Generating definitions for %d columns...", len(col_names))

    dtypes_out: Dict[str, Any] = {}
    ddf_dtypes = ddf.dtypes
    for col in col_names:
        if col in ddf_dtypes:
            dtypes_out[col] = ddf_dtypes[col]
        else:
            dtypes_out[col] = sample_pdf[col].dtype

    cache_path = Path(cache_dir) if use_cache else None
    definitions: Dict[str, str] = {}
    total_cols = len(col_names)

    def _work(idx: int, total: int, profile: Dict[str, Any]) -> tuple[int, str, str]:
        name = profile["name"]
        key = _hash_key(
            profile["name"],
            profile["dtype"],
            str(profile["nulls"]),
            str(profile["unique_count"]),
            json.dumps(profile.get("cache_key_sample", []), default=str),
        )

        if cache_path:
            cached = _cache_load(cache_path, key)
            if cached:
                logger.info("[cache] %s/%s %s", idx, total, name)
                return idx, name, cached

        llm_logger.info("[start] %s/%s %s", idx, total, name)
        delay = _current_rate_limit_delay()
        if delay:
            time.sleep(delay)
        prompt = _prompt_for_column(profile, context)
        desc = _describe_column(client, prompt, model=model, col_name=name)
        definition = desc.definition
        _relax_rate_limit()
        if cache_path:
            _cache_save(cache_path, key, definition)
        return idx, name, definition

    completed = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futs = {
            ex.submit(_work, idx, total_cols, profile): profile["name"]
            for idx, profile in enumerate(profiles, 1)
        }
        for fut in as_completed(futs):
            try:
                idx, name, definition = fut.result()
                safe_def = _ascii7(definition)
            except Exception as e:
                data_dict_logger.exception("Definition failed for column %s", futs[fut])
                name = futs[fut]
                safe_def = _ascii7(f"Definition unavailable; {type(e).__name__}")
            definitions[name] = safe_def
            completed += 1
            if completed == 1 or completed % 10 == 0:
                llm_logger.info(
                    "[llm] ✨ Progress: %s/%s columns completed", completed, total_cols
                )

    del sample_pdf, counts, null_counts, unique_counts
    gc.collect()

    llm_logger.info("[llm] ✅ Complete! Generated %s definitions", len(definitions))

    yaml_out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "columns": [
            {"Column Name": n, "Definition": definitions.get(n, "N/A")}
            for n in col_names
        ],
    }
    yaml_text = yaml.safe_dump(
        _ascii_deep(yaml_out), sort_keys=False, allow_unicode=False
    )

    return (
        _ascii_deep({n: definitions.get(n, "N/A") for n in col_names}),
        dtypes_out,
        yaml_text,
    )


# ---------------------------------------------------------------------------
# Table comment generation
# ---------------------------------------------------------------------------
DEFAULT_TABLE_COMMENT_MODEL = "gpt-5-nano"
_default_client = from_openai(openai, mode=Mode.JSON)


def _ensure_instructor_client(client):
    return client if hasattr(client, "mode") else from_openai(client, mode=Mode.JSON)


class CommentModel(BaseModel):
    comment: constr(max_length=750, pattern=ASCII) = Field(..., alias="Comment")
    model_config = ConfigDict(populate_by_name=True)

    @field_validator("comment", mode="after")
    @classmethod
    def _ascii_guard(cls, v: str) -> str:
        v = _ascii7(v).strip()
        return v[0].upper() + v[1:] if v else v


def _summarize_data_dict_for_comment(yaml_dd: str, max_columns: int = 15) -> str:
    """
    Create condensed data dictionary for table comment generation.

    Reduces token usage by 70-90% while preserving key information:
    - Summary statistics (total columns, type counts)
    - Date range if available
    - Most important columns (IDs, dates, amounts, first N columns)
    """
    try:
        full_dd = yaml.safe_load(yaml_dd)
        columns = full_dd.get("columns", [])

        if not columns:
            return yaml_dd  # Return original if parsing fails

        # Count column types by inspecting definitions
        type_counts = {}
        date_columns = []
        id_columns = []
        important_columns = []

        for i, col_entry in enumerate(columns):
            col_name = col_entry.get("Column Name", "")
            col_def = col_entry.get("Definition", "").lower()

            # Identify column types from name/definition
            if any(x in col_name.lower() for x in ["_id", "id_", "_key", "key_"]):
                id_columns.append(col_entry)
            elif any(
                x in col_name.lower()
                for x in ["date", "time", "timestamp", "_at", "_on"]
            ):
                date_columns.append(col_entry)
            elif any(
                x in col_name.lower()
                for x in ["amount", "total", "price", "cost", "revenue"]
            ):
                important_columns.append(col_entry)
            elif i < 5:  # First 5 columns
                important_columns.append(col_entry)

            # Rough type classification from definition
            if any(x in col_def for x in ["date", "time", "timestamp"]):
                type_counts["datetime"] = type_counts.get("datetime", 0) + 1
            elif any(
                x in col_def
                for x in ["number", "numeric", "integer", "float", "count", "amount"]
            ):
                type_counts["numeric"] = type_counts.get("numeric", 0) + 1
            elif any(x in col_def for x in ["boolean", "true/false", "yes/no", "flag"]):
                type_counts["boolean"] = type_counts.get("boolean", 0) + 1
            else:
                type_counts["string"] = type_counts.get("string", 0) + 1

        # Build sampled column list (prioritize important columns)
        sampled = []
        seen = set()

        # Add IDs first
        for col in id_columns:
            if col["Column Name"] not in seen and len(sampled) < max_columns:
                sampled.append(col)
                seen.add(col["Column Name"])

        # Add dates
        for col in date_columns:
            if col["Column Name"] not in seen and len(sampled) < max_columns:
                sampled.append(col)
                seen.add(col["Column Name"])

        # Add other important columns
        for col in important_columns:
            if col["Column Name"] not in seen and len(sampled) < max_columns:
                sampled.append(col)
                seen.add(col["Column Name"])

        # Fill remaining slots with first unseen columns
        for col in columns:
            if col["Column Name"] not in seen and len(sampled) < max_columns:
                sampled.append(col)
                seen.add(col["Column Name"])

        # Build condensed YAML
        summary = {
            "summary": {
                "total_columns": len(columns),
                "by_type": type_counts,
            },
            "key_columns": sampled,
        }

        if len(columns) > max_columns:
            summary["note"] = (
                f"Showing {len(sampled)} of {len(columns)} columns (IDs, dates, and key fields prioritized)"
            )

        return yaml.safe_dump(summary, sort_keys=False, allow_unicode=False)

    except Exception:
        # If summarization fails, return original (fallback)
        return yaml_dd


def build_prompt(*, yaml_dd: str, context: str) -> str:
    # Summarize the data dictionary to reduce token usage
    summarized_dd = _summarize_data_dict_for_comment(yaml_dd, max_columns=15)

    return textwrap.dedent(f"""\
You are a senior data scientist. Write ONE compact paragraph (≤500 ASCII characters, max 5 sentences) that states **what is in this table**:
key columns or data themes, data types, and overall time span (earliest to latest dates present).
Do not mention SQL, analysis use-cases, KPIs, or any jargon or abbreviations.

ORGANIZATION CONTEXT:
{context}

DATA DICTIONARY:
{summarized_dd}

Respond with exactly this single-line JSON (no markdown, no extra keys):
{{"Comment":"<your paragraph>"}}""")


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(exp_base=2, max=16),
    reraise=True,
)
def _call_comment(client, model, prompt) -> str:
    llm_logger.info("[llm] Generating table comment...")
    llm_logger.debug("[llm] Calling model=%s for table comment", model)
    # Only show first 500 chars of prompt to avoid log spam
    prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
    llm_logger.debug("[llm] Prompt preview:\n%s", prompt_preview)

    result = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Respond only with valid json."},
            {"role": "user", "content": prompt},
        ],
        response_model=CommentModel,  # Instructor validation
        top_p=1.0,
    )

    # Log token usage if available
    if hasattr(result, "_raw_response") and hasattr(result._raw_response, "usage"):
        usage = result._raw_response.usage
        llm_logger.debug(
            "[llm] Table comment response | tokens: prompt=%d, completion=%d, total=%d",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )
    else:
        llm_logger.debug("[llm] Table comment response: %s...", result.comment[:100])

    llm_logger.info("[llm] ✅ Table comment generated")
    return result.comment


def create_table_comment(
    *,
    openai_client: openai.OpenAI,
    model_name: str = DEFAULT_TABLE_COMMENT_MODEL,
    file_context: str | None = None,
    yaml_dd: str | None = None,
    weave_project: str | None = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Build an ASCII, SQL-safe table comment. Returns (meta_dict, comment_str).
    """
    if weave_project:
        import weave

        weave.init(weave_project)

    prompt = build_prompt(yaml_dd=yaml_dd or "", context=file_context or "")
    client = _ensure_instructor_client(openai_client)
    comment = _call_comment(client, model_name, prompt)
    meta: Dict[str, Any] = {
        "generated_utc": os.environ.get("TZ", "UTC"),
        "length": len(comment),
    }
    return meta, _ascii_deep(comment)


# ---------------------------------------------------------------------------
# Column tagging + schema alignment
# ---------------------------------------------------------------------------
def build_column_tags_yaml_dask(
    ddf: dd.DataFrame,
    *,
    id_cols: Iterable[str] = (),
    kpi_cols: Iterable[str] = (),
    missing_indicator_cols: Iterable[str] | None = None,
    max_card: int = 20,
    use_approx_unique: bool = False,
    approx_gray_band: int = 5,
    exact_unique_limit: int | None = None,
    extra_tags_all: Mapping[str, str] | None = None,
    extra_tags_by_column: Mapping[str, Mapping[str, str]] | None = None,
    progress_every: int = 10,
    approx_row_threshold: int = 2_000_000,
    debug: bool = False,
    stats: ColumnStats | None = None,
) -> Tuple[Dict[str, Dict[str, str]], str]:
    """
    Classify columns and emit a YAML for tags (no UC/SQL).

    RULES (Type + cardinality only):
      - id/kpi override everything
      - BOOLEAN => categorical
      - STRING/object/category => categorical
      - DATETIME/DATE => categorical
      - Numeric => categorical if uniq <= max_card else continuous
      - fallback => categorical

    If stats is provided, counts/uniques are reused to avoid recomputation.
    """
    cols = list(ddf.columns)
    id_set = {c.lower() for c in id_cols}
    kpi_set = {c.lower() for c in kpi_cols}
    miss_set = {c.lower() for c in (missing_indicator_cols or [])}

    if stats is None:
        logger.info("[tags] computing non-null counts (and inferring row_count)...")
        stats = compute_column_stats(
            ddf,
            columns=cols,
            use_approx_unique=use_approx_unique,
            approx_row_threshold=approx_row_threshold,
        )
    else:
        stats = stats.subset(cols)
        logger.info(
            "[tags] using precomputed stats (rows≈%s, cols=%s)",
            f"{stats.row_count:,}",
            len(cols),
        )

    row_count = stats.row_count
    null_counts = stats.null_counts
    dtypes = stats.dtypes
    unique_counts = stats.unique_counts
    stats_unique_is_approx = stats.unique_is_approx

    logger.info(
        "[tags] stats: rows≈%s, cols=%s, use_approx_unique=%s, approx_row_threshold=%s",
        f"{row_count:,}",
        len(cols),
        use_approx_unique,
        approx_row_threshold,
    )

    use_approx = bool(use_approx_unique and row_count > approx_row_threshold)

    if unique_counts is None or stats_unique_is_approx != use_approx:
        if use_approx:
            logger.info(
                "[tags] using approximate uniques (per-column nunique_approx) for large table"
            )
            unique_counts = _compute_approx_unique_counts(ddf, cols)
            stats_unique_is_approx = True
        else:
            logger.info("[tags] computing exact uniques (nunique) for all columns...")
            unique_counts = _compute_exact_unique_counts(ddf, cols)
            stats_unique_is_approx = False
    else:
        if use_approx:
            logger.info("[tags] using precomputed approximate uniques")
        else:
            logger.info("[tags] using precomputed exact uniques")

    unique_counts = unique_counts.reindex(cols).fillna(0).astype(int)

    # Optional exact uniques for gray-band columns when using approx
    if use_approx and approx_gray_band >= 0:
        gray_cols: list[str] = []
        for c in cols:
            est = int(unique_counts.get(c, 0))
            if abs(est - max_card) <= approx_gray_band:
                gray_cols.append(c)
        if exact_unique_limit is not None:
            gray_cols = gray_cols[:exact_unique_limit]

        if gray_cols:
            logger.info("[tags] exact uniques for gray-band cols: %s", gray_cols)
            exact_uniques = ddf[gray_cols].nunique(dropna=True).compute()
            if isinstance(exact_uniques, pd.DataFrame) and exact_uniques.shape[0] == 1:
                exact_uniques = exact_uniques.iloc[0]
            if not isinstance(exact_uniques, pd.Series):
                raise TypeError(
                    f"Exact uniques must be a Series; got {type(exact_uniques)}"
                )
            exact_uniques = exact_uniques.reindex(gray_cols).fillna(0).astype(int)
            for c in gray_cols:
                unique_counts[c] = int(exact_uniques[c])

    tags_by_col: Dict[str, Dict[str, str]] = {}
    total_cols = len(cols)

    for i, col in enumerate(cols, start=1):
        if i == 1 or i % progress_every == 0 or i == total_cols:
            logger.info("[tags] profiling %s/%s", i, total_cols)

        col_l = col.lower()
        uniq = int(unique_counts.get(col, 0))
        dtype_str = str(dtypes[col]).lower()

        dt = pd.api.types.pandas_dtype(dtypes[col])

        # ---- classification: TYPE + CARDINALITY ONLY ----
        if col_l in id_set:
            ctype = "id"
        elif col_l in kpi_set:
            ctype = "kpi"
        else:
            if is_bool_dtype(dt):
                ctype = "categorical"
            elif (
                is_string_dtype(dt) or "object" in dtype_str or "category" in dtype_str
            ):
                ctype = "categorical"
            elif is_datetime64_any_dtype(dt):
                ctype = "categorical"
            elif is_numeric_dtype(dt):
                ctype = "categorical" if uniq <= max_card else "continuous"
            else:
                ctype = "categorical"

        tags: Dict[str, Any] = {
            "name": col,
            "type": ctype,
            "dtype": dtype_str,
            "null_count": int(null_counts.get(col, 0)),
            "unique_count": uniq,
        }

        if col_l in miss_set:
            tags["missing_indicator"] = "true"

        if extra_tags_all:
            tags.update(extra_tags_all)
        if extra_tags_by_column and col in extra_tags_by_column:
            tags.update(extra_tags_by_column[col])

        tags_by_col[col] = tags

        if debug:
            is_bool = bool(is_bool_dtype(dt))
            is_str = bool(
                is_string_dtype(dt) or "object" in dtype_str or "category" in dtype_str
            )
            is_cat = "category" in dtype_str
            logger.debug(
                "[tag] %s dtype=%s uniq=%s max_card=%s is_bool=%s is_str=%s is_cat=%s -> type=%s",
                col,
                dtype_str,
                uniq,
                max_card,
                is_bool,
                is_str,
                is_cat,
                ctype,
            )

    columns_yaml = [{"name": c, **tags_by_col[c]} for c in cols]
    yaml_obj = {"columns": columns_yaml}
    yaml_text = yaml.safe_dump(
        _ascii_deep(yaml_obj), sort_keys=False, allow_unicode=False
    )
    return tags_by_col, yaml_text


def _map_dtype_to_data_type(dtype: Any) -> str:
    """
    Map a pandas/dask dtype to one of:
    STRING, INTEGER, FLOAT, BOOLEAN, DOUBLE.

    - All bools -> BOOLEAN
    - All ints  -> INTEGER
    - All floats -> DOUBLE
    - Everything else -> STRING
    """
    try:
        dt = pd.api.types.pandas_dtype(dtype)
    except TypeError:
        return "STRING"

    if is_bool_dtype(dt):
        return "BOOLEAN"
    if is_integer_dtype(dt):
        return "INTEGER"
    if is_float_dtype(dt):
        return "DOUBLE"
    if is_string_dtype(dt) or str(dt) in {"object", "string"}:
        return "STRING"
    return "STRING"


def build_data_dictionary_json(
    *,
    table_name: str,
    table_comment: Optional[str],
    column_definitions: Mapping[str, str],
    column_tags: Mapping[str, Mapping[str, str]],
    column_dtypes: Mapping[str, Any] | None = None,
    column_order: Iterable[str] | None = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Build a flat JSON data dictionary.

    FIXED RULES ENFORCED HERE (safety net):
      - If data_type is STRING or BOOLEAN, column_type is forced to categorical
        unless the column is explicitly id/kpi.
    """
    if column_order is not None:
        col_names = list(column_order)
    else:
        col_names = sorted(
            set(column_definitions.keys())
            | set(column_tags.keys())
            | (set(column_dtypes.keys()) if column_dtypes is not None else set())
        )

    columns_json: list[Dict[str, Any]] = []

    for col in col_names:
        definition = column_definitions.get(col, "N/A")
        tags = column_tags.get(col, {}) or {}

        col_type = tags.get("type", "categorical")

        if column_dtypes is not None and col in column_dtypes:
            data_type = _map_dtype_to_data_type(column_dtypes[col])
        else:
            data_type = "STRING"

        # ---- HARD INVARIANTS (your rules) ----
        # STRING always categorical; BOOLEAN always categorical; unless explicitly id/kpi
        if data_type in {"STRING", "BOOLEAN"} and col_type not in {"id", "kpi"}:
            col_type = "categorical"

        # Lift fields (flattened)
        value_sum_col = tags.get("value_sum_column") or tags.get(
            "lift_value_sum_column"
        )
        value_sum_unit = tags.get("value_sum_unit") or tags.get("lift_value_sum_unit")
        event_sum_col = tags.get("event_sum_column") or tags.get(
            "lift_event_sum_column"
        )
        event_sum_unit = tags.get("event_sum_unit") or tags.get("lift_event_sum_unit")

        lift_enabled = any(
            [value_sum_col, value_sum_unit, event_sum_col, event_sum_unit]
        )

        col_entry: Dict[str, Any] = {
            "comment": definition,
            "data_type": data_type,
            "column_name": col,
            "column_type": col_type,
            "lift_enabled": bool(lift_enabled),
        }

        if value_sum_col:
            col_entry["lift_value_sum_column"] = value_sum_col
        if value_sum_unit:
            col_entry["lift_value_sum_unit"] = value_sum_unit
        if event_sum_col:
            col_entry["lift_event_sum_column"] = event_sum_col
        if event_sum_unit:
            col_entry["lift_event_sum_unit"] = event_sum_unit

        # Include unique_count to avoid expensive fallback in config_builder
        unique_count = tags.get("unique_count")
        if unique_count is not None:
            col_entry["unique_count"] = unique_count

        columns_json.append(col_entry)

    meta_json: Dict[str, Any] = {
        "comment": table_comment or "",
        "table_name": table_name,
        "columns": columns_json,
    }

    json_text = json.dumps(meta_json, ensure_ascii=False, indent=4)
    return meta_json, json_text


def inspect_schema_alignment(
    ddf: dd.DataFrame,
    dtypes_map: dict[str, Any],
    tags_by_col: dict[str, dict[str, str]],
    meta_json: dict[str, Any],
    tags_yaml: str,
    max_card: int,
    *,
    only_suspect: bool = True,
) -> None:
    """
    Print a per-column summary comparing dtypes/tags/json.
    """
    tags_obj = yaml.safe_load(tags_yaml)
    tags_columns = {c["name"]: c for c in tags_obj.get("columns", [])}
    json_cols = {c["column_name"]: c for c in meta_json.get("columns", [])}

    all_cols = list(ddf.columns)
    logger.info("Inspecting %s column(s)...", len(all_cols))

    for col in all_cols:
        dask_dtype = str(ddf[col].dtype)
        dtypes_entry = dtypes_map.get(col)
        tags = tags_by_col.get(col, {}) or {}
        col_type = tags.get("type", "categorical")

        json_entry = json_cols.get(col)
        if json_entry is None:
            json_data_type = None
            json_col_type = None
        else:
            json_data_type = json_entry.get("data_type")
            json_col_type = json_entry.get("column_type")

        col_meta = tags_columns.get(col, {})
        unique_count = col_meta.get("unique_count")
        null_count = col_meta.get("null_count")
        try:
            unique_count_int = int(unique_count) if unique_count is not None else None
        except Exception:
            unique_count_int = None

        is_dask_numeric = dask_dtype.startswith(("int", "float", "UInt", "Int"))
        json_says_string = json_data_type == "STRING"
        cat_vs_cont_mismatch = False

        if unique_count_int is not None:
            if unique_count_int <= max_card and col_type == "continuous":
                cat_vs_cont_mismatch = True
            if unique_count_int > max_card and col_type == "categorical":
                cat_vs_cont_mismatch = True

        suspect = False
        if is_dask_numeric and json_says_string:
            suspect = True
        if cat_vs_cont_mismatch:
            suspect = True

        if only_suspect and not suspect:
            continue

        logger.info("--- %s ---", col)
        logger.info("  Dask dtype        : %s", dask_dtype)
        logger.info("  dtypes_map entry  : %s", dtypes_entry)
        logger.info("  tags['type']      : %s", col_type)
        logger.info("  JSON.data_type    : %s", json_data_type)
        logger.info("  JSON.column_type  : %s", json_col_type)
        logger.info("  null_count        : %s", null_count)
        logger.info("  unique_count      : %s", unique_count)
        logger.info("  max_card          : %s", max_card)
        logger.info("  is_dask_numeric   : %s", is_dask_numeric)
        logger.info("  json_says_string  : %s", json_says_string)
        logger.info("  cat/cont mismatch : %s", cat_vs_cont_mismatch)


# ---------------------------------------------------------------------------
# Config builder helpers (schema-aligned with neuralift_segmenter/config.py)
# ---------------------------------------------------------------------------
# See build_pretty_config_from_data_dict() docstring for heuristic details and
# paper links in plain text.

_DEFAULT_OUTPUT_PATH = str(Path(tempfile.gettempdir()) / str(uuid4()))

SEGMENTER_CONFIG_DEFAULTS: Dict[str, Any] = {
    "use_gpu": True,
    "is_container": False,
    "use_wandb": False,
    "input_uri": None,
    "input_path": None,
    "config_file_name": "config.yaml",
    "output_path": _DEFAULT_OUTPUT_PATH,
    "delete_existing_artifacts": False,
    "labels_file_name": None,
    "ranked_points_file_name": None,
    "verbose": 0,
    "headless": False,
    "resume": False,
    "json_logging": None,
    "data": {
        "sample_frac": None,
    },
    "dae": {
        "dataset_stats_file_name": "dataset_stats.joblib",
        "estimate_batch_size": True,
        "rmm_allocator": True,
        "compile": False,
        "scale_batch_size": False,
        "weight_averaging": True,
        "matmul_precision": "high",
        "data_module": {
            "val_split": 0.2,
            "batch_size": 256,
            "compute_stats_from": "full",
            "num_sample_partitions": 5,
            "optimize_memory": False,
        },
        "model": {
            "learning_rate": 4e-3,
            "backbone_type": "mlp",
            "encoder_hidden_dims": [256, 128],
            "decoder_hidden_dims": [128, 256],
            "latent_dim": 64,
            "feature_embed_dim": 32,
            "scheduler": "onecycle",
            "optimizer": "adam",
            "gradient_checkpointing": False,
            "use_sparse_categorical": False,
            "use_grouped_categorical_head": False,
            "boolean_cardinality_threshold": 2,
            "use_mixed_categorical": True,
            "max_onehot_cardinality": 4,
            "batch_norm_continuous": True,
            "batch_norm_embeddings": True,
            "embedding_dropout": 0.1,
            "robust_scaler": False,
            "num_swap_prob": 0.35,
            "cat_swap_prob": 0.35,
        },
        "trainer": {
            "max_epochs": 50,
            "accelerator": "auto",
            "precision": "bf16-mixed",
            "fast_dev_run": False,
            "gradient_clip_val": 1.0,
            "devices": -1,
            "enable_model_summary": True,
            "sync_batchnorm": False,
        },
        "distributed": {
            "enabled": False,
            "backend": "nccl",
        },
    },
    "segmenter": {
        "cluster_selection_method": "eom",
        "min_cluster_pct": None,
        "min_cluster_size": None,
        "min_cluster_threshold": None,
        "min_samples": 10,
        "min_samples_pct": None,
        "min_dist": 0.0,
        "soft_clustering_batch_size": None,
        "noise_threshold": None,
        "n_neighbors": 15,
        "n_components": None,
        "nnd_n_clusters": 4,
        "nnd_overlap_factor": 2,
        "knn_n_clusters": 4,
        "knn_overlap_factor": 2,
        "metric": "euclidean",
        "prediction_data": True,
    },
    "xgboost": {
        "scale_pos_weight": True,
        "max_bin": 256,
        "rmm_pool_frac": 0.8,
        "jit_unspill": True,
        "enable_cudf_spill": False,
        "protocol": None,
    },
    "explainability": {
        "top_n": 5,
        "num_features": 25,
    },
    "wandb": {
        "entity": "neuralift-ai",
        "project": None,
        "group": None,
        "mode": "online",
    },
}

_MISSING_DEFAULT = object()


def _flatten_defaults(values: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_defaults(value, path))
        else:
            flat[path] = value
    return flat


def _format_default_value(value: Any) -> str:
    text = yaml.safe_dump(
        value,
        sort_keys=False,
        default_flow_style=True,
        allow_unicode=False,
    ).strip()
    if text.endswith("..."):
        text = text[:-3].strip()
    return text.replace("\n", " ")


def _get_value_by_path(config: Mapping[str, Any], path: str) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING_DEFAULT
        current = current[part]
    return current


def _strip_nulls_and_defaults(value: Any, defaults: Any = _MISSING_DEFAULT) -> Any:
    if isinstance(value, Mapping):
        cleaned: Dict[str, Any] = {}
        defaults_map = defaults if isinstance(defaults, Mapping) else {}
        for key, val in value.items():
            default_val = defaults_map.get(key, _MISSING_DEFAULT)
            cleaned_val = _strip_nulls_and_defaults(val, default_val)
            if cleaned_val is None:
                continue
            if default_val is not _MISSING_DEFAULT and cleaned_val == default_val:
                continue
            cleaned[key] = cleaned_val
        return cleaned or None
    if isinstance(value, list):
        cleaned_list = [_strip_nulls_and_defaults(val) for val in value]
        cleaned_list = [val for val in cleaned_list if val is not None]
        if isinstance(defaults, list) and cleaned_list == defaults:
            return None
        return cleaned_list
    if value is None:
        return None
    if defaults is not _MISSING_DEFAULT and value == defaults:
        return None
    return value


def _filter_rationale(
    rationale: Mapping[str, str], config: Mapping[str, Any]
) -> Dict[str, str]:
    filtered: Dict[str, str] = {}
    for path, msg in rationale.items():
        if _get_value_by_path(config, path) is not _MISSING_DEFAULT:
            filtered[path] = msg
    return filtered


def _annotate_yaml_with_defaults(
    yaml_text: str,
    *,
    config: Mapping[str, Any],
    defaults: Mapping[str, Any] | None,
) -> str:
    if not defaults:
        return yaml_text

    flat_defaults = _flatten_defaults(defaults)
    lines = yaml_text.splitlines()
    output: List[str] = []
    stack: List[Tuple[int, str]] = []

    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            output.append(line)
            continue

        match = re.match(r"^(\s*)([^:#]+):(?:\s*.*)?$", line)
        if not match:
            output.append(line)
            continue

        indent = len(match.group(1))
        key = match.group(2).strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        path = ".".join([item[1] for item in stack] + [key])
        value = _get_value_by_path(config, path)
        if isinstance(value, Mapping):
            output.append(line)
            stack.append((indent, key))
            continue

        default_value = flat_defaults.get(path, _MISSING_DEFAULT)
        if default_value is not _MISSING_DEFAULT and value != default_value:
            line = f"{line} # default: {_format_default_value(default_value)}"

        output.append(line)
    return "\n".join(output)


def _read_env_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# Treat bool as categorical; treat datetime as categorical for tabular segmentation pipelines.
def _dtype_is_numeric(dtype: Any) -> bool:
    try:
        dt = pd.api.types.pandas_dtype(dtype)
        if is_bool_dtype(dt):
            return False
        return bool(is_numeric_dtype(dt))
    except Exception:
        s = str(dtype).lower()
        return s.startswith(("int", "uint", "float", "decimal"))


def _safe_row_count(ddf: dd.DataFrame) -> int:
    try:
        return int(ddf.shape[0].compute())
    except Exception:
        try:
            return int(ddf.map_partitions(len).compute().sum())
        except Exception:
            return 0


def _approx_nunique_dask(
    ddf: dd.DataFrame,
    col: str,
    *,
    n_rows: int,
    max_sample: int = 100_000,
) -> int:
    """
    Approximate nunique using HyperLogLog (nunique_approx) first, with sampling fallback.

    OPTIMIZATION: Always try nunique_approx first (O(1) memory, fast) before falling
    back to sampling or exact computation. Avoids expensive .astype(str) conversion.
    """
    if col not in ddf.columns:
        return 0

    try:
        series = ddf[col]

        # OPTIMIZATION: Try nunique_approx first - it's fast and memory-efficient
        if hasattr(series, "nunique_approx"):
            try:
                return int(series.nunique_approx().compute())
            except Exception:
                pass

        # Fallback to sampled exact nunique
        if n_rows > max_sample:
            frac = max_sample / float(n_rows)
            series = series.sample(frac=frac, random_state=42)

        try:
            return int(series.nunique().compute())
        except Exception:
            # Last resort - but avoid .astype(str) which is very expensive
            return 0
    except Exception:
        return 0


def infer_column_roles_from_data_dict(
    data_dict: Dict[str, Any],
    ddf: dd.DataFrame,
    *,
    max_card_for_cat: int = 20,
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, int]]:
    """Infer column roles and collect categorical cardinalities.

    Inputs:
      data_dict["columns"] is expected to be compatible with either:
        - tags YAML produced by build_column_tags_yaml_dask() (fields: name, type, dtype, unique_count)
        - data dictionary JSON produced by build_data_dictionary_json() (fields: column_name, column_type, data_type, ...)

    Returns:
      (id_cols, kpi_cols, cat_cols, cont_cols, cat_cardinalities_by_col)
    """
    cols_meta = data_dict.get("columns") or []

    n_rows = _safe_row_count(ddf)

    id_cols: List[str] = []
    kpi_cols: List[str] = []
    cat_cols: List[str] = []
    cont_cols: List[str] = []
    cat_cards: Dict[str, int] = {}

    for col_meta in cols_meta:
        name = (
            col_meta.get("name")
            or col_meta.get("column_name")
            or col_meta.get("Column Name")
        )
        if not name:
            continue

        # tags.yaml uses "type"; json data dict uses "column_type"
        ctype = (col_meta.get("type") or col_meta.get("column_type") or "").lower()

        if ctype == "id":
            id_cols.append(name)
            continue
        if ctype == "kpi":
            kpi_cols.append(name)
            continue

        # Prefer explicit types
        if ctype == "categorical":
            cat_cols.append(name)
        elif ctype == "continuous":
            cont_cols.append(name)
        else:
            # Infer from dtype + cardinality
            if name in ddf.columns:
                dtype_val = ddf[name].dtype
            else:
                dtype_val = (
                    col_meta.get("dtype") or col_meta.get("data_type") or "STRING"
                )

            if not _dtype_is_numeric(dtype_val):
                cat_cols.append(name)
            else:
                # Prefer unique_count from tags if present
                uniq = col_meta.get("unique_count") or col_meta.get("nunique")
                try:
                    uniq_int = int(uniq) if uniq is not None else None
                except Exception:
                    uniq_int = None

                card = (
                    uniq_int
                    if uniq_int is not None
                    else _approx_nunique_dask(ddf, name, n_rows=n_rows)
                )
                if int(card) <= int(max_card_for_cat):
                    cat_cols.append(name)
                else:
                    cont_cols.append(name)

        # Track categorical cardinality for embedding heuristics
        if name in cat_cols:
            uniq = col_meta.get("unique_count") or col_meta.get("nunique")
            try:
                cat_cards[name] = int(uniq)
            except Exception:
                # If missing, compute approximate
                cat_cards[name] = _approx_nunique_dask(ddf, name, n_rows=n_rows)

    def dedupe_keep_order(xs: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return (
        dedupe_keep_order(id_cols),
        dedupe_keep_order(kpi_cols),
        dedupe_keep_order(cat_cols),
        dedupe_keep_order(cont_cols),
        cat_cards,
    )


def _next_pow2(x: int) -> int:
    x = int(max(1, x))
    p = 1
    while p < x:
        p *= 2
    return p


def suggest_autoencoder_dims(
    n_features: int,
    *,
    max_width: int = 1024,
    max_latent: int = 128,
    min_latent: int = 16,
    max_layers: int = 4,
) -> Tuple[List[int], List[int], int]:
    """Suggest encoder/decoder hidden dims + latent_dim for a DAE.

    Heuristic:
      latent_dim ≈ next_pow2(4 * sqrt(n_features)), clipped to [min_latent, max_latent]
      top_width  ≈ next_pow2(min(max_width, max(128, 4 * latent_dim)))
      encoder dims halve down toward latent_dim (latent_dim is *not* included in hidden dims).
      decoder is symmetric to the encoder hidden dims.
    """
    n_features = max(1, int(n_features))

    latent_target = int(round(4.0 * math.sqrt(n_features)))
    latent_dim = _next_pow2(latent_target)
    latent_dim = max(min_latent, min(max_latent, latent_dim))

    top_width = min(max_width, max(128, 4 * latent_dim))
    top_width = _next_pow2(top_width)

    encoder: List[int] = []
    w = top_width
    while w > latent_dim and len(encoder) < max_layers:
        encoder.append(int(w))
        next_w = int(w // 2)
        if next_w <= latent_dim:
            break
        w = next_w

    if not encoder:
        encoder = [int(top_width)]

    decoder = list(reversed(encoder))
    return encoder, decoder, int(latent_dim)


def suggest_feature_embed_dim(
    cat_cardinalities: Sequence[int],
    *,
    default_if_unknown: int = 8,
    max_embed_dim: int = 32,
) -> int:
    """Global embedding dim for categorical features.

    Based on entity-embedding practice: small dims work well for most categories; very
    high-cardinality features may benefit from larger embedding dims. (Guo & Berkhahn, 2016)
    """
    cards = [int(c) for c in cat_cardinalities if c is not None and int(c) > 0]
    if not cards:
        return int(default_if_unknown)

    med = statistics.median(cards)

    if med <= 10:
        return 4
    if med <= 50:
        return 8
    if med <= 200:
        return 16
    return int(max_embed_dim)


def suggest_corruption_probs(
    *,
    n_features: int,
    cat_cardinalities: Sequence[int],
) -> Tuple[float, float]:
    """Swap corruption probabilities for numeric and categorical inputs.

    Denoising AEs rely on deliberately corrupting the input and learning to reconstruct
    the clean signal (Vincent et al., 2008). We choose moderate noise by default.
    """
    n_features = max(1, int(n_features))

    num_swap_prob = 0.35 if n_features >= 50 else 0.30

    cards = [int(c) for c in cat_cardinalities if c is not None and int(c) > 0]
    med_card = statistics.median(cards) if cards else 0

    # For very high-cardinality categorical features, too much corruption can erase
    # rare-but-meaningful signals; back off slightly.
    if med_card >= 1000:
        cat_swap_prob = 0.20
    elif med_card >= 100:
        cat_swap_prob = 0.30
    else:
        cat_swap_prob = 0.35

    return float(num_swap_prob), float(cat_swap_prob)


def suggest_backbone_type(
    *,
    n_rows: int,
    n_features: int,
    n_categorical: int,
) -> str:
    """Pick a backbone for the DAE.

    Heuristic: default to MLP; choose FT-Transformer only for large, wide, categorical-heavy
    tables where attention-based mixing tends to help.
    """
    n_rows = max(1, int(n_rows))
    n_features = max(1, int(n_features))
    n_categorical = max(0, int(n_categorical))
    cat_ratio = n_categorical / float(n_features)

    if n_rows >= 1_000_000 and n_features >= 200 and cat_ratio >= 0.25:
        return "ft_transformer"
    return "mlp"


def suggest_batch_size(
    *,
    use_gpu: bool,
    n_features: int,
    n_rows: int,
    device_mem_gb: float | None = None,
    target_steps_per_epoch: Tuple[int, int] = (10, 20),
) -> int:
    """Batch-size heuristic based on memory and 10-20 steps per epoch.

    Strategy:
      - Target 10-20 steps per epoch (batch_size ~= n_rows / steps_mid).
      - Cap by a memory-derived ceiling that scales with device RAM and table width.
    """
    n_rows = max(1, int(n_rows))
    n_features = max(1, int(n_features))

    min_steps, max_steps = sorted([int(x) for x in target_steps_per_epoch])
    min_steps = max(1, min_steps)
    max_steps = max(min_steps, max_steps)
    steps_mid = int(round((min_steps + max_steps) / 2.0))

    bs_target = max(1, int(round(n_rows / float(steps_mid))))
    bs_min = max(1, int(math.floor(n_rows / float(max_steps))))
    bs_max = max(1, int(math.ceil(n_rows / float(min_steps))))
    bs_by_steps = max(bs_min, min(bs_target, bs_max))

    if not use_gpu:
        base_cap = 256
    else:
        base_cap = 2048
        if device_mem_gb:
            base_cap = int(round(base_cap * (device_mem_gb / 16.0)))

    if n_features > 200:
        base_cap = int(base_cap * 0.5)
    if n_features > 500:
        base_cap = int(base_cap * 0.5)

    base_cap = max(1, int(base_cap))
    return int(min(bs_by_steps, base_cap))


def suggest_scheduler(
    *,
    use_gpu: bool,
    steps_per_epoch: int,
) -> str:
    """Choose a LR scheduler based on step count and hardware."""
    steps_per_epoch = max(1, int(steps_per_epoch))
    if not use_gpu:
        return "cosine"
    if steps_per_epoch < 10:
        return "cosine"
    return "onecycle"


def suggest_learning_rate(
    *,
    batch_size: int,
    use_gpu: bool,
    scheduler: str,
) -> float:
    """Learning-rate heuristic aligned with the chosen scheduler."""
    if not use_gpu:
        return 1e-3
    bs = max(1, int(batch_size))
    scheduler = (scheduler or "").lower()
    base = 4e-3 if scheduler == "onecycle" else 2e-3
    lr = base * math.sqrt(bs / 2048.0)
    return float(max(5e-4, min(8e-3, lr)))


def suggest_compute_stats_plan(n_rows: int) -> Tuple[str, int]:
    """How to compute dataset stats (scaling, etc.) without biasing on ordered partitions."""
    n_rows = max(1, int(n_rows))
    if n_rows <= 1_000_000:
        return "full", 5
    if n_rows <= 20_000_000:
        return "sample", 10
    return "sample", 10


def suggest_target_segments(n_rows: int, target_range: Tuple[int, int]) -> int:
    """Pick a midpoint target segment count in [low, high] based on dataset size."""
    low, high = target_range
    n_rows = max(1, int(n_rows))
    if n_rows < 200_000:
        return max(low, min(high, 8))
    if n_rows < 5_000_000:
        return max(low, min(high, 12))
    return max(low, min(high, 16))


def suggest_segmenter_hparams(
    n_rows: int,
    *,
    target_segments_range: Tuple[int, int] = (5, 20),
) -> Dict[str, Any]:
    """Heuristics for UMAP + HDBSCAN segmentation targeting ~5–20 segments.

    Key ideas:
      - UMAP parameters:
          * n_neighbors trades local vs global structure (UMAP docs/paper).
          * min_dist near 0 encourages clumpier embeddings, often better for clustering.
      - HDBSCAN parameters:
          * min_cluster_size is the primary "smallest cluster you care about" knob.
          * prediction_data=True enables fast approximate_predict for out-of-sample labeling.
    """
    n_rows = max(1, int(n_rows))
    target_mid = suggest_target_segments(n_rows, target_segments_range)

    # To aim for ~target_mid segments, enforce a minimum "kept" cluster threshold near n/target_mid.
    min_cluster_threshold = max(30, int(n_rows / float(target_mid)))

    # Set the smallest cluster of interest to ~1/(target_mid*6) of the dataset.
    # For target_mid=16 => n/96 (~1.0%) which matches the example config_ominous.yaml.
    min_cluster_size = max(30, int(n_rows / float(target_mid * 6)))

    min_samples = max(10, int(min_cluster_size / 3))

    if n_rows < 100_000:
        n_neighbors = 30
        n_components = 8
    elif n_rows < 1_000_000:
        n_neighbors = 40
        n_components = 10
    else:
        n_neighbors = 45
        n_components = 15

    # For very large N, fit HDBSCAN on a sample and then assign remaining points
    # using approximate_predict (requires prediction_data=True).
    sample_target = 2_000_000
    if n_rows <= sample_target:
        sample_hdbscan = 1.0
    else:
        sample_hdbscan = min(1.0, sample_target / float(n_rows))
        # Ensure we don't go absurdly small; 0.2% is a minimum guardrail.
        sample_hdbscan = max(sample_hdbscan, 0.002)

    # Chunked soft clustering (membership vectors) to avoid GPU/host OOM.
    if n_rows >= 10_000_000:
        soft_batch = 100_000
    elif n_rows >= 1_000_000:
        soft_batch = 50_000
    else:
        soft_batch = None

    return {
        "cluster_selection_method": "eom",
        "min_cluster_size": int(min_cluster_size),
        "min_cluster_pct": None,
        "min_cluster_threshold": int(min_cluster_threshold),
        "min_samples": int(min_samples),
        "min_samples_pct": None,
        "n_neighbors": int(n_neighbors),
        "n_components": int(n_components),
        "min_dist": 0.0,
        "metric": "euclidean",
        "sample_hdbscan": float(sample_hdbscan),
        "prediction_data": True,
        "soft_clustering_batch_size": soft_batch,
        "noise_threshold": 0.05 if n_rows >= 1_000_000 else None,
        # Leave these at schema defaults unless you have a specific overlap strategy.
        "nnd_n_clusters": 4,
        "nnd_overlap_factor": 2,
        "knn_n_clusters": 4,
        "knn_overlap_factor": 2,
    }


def suggest_explainability_hparams(n_features: int) -> Dict[str, int]:
    n_features = max(1, int(n_features))
    num_features = int(min(100, max(25, round(0.5 * n_features))))
    top_n = int(min(num_features, max(10, round(0.2 * num_features))))
    return {"num_features": num_features, "top_n": top_n}


def build_pretty_config_from_data_dict(
    data_dict: Dict[str, Any],
    ddf: dd.DataFrame,
    *,
    use_gpu: bool = True,
    use_wandb: bool = False,
    config_debug: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_mode: str | None = None,
    delete_existing_artifacts: bool = False,
    robust_scaler: bool = False,
    max_card_for_cat: int = 20,
    max_epochs: int = 50,
    backbone_type: str | None = None,
    scheduler: str | None = None,
    optimizer: str = "adam",
    device_mem_gb: float | None = None,
    target_steps_per_epoch: Tuple[int, int] = (10, 20),
    target_segments_range: Tuple[int, int] = (5, 20),
    return_rationale: bool = False,
) -> Any:
    """Build a schema-aligned config for Neuralift segmentation.

    Heuristic summary:
      - DAE latent_dim = next_pow2(4*sqrt(n_features)), clipped to [16, 128].
        Hidden dims halve toward latent_dim, and latent_dim is NOT included in
        encoder/decoder hidden layers.
      - Embedding dim from median categorical cardinality (Guo & Berkhahn 2016).
      - Corruption probabilities for denoising AEs (Vincent et al. 2008).
      - Batch size targets 10-20 steps per epoch and is capped by device memory
        plus feature width (set device_mem_gb or NL_DEVICE_MEM_GB for accuracy).
      - Scheduler: OneCycle for GPU runs with >=10 steps/epoch; cosine otherwise.
        LR scales ~sqrt(batch_size) with a scheduler-specific base (Smith 2017).
      - Backbone: default MLP; switch to FT-Transformer only for large, wide,
        categorical-heavy tables.
      - Segmenter: UMAP + HDBSCAN heuristics targeting ~5-20 segments.
      - Explainability: XGBoost + SHAP, with conservative defaults.

    References (plain text):
      DAE: https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf
      UMAP: https://arxiv.org/abs/1802.03426
      HDBSCAN: https://arxiv.org/abs/1705.07321
      OneCycle: https://arxiv.org/abs/1708.07120
      AdamW: https://arxiv.org/abs/1711.05101
      Gradient checkpointing: https://arxiv.org/abs/1604.06174
      Mixed precision: https://arxiv.org/abs/1710.03740
      BF16: https://arxiv.org/abs/1905.12322
      Entity embeddings: https://arxiv.org/abs/1604.06737
      XGBoost: https://arxiv.org/abs/1603.02754
      SHAP: https://arxiv.org/abs/1705.07874

    If return_rationale=True, returns (config, rationale_dict) where rationale_dict maps
    'yaml.path.key' -> explanation string.
    """
    n_rows = _safe_row_count(ddf)

    id_cols, kpi_cols, cat_cols, cont_cols, cat_cards_by_col = (
        infer_column_roles_from_data_dict(
            data_dict, ddf, max_card_for_cat=max_card_for_cat
        )
    )
    n_features = int(len(cat_cols) + len(cont_cols))

    encoder_hidden_dims, decoder_hidden_dims, latent_dim = suggest_autoencoder_dims(
        n_features
    )

    feature_embed_dim = suggest_feature_embed_dim(list(cat_cards_by_col.values()))
    num_swap_prob, cat_swap_prob = suggest_corruption_probs(
        n_features=n_features, cat_cardinalities=list(cat_cards_by_col.values())
    )

    if device_mem_gb is None:
        device_mem_gb = _read_env_float("NL_DEVICE_MEM_GB") or _read_env_float(
            "NL_GPU_MEM_GB"
        )

    batch_size = suggest_batch_size(
        use_gpu=use_gpu,
        n_features=n_features,
        n_rows=n_rows,
        device_mem_gb=device_mem_gb,
        target_steps_per_epoch=target_steps_per_epoch,
    )
    steps_per_epoch = max(1, int(math.ceil(n_rows / float(batch_size))))
    if scheduler is None:
        scheduler = suggest_scheduler(use_gpu=use_gpu, steps_per_epoch=steps_per_epoch)

    learning_rate = suggest_learning_rate(
        batch_size=batch_size, use_gpu=use_gpu, scheduler=scheduler
    )

    compute_stats_from, num_sample_partitions = suggest_compute_stats_plan(n_rows)

    if backbone_type is None:
        backbone_type = suggest_backbone_type(
            n_rows=n_rows, n_features=n_features, n_categorical=len(cat_cols)
        )

    primary_width = encoder_hidden_dims[0] if encoder_hidden_dims else 0
    gradient_checkpointing = bool(
        use_gpu and (primary_width >= 512 or n_features > 300)
    )

    use_wandb_flag = bool(use_wandb)
    labels_file_name: str | None = "labels.npy"
    ranked_points_file_name: str | None = "precomputed_points.npy"
    if use_wandb_flag:
        labels_file_name = None
        ranked_points_file_name = None

    seg = suggest_segmenter_hparams(n_rows, target_segments_range=target_segments_range)

    explain = suggest_explainability_hparams(n_features)
    default_output_path = SEGMENTER_CONFIG_DEFAULTS["output_path"]

    # -----------------------------
    # Build config dict (schema per config.py)
    # -----------------------------
    config: Dict[str, Any] = {
        "use_gpu": bool(use_gpu),
        "is_container": False,
        "use_wandb": use_wandb_flag,
        "wandb": {
            "project": wandb_project,
        },
        "input_uri": None,
        "input_path": None,
        "config_file_name": "config.yaml",
        "output_path": default_output_path,
        # Optional output artifact file names (null when W&B is enabled).
        "labels_file_name": labels_file_name,
        "ranked_points_file_name": ranked_points_file_name,
        "verbose": 0,
        "headless": False,
        "resume": False,
        "json_logging": None,
        "data": {
            "sample_frac": None,
        },
        "dae": {
            "matmul_precision": "high" if use_gpu else "medium",
            "dataset_stats_file_name": "dataset_stats.joblib",
            "estimate_batch_size": True,
            "rmm_allocator": bool(use_gpu),
            "compile": False,
            "scale_batch_size": False,
            "weight_averaging": True,
            "data_module": {
                "val_split": 0.2,
                "batch_size": int(batch_size),
                "compute_stats_from": str(compute_stats_from),
                "num_sample_partitions": int(num_sample_partitions),
                "optimize_memory": False,
            },
            "model": {
                "learning_rate": float(learning_rate),
                "backbone_type": str(backbone_type),
                "encoder_hidden_dims": [int(x) for x in encoder_hidden_dims],
                "decoder_hidden_dims": [int(x) for x in decoder_hidden_dims],
                "latent_dim": int(latent_dim),
                "feature_embed_dim": int(feature_embed_dim),
                "num_swap_prob": float(num_swap_prob),
                "cat_swap_prob": float(cat_swap_prob),
                "scheduler": str(scheduler),
                "optimizer": str(optimizer),
                "gradient_checkpointing": bool(gradient_checkpointing),
                "use_sparse_categorical": False,
                "use_grouped_categorical_head": False,
                "boolean_cardinality_threshold": 2,
                "use_mixed_categorical": True,
                "max_onehot_cardinality": 4,
                "batch_norm_continuous": True,
                "batch_norm_embeddings": True,
                "embedding_dropout": 0.1,
                "robust_scaler": bool(robust_scaler),
            },
            "trainer": {
                "max_epochs": int(max_epochs),
                "accelerator": "auto",
                "precision": "bf16-mixed" if use_gpu else "32",
                "fast_dev_run": False,
                "gradient_clip_val": 1.0,
                "devices": -1 if use_gpu else 1,
                "enable_model_summary": True,
                "sync_batchnorm": False,
            },
            "distributed": {
                "enabled": False,
                "backend": "nccl",
            },
        },
        "segmenter": {
            "cluster_selection_method": seg["cluster_selection_method"],
            "min_cluster_pct": seg["min_cluster_pct"],
            "min_cluster_size": seg["min_cluster_size"],
            "min_cluster_threshold": seg["min_cluster_threshold"],
            "min_samples": seg["min_samples"],
            "min_samples_pct": seg["min_samples_pct"],
            "min_dist": seg["min_dist"],
            "soft_clustering_batch_size": seg["soft_clustering_batch_size"],
            "noise_threshold": seg["noise_threshold"],
            "n_neighbors": seg["n_neighbors"],
            "n_components": seg["n_components"],
            "nnd_n_clusters": seg["nnd_n_clusters"],
            "nnd_overlap_factor": seg["nnd_overlap_factor"],
            "knn_n_clusters": seg["knn_n_clusters"],
            "knn_overlap_factor": seg["knn_overlap_factor"],
            "metric": seg["metric"],
            "prediction_data": seg["prediction_data"],
        },
        "explainability": {
            "num_features": int(explain["num_features"]),
            "top_n": int(explain["top_n"]),
        },
    }

    # -----------------------------
    # Rationale (YAML comments)
    # -----------------------------
    rationale: Dict[str, str] = {}

    # Root
    rationale["use_gpu"] = (
        "Enable GPU acceleration (A10-class GPUs+ support fast bf16 mixed precision)."
    )
    rationale["use_wandb"] = (
        "Track runs/metrics/artifacts; disable for air-gapped/offline runs."
    )
    rationale["labels_file_name"] = (
        "Standard output filename for final HDBSCAN labels; omitted when W&B is enabled."
    )
    rationale["ranked_points_file_name"] = (
        "Standard output filename for precomputed or ranked embedding points; omitted when W&B is enabled."
    )

    # Data
    rationale["data.sample_frac"] = (
        "Optional subsample ratio; leave unset to use the full dataset."
    )

    # DAE
    rationale["dae.matmul_precision"] = (
        "Use high matmul precision on GPU for stable/faster tensor cores; medium on CPU."
    )
    rationale["dae.dataset_stats_file_name"] = (
        "Persist dataset scaling/statistics for reproducible transforms."
    )
    rationale["dae.estimate_batch_size"] = (
        "Let the trainer shrink batch size automatically if out-of-memory."
    )
    rationale["dae.rmm_allocator"] = (
        "Use RAPIDS RMM allocator on GPU runs to reduce fragmentation when cuDF/RAPIDS is in play."
    )
    rationale["dae.compile"] = (
        "torch.compile can speed training, but is kept off by default for maximum stability."
    )
    rationale["dae.scale_batch_size"] = (
        "Disable implicit batch scaling; use explicit batch size with memory + steps/epoch heuristics."
    )
    rationale["dae.weight_averaging"] = (
        "Weight averaging can smooth training; disable if you need strict checkpoint comparability."
    )

    rationale["dae.data_module.val_split"] = (
        "20% validation split is a robust default for unsupervised early sanity checks."
    )
    rationale["dae.data_module.batch_size"] = (
        "Batch size targets 10-20 steps/epoch and is capped by device memory and table width."
    )
    rationale["dae.data_module.compute_stats_from"] = (
        "Use full stats for small data; sample stats for large data to avoid expensive full scans."
    )
    rationale["dae.data_module.num_sample_partitions"] = (
        "Number of partitions sampled to estimate dataset stats on large tables."
    )
    rationale["dae.data_module.optimize_memory"] = (
        "Reduce memory pressure during data module setup when enabled."
    )

    rationale["dae.model.learning_rate"] = (
        "LR scales ~sqrt(batch_size) with scheduler-specific base (Smith 2017)."
    )
    rationale["dae.model.scheduler"] = (
        "OneCycle for GPU runs with enough steps/epoch; cosine otherwise."
    )
    rationale["dae.model.optimizer"] = (
        "Adam is a stable baseline; switch to AdamW/SGD if you need different regularization."
    )
    rationale["dae.model.backbone_type"] = (
        "MLP default; switch to FT-Transformer for large, wide, categorical-heavy tables."
    )
    rationale["dae.model.encoder_hidden_dims"] = (
        "Geometric compression toward latent space; latent_dim is not part of hidden dims."
    )
    rationale["dae.model.decoder_hidden_dims"] = (
        "Symmetric decoder mirroring encoder hidden dims."
    )
    rationale["dae.model.latent_dim"] = (
        "Latent dimension scales with sqrt(feature_count) to balance compression and capacity."
    )
    rationale["dae.model.feature_embed_dim"] = (
        "Embedding dimension chosen from median categorical cardinality (Guo & Berkhahn 2016)."
    )
    rationale["dae.model.num_swap_prob"] = (
        "Numeric corruption probability for denoising objective (Vincent et al. 2008)."
    )
    rationale["dae.model.cat_swap_prob"] = (
        "Categorical corruption probability; reduced when categorical cardinalities are very high."
    )
    rationale["dae.model.gradient_checkpointing"] = (
        "Checkpointing trades compute for memory; useful for big batches/width (Chen et al. 2016)."
    )
    rationale["dae.model.use_sparse_categorical"] = (
        "Disable sparse categorical path by default; enable only if your model supports it."
    )
    rationale["dae.model.use_grouped_categorical_head"] = (
        "Group categorical reconstruction heads to reduce parameters and overfitting risk."
    )
    rationale["dae.model.boolean_cardinality_threshold"] = (
        "Treat boolean-like features as categoricals with cardinality <= 2."
    )
    rationale["dae.model.robust_scaler"] = (
        "Robust scaling is resistant to heavy-tailed numeric features common in marketing/adtech fact tables."
    )

    rationale["dae.trainer.max_epochs"] = (
        "50 epochs is a stable default; reduce for fast iteration, increase for harder domains."
    )
    rationale["dae.trainer.accelerator"] = (
        "Auto accelerator chooses GPU when available."
    )
    rationale["dae.trainer.precision"] = (
        "bf16-mixed is a strong GPU default; BF16 has FP32-like exponent range (Kalamkar et al. 2019)."
    )
    rationale["dae.trainer.gradient_clip_val"] = (
        "Clip gradients to stabilize training under noise and large LR schedules."
    )
    rationale["dae.trainer.devices"] = (
        "Use all visible GPUs for throughput; set to 1 to debug deterministically."
    )
    rationale["dae.trainer.enable_model_summary"] = (
        "Keep model summary for sanity checking layer sizes."
    )

    rationale["dae.distributed.enabled"] = (
        "Disabled by default; enable when running multi-node DDP explicitly."
    )
    rationale["dae.distributed.backend"] = (
        "NCCL is standard for multi-GPU NVIDIA distributed training."
    )

    # Segmenter
    rationale["segmenter.cluster_selection_method"] = (
        "EOM tends to produce a stable set of clusters; 'leaf' often yields more clusters."
    )
    rationale["segmenter.min_cluster_size"] = (
        "Primary HDBSCAN knob: smallest grouping considered a cluster (HDBSCAN docs)."
    )
    rationale["segmenter.min_cluster_pct"] = (
        "Pct version of min_cluster_size for readability across table sizes."
    )
    rationale["segmenter.min_cluster_threshold"] = (
        "Pipeline-level threshold to steer toward ~5–20 segments (≈ n_rows/target_mid)."
    )
    rationale["segmenter.min_samples"] = (
        "Density conservativeness; set lower than min_cluster_size to reduce outliers."
    )
    rationale["segmenter.min_samples_pct"] = (
        "Pct version of min_samples for readability."
    )
    rationale["segmenter.n_neighbors"] = (
        "UMAP neighborhood size; larger values emphasize global structure (UMAP docs)."
    )
    rationale["segmenter.min_dist"] = (
        "UMAP min_dist=0 makes clumpier embeddings, often better for clustering (UMAP docs)."
    )
    rationale["segmenter.n_components"] = (
        "Cluster in 5–10D UMAP space to preserve structure while avoiding high-dim clustering."
    )
    rationale["segmenter.metric"] = (
        "Euclidean is appropriate for DAE latent spaces after scaling."
    )
    rationale["segmenter.prediction_data"] = (
        "Required for fast approximate_predict labeling of new points (HDBSCAN docs)."
    )
    rationale["segmenter.soft_clustering_batch_size"] = (
        "Chunk membership computations to avoid OOM on very large N."
    )
    rationale["segmenter.noise_threshold"] = (
        "Low threshold keeps most points assigned while reserving a noise tail."
    )
    rationale["segmenter.nnd_n_clusters"] = (
        "Keep schema default unless you have an overlap strategy."
    )
    rationale["segmenter.nnd_overlap_factor"] = (
        "Keep schema default unless you have an overlap strategy."
    )
    rationale["segmenter.knn_n_clusters"] = (
        "Keep schema default unless you have an overlap strategy."
    )
    rationale["segmenter.knn_overlap_factor"] = (
        "Keep schema default unless you have an overlap strategy."
    )

    rationale["explainability.num_features"] = (
        "Number of top features to compute SHAP/global importances over (tradeoff: time vs detail)."
    )
    rationale["explainability.top_n"] = "How many top features to surface in reports."

    rationale["wandb.project"] = "W&B project name."

    if not config_debug:
        config = _strip_nulls_and_defaults(config, SEGMENTER_CONFIG_DEFAULTS)
        rationale = _filter_rationale(rationale, config)

    if return_rationale:
        return config, rationale
    return config


def render_config_yaml_with_comments(
    config: Dict[str, Any],
    *,
    header_comment_lines: Sequence[str] | None = None,
    rationale: Mapping[str, str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> str:
    """Render YAML with a top-of-file comment block describing heuristics.

    This avoids needing a YAML library that preserves inline comments.
    When defaults is provided, inline "# default: ..." annotations are added.
    """
    header_comment_lines = list(header_comment_lines or [])
    rationale = dict(rationale or {})

    lines: List[str] = []
    for ln in header_comment_lines:
        ln = str(ln).rstrip("\n")
        if not ln.startswith("#"):
            ln = "# " + ln
        lines.append(ln)

    if rationale:
        lines.append("#")
        lines.append(
            "# -----------------------------------------------------------------------------"
        )
        lines.append("# HEURISTIC NOTES (why these defaults)")
        lines.append(
            "# -----------------------------------------------------------------------------"
        )
        for k in sorted(rationale.keys()):
            msg = str(rationale[k]).replace("\n", " ").strip()
            lines.append(f"# {k}: {msg}")
        lines.append(
            "# -----------------------------------------------------------------------------"
        )
        lines.append("#")

    yaml_text = yaml.safe_dump(
        config,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
    yaml_text = _annotate_yaml_with_defaults(
        yaml_text,
        config=config,
        defaults=defaults,
    )
    lines.append(yaml_text.rstrip("\n"))
    lines.append("")  # final newline
    return "\n".join(lines)


def print_config_yaml(config: Dict[str, Any]) -> None:
    text = yaml.safe_dump(
        config,
        sort_keys=False,
        default_flow_style=False,  # block style
        allow_unicode=False,
    )
    logger.info(text)


def save_config_yaml(
    config: Dict[str, Any],
    path: Union[str, Path],
    *,
    header_comment_lines: Sequence[str] | None = None,
    rationale: Mapping[str, str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> None:
    """Save config YAML.

    Backward-compatible: if header_comment_lines/rationale are omitted, behaves like
    the original save_config_yaml (plain YAML without comments).
    """
    path = Path(path)
    if header_comment_lines or rationale or defaults:
        text = render_config_yaml_with_comments(
            config,
            header_comment_lines=header_comment_lines,
            rationale=rationale,
            defaults=defaults,
        )
    else:
        text = yaml.safe_dump(
            config, sort_keys=False, default_flow_style=False, allow_unicode=False
        )
    path.write_text(text)


# ---------------------------------------------------------------------------
# Minimal metadata builder used by pipeline/tests
# ---------------------------------------------------------------------------
def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                return trimmed
        else:
            return str(value)
    return ""


def resolve_output_table_name(cfg, override: str | None = None) -> str:
    """
    Resolve a stable logical table name for metadata/tags.
    """
    name = _first_nonempty(
        override,
        getattr(cfg.output, "uc_table", None),
        getattr(cfg.output, "uc_volume_name", None),
    )
    return name or "unknown_table"


def build_metadata(
    ddf, cfg, *, table_name_override: str | None = None
) -> Tuple[dict, str]:
    """
    Full metadata builder:
      1) Build tags YAML (id/kpi/cat/cont, uniques, extras)
      2) Build LLM column definitions (data dictionary)
      3) Build table comment
      4) Build JSON data dictionary (comment/data_type/column_name/column_type/lift_*)
    """
    table_name = resolve_output_table_name(cfg, table_name_override)
    skip_llm = os.getenv("NL_SKIP_LLM") == "1" or os.getenv(
        "OPENAI_API_KEY", ""
    ).startswith("sk-test")

    # ---- stats (shared by tags + data dictionary) ----
    tags_cfg = cfg.metadata.tags
    skip_unique_counts = getattr(tags_cfg, "skip_unique_counts", False)

    if skip_unique_counts:
        logger.info(
            "[stats] skip_unique_counts=True; computing counts only (fast mode)"
        )

    stats = compute_column_stats(
        ddf,
        use_approx_unique=tags_cfg.use_approx_unique,
        approx_row_threshold=getattr(tags_cfg, "approx_row_threshold", 2_000_000),
        compute_unique_counts=not skip_unique_counts,
    )

    # ---- tags ----
    extra_all = tags_cfg.extra_tags_all or {}
    extra_by_col = tags_cfg.extra_tags_by_column or {}
    tags_by_col, tags_yaml = build_column_tags_yaml_dask(
        ddf,
        id_cols=tags_cfg.id_cols,
        kpi_cols=tags_cfg.kpi_cols,
        missing_indicator_cols=tags_cfg.missing_indicator_cols,
        max_card=tags_cfg.max_card,
        use_approx_unique=tags_cfg.use_approx_unique,
        approx_gray_band=getattr(tags_cfg, "approx_gray_band", 5),
        approx_row_threshold=getattr(tags_cfg, "approx_row_threshold", 2_000_000),
        extra_tags_all=extra_all,
        extra_tags_by_column=extra_by_col,
        debug=cfg.logging.level == "debug",
        stats=stats,
    )

    # ---- definitions (LLM) with fallback ----
    try:
        if skip_llm:
            raise RuntimeError("LLM skipped by config")
        definitions, dtypes_out, dd_yaml = create_intelligent_data_dictionary(
            ddf,
            model=cfg.metadata.model,
            context=cfg.metadata.context,
            sample_rows=cfg.metadata.sample_rows,
            max_concurrency=cfg.metadata.max_concurrency,
            stats=stats,
            # LLM cache settings from config
            use_cache=getattr(cfg.metadata, "use_llm_cache", False),
            cache_dir=getattr(cfg.metadata, "llm_cache_dir", ".nl_dd_cache"),
            # Limit columns for LLM if configured
            max_cols=getattr(cfg.metadata, "max_columns_for_comment", None),
        )
    except Exception as exc:  # pragma: no cover - network/LLM failures
        logger.warning(
            "[meta] LLM data dictionary skipped/failed (%s); using fallback definitions",
            exc,
        )
        definitions = {c: f"{c} column" for c in ddf.columns}
        dtypes_out = {c: ddf.dtypes[c] for c in ddf.columns}
        dd_yaml = yaml.safe_dump(
            {
                "columns": [
                    {"Column Name": c, "Definition": definitions[c]}
                    for c in ddf.columns
                ]
            },
            sort_keys=False,
            allow_unicode=False,
        )

    # ---- table comment (LLM) with fallback ----
    skip_table_comment = getattr(cfg.metadata, "skip_table_comment", False)
    try:
        if skip_llm or skip_table_comment:
            raise RuntimeError("Table comment skipped by config")
        comment_meta, table_comment = create_table_comment(
            openai_client=openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
            model_name=cfg.metadata.model,
            file_context=cfg.metadata.context,
            yaml_dd=dd_yaml,
        )
    except Exception as exc:  # pragma: no cover - network/LLM failures
        logger.warning(
            "[meta] table comment generation skipped/failed (%s); using fallback", exc
        )
        table_comment = (
            cfg.metadata.context
            or getattr(cfg.output, "run_name", None)
            or f"Table {table_name}"
        )

    # ---- JSON data dictionary ----
    meta_json, meta_json_text = build_data_dictionary_json(
        table_name=table_name,
        table_comment=table_comment,
        column_definitions=definitions,
        column_tags=tags_by_col,
        column_dtypes=dtypes_out,
        column_order=ddf.columns,
    )
    if cfg.logging.level == "debug":
        json_cols = {c["column_name"]: c for c in meta_json.get("columns", [])}
        for col in ddf.columns:
            json_entry = json_cols.get(col, {})
            uniq = tags_by_col.get(col, {}).get("unique_count")
            logger.debug(
                "[final] %s data_type=%s unique_count=%s column_type=%s",
                col,
                json_entry.get("data_type"),
                uniq,
                json_entry.get("column_type"),
            )

    meta_with_table = dict(meta_json)
    meta_with_table["table"] = table_name
    columns_map: Dict[str, dict] = {}
    for col in meta_json.get("columns", []):
        name = col.get("column_name")
        cdict = dict(col)
        tags_entry = tags_by_col.get(name, {}) if name else {}
        tag_list: list[str] = []
        ctype = tags_entry.get("type")
        if ctype:
            tag_list.append(ctype)
        # carry through any other tags we have
        for k, v in tags_entry.items():
            if k != "type":
                cdict[k] = v
        cdict["tags"] = tag_list
        if name:
            columns_map[name] = cdict
    meta_with_table["columns"] = columns_map

    return meta_with_table, meta_json_text


def build_table_comment(cfg):
    table_name = resolve_output_table_name(cfg)
    return (
        cfg.metadata.context
        or getattr(cfg.output, "run_name", None)
        or f"Table {table_name}"
    )


__all__ = [
    "SEGMENTER_CONFIG_DEFAULTS",
    "ColumnStats",
    "compute_column_stats",
    "create_intelligent_data_dictionary",
    "create_table_comment",
    "build_column_tags_yaml_dask",
    "build_data_dictionary_json",
    "inspect_schema_alignment",
    "build_pretty_config_from_data_dict",
    "infer_column_roles_from_data_dict",
    "print_config_yaml",
    "render_config_yaml_with_comments",
    "save_config_yaml",
    "build_metadata",
    "build_table_comment",
    "resolve_output_table_name",
]
