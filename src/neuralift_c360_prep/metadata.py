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
import json
import logging
import math
import os
import re
import textwrap
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import dask.dataframe as dd
import openai
import pandas as pd
import yaml
from dask import compute as dask_compute
from instructor import Mode, from_openai  # type: ignore
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
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

    Note: The pipeline prefers _compute_stats_and_sample() to batch stats + sample
    in a single pass. This remains for compatibility and direct callers.
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


def _build_partition_sample(
    ddf: dd.DataFrame,
    *,
    sample_rows: int,
    sample_cols: Sequence[str],
    random_state: int = 42,
) -> dd.DataFrame | None:
    if sample_rows <= 0 or not sample_cols:
        return None

    cols = [c for c in sample_cols if c in ddf.columns]
    if not cols:
        return None

    n_parts = max(1, int(ddf.npartitions))
    per_part = max(1, int(math.ceil(sample_rows / float(n_parts))))

    def _sample_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        subset = pdf[cols]
        if subset.empty:
            return subset.head(0)
        n = min(len(subset), per_part)
        if len(subset) <= n:
            return subset
        return subset.sample(n=n, random_state=random_state)

    return ddf.map_partitions(_sample_partition, meta=ddf[cols]._meta)


def _compute_stats_and_sample(
    ddf: dd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    sample_rows: int = 0,
    sample_cols: Sequence[str] | None = None,
    compute_unique_counts: bool = True,
    use_approx_unique: bool = True,
    approx_row_threshold: int | None = 2_000_000,
) -> tuple[ColumnStats, pd.DataFrame | None]:
    cols = list(columns) if columns is not None else list(ddf.columns)
    dtypes = ddf.dtypes.reindex(cols)

    if not cols:
        empty = _empty_int_series(cols)
        stats = ColumnStats(
            columns=cols,
            row_count=0,
            counts=empty,
            null_counts=empty,
            dtypes=dtypes,
            unique_counts=empty if compute_unique_counts else None,
            unique_is_approx=False,
        )
        return stats, None

    counts_delayed = ddf[cols].count()
    tasks: list[Any] = [counts_delayed]

    unique_counts = None
    unique_is_approx = False
    approx_tasks: list[Any] = []
    if compute_unique_counts:
        if use_approx_unique:
            approx_tasks = [ddf[c].nunique_approx() for c in cols]
            tasks.extend(approx_tasks)
            unique_is_approx = True
        else:
            tasks.append(ddf[cols].nunique(dropna=True))

    sample_ddf = _build_partition_sample(
        ddf,
        sample_rows=sample_rows,
        sample_cols=sample_cols or cols,
    )
    if sample_ddf is not None:
        tasks.append(sample_ddf)

    results = dask_compute(*tasks)
    idx = 0

    counts = results[idx]
    idx += 1

    if not isinstance(counts, pd.Series):
        counts = pd.Series(counts, index=cols)
    counts = counts.reindex(cols).fillna(0).astype(int)
    row_count = int(counts.max()) if len(counts) else 0
    null_counts = (row_count - counts).astype(int)

    if compute_unique_counts:
        if use_approx_unique:
            approx_vals = results[idx : idx + len(cols)]
            idx += len(cols)
            unique_counts = pd.Series(approx_vals, index=cols).astype(int)
        else:
            exact_res = results[idx]
            idx += 1
            if isinstance(exact_res, pd.DataFrame):
                if exact_res.shape[0] == 1:
                    exact_res = exact_res.iloc[0]
                else:
                    raise TypeError(
                        "Expected unique counts as Series/1-row DF; got "
                        f"DF shape={exact_res.shape}"
                    )
            if not isinstance(exact_res, pd.Series):
                exact_res = pd.Series(exact_res, index=cols)
            unique_counts = exact_res.reindex(cols).fillna(0).astype(int)

    sample_pdf = results[idx] if sample_ddf is not None else None
    if sample_pdf is not None and sample_rows > 0 and len(sample_pdf) > sample_rows:
        sample_pdf = sample_pdf.sample(n=sample_rows, random_state=42)

    if (
        compute_unique_counts
        and use_approx_unique
        and approx_row_threshold is not None
        and row_count > 0
        and row_count <= approx_row_threshold
    ):
        logger.info(
            "[stats] row_count=%s < threshold=%s; computing exact uniques...",
            f"{row_count:,}",
            f"{approx_row_threshold:,}",
        )
        unique_counts = _compute_exact_unique_counts(ddf, cols)
        unique_is_approx = False

    stats = ColumnStats(
        columns=cols,
        row_count=row_count,
        counts=counts,
        null_counts=null_counts,
        dtypes=dtypes,
        unique_counts=unique_counts if compute_unique_counts else None,
        unique_is_approx=unique_is_approx,
    )
    return stats, sample_pdf


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
    sample_pdf: pd.DataFrame | None = None,
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
    if sample_rows <= 0:
        sample_pdf = pd.DataFrame(columns=col_names)
    elif sample_pdf is None:
        logger.info("🎯 Building sample via ddf.sample(...) (no head())...")
        sample_frac = min(1.0, sample_rows / max(row_count, 1))
        sample_ddf = ddf[col_names].sample(frac=sample_frac, random_state=42)
        sample_pdf = sample_ddf.compute()
    else:
        logger.info("🎯 Using precomputed sample (%s rows)...", len(sample_pdf))
        sample_pdf = sample_pdf.reindex(columns=col_names)
        if len(sample_pdf) > sample_rows:
            sample_pdf = sample_pdf.sample(n=sample_rows, random_state=42)

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
        gray_col_info: dict[str, int] = {}  # col -> approx unique count
        for c in cols:
            est = int(unique_counts.get(c, 0))
            if abs(est - max_card) <= approx_gray_band:
                gray_cols.append(c)
                gray_col_info[c] = est
        if exact_unique_limit is not None:
            gray_cols = gray_cols[:exact_unique_limit]
            gray_col_info = {c: gray_col_info[c] for c in gray_cols}

        if gray_cols:
            logger.warning(
                "[tags] gray-band exact uniques for %d column(s) trigger an extra compute pass; "
                "set metadata.tags.approx_gray_band=-1 to keep approximate uniques only.",
                len(gray_cols),
            )
            # Log detailed info about which columns and why
            detail_lines = [
                f"  - {c}: ~{gray_col_info[c]} approx uniques (within {approx_gray_band} of max_card={max_card})"
                for c in gray_cols
            ]
            logger.info(
                "[tags] gray-band columns being recomputed for exact uniques:\n%s",
                "\n".join(detail_lines),
            )
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
    row_count: int | None = None,
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

        columns_json.append(col_entry)

    column_count = len(columns_json)
    row_count_int = int(row_count) if row_count is not None else None
    row_text = f"{row_count_int:,}" if row_count_int is not None else "unknown"
    col_text = f"{column_count:,}"

    meta_json: Dict[str, Any] = {
        "comment": table_comment or "",
        "table_name": table_name,
        "row_count": row_count_int,
        "column_count": column_count,
        "shape": f"{row_text} rows, {col_text} columns",
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
# Minimal config builder (trimmed config.yaml output)
# ---------------------------------------------------------------------------
_MIN_CLUSTER_THRESHOLD_FLOOR = 50
_MIN_CLUSTER_THRESHOLD_DIVISOR = 250


def _min_cluster_threshold_min(row_count: int | None) -> int:
    if row_count is None or row_count <= 0:
        return _MIN_CLUSTER_THRESHOLD_FLOOR
    scaled = int(math.ceil(row_count / float(_MIN_CLUSTER_THRESHOLD_DIVISOR)))
    return max(scaled, _MIN_CLUSTER_THRESHOLD_FLOOR)


def build_minimal_config(
    *,
    row_count: int | None,
    run_name: str | None,
) -> Dict[str, Any]:
    project_name = run_name if run_name else None
    return {
        "use_wandb": True,
        "use_tuner": True,
        "use_ensemble": True,
        "use_auto_config": True,
        "wandb": {
            "project": project_name,
        },
        "dae": {
            "data_module": {
                "batch_size": 8192,
                "compute_stats_from": "full",
            },
            "trainer": {
                "max_epochs": 50,
            },
        },
        "explainability": {
            "num_features": 50,
            "top_n": 10,
        },
        "tuner": {
            "segment_size": "M",
            "min_cluster_threshold_min": _min_cluster_threshold_min(row_count),
        },
    }


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

    max_cols = getattr(cfg.metadata, "max_columns_for_comment", None)
    llm_cols = list(ddf.columns[:max_cols]) if max_cols else list(ddf.columns)
    sample_rows = cfg.metadata.sample_rows if not skip_llm else 0
    stats, sample_pdf = _compute_stats_and_sample(
        ddf,
        columns=list(ddf.columns),
        sample_rows=sample_rows,
        sample_cols=llm_cols,
        compute_unique_counts=not skip_unique_counts,
        use_approx_unique=tags_cfg.use_approx_unique,
        approx_row_threshold=getattr(tags_cfg, "approx_row_threshold", 2_000_000),
    )
    logger.info(
        "[stats] single-pass stats complete (rows≈%s, cols=%s, sample_rows=%s, approx_uniques=%s)",
        f"{stats.row_count:,}",
        len(stats.columns),
        sample_rows,
        stats.unique_is_approx,
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
    logger.info("[meta] building data dictionary definitions...")
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
            sample_pdf=sample_pdf,
            # LLM cache settings from config
            use_cache=getattr(cfg.metadata, "use_llm_cache", False),
            cache_dir=getattr(cfg.metadata, "llm_cache_dir", ".nl_dd_cache"),
            # Limit columns for LLM if configured
            max_cols=max_cols,
        )
        logger.info(
            "[meta] data dictionary definitions ready (%s columns)",
            len(definitions),
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
        logger.info(
            "[meta] data dictionary definitions ready (%s columns, fallback)",
            len(definitions),
        )

    # ---- table comment (LLM) with fallback ----
    skip_table_comment = getattr(cfg.metadata, "skip_table_comment", False)
    logger.info("[meta] building table comment...")
    try:
        if skip_llm or skip_table_comment:
            raise RuntimeError("Table comment skipped by config")
        comment_meta, table_comment = create_table_comment(
            openai_client=openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
            model_name=cfg.metadata.model,
            file_context=cfg.metadata.context,
            yaml_dd=dd_yaml,
        )
        logger.info("[meta] table comment ready")
    except Exception as exc:  # pragma: no cover - network/LLM failures
        logger.warning(
            "[meta] table comment generation skipped/failed (%s); using fallback", exc
        )
        table_comment = (
            cfg.metadata.context
            or getattr(cfg.output, "run_name", None)
            or f"Table {table_name}"
        )
        logger.info("[meta] table comment ready (fallback)")

    # ---- JSON data dictionary ----
    logger.info("[meta] building data dictionary JSON...")
    meta_json, meta_json_text = build_data_dictionary_json(
        table_name=table_name,
        table_comment=table_comment,
        column_definitions=definitions,
        column_tags=tags_by_col,
        column_dtypes=dtypes_out,
        column_order=ddf.columns,
        row_count=stats.row_count,
    )
    logger.info(
        "[meta] data dictionary JSON ready (%s columns)",
        len(meta_json.get("columns", [])),
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
    meta_with_table["_row_count"] = stats.row_count
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
    "ColumnStats",
    "compute_column_stats",
    "create_intelligent_data_dictionary",
    "create_table_comment",
    "build_column_tags_yaml_dask",
    "build_data_dictionary_json",
    "inspect_schema_alignment",
    "build_minimal_config",
    "build_metadata",
    "build_table_comment",
    "resolve_output_table_name",
]
