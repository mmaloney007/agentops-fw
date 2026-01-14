#!/usr/bin/env python3
"""
metadata/llm.py
---------------
LLM-based column definitions and table comments.

Contains:
    - create_intelligent_data_dictionary: Generate column definitions via LLM
    - create_table_comment: Generate table-level comments via LLM

Author: Mike Maloney - Neuralift, Inc.
Updated: 2026-01-13
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dask.dataframe as dd
import openai
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, constr, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .helpers import (
    ASCII,
    _ascii7,
    _ascii_deep,
    _cache_load,
    _cache_save,
    _current_rate_limit_delay,
    _gc_collect,
    _hash_key,
    _profile_single_column,
    _rate_limit_before_sleep,
    _relax_rate_limit,
    _wrap_instructor,
    _ensure_instructor_client,
)
from ..stats_cache import StatsCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM column description schema + prompt
# ---------------------------------------------------------------------------
class ColumnDescription(BaseModel):
    """Pydantic model for LLM-generated column descriptions."""

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
    """Generate a prompt for a single column."""
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


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(exp_base=2, max=60),
    reraise=True,
    before_sleep=_rate_limit_before_sleep,
)
def _describe_column(client, prompt: str, model: str) -> ColumnDescription:
    """Call LLM to describe a single column."""
    return client.chat.completions.create(
        model=model,
        response_model=ColumnDescription,
        messages=[
            {"role": "system", "content": "Respond only with valid json."},
            {"role": "user", "content": prompt},
        ],
        top_p=1.0,
    )


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
    show_progress: bool = True,
    debug: bool = False,
    use_approx_unique: bool = True,
    use_sample_for_uniques: bool = False,
) -> Tuple[Dict[str, str], Dict[str, Any], str]:
    """MEMORY-OPTIMIZED Dask data dictionary generator.

    Dask-side optimizations:
      - Single stats pass for non-null counts and uniques (approx by default).
      - Sampling via ddf.sample(...), no head().
      - All per-column profiling only uses the in-memory pandas sample.

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

    if show_progress:
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

    if show_progress:
        logger.info(
            "[data-dict] Starting data dictionary for %d column(s)...", len(col_names)
        )
        logger.info("[data-dict] Computing Dask stats (counts, uniques)...")

    # Use centralized stats_cache for all stats computation
    cache = StatsCache(show_progress=show_progress)

    # Compute all stats in one pass via stats_cache
    if use_sample_for_uniques:
        # Only get row count and null counts; uniques from sample later
        row_count = cache.get_row_count(ddf)
        null_counts = cache.get_null_counts(ddf, col_names)
        unique_counts = None
    else:
        # Get all stats including uniques
        row_count, null_counts, unique_counts = cache.compute_all_stats(
            ddf, col_names, approx_unique=use_approx_unique
        )

    if show_progress:
        logger.info(
            "[data-dict] Stats done: ~%d rows over %d columns",
            row_count,
            len(col_names),
        )
        logger.info("[data-dict] Building sample via ddf.sample(...) (no head())...")

    sample_frac = min(1.0, sample_rows / max(row_count, 1))
    sample_ddf = ddf[col_names].sample(frac=sample_frac, random_state=42)
    sample_pdf = sample_ddf.compute()

    if use_sample_for_uniques or unique_counts is None:
        unique_counts = sample_pdf.nunique(dropna=True)
        if not isinstance(unique_counts, pd.Series):
            unique_counts = pd.Series(unique_counts, index=col_names)
        unique_counts = unique_counts.astype(int)

    if show_progress:
        logger.info(
            "[data-dict] Sample ready: %d rows, %d columns (used for profiling)",
            len(sample_pdf),
            len(col_names),
        )
        logger.info("[data-dict] Profiling columns and generating LLM definitions...")

    profiles: List[Dict[str, Any]] = []
    for i, name in enumerate(col_names, start=1):
        if show_progress and (i == 1 or i % 10 == 0 or i == len(col_names)):
            logger.info("[profile] %d/%d %s", i, len(col_names), name)
        prof = _profile_single_column(
            sample_pdf[name],
            name,
            row_count=row_count,
            null_count=int(null_counts.get(name, 0)),
            unique_count=int(unique_counts.get(name, 0)),
            top_n_values=top_n_values,
        )
        profiles.append(prof)

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
                if show_progress:
                    logger.info("[cache] %d/%d %s", idx, total, name)
                return idx, name, cached

        if show_progress:
            logger.info("[start] %d/%d %s", idx, total, name)
        delay = _current_rate_limit_delay()
        if delay:
            time.sleep(delay)
        prompt = _prompt_for_column(profile, context)
        if debug:
            logger.debug("[DEBUG %d/%d] Prompt for '%s': %s", idx, total, name, prompt)
        desc = _describe_column(client, prompt, model=model)
        definition = desc.definition
        if debug:
            logger.debug(
                "[DEBUG %d/%d] Definition for '%s': %s", idx, total, name, definition
            )
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
                logging.getLogger("data-dict").exception(
                    "Definition failed for column %s", futs[fut]
                )
                name = futs[fut]
                safe_def = _ascii7(f"Definition unavailable; {type(e).__name__}")
            definitions[name] = safe_def
            completed += 1
            if show_progress and (completed == 1 or completed % 10 == 0):
                logger.info(
                    "[data-dict] LLM progress: %d/%d columns", completed, total_cols
                )

    del sample_pdf, null_counts, unique_counts
    _gc_collect()

    if show_progress:
        logger.info("[data-dict] Complete! Generated %d definitions", len(definitions))

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


class CommentModel(BaseModel):
    """Pydantic model for LLM-generated table comments."""

    comment: constr(max_length=750, pattern=ASCII) = Field(..., alias="Comment")
    model_config = ConfigDict(populate_by_name=True)

    @field_validator("comment", mode="after")
    @classmethod
    def _ascii_guard(cls, v: str) -> str:
        v = _ascii7(v).strip()
        return v[0].upper() + v[1:] if v else v


def build_prompt(*, yaml_dd: str, context: str) -> str:
    """Build prompt for table comment generation."""
    return textwrap.dedent(f"""\
You are a senior data scientist. Write ONE compact paragraph (≤500 ASCII characters, max 5 sentences) that states **what is in this table**:
key columns or data themes, data types, and overall time span (earliest to latest dates present).
Do not mention SQL, analysis use-cases, KPIs, or any jargon or abbreviations.

ORGANIZATION CONTEXT:
{context}

DATA DICTIONARY:
{yaml_dd}

Respond with exactly this single-line JSON (no markdown, no extra keys):
{{"Comment":"<your paragraph>"}}""")


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(exp_base=2, max=16),
    reraise=True,
)
def _call_comment(client, model, prompt) -> str:
    """Call LLM to generate table comment."""
    result = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Respond only with valid json."},
            {"role": "user", "content": prompt},
        ],
        response_model=CommentModel,
        top_p=1.0,
    )
    return result.comment


def create_table_comment(
    *,
    openai_client: openai.OpenAI,
    model_name: str = DEFAULT_TABLE_COMMENT_MODEL,
    file_context: str | None = None,
    yaml_dd: str | None = None,
    weave_project: str | None = None,
) -> Tuple[Dict[str, Any], str]:
    """Build an ASCII, SQL-safe table comment.

    Returns:
        (meta_dict, comment_str)
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


__all__ = [
    "ColumnDescription",
    "CommentModel",
    "create_intelligent_data_dictionary",
    "create_table_comment",
    "DEFAULT_TABLE_COMMENT_MODEL",
    "build_prompt",
]
