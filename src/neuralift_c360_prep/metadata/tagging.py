#!/usr/bin/env python3
"""
metadata/tagging.py
-------------------
Column tagging and data dictionary JSON building.

Contains:
    - build_column_tags_yaml_dask: Classify columns and emit tags YAML
    - build_data_dictionary_json: Build flat JSON data dictionary
    - inspect_schema_alignment: Debug schema/tag alignment

Author: Mike Maloney - Neuralift, Inc.
Updated: 2026-01-13
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import dask.dataframe as dd
import pandas as pd
import yaml
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from .helpers import _ascii_deep
from ..stats_cache import StatsCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column tagging
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
    show_progress: bool = True,
    progress_every: int = 10,
    approx_row_threshold: int = 2_000_000,
    debug: bool = False,
) -> Tuple[Dict[str, Dict[str, str]], str]:
    """Classify columns and emit a YAML for tags.

    RULES (Type + cardinality only):
      - id/kpi override everything
      - BOOLEAN => categorical
      - STRING/object/category => categorical
      - DATETIME/DATE => categorical
      - Numeric => categorical if uniq <= max_card else continuous
      - fallback => categorical

    Returns:
        (tags_by_col, yaml_text)
    """
    cols = list(ddf.columns)
    id_set = {c.lower() for c in id_cols}
    kpi_set = {c.lower() for c in kpi_cols}
    miss_set = {c.lower() for c in (missing_indicator_cols or [])}

    # Use centralized stats_cache for all stats computation
    cache = StatsCache(show_progress=show_progress)

    if show_progress:
        logger.info("[tags] computing non-null counts (and inferring row_count)...")

    # Get row count first (needed to determine approx vs exact)
    row_count = cache.get_row_count(ddf)
    dtypes = ddf.dtypes

    if show_progress:
        logger.info(
            "[tags] stats: rows=%d, cols=%d, use_approx_unique=%s, approx_row_threshold=%d",
            row_count,
            len(cols),
            use_approx_unique,
            approx_row_threshold,
        )

    use_approx = bool(use_approx_unique and row_count > approx_row_threshold)

    # Compute null counts and unique counts via stats_cache
    null_counts = cache.get_null_counts(ddf, cols)

    if use_approx:
        if show_progress:
            logger.info(
                "[tags] using approximate uniques (per-column nunique_approx) for large table"
            )
        unique_counts = cache.get_unique_counts(ddf, cols, approx=True)
    else:
        if show_progress:
            logger.info("[tags] computing exact uniques (nunique) for all columns...")
        unique_counts = cache.get_unique_counts(ddf, cols, approx=False)

    # Optional exact uniques for gray-band columns when using approx
    # (columns near the max_card threshold need exact counts for accurate classification)
    if use_approx and approx_gray_band >= 0:
        gray_cols: list[str] = []
        for c in cols:
            est = int(unique_counts.get(c, 0))
            if abs(est - max_card) <= approx_gray_band:
                gray_cols.append(c)
        if exact_unique_limit is not None:
            gray_cols = gray_cols[:exact_unique_limit]

        if gray_cols:
            if show_progress:
                logger.info("[tags] exact uniques for gray-band cols: %s", gray_cols)
            # Compute exact uniques for gray-band columns (bypass cache for precision)
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
        if show_progress and (i == 1 or i % progress_every == 0 or i == total_cols):
            logger.info("[tags] profiling %d/%d", i, total_cols)

        col_l = col.lower()
        uniq = int(unique_counts.get(col, 0))
        dtype_str = str(dtypes[col]).lower()

        dt = pd.api.types.pandas_dtype(dtypes[col])

        # Classification: TYPE + CARDINALITY ONLY
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
                "[tag] %s dtype=%s uniq=%d max_card=%d is_bool=%s is_str=%s is_cat=%s -> type=%s",
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


# ---------------------------------------------------------------------------
# Data type mapping
# ---------------------------------------------------------------------------
def _map_dtype_to_data_type(dtype: Any) -> str:
    """Map a pandas/dask dtype to STRING, INTEGER, FLOAT, BOOLEAN, DOUBLE."""
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


# ---------------------------------------------------------------------------
# JSON data dictionary
# ---------------------------------------------------------------------------
def build_data_dictionary_json(
    *,
    table_name: str,
    table_comment: Optional[str],
    column_definitions: Mapping[str, str],
    column_tags: Mapping[str, Mapping[str, str]],
    column_dtypes: Mapping[str, Any] | None = None,
    column_order: Iterable[str] | None = None,
) -> Tuple[Dict[str, Any], str]:
    """Build a flat JSON data dictionary.

    FIXED RULES ENFORCED HERE (safety net):
      - If data_type is STRING or BOOLEAN, column_type is forced to categorical
        unless the column is explicitly id/kpi.

    Returns:
        (meta_json, json_text)
    """
    import json

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

        # HARD INVARIANTS
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


# ---------------------------------------------------------------------------
# Schema alignment inspection
# ---------------------------------------------------------------------------
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
    """Print a per-column summary comparing dtypes/tags/json."""
    tags_obj = yaml.safe_load(tags_yaml)
    tags_columns = {c["name"]: c for c in tags_obj.get("columns", [])}
    json_cols = {c["column_name"]: c for c in meta_json.get("columns", [])}

    all_cols = list(ddf.columns)
    logger.info("[schema] Inspecting %d column(s)...", len(all_cols))

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

        logger.info(
            "[schema] %s: dask_dtype=%s, dtypes_map=%s, tags_type=%s, "
            "json_data_type=%s, json_col_type=%s, null_count=%s, unique_count=%s, "
            "max_card=%d, is_numeric=%s, json_says_string=%s, cat_cont_mismatch=%s",
            col,
            dask_dtype,
            dtypes_entry,
            col_type,
            json_data_type,
            json_col_type,
            null_count,
            unique_count,
            max_card,
            is_dask_numeric,
            json_says_string,
            cat_vs_cont_mismatch,
        )


__all__ = [
    "build_column_tags_yaml_dask",
    "build_data_dictionary_json",
    "inspect_schema_alignment",
    "_map_dtype_to_data_type",
]
