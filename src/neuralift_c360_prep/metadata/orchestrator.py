#!/usr/bin/env python3
"""
metadata/orchestrator.py
------------------------
High-level metadata orchestration functions.

Contains:
    - resolve_output_table_name: Resolve table name from config
    - build_metadata: Full metadata builder (tags + LLM definitions + JSON)
    - build_table_comment: Simple table comment builder

Author: Mike Maloney - Neuralift, Inc.
Updated: 2026-01-13
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import dask.dataframe as dd
import openai
import pandas as pd
import yaml

from .llm import (
    create_intelligent_data_dictionary,
    create_table_comment as _llm_table_comment,
)
from .tagging import build_column_tags_yaml_dask, build_data_dictionary_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _first_nonempty(*args: Any) -> Any:
    """Return the first non-empty/non-None argument."""
    for a in args:
        if a:
            return a
    return None


def _to_snake(name: str, max_len: int = 64) -> str:
    """CamelCase + punctuation → ASCII snake_case."""
    if pd.isna(name):
        return "_"
    txt = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    txt = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", txt)
    txt = re.sub(r"[^\w]+", "_", txt).strip("_").lower()
    if txt and txt[0].isdigit():
        txt = f"_{txt}"
    return (txt or "_")[:max_len]


def _lift_to_tags(lift) -> Dict[str, str]:
    """Extract lift metadata from a lift config object."""
    col_tags: Dict[str, str] = {}
    if getattr(lift, "value_sum_column", None):
        col_tags["lift_value_sum_column"] = lift.value_sum_column
    if getattr(lift, "value_sum_unit", None):
        col_tags["lift_value_sum_unit"] = lift.value_sum_unit
    if getattr(lift, "event_sum_column", None):
        col_tags["lift_event_sum_column"] = lift.event_sum_column
    if getattr(lift, "event_sum_unit", None):
        col_tags["lift_event_sum_unit"] = lift.event_sum_unit
    return col_tags


def _resolve_feature_kpi_cols(feat) -> List[str]:
    """Resolve KPI column names from a feature function config."""
    if getattr(feat, "kpi_columns", None):
        return [_to_snake(c) for c in feat.kpi_columns or []]
    if not getattr(feat, "kpi", False):
        return []
    feat_type = getattr(feat, "type", None)
    if feat_type == "date_parts":
        logger.warning(
            "[metadata] date_parts produces multiple columns; use kpi_columns to tag KPIs"
        )
        return []
    if feat_type == "binning":
        source_col = getattr(feat, "source_col", None) or (
            feat.inputs[0] if getattr(feat, "inputs", None) else None
        )
        if source_col:
            suffix = getattr(feat, "output_suffix", None) or "_bin"
            out_col = getattr(feat, "out_col", None) or f"{source_col}{suffix}"
            return [_to_snake(out_col)]
    if feat_type == "winsorize":
        source_col = getattr(feat, "source_col", None) or (
            feat.inputs[0] if getattr(feat, "inputs", None) else None
        )
        if source_col:
            out_col = getattr(feat, "out_col", None) or f"{source_col}_winsorized"
            return [_to_snake(out_col)]
    if feat_type == "log_transform":
        source_col = getattr(feat, "source_col", None) or (
            feat.inputs[0] if getattr(feat, "inputs", None) else None
        )
        if source_col:
            out_col = getattr(feat, "out_col", None) or f"{source_col}_log"
            return [_to_snake(out_col)]
    if feat_type == "categorical_bucket":
        source_col = getattr(feat, "source_col", None) or (
            feat.inputs[0] if getattr(feat, "inputs", None) else None
        )
        if source_col:
            out_col = getattr(feat, "out_col", None) or f"{source_col}_bucket"
            return [_to_snake(out_col)]
    if feat_type == "string_normalize":
        source_col = getattr(feat, "source_col", None) or (
            feat.inputs[0] if getattr(feat, "inputs", None) else None
        )
        if source_col:
            out_col = getattr(feat, "out_col", None) or f"{source_col}_normalized"
            return [_to_snake(out_col)]
    if feat_type == "frequency_encode":
        source_col = getattr(feat, "source_col", None) or (
            feat.inputs[0] if getattr(feat, "inputs", None) else None
        )
        if source_col:
            out_col = getattr(feat, "out_col", None) or f"{source_col}_freq"
            return [_to_snake(out_col)]
    if feat_type == "days_since":
        source_col = getattr(feat, "source_col", None) or (
            feat.inputs[0] if getattr(feat, "inputs", None) else None
        )
        if source_col:
            out_col = getattr(feat, "out_col", None) or f"{source_col}_days_since"
            return [_to_snake(out_col)]
    if feat_type == "ratio":
        numerator = getattr(feat, "numerator_col", None) or getattr(
            feat, "source_col", None
        )
        if not numerator and getattr(feat, "inputs", None):
            numerator = feat.inputs[0]
        denominator = getattr(feat, "denominator_col", None)
        if not denominator and getattr(feat, "inputs", None) and len(feat.inputs) > 1:
            denominator = feat.inputs[1]
        if numerator and denominator:
            out_col = getattr(feat, "out_col", None) or f"{numerator}_per_{denominator}"
            return [_to_snake(out_col)]
    if getattr(feat, "out_col", None):
        return [_to_snake(feat.out_col)]
    return []


def _collect_kpi_and_lift_from_functions(
    cfg,
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """Extract KPI columns and lift metadata from functions config.

    Supports both new flat list format and legacy 3-section format.

    Returns:
        (kpi_cols, extra_tags_by_column)
        - kpi_cols: list of KPI column names (output cols from functions with kpi=True)
        - extra_tags_by_column: dict mapping column names to lift tags
    """
    kpi_cols: List[str] = []
    extra_tags: Dict[str, Dict[str, str]] = {}

    funcs = getattr(cfg, "functions", None)
    if funcs is None:
        return kpi_cols, extra_tags

    # Get the flat list of functions (new format)
    functions = getattr(funcs, "functions", []) or []

    for func in functions:
        # Skip functions not marked as KPI
        if not getattr(func, "kpi", False):
            continue

        func_type = getattr(func, "type", None)

        # Resolve output column name based on function type
        if func_type == "zsml":
            source_col = getattr(func, "source_col", None)
            out_col = getattr(func, "out_col", None) or f"{source_col}_tier"
            out_col = _to_snake(out_col)
            kpi_cols.append(out_col)

            # Extract lift metadata
            lift = getattr(func, "lift", None)
            if lift:
                col_tags = _lift_to_tags(lift)
                if col_tags:
                    extra_tags[out_col] = col_tags

        elif func_type == "identity":
            col = getattr(func, "column", None)
            if col:
                col_snake = _to_snake(col)
                kpi_cols.append(col_snake)

                # Extract lift metadata
                lift = getattr(func, "lift", None)
                if lift:
                    col_tags = _lift_to_tags(lift)
                    if col_tags:
                        extra_tags[col_snake] = col_tags

        else:
            # For other function types, use the existing resolution logic
            feat_kpis = _resolve_feature_kpi_cols(func)
            for col in feat_kpis:
                kpi_cols.append(col)
                lift = getattr(func, "lift", None)
                if lift:
                    col_tags = _lift_to_tags(lift)
                    if col_tags:
                        extra_tags[col] = col_tags

    return kpi_cols, extra_tags


def _resolve_lift_column(name: str, columns: Iterable[str]) -> str | None:
    """Resolve a lift column name against available columns."""
    if name in columns:
        return name
    snake = _to_snake(name)
    if snake in columns:
        return snake
    return None


def _validate_lift_tags(
    tags_by_col: Mapping[str, Mapping[str, str]],
    columns: Iterable[str],
    *,
    strict: bool,
) -> Dict[str, Dict[str, str]]:
    """Validate lift tags against available columns."""
    validated: Dict[str, Dict[str, str]] = {}
    cols_set = set(columns)
    lift_keys = {
        "lift_value_sum_column",
        "lift_value_sum_unit",
        "lift_event_sum_column",
        "lift_event_sum_unit",
    }

    for col, tags in tags_by_col.items():
        issues: list[str] = []
        next_tags = dict(tags)

        for col_key, unit_key in (
            ("lift_value_sum_column", "lift_value_sum_unit"),
            ("lift_event_sum_column", "lift_event_sum_unit"),
        ):
            col_val = next_tags.get(col_key)
            unit_val = next_tags.get(unit_key)

            if col_val:
                resolved = _resolve_lift_column(col_val, cols_set)
                if not resolved:
                    issues.append(f"{col_key} '{col_val}' not found in columns")
                else:
                    next_tags[col_key] = resolved
                if not unit_val:
                    issues.append(f"{unit_key} is required when {col_key} is set")
            elif unit_val:
                issues.append(f"{unit_key} set without {col_key}")

        if issues:
            msg = f"[lift] invalid lift metadata for '{col}': {', '.join(issues)}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            non_lift = {k: v for k, v in next_tags.items() if k not in lift_keys}
            if non_lift:
                validated[col] = non_lift
            continue

        validated[col] = next_tags

    return validated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def resolve_output_table_name(cfg, override: str | None = None) -> str:
    """Resolve a stable logical table name for metadata/tags.

    Args:
        cfg: Configuration object with output settings.
        override: Optional explicit table name to use.

    Returns:
        The resolved table name.
    """
    name = _first_nonempty(
        override,
        getattr(cfg.output, "uc_table", None),
        getattr(cfg.output, "uc_volume_name", None),
    )
    return name or "unknown_table"


def build_metadata(
    ddf: dd.DataFrame,
    cfg,
    *,
    table_name_override: str | None = None,
) -> Tuple[dict, str]:
    """Full metadata builder.

    Orchestrates the full metadata generation pipeline:
      1) Build tags YAML (id/kpi/cat/cont, uniques, extras)
      2) Build LLM column definitions (data dictionary)
      3) Build table comment
      4) Build JSON data dictionary (comment/data_type/column_name/column_type/lift_*)

    Args:
        ddf: Dask DataFrame to generate metadata for.
        cfg: Configuration object with metadata settings.
        table_name_override: Optional explicit table name.

    Returns:
        Tuple of (metadata_dict, json_text).
    """
    show_progress = getattr(cfg.logging, "show_progress", True)
    table_name = resolve_output_table_name(cfg, table_name_override)
    skip_llm = os.getenv("NL_SKIP_LLM") == "1" or os.getenv(
        "OPENAI_API_KEY", ""
    ).startswith("sk-test")

    if show_progress:
        logger.info("[metadata] Building metadata for table: %s", table_name)

    # ---- tags ----
    tags_cfg = cfg.metadata.tags
    extra_all = tags_cfg.extra_tags_all or {}
    extra_by_col = dict(tags_cfg.extra_tags_by_column or {})

    # Collect KPI cols and lift metadata from functions config
    funcs_kpi_cols, funcs_extra_tags = _collect_kpi_and_lift_from_functions(cfg)
    funcs_extra_tags = _validate_lift_tags(
        funcs_extra_tags,
        ddf.columns,
        strict=getattr(cfg.metadata, "lift_strict", False),
    )

    # Merge KPI cols from functions with legacy tags_cfg.kpi_cols
    all_kpi_cols = list(tags_cfg.kpi_cols) + funcs_kpi_cols
    # Remove duplicates while preserving order
    seen_kpis: set = set()
    deduped_kpi_cols = []
    for k in all_kpi_cols:
        if k.lower() not in seen_kpis:
            seen_kpis.add(k.lower())
            deduped_kpi_cols.append(k)

    # Merge lift tags from functions with legacy extra_tags_by_column
    for col, tags in funcs_extra_tags.items():
        if col in extra_by_col:
            extra_by_col[col] = {**extra_by_col[col], **tags}
        else:
            extra_by_col[col] = tags

    # Validate lift tags across all extra tags
    extra_by_col = _validate_lift_tags(
        extra_by_col,
        ddf.columns,
        strict=getattr(cfg.metadata, "lift_strict", False),
    )

    # Also get ID cols from ids config (new style)
    ids_cfg = getattr(cfg, "ids", None)
    id_cols_from_ids = list(getattr(ids_cfg, "columns", []) or []) if ids_cfg else []
    all_id_cols = list(tags_cfg.id_cols) + id_cols_from_ids
    # Remove duplicates
    seen_ids: set = set()
    deduped_id_cols = []
    for i in all_id_cols:
        if i.lower() not in seen_ids:
            seen_ids.add(i.lower())
            deduped_id_cols.append(i)

    if show_progress:
        logger.info(
            "[metadata] Building tags for %d columns (ids=%d, kpis=%d)",
            len(ddf.columns),
            len(deduped_id_cols),
            len(deduped_kpi_cols),
        )

    tags_by_col, tags_yaml = build_column_tags_yaml_dask(
        ddf,
        id_cols=deduped_id_cols,
        kpi_cols=deduped_kpi_cols,
        missing_indicator_cols=tags_cfg.missing_indicator_cols,
        max_card=tags_cfg.max_card,
        use_approx_unique=tags_cfg.use_approx_unique,
        extra_tags_all=extra_all,
        extra_tags_by_column=extra_by_col,
        show_progress=show_progress,
        debug=cfg.logging.level == "debug",
    )

    # ---- definitions (LLM) with fallback ----
    try:
        if skip_llm:
            raise RuntimeError("LLM skipped by config")
        if show_progress:
            logger.info("[metadata] Generating LLM column definitions...")
        definitions, dtypes_out, dd_yaml = create_intelligent_data_dictionary(
            ddf,
            model=cfg.metadata.model,
            context=cfg.metadata.context,
            sample_rows=cfg.metadata.sample_rows,
            max_concurrency=cfg.metadata.max_concurrency,
            show_progress=show_progress,
        )
    except Exception as exc:  # pragma: no cover - network/LLM failures
        if show_progress:
            logger.warning(
                "[metadata] LLM data dictionary skipped/failed (%s); using fallback definitions",
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
    try:
        if skip_llm:
            raise RuntimeError("LLM skipped by config")
        if show_progress:
            logger.info("[metadata] Generating LLM table comment...")
        comment_meta, table_comment = _llm_table_comment(
            openai_client=openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
            model_name=cfg.metadata.model,
            file_context=cfg.metadata.context,
            yaml_dd=dd_yaml,
        )
    except Exception as exc:  # pragma: no cover - network/LLM failures
        if show_progress:
            logger.warning(
                "[metadata] Table comment generation skipped/failed (%s); using fallback",
                exc,
            )
        table_comment = (
            cfg.metadata.context
            or getattr(cfg.output, "run_name", None)
            or f"Table {table_name}"
        )

    # ---- JSON data dictionary ----
    if show_progress:
        logger.info("[metadata] Building JSON data dictionary...")
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

    if show_progress:
        logger.info(
            "[metadata] Metadata generation complete for %d columns", len(ddf.columns)
        )

    return meta_with_table, meta_json_text


def build_table_comment(cfg) -> str:
    """Build a simple table comment from config.

    Args:
        cfg: Configuration object with metadata and output settings.

    Returns:
        A table comment string.
    """
    table_name = resolve_output_table_name(cfg)
    return (
        cfg.metadata.context
        or getattr(cfg.output, "run_name", None)
        or f"Table {table_name}"
    )


__all__ = [
    "resolve_output_table_name",
    "build_metadata",
    "build_table_comment",
]
