#!/usr/bin/env python3
"""
Function registry and application for agentic workflows.

Purpose:
    - Apply KPI and feature functions in a consistent, row-level way.
    - Enforce snake_case outputs and single-DF in/out behavior.
    - Support built-ins (zsml, binning, identity_kpi) and callable hooks.
"""

from __future__ import annotations

import importlib
import logging
from typing import Iterable, Sequence

import dask.dataframe as dd
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from dask.dataframe.utils import meta_nonempty

from .config import FunctionConfig
from .preprocess import ensure_zsml_kpi_dask, _to_snake

logger = logging.getLogger(__name__)

BUILTIN_FUNCTION_DOCS = {
    "zsml": (
        "ZSML KPI tiering. Creates Low/Medium/High segments from numeric column.\n"
        "  Inputs: source_col (numeric)\n"
        "  Output: <source>_tier (snake_case)\n"
        "  Options: quantiles=[0.33, 0.66], zero_threshold=0, clip_high_quantile=0.95\n"
        "  Can mark as kpi=true with lift metadata."
    ),
    "identity": (
        "Identity KPI tag. No transformation; tags existing column as KPI.\n"
        "  Inputs: column (existing column name)\n"
        "  Output: (none - uses existing column)\n"
        "  Use case: Mark existing boolean/categorical as KPI with lift metadata."
    ),
    "binning": (
        "Numeric binning. Creates categorical bins from continuous values.\n"
        "  Inputs: source_col (numeric)\n"
        "  Output: <source>_bin or out_col\n"
        "  Options:\n"
        "    - bins: [0, 18, 35, 55, 100] (custom edges)\n"
        "    - quantiles: [0.33, 0.66] (quantile-based, default)\n"
        "    - labels: ['Low', 'Medium', 'High'] (custom labels)"
    ),
    "winsorize": (
        "Outlier capping (winsorization). Clips extreme values.\n"
        "  Inputs: source_col (numeric)\n"
        "  Output: <source>_winsorized or out_col\n"
        "  Options:\n"
        "    - lower_quantile/upper_quantile: 0.01/0.99 (percentile bounds)\n"
        "    - lower_bound/upper_bound: fixed values"
    ),
    "log_transform": (
        "Safe logarithmic transform. Normalizes right-skewed distributions.\n"
        "  Inputs: source_col (numeric)\n"
        "  Output: <source>_log or out_col\n"
        "  Options: log_method='log1p'|'log', log_clip_min=0, log_offset=0"
    ),
    "date_parts": (
        "Date part extraction. Extracts year, month, day, etc. from dates.\n"
        "  Inputs: source_col (datetime)\n"
        "  Output: multiple columns (<source>_year, _month, _day, etc.)\n"
        "  Options: date_parts=['year','month',...], auto_daypart=true, drop_source=true"
    ),
    "categorical_bucket": (
        "Categorical bucketing. Groups infrequent values into 'other'.\n"
        "  Inputs: source_col (categorical/string)\n"
        "  Output: <source>_bucket or out_col\n"
        "  Options: top_k=5 (keep top K), min_count=100, other_label='other'"
    ),
    "string_normalize": (
        "String normalization. Cleans and standardizes text values.\n"
        "  Inputs: source_col (string)\n"
        "  Output: <source>_clean or out_col\n"
        "  Options: string_case='lower', strip=true, collapse_whitespace=true"
    ),
    "frequency_encode": (
        "Frequency encoding. Replaces categories with their frequency.\n"
        "  Inputs: source_col (categorical)\n"
        "  Output: <source>_freq or out_col\n"
        "  Options: normalize=true (0-1 scale)"
    ),
    "days_since": (
        "Days since calculation. Computes days between date and reference.\n"
        "  Inputs: source_col (datetime)\n"
        "  Output: days_since_<source> or out_col\n"
        "  Options: reference_date='2024-01-01' (defaults to today)"
    ),
    "ratio": (
        "Ratio feature. Divides numerator by denominator.\n"
        "  Inputs: numerator_col, denominator_col (or inputs=[num, denom])\n"
        "  Output: <num>_per_<denom> or out_col\n"
        "  Options: on_zero='zero'|'nan'|'epsilon', epsilon=1e-9"
    ),
    "callable": (
        "Custom callable. Invokes external Python function.\n"
        "  Inputs: callable='module:function', inputs=[...]\n"
        "  Output: defined by callable\n"
        "  Options: params={...} (passed to callable)"
    ),
}

DEFAULT_DAYPART_BINS = {
    "morning": (5, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 5),
}


def resolve_id_columns(ddf: dd.DataFrame, cfg) -> list[str]:
    """Resolve configured ID columns to snake_case names that exist in the frame."""
    ids_cfg = getattr(cfg, "ids", None)
    explicit = list(getattr(ids_cfg, "columns", []) or []) if ids_cfg else []
    if not explicit:
        explicit = list(getattr(cfg.input, "id_cols", []) or [])

    seen: set[str] = set()
    resolved: list[str] = []
    for col in explicit:
        if col in ddf.columns and col not in seen:
            resolved.append(col)
            seen.add(col)
            continue
        snake = _to_snake(col)
        if snake in ddf.columns and snake not in seen:
            resolved.append(snake)
            seen.add(snake)
    return resolved


def apply_functions(ddf: dd.DataFrame, cfg) -> dd.DataFrame:
    """Apply all functions from config in order, returning updated DataFrame.

    Processes each function in the `functions:` list sequentially. Functions can be:
    - KPI tiers: zsml (Low/Med/High), identity (tag existing column)
    - Feature engineering: binning, winsorize, log_transform, date_parts,
      categorical_bucket, string_normalize, frequency_encode, days_since, ratio
    - Custom: callable (external Python function)

    Any function can have `kpi: true` and `lift: {...}` for analytics metadata.

    Args:
        ddf: Dask DataFrame to transform.
        cfg: BundleConfig with `functions` section.

    Returns:
        DataFrame with all transformations applied.

    Example config:
        functions:
          - type: zsml
            source_col: Revenue
            kpi: true
            lift: {value_sum_column: Revenue, event_sum_column: Orders}
          - type: binning
            source_col: Age
            bins: [0, 18, 35, 55, 100]
    """
    funcs_cfg = getattr(cfg, "functions", None)
    if not funcs_cfg:
        return ddf

    verbose = getattr(cfg.logging, "level", "info") == "debug"
    id_cols = resolve_id_columns(ddf, cfg)

    # Get the flat list of functions
    functions = list(getattr(funcs_cfg, "functions", []) or [])

    # Also check for legacy preprocessing.zsml
    pre_cfg = getattr(cfg, "preprocessing", None) or getattr(cfg, "cleaning", None)
    zsml_cfg = getattr(pre_cfg, "zsml", None)
    if getattr(zsml_cfg, "enabled", False) and getattr(zsml_cfg, "source_col", None):
        from .config import FunctionConfig

        functions.insert(
            0,
            FunctionConfig(
                type="zsml",
                source_col=zsml_cfg.source_col,
                out_col=zsml_cfg.out_col,
                zero_threshold=zsml_cfg.zero_threshold,
                quantiles=zsml_cfg.quantiles,
                clip_high_quantile=zsml_cfg.clip_high_quantile,
                unit=zsml_cfg.unit,
                range_style=zsml_cfg.range_style,
                add_prefix=zsml_cfg.add_prefix,
                kpi=True,  # Legacy ZSML is always a KPI
            ),
        )

    # Single loop over all functions
    for func in functions:
        before_cols = list(ddf.columns)
        func_type = getattr(func, "type", "callable")

        if func_type == "zsml":
            ddf = _apply_zsml(ddf, func)
        elif func_type == "identity":
            ddf = _apply_identity(ddf, func)
        elif func_type == "binning":
            ddf = _apply_binning(ddf, func)
        elif func_type == "winsorize":
            ddf = _apply_winsorize(ddf, func)
        elif func_type == "log_transform":
            ddf = _apply_log_transform(ddf, func)
        elif func_type == "date_parts":
            ddf = _apply_date_parts(ddf, func)
        elif func_type == "categorical_bucket":
            ddf = _apply_categorical_bucket(ddf, func)
        elif func_type == "string_normalize":
            ddf = _apply_string_normalize(ddf, func)
        elif func_type == "frequency_encode":
            ddf = _apply_frequency_encode(ddf, func)
        elif func_type == "days_since":
            ddf = _apply_days_since(ddf, func)
        elif func_type == "ratio":
            ddf = _apply_ratio(ddf, func)
        elif func_type == "callable":
            ddf = _apply_callable(ddf, func)
        else:
            raise ValueError(f"Unsupported function type: {func_type}")

        ddf, new_cols = _rename_new_cols_snake(ddf, before_cols)
        ddf = _apply_return_mode(
            ddf,
            return_mode=getattr(func, "return_mode", "all"),
            return_columns=getattr(func, "return_columns", []),
            id_cols=id_cols,
            new_cols=new_cols,
            verbose=verbose,
        )

    return ddf


def _apply_zsml(ddf: dd.DataFrame, func: FunctionConfig) -> dd.DataFrame:
    """Apply ZSML KPI tiering transformation.

    Creates Low/Medium/High customer segments from numeric column using quantile-based
    thresholds. Typically used with kpi=true and lift metadata for analytics.

    Args:
        ddf: Dask DataFrame.
        func: FunctionConfig with:
            - source_col: Numeric column to tier (required)
            - out_col: Output column name (default: <source>_tier)
            - quantiles: Breakpoints (default: [0.33, 0.66] for tertiles)
            - zero_threshold: Values below this become 'Zero' tier
            - clip_high_quantile: Cap outliers at this percentile (default: 0.95)

    Returns:
        DataFrame with new tier column (e.g., 'Low', 'Medium', 'High').
    """
    source_col = _resolve_existing_column(ddf, func.source_col)
    if source_col is None:
        raise KeyError(f"ZSML source_col '{func.source_col}' not found")

    out_col = func.out_col or f"{source_col}_tier"
    out_col = _to_snake(out_col)

    ensure_zsml_kpi_dask(
        ddf,
        source_col=source_col,
        out_col=out_col,
        zero_threshold=getattr(func, "zero_threshold", 0.0),
        quantiles=getattr(func, "quantiles", (0.33, 0.66)),
        clip_high_quantile=getattr(func, "clip_high_quantile", 0.95),
        unit=getattr(func, "unit", ""),
        range_style=getattr(func, "range_style", "text"),
        add_prefix=getattr(func, "add_prefix", True),
    )

    return ddf


def _apply_identity(ddf: dd.DataFrame, func: FunctionConfig) -> dd.DataFrame:
    """Identity function - validates column exists, no transformation.

    Used to tag existing columns as KPIs without any transformation.
    """
    col = func.column
    resolved = _resolve_existing_column(ddf, col)
    if resolved is None:
        raise KeyError(f"identity column '{col}' not found in dataframe")
    # No transformation - column is just tagged as KPI via metadata
    return ddf


# Legacy function aliases for backward compatibility
def apply_kpi_functions(
    ddf: dd.DataFrame,
    kpis: Sequence[FunctionConfig],
    *,
    id_cols: Sequence[str],
    verbose: bool = False,
) -> dd.DataFrame:
    """DEPRECATED: Use apply_functions instead."""
    for kpi in kpis or []:
        before_cols = list(ddf.columns)
        ddf = _apply_zsml(ddf, kpi)
        ddf, new_cols = _rename_new_cols_snake(ddf, before_cols)
        ddf = _apply_return_mode(
            ddf,
            return_mode=getattr(kpi, "return_mode", "all"),
            return_columns=getattr(kpi, "return_columns", []),
            id_cols=id_cols,
            new_cols=new_cols,
            verbose=verbose,
        )
    return ddf


def apply_feature_functions(
    ddf: dd.DataFrame,
    features: Sequence[FunctionConfig],
    *,
    id_cols: Sequence[str],
    verbose: bool = False,
) -> dd.DataFrame:
    """DEPRECATED: Use apply_functions instead."""
    for feat in features or []:
        before_cols = list(ddf.columns)
        func_type = getattr(feat, "type", "callable")
        if func_type == "binning":
            ddf = _apply_binning(ddf, feat)
        elif func_type == "winsorize":
            ddf = _apply_winsorize(ddf, feat)
        elif func_type == "log_transform":
            ddf = _apply_log_transform(ddf, feat)
        elif func_type == "date_parts":
            ddf = _apply_date_parts(ddf, feat)
        elif func_type == "categorical_bucket":
            ddf = _apply_categorical_bucket(ddf, feat)
        elif func_type == "string_normalize":
            ddf = _apply_string_normalize(ddf, feat)
        elif func_type == "frequency_encode":
            ddf = _apply_frequency_encode(ddf, feat)
        elif func_type == "days_since":
            ddf = _apply_days_since(ddf, feat)
        elif func_type == "ratio":
            ddf = _apply_ratio(ddf, feat)
        elif func_type == "callable":
            ddf = _apply_callable(ddf, feat)
        else:
            raise ValueError(f"Unsupported feature function type: {func_type}")

        ddf, new_cols = _rename_new_cols_snake(ddf, before_cols)
        ddf = _apply_return_mode(
            ddf,
            return_mode=getattr(feat, "return_mode", "all"),
            return_columns=getattr(feat, "return_columns", []),
            id_cols=id_cols,
            new_cols=new_cols,
            verbose=verbose,
        )
    return ddf


def _apply_callable(
    ddf: dd.DataFrame,
    feat: FunctionConfig,
) -> dd.DataFrame:
    if not feat.callable:
        raise ValueError("callable is required for type=callable")

    fn = _load_callable(feat.callable)
    if not (fn.__doc__ or "").strip():
        logger.warning("[functions] callable %s is missing a docstring", feat.callable)

    _validate_inputs(ddf, feat.inputs)

    def _run(pdf: pd.DataFrame) -> pd.DataFrame:
        return fn(pdf, **(feat.params or {}))

    sample_meta = meta_nonempty(ddf._meta)
    sample_meta = _run(sample_meta)
    ddf = ddf.map_partitions(_run, meta=sample_meta.head(0))

    return ddf


def _apply_binning(
    ddf: dd.DataFrame,
    feat: FunctionConfig,
) -> dd.DataFrame:
    """Apply numeric binning to create categorical bins.

    Creates bins from continuous values using either fixed edges or quantile-based
    breakpoints. Supports custom labels for each bin.

    Args:
        ddf: Dask DataFrame.
        feat: FunctionConfig with:
            - source_col: Numeric column to bin (required)
            - out_col: Output column name (default: <source>_bin)
            - bins: Custom edges like [0, 18, 35, 55, 100] (optional)
            - quantiles: Quantile breakpoints like [0.33, 0.66] (default)
            - labels: Custom labels like ['Young', 'Middle', 'Senior'] (optional)
            - right: Close bins on right (default: True)
            - include_lowest: Include lowest value (default: True)

    Returns:
        DataFrame with new binned column as string type.

    Examples:
        # Age bins with custom edges:
        type: binning
        source_col: Age
        bins: [0, 18, 35, 55, 100]
        labels: ['Young', 'Adult', 'Middle-aged', 'Senior']

        # Income tertiles (quantile-based):
        type: binning
        source_col: Income
        quantiles: [0.33, 0.66]
    """
    source_col = _resolve_source_col(ddf, feat, err_name="Binning")

    out_col = feat.out_col or _build_out_col(source_col, feat.output_suffix)
    out_col = _to_snake(out_col)

    edges = _resolve_binning_edges(ddf, source_col, feat)
    labels = _resolve_binning_labels(edges, feat.labels)

    def _format_interval(interval) -> str | pd.NA:
        if pd.isna(interval):
            return pd.NA
        if isinstance(interval, pd.Interval):
            left = interval.left
            right = interval.right
            return f"{left}-{right}"
        return str(interval)

    def _bin_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        out = pdf.copy()
        cut = pd.cut(
            out[source_col],
            bins=edges,
            labels=labels,
            include_lowest=feat.include_lowest,
            right=feat.right,
            duplicates="drop",
        )
        if labels is None:
            cut = cut.map(_format_interval)
        out[out_col] = cut.astype(str)
        return out

    meta = ddf._meta.copy()
    meta[out_col] = pd.Series(dtype="object")
    ddf = ddf.map_partitions(_bin_partition, meta=meta)

    return ddf


def _apply_winsorize(
    ddf: dd.DataFrame,
    feat: FunctionConfig,
) -> dd.DataFrame:
    source_col = _resolve_source_col(ddf, feat, err_name="Winsorize")
    out_col = feat.out_col or f"{source_col}_winsorized"
    out_col = _to_snake(out_col)

    series = _to_numeric_series(ddf, source_col)
    lower = feat.lower_bound
    upper = feat.upper_bound
    quantiles = []
    if lower is None and feat.lower_quantile is not None:
        quantiles.append(float(feat.lower_quantile))
    if upper is None and feat.upper_quantile is not None:
        quantiles.append(float(feat.upper_quantile))

    if quantiles:
        q_vals = series.quantile(sorted(set(quantiles))).compute()
        if isinstance(q_vals, pd.Series):
            if lower is None and feat.lower_quantile in q_vals.index:
                lower = float(q_vals.loc[feat.lower_quantile])
            if upper is None and feat.upper_quantile in q_vals.index:
                upper = float(q_vals.loc[feat.upper_quantile])

    if lower is None and upper is None:
        raise ValueError("winsorize requires at least one bound or quantile")

    def _winsorize_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        out = pdf.copy()
        values = pd.to_numeric(out[source_col], errors="coerce")
        out[out_col] = values.clip(lower=lower, upper=upper)
        return out

    meta = ddf._meta.copy()
    meta[out_col] = pd.Series(dtype="float64")
    return ddf.map_partitions(_winsorize_partition, meta=meta)


def _apply_log_transform(
    ddf: dd.DataFrame,
    feat: FunctionConfig,
) -> dd.DataFrame:
    source_col = _resolve_source_col(ddf, feat, err_name="Log transform")
    out_col = feat.out_col or f"{source_col}_log"
    out_col = _to_snake(out_col)

    def _log_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        out = pdf.copy()
        values = pd.to_numeric(out[source_col], errors="coerce")
        if feat.log_clip_min is not None:
            values = values.clip(lower=feat.log_clip_min)
        if feat.log_clip_max is not None:
            values = values.clip(upper=feat.log_clip_max)
        values = values + float(feat.log_offset or 0.0)
        if feat.log_method == "log":
            mask = values > 0
            logged = np.where(mask, np.log(values), np.nan)
            if feat.log_on_nonpositive == "zero":
                logged = np.where(mask, logged, 0.0)
        else:
            mask = values > -1
            logged = np.where(mask, np.log1p(values), np.nan)
            if feat.log_on_nonpositive == "zero":
                logged = np.where(mask, logged, 0.0)
        out[out_col] = logged
        return out

    meta = ddf._meta.copy()
    meta[out_col] = pd.Series(dtype="float64")
    return ddf.map_partitions(_log_partition, meta=meta)


def _apply_date_parts(
    ddf: dd.DataFrame,
    feat: FunctionConfig,
) -> dd.DataFrame:
    source_col = _resolve_source_col(ddf, feat, err_name="Date parts")
    prefix = _to_snake(feat.out_col or source_col)
    parts = _normalize_date_parts(feat.date_parts)
    unit = _resolve_timestamp_unit(ddf, source_col, feat.timestamp_unit)
    if feat.auto_daypart:
        if _detect_has_time(ddf, source_col, unit):
            if "hour" not in parts:
                parts.append("hour")
            if "daypart" not in parts:
                parts.append("daypart")
    parts = _order_date_parts(parts)

    def _date_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        out = pdf.copy()
        dt = _parse_datetime_series(
            out[source_col],
            unit=unit,
            timezone=feat.timezone,
        )
        if "year" in parts:
            out[f"{prefix}_year"] = dt.dt.year.astype("Int64")
        if "quarter" in parts:
            out[f"{prefix}_quarter"] = dt.dt.quarter.astype("Int64")
        if "month" in parts:
            out[f"{prefix}_month"] = dt.dt.month.astype("Int64")
        if "day" in parts:
            out[f"{prefix}_day"] = dt.dt.day.astype("Int64")
        if "hour" in parts:
            out[f"{prefix}_hour"] = dt.dt.hour.astype("Int64")
        if "day_of_week" in parts:
            out[f"{prefix}_day_of_week"] = dt.dt.weekday.astype("Int64")
        if "day_of_year" in parts:
            out[f"{prefix}_day_of_year"] = dt.dt.dayofyear.astype("Int64")
        if "week_of_year" in parts:
            out[f"{prefix}_week_of_year"] = dt.dt.isocalendar().week.astype("Int64")
        if "daypart" in parts:
            out[f"{prefix}_daypart"] = _build_daypart(
                dt.dt.hour,
                feat.daypart_bins,
            )
        if "is_weekend" in parts:
            out[f"{prefix}_is_weekend"] = (dt.dt.weekday >= 5).astype("boolean")
        if "is_month_start" in parts:
            out[f"{prefix}_is_month_start"] = dt.dt.is_month_start.astype("boolean")
        if "is_month_end" in parts:
            out[f"{prefix}_is_month_end"] = dt.dt.is_month_end.astype("boolean")
        if "is_quarter_start" in parts:
            out[f"{prefix}_is_quarter_start"] = dt.dt.is_quarter_start.astype("boolean")
        if "is_quarter_end" in parts:
            out[f"{prefix}_is_quarter_end"] = dt.dt.is_quarter_end.astype("boolean")
        if "is_year_start" in parts:
            out[f"{prefix}_is_year_start"] = dt.dt.is_year_start.astype("boolean")
        if "is_year_end" in parts:
            out[f"{prefix}_is_year_end"] = dt.dt.is_year_end.astype("boolean")
        if feat.drop_source:
            out = out.drop(columns=[source_col])
        return out

    meta = ddf._meta.copy()
    for part in parts:
        col = f"{prefix}_{part}"
        if part.startswith("is_"):
            meta[col] = pd.Series(dtype="boolean")
        elif part == "daypart":
            meta[col] = pd.Series(dtype="object")  # object for distributed compat
        else:
            meta[col] = pd.Series(dtype="Int64")
    if feat.drop_source and source_col in meta.columns:
        meta = meta.drop(columns=[source_col])
    return ddf.map_partitions(_date_partition, meta=meta)


def _apply_categorical_bucket(
    ddf: dd.DataFrame,
    feat: FunctionConfig,
) -> dd.DataFrame:
    source_col = _resolve_source_col(ddf, feat, err_name="Categorical bucket")
    out_col = feat.out_col or f"{source_col}_bucket"
    out_col = _to_snake(out_col)

    series = ddf[source_col].astype(
        "object"
    )  # object dtype serializes cleanly in distributed
    counts = series.value_counts().compute()
    keep = None
    if feat.top_k:
        keep = set(counts.nlargest(int(feat.top_k)).index.astype(str))
    if feat.min_count:
        min_keep = set(counts[counts >= int(feat.min_count)].index.astype(str))
        keep = min_keep if keep is None else keep.intersection(min_keep)

    keep = keep or set()
    other_label = feat.other_label or "other"

    def _bucket_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        out = pdf.copy()
        vals = out[source_col].astype(str)
        bucketed = vals.where(vals.isin(keep), other_label)
        bucketed = bucketed.where(~vals.isna(), pd.NA)
        out[out_col] = bucketed
        return out

    meta = ddf._meta.copy()
    meta[out_col] = pd.Series(dtype="object")  # object for distributed compat
    return ddf.map_partitions(_bucket_partition, meta=meta)


def _apply_string_normalize(
    ddf: dd.DataFrame,
    feat: FunctionConfig,
) -> dd.DataFrame:
    """Standardize string text by cleaning whitespace and casing."""
    source_col = _resolve_source_col(ddf, feat, err_name="String normalize")
    out_col = feat.out_col or f"{source_col}_normalized"
    out_col = _to_snake(out_col)

    def _normalize_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        out = pdf.copy()
        series = out[source_col].astype(str)
        if feat.strip:
            series = series.str.strip()
        if feat.replace_regex:
            series = series.str.replace(
                feat.replace_regex,
                feat.replace_with or "",
                regex=True,
            )
        if feat.collapse_whitespace:
            series = series.str.replace(r"\s+", " ", regex=True)
        if feat.string_case == "lower":
            series = series.str.lower()
        elif feat.string_case == "upper":
            series = series.str.upper()
        elif feat.string_case == "title":
            series = series.str.title()
        out[out_col] = series
        return out

    meta = ddf._meta.copy()
    meta[out_col] = pd.Series(dtype="object")  # object for distributed compat
    return ddf.map_partitions(_normalize_partition, meta=meta)


def _apply_frequency_encode(
    ddf: dd.DataFrame,
    feat: FunctionConfig,
) -> dd.DataFrame:
    """Encode categorical values as counts or frequencies."""
    source_col = _resolve_source_col(ddf, feat, err_name="Frequency encode")
    out_col = feat.out_col or f"{source_col}_freq"
    out_col = _to_snake(out_col)

    series = ddf[source_col].astype(
        "object"
    )  # object dtype serializes cleanly in distributed
    counts = series.value_counts(dropna=True).compute()
    total = int(ddf.map_partitions(len).sum().compute())
    if feat.normalize and total:
        freq_map = (counts / total).to_dict()
    else:
        freq_map = counts.to_dict()

    def _freq_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        out = pdf.copy()
        vals = out[source_col].astype(str)
        out[out_col] = vals.map(freq_map)
        return out

    meta = ddf._meta.copy()
    meta[out_col] = pd.Series(dtype="float64")
    return ddf.map_partitions(_freq_partition, meta=meta)


def _apply_days_since(
    ddf: dd.DataFrame,
    feat: FunctionConfig,
) -> dd.DataFrame:
    """Compute days between a timestamp column and a reference date."""
    source_col = _resolve_source_col(ddf, feat, err_name="Days since")
    out_col = feat.out_col or f"{source_col}_days_since"
    out_col = _to_snake(out_col)
    unit = _resolve_timestamp_unit(ddf, source_col, feat.timestamp_unit)

    def _days_since_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        out = pdf.copy()
        dt = _parse_datetime_series(
            out[source_col],
            unit=unit,
            timezone=feat.timezone,
        )
        ref = _resolve_reference_timestamp(dt, feat.reference_date, feat.timezone)
        delta = ref - dt
        out[out_col] = delta.dt.total_seconds() / 86400.0
        return out

    meta = ddf._meta.copy()
    meta[out_col] = pd.Series(dtype="float64")
    return ddf.map_partitions(_days_since_partition, meta=meta)


def _apply_ratio(
    ddf: dd.DataFrame,
    feat: FunctionConfig,
) -> dd.DataFrame:
    numerator_col = (
        feat.numerator_col
        or feat.source_col
        or (feat.inputs[0] if feat.inputs else None)
    )
    denominator_col = feat.denominator_col or (
        feat.inputs[1] if len(feat.inputs) > 1 else None
    )
    if not numerator_col or not denominator_col:
        raise ValueError("ratio requires numerator/denominator or inputs")

    numerator_col = _resolve_existing_column(ddf, numerator_col)
    denominator_col = _resolve_existing_column(ddf, denominator_col)
    if numerator_col is None or denominator_col is None:
        raise KeyError("ratio inputs not found in dataframe")

    out_col = feat.out_col or f"{numerator_col}_per_{denominator_col}"
    out_col = _to_snake(out_col)

    def _ratio_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        out = pdf.copy()
        num = pd.to_numeric(out[numerator_col], errors="coerce")
        den = pd.to_numeric(out[denominator_col], errors="coerce")
        if feat.on_zero == "epsilon":
            den = den.replace(0, feat.epsilon)
            ratio = num / den
        else:
            ratio = num / den
            if feat.on_zero == "zero":
                ratio = ratio.where(den != 0, 0.0)
            else:
                ratio = ratio.where(den != 0)
        out[out_col] = ratio
        return out

    meta = ddf._meta.copy()
    meta[out_col] = pd.Series(dtype="float64")
    return ddf.map_partitions(_ratio_partition, meta=meta)


def _resolve_binning_edges(
    ddf: dd.DataFrame, source_col: str, feat: FunctionConfig
) -> list[float]:
    if feat.bins:
        edges = list(feat.bins)
    else:
        quantiles = list(feat.quantiles or [0.2, 0.4, 0.6, 0.8])
        quantiles = [q for q in quantiles if 0 < q < 1]
        quantiles = sorted(set(quantiles))
        if not quantiles:
            raise ValueError("binning requires bins or non-empty quantiles")
        qs = [0.0] + quantiles + [1.0]
        q_vals = ddf[source_col].quantile(qs).compute()
        edges = [float(v) for v in q_vals.values]
    edges = sorted(set(edges))
    if len(edges) < 2:
        raise ValueError("binning requires at least two distinct edges")
    return edges


def _resolve_source_col(
    ddf: dd.DataFrame,
    feat: FunctionConfig,
    *,
    err_name: str,
) -> str:
    source_col = feat.source_col or (feat.inputs[0] if feat.inputs else None)
    if not source_col:
        raise ValueError(f"{err_name} requires source_col or inputs")
    resolved = _resolve_existing_column(ddf, source_col)
    if resolved is None:
        raise KeyError(f"{err_name} source_col '{source_col}' not found")
    return resolved


def _to_numeric_series(ddf: dd.DataFrame, col: str) -> dd.Series:
    return ddf[col].map_partitions(pd.to_numeric, errors="coerce")


def _normalize_date_parts(parts: Sequence[str] | None) -> list[str]:
    defaults = [
        "year",
        "quarter",
        "month",
        "day",
        "hour",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "daypart",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
    ]
    if parts is None:
        return [p for p in defaults if p not in {"hour", "daypart"}]
    normalized = []
    for part in parts:
        key = _to_snake(part)
        if key == "weekday":
            key = "day_of_week"
        if key == "week":
            key = "week_of_year"
        normalized.append(key)
    out = [p for p in normalized if p in set(defaults)]
    if not out:
        raise ValueError("date_parts must include at least one supported part")
    return out


def _order_date_parts(parts: Sequence[str]) -> list[str]:
    order = {
        "year": 0,
        "quarter": 1,
        "month": 2,
        "day": 3,
        "hour": 4,
        "day_of_week": 5,
        "day_of_year": 6,
        "week_of_year": 7,
        "daypart": 8,
        "is_weekend": 9,
        "is_month_start": 10,
        "is_month_end": 11,
        "is_quarter_start": 12,
        "is_quarter_end": 13,
        "is_year_start": 14,
        "is_year_end": 15,
    }
    return sorted(parts, key=lambda p: order.get(p, 999))


def _resolve_timestamp_unit(
    ddf: dd.DataFrame,
    source_col: str,
    unit: str | None,
) -> str | None:
    if unit is None or unit == "auto":
        if not is_numeric_dtype(ddf[source_col].dtype):
            return None
        sample = ddf[source_col].dropna().quantile(0.5).compute()
        if pd.isna(sample):
            return None
        sample = float(sample)
        if sample > 1.0e17:
            return "ns"
        if sample > 1.0e14:
            return "us"
        if sample > 1.0e11:
            return "ms"
        if sample > 1.0e9:
            return "s"
        return None
    return unit


def _detect_has_time(
    ddf: dd.DataFrame,
    source_col: str,
    unit: str | None,
    sample_rows: int = 1000,
) -> bool:
    try:
        sample = ddf[source_col].dropna().head(sample_rows, compute=True)
    except Exception:
        return False
    if sample.empty:
        return False
    if unit:
        dt = pd.to_datetime(sample, errors="coerce", unit=unit, utc=True)
    else:
        dt = pd.to_datetime(sample, errors="coerce", format="mixed")
    if dt.isna().all():
        return False
    return bool(
        (dt.dt.hour != 0).any()
        or (dt.dt.minute != 0).any()
        or (dt.dt.second != 0).any()
    )


def _parse_datetime_series(
    series: pd.Series,
    *,
    unit: str | None,
    timezone: str | None,
) -> pd.Series:
    if unit:
        dt = pd.to_datetime(series, errors="coerce", unit=unit, utc=True)
    else:
        dt = pd.to_datetime(series, errors="coerce", format="mixed")

    if timezone:
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(timezone)
        else:
            dt = dt.dt.tz_convert(timezone)
    return dt


def _resolve_reference_timestamp(
    series: pd.Series,
    reference_date: str | None,
    timezone: str | None,
) -> pd.Timestamp:
    tzinfo = getattr(series.dt, "tz", None)
    if reference_date:
        if tzinfo is not None:
            ref = pd.to_datetime(reference_date, errors="coerce", utc=True)
            if ref.tzinfo is None:
                ref = ref.tz_localize(tzinfo)
            else:
                ref = ref.tz_convert(tzinfo)
        else:
            ref = pd.to_datetime(reference_date, errors="coerce")
        return ref
    if tzinfo is not None:
        return pd.Timestamp.now(tz=tzinfo).normalize()
    if timezone:
        return pd.Timestamp.now(tz=timezone).normalize()
    return pd.Timestamp.now().normalize()


def _build_daypart(
    hours: pd.Series,
    bins: dict[str, Sequence[int]] | None,
) -> pd.Series:
    bins = bins or DEFAULT_DAYPART_BINS
    labels = []
    conditions = []
    for label, bounds in bins.items():
        start, end = bounds[0], bounds[1]
        if start < end:
            cond = (hours >= start) & (hours < end)
        else:
            cond = (hours >= start) | (hours < end)
        conditions.append(cond)
        labels.append(label)
    return pd.Series(np.select(conditions, labels, default=pd.NA), index=hours.index)


def _resolve_binning_labels(
    edges: Sequence[float], labels: Sequence[str] | None
) -> Sequence[str] | None:
    if labels is None:
        return None
    if len(labels) != len(edges) - 1:
        raise ValueError("labels length must match number of bins")
    return labels


def _build_out_col(source_col: str, suffix: str | None) -> str:
    suffix = suffix or "_bin"
    return f"{source_col}{suffix}"


def _load_callable(path: str):
    if ":" not in path:
        raise ValueError("callable must be in 'module:function' form")
    mod_name, fn_name = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn


def _validate_inputs(ddf: dd.DataFrame, inputs: Iterable[str]) -> None:
    for col in inputs or []:
        resolved = _resolve_existing_column(ddf, col)
        if resolved is None:
            raise KeyError(f"required input column '{col}' not found")


def _resolve_existing_column(ddf: dd.DataFrame, col: str) -> str | None:
    if col in ddf.columns:
        return col
    snake = _to_snake(col)
    if snake in ddf.columns:
        return snake
    if "_on_" in snake:
        alt = snake.replace("_on_", "_")
        if alt in ddf.columns:
            return alt
    return None


def _rename_new_cols_snake(
    ddf: dd.DataFrame, before_cols: Sequence[str]
) -> tuple[dd.DataFrame, list[str]]:
    new_cols = [c for c in ddf.columns if c not in before_cols]
    if not new_cols:
        return ddf, []

    mapping = {c: _to_snake(c) for c in new_cols}
    targets = list(mapping.values())
    if len(set(targets)) != len(targets):
        raise ValueError("snake_case conversion produced duplicate column names")

    existing = set(ddf.columns) - set(new_cols)
    for src, dst in mapping.items():
        if dst in existing:
            raise ValueError(
                f"snake_case output '{dst}' would overwrite existing column"
            )

    if any(src != dst for src, dst in mapping.items()):
        ddf = ddf.rename(columns=mapping)
        new_cols = [mapping[c] for c in new_cols]

    return ddf, new_cols


def _apply_return_mode(
    ddf: dd.DataFrame,
    *,
    return_mode: str,
    return_columns: Sequence[str],
    id_cols: Sequence[str],
    new_cols: Sequence[str],
    verbose: bool,
) -> dd.DataFrame:
    if return_mode == "all":
        return ddf

    if return_mode == "new_only":
        keep = list(id_cols) + list(new_cols)
    elif return_mode == "list":
        resolved = [_resolve_existing_column(ddf, c) for c in (return_columns or [])]
        keep = list(id_cols) + [c for c in resolved if c]
    else:
        raise ValueError(f"unsupported return_mode '{return_mode}'")

    keep = _dedupe_keep(keep, ddf.columns)
    if verbose:
        logger.info("[functions] return_mode=%s keep=%s", return_mode, keep)
    return ddf[keep]


def _dedupe_keep(keep: Sequence[str], columns: Iterable[str]) -> list[str]:
    cols = set(columns)
    out: list[str] = []
    seen: set[str] = set()
    for col in keep:
        if col in cols and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def list_builtin_functions() -> dict[str, str]:
    """Return builtin function docs for agent discovery."""
    return dict(BUILTIN_FUNCTION_DOCS)


__all__ = [
    "apply_functions",
    "apply_feature_functions",
    "apply_kpi_functions",
    "resolve_id_columns",
    "list_builtin_functions",
]
