#!/usr/bin/env python3
"""
preprocess.py (Dask)
--------------------
Preprocessing + ZSML helpers

Purpose:
    - Apply core preprocessing only (rename→bool-fix→missing→drop rules).
    - Generate missing-value reports and fills, add missing flags, and drop empty/constant/unique-only columns.
    - Fit/apply ZSML tiers for KPIs and produce summary reports.

Usage:
    from neuralift_c360_prep.preprocess import preprocess, preprocess_dask_scaled
    ddf = preprocess(ddf, cfg)
    ddf = preprocess_dask_scaled(ddf, fill_missing=True, apply_boolfix=True)

Dependencies:
    - dask[dataframe]
    - pandas
    - numpy
    - uuid6 (indirect via ingest pipeline hooks)

Author: Mike Maloney - Neuralift, Inc.
Updated: 2025-12-08
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import compute as dask_compute

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_step(msg: str) -> None:
    logger.info(msg)


def _log_head(ddf: dd.DataFrame, cols: list[str], debug: bool, label: str) -> None:
    if not debug or not cols:
        return
    try:
        sample = ddf[cols].head(5, compute=True)
        logger.debug("[debug] %s sample (first 5 rows):\n%s", label, sample)
    except Exception:
        logger.debug("[debug] %s sample unavailable (head() failed)", label)


# ---------------------------------------------------------------------------
# Column/value policies
# ---------------------------------------------------------------------------
_COL_SANITISE_RE = re.compile(r"[^\w]+")
_VALUE_CLEAN_RE = re.compile(r"[\x00-\x1F\x7F]")
_ALLOWED_VALUE_CHR = "-+&$%./:<>();,:'\"?@#|±€£•*≤≥!=[]"
_TRUTHY: Set[str] = {"true", "t", "yes", "y", "1", "on"}
_FALSY: Set[str] = {"false", "f", "no", "n", "0", "off"}
_BOOL_MAX_UNIQUES = 3


# ---------------------------------------------------------------------------
# Column rename
# ---------------------------------------------------------------------------
def _to_snake(name: str, max_len: int = 64) -> str:
    """CamelCase + punctuation → ASCII snake_case, clipped to max_len."""
    if pd.isna(name):
        return "_"
    txt = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    txt = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", txt)
    txt = _COL_SANITISE_RE.sub("_", txt).strip("_").lower()
    if txt and txt[0].isdigit():
        txt = f"_{txt}"
    return (txt or "_")[:max_len]


def rename_columns_snake_ddf(
    ddf: dd.DataFrame, max_len: int = 64, dedupe: bool = True
) -> dd.DataFrame:
    """Rename columns lazily; optional dedupe adds _1, _2 on collisions."""
    mapping = {c: _to_snake(c, max_len=max_len) for c in ddf.columns}
    if dedupe:
        seen: Dict[str, int] = {}
        for old, new in mapping.items():
            seen[new] = seen.get(new, -1) + 1
            if seen[new] > 0:
                mapping[old] = f"{new}_{seen[new]}"
    return ddf.rename(columns=mapping)


# ---------------------------------------------------------------------------
# Value sanitization (partition-wise)
# ---------------------------------------------------------------------------
def _clean_value_partition(pdf: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for c in columns:
        s = pdf[c]
        mask = s.notna()
        cleaned = (
            s[mask]
            .astype(str)
            .str.replace(_VALUE_CLEAN_RE, "", regex=True)
            .str.replace(f"[^{re.escape(_ALLOWED_VALUE_CHR)}\\w\\s]", "", regex=True)
            .str.strip()
            .str.replace(r"\s{2,}", " ", regex=True)
        )
        pdf.loc[mask, c] = cleaned
    return pdf


# ---------------------------------------------------------------------------
# Bool fix (partition-wise)
# ---------------------------------------------------------------------------
def _bool_token(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, Number):
        return {1: True, 0: False}.get(int(val), None)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in _TRUTHY:
            return True
        if s in _FALSY:
            return False
    return None


def _boolfix_partition(pdf: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    for col in candidate_cols:
        if pdf[col].dropna().empty:
            continue
        tokens = pdf[col].dropna().astype(str).str.strip().str.lower().unique()
        if 0 < len(tokens) <= _BOOL_MAX_UNIQUES and set(tokens) <= (_TRUTHY | _FALSY):
            mapped = pdf[col].map(_bool_token)
            if not mapped.isna().any():  # only if all map cleanly
                pdf[col] = mapped.astype("boolean")
    return pdf


# ---------------------------------------------------------------------------
# Numeric medians (big-table aware)
# ---------------------------------------------------------------------------
def _compute_numeric_medians_bigaware(
    ddf: dd.DataFrame,
    num_cols: list[str],
    *,
    total_rows: int,
    big_row_threshold: int = 5_000_000,
    sample_rows: int = 500_000,
) -> Dict[str, float]:
    """
    Compute medians for numeric columns that have nulls.
    """
    medians: Dict[str, float] = {}
    if not num_cols:
        return medians

    if total_rows <= big_row_threshold:
        _log_step(
            f"[missing/medians] small table (rows≈{total_rows:,}), "
            f"batched quantile() for {len(num_cols)} numeric cols"
        )
        q = ddf[num_cols].quantile(0.5).compute()
        for col in num_cols:
            try:
                medians[col] = float(q[col])
            except Exception:
                medians[col] = 0.0
        return medians

    _log_step(
        f"[missing/medians] big table (rows≈{total_rows:,}), "
        f"shared sample for {len(num_cols)} numeric cols "
        f"(sample_rows={sample_rows:,})"
    )

    frac = min(1.0, sample_rows / total_rows if total_rows > 0 else 1.0)
    sample_ddf = ddf[num_cols].sample(frac=frac, random_state=42)
    sample_pdf = sample_ddf.compute()

    for col in num_cols:
        try:
            medians[col] = float(sample_pdf[col].median())
        except Exception:
            medians[col] = 0.0

    return medians


# ---------------------------------------------------------------------------
# Missing report + optional fills
# ---------------------------------------------------------------------------
def missing_report_and_fill_dask(
    ddf: dd.DataFrame,
    *,
    row_count: int | None = None,
    null_counts: pd.Series | None = None,
    dtypes: pd.Series | None = None,
    fill: bool = False,
    fill_strings_with: str = "const",  # "const" | "mode"
    fill_strings_const: str = "Unknown",
    fill_numbers_with: str = "median",  # "median" | "mean" | "zero" | number
    add_flags: bool = False,
    verbose: bool = True,
    big_row_threshold: int = 5_000_000,
    numeric_sample_rows: int = 500_000,
    progress_every: int = 25,
    fill_overrides: dict[str, Any] | None = None,  # NEW: per-column overrides
) -> tuple[dd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Memory-efficient missing report + optional fills with progress logging.
    """
    debug = verbose
    if row_count is None:
        try:
            row_count = int(ddf.map_partitions(len).sum().compute())
        except Exception:
            row_count = int(ddf.shape[0].compute())

    if dtypes is None:
        dtypes = ddf.dtypes

    if null_counts is None:
        _log_step("[missing] computing null counts via isna().sum()...")
        null_counts = ddf.isna().sum().compute()

    total_rows = int(row_count)
    if total_rows == 0:
        report_df = pd.DataFrame(
            columns=["column", "dtype", "null_count", "pct_null", "suggest_fill"]
        )
        logger.warning("⚠️ DataFrame is empty; nothing to fill.")
        return ddf, report_df, {}

    _log_step(f"[missing] start: rows≈{total_rows:,}, cols={len(ddf.columns)}")

    cols_with_nulls = [c for c, n in null_counts.items() if n > 0]
    _log_step(f"[missing] {len(cols_with_nulls)} column(s) with ≥1 null")

    if not cols_with_nulls:
        report_df = pd.DataFrame(
            columns=["column", "dtype", "null_count", "pct_null", "suggest_fill"]
        )
        _log_step("✅ No missing values detected.")
        return ddf, report_df, {}

    num_cols = [c for c in cols_with_nulls if pd.api.types.is_numeric_dtype(dtypes[c])]
    dt_cols = [
        c for c in cols_with_nulls if pd.api.types.is_datetime64_any_dtype(dtypes[c])
    ]
    str_cols = [c for c in cols_with_nulls if c not in num_cols and c not in dt_cols]

    _log_step(
        f"[missing] split by type: numeric={len(num_cols)}, "
        f"datetime={len(dt_cols)}, string/other={len(str_cols)}"
    )

    # Handle per-column overrides
    overrides = fill_overrides or {}

    medians: Dict[str, float] = {}
    means: Dict[str, float] = {}
    # Determine which numeric columns need median vs mean
    num_cols_for_median = [
        c for c in num_cols if c not in overrides and fill_numbers_with == "median"
    ]
    num_cols_for_mean = [
        c for c in num_cols if c not in overrides and fill_numbers_with == "mean"
    ]

    if fill and num_cols_for_median:
        medians = _compute_numeric_medians_bigaware(
            ddf,
            num_cols_for_median,
            total_rows=total_rows,
            big_row_threshold=big_row_threshold,
            sample_rows=numeric_sample_rows,
        )

if fill and num_cols_for_mean:
        _log_step(
            f"[missing/means] computing means for {len(num_cols_for_mean)} numeric cols",
            verbose,
        )
        mean_vals = ddf[num_cols_for_mean].mean().compute()
        for col in num_cols_for_mean:
            try:
                means[col] = float(mean_vals[col])
            except Exception:
                means[col] = 0.0

    # Filter string cols that don't have overrides for mode calculation
    str_cols_for_mode = [
        c for c in str_cols if c not in overrides and fill_strings_with == "mode"
    ]

    string_modes: dict[str, Any] = {}
    if fill and str_cols_for_mode:
        _log_step(
            f"[missing/modes] computing string modes for {len(str_cols_for_mode)} columns (batched)...",
            verbose,
        )
        # OPTIMIZATION: Batch all value_counts computations into a single dask_compute call
        # This reduces scheduler round-trips from O(n) to O(1)
        mode_tasks = [ddf[col].dropna().value_counts().head(1) for col in str_cols_for_mode]
        mode_results = dask_compute(*mode_tasks)

        for col, vc in zip(str_cols_for_mode, mode_results):
            if len(vc):
                string_modes[col] = vc.index[0]
            else:
                string_modes[col] = fill_strings_const
        _log_step(f"[missing/modes] computed modes for {len(str_cols_for_mode)} columns")

    suggestions: list[dict[str, Any]] = []
    fill_map: dict[str, Any] = {}
    flag_cols: list[str] = []

    _log_step(f"[missing] building fill_map over {len(cols_with_nulls)} column(s)...")

    for i, col in enumerate(cols_with_nulls, start=1):
        nulls = int(null_counts[col])
        pct = round((nulls / total_rows) * 100, 2)
        dtype = dtypes[col]

        suggest_fill: Any = None

        if fill:
            # Check for per-column override first
            if col in overrides:
                fill_val = overrides[col]
                fill_map[col] = fill_val
                suggest_fill = fill_val
            elif pd.api.types.is_numeric_dtype(dtype):
                if nulls == total_rows:
                    fill_val = 0
                elif fill_numbers_with == "median":
                    fill_val = medians.get(col, 0.0)
                elif fill_numbers_with == "mean":
                    fill_val = means.get(col, 0.0)
                elif isinstance(fill_numbers_with, (int, float)):
                    fill_val = fill_numbers_with
                else:
                    # "zero" or fallback
                    fill_val = 0
                fill_map[col] = fill_val
                suggest_fill = fill_val

            elif pd.api.types.is_datetime64_any_dtype(dtype):
                fill_val = pd.NaT

            else:
                if fill_strings_with == "const":
                    fill_val = fill_strings_const
                elif fill_strings_with == "mode":
                    fill_val = string_modes.get(col, fill_strings_const)
                else:
                    # Custom string value
                    fill_val = fill_strings_with
                fill_map[col] = fill_val
                suggest_fill = fill_val

        if add_flags:
            flag_cols.append(col)

        suggestions.append(
            dict(
                column=col,
                dtype=str(dtype),
                null_count=int(nulls),
                pct_null=pct,
                suggest_fill=suggest_fill,
            )
        )

        if i == 1 or i % progress_every == 0 or i == len(cols_with_nulls):
            _log_step(
                f"[missing] processed {i}/{len(cols_with_nulls)} (last={col}, "
                f"nulls={nulls}, pct={pct}%)"
            )

    out = ddf

    if fill and fill_map:
        fill_rows = []
        for col, value in fill_map.items():
            fill_rows.append(
                {
                    "column": col,
                    "null_count": int(null_counts.get(col, 0)),
                    "fill_value": value,
                }
            )
        fill_df = pd.DataFrame(fill_rows).sort_values("null_count", ascending=False)
        if verbose:
            _log_step("[missing] fill summary:", verbose)
            with pd.option_context("display.max_colwidth", None, "display.width", 200):
                print(fill_df.to_markdown(index=False))
        else:
            logger.info(
                "[missing] fill summary:\n%s",
                fill_df.to_string(index=False),
            )
        _log_step(
            f"[missing] applying fills for {len(fill_map)} column(s) via .fillna(...)"
        )
        out = out.fillna(fill_map)

    if add_flags and flag_cols:
        _log_step(f"[missing] adding missing flags for {len(flag_cols)} column(s)...")
        flag_exprs = {f"{c}_is_missing": out[c].isna() for c in flag_cols}
        out = out.assign(**flag_exprs)

    report_df = pd.DataFrame(suggestions).sort_values("pct_null", ascending=False)
    if debug:
        with pd.option_context("display.max_colwidth", None, "display.width", 200):
            logger.debug("\n%s", report_df.to_markdown(index=False))

    _log_step("[missing] done.")

    return out, report_df, fill_map


# ---------------------------------------------------------------------------
# Main scalable preprocess
# ---------------------------------------------------------------------------
def preprocess_dask_scaled(
    ddf: dd.DataFrame,
    *,
    sanitize_values: bool = False,  # partition-wise string clean; safe
    sanitize_cols: Iterable[str] | None = None,
    apply_boolfix: bool = True,
    boolfix_cols: Iterable[str] | None = None,
    drop_columns: Iterable[str] | None = None,
    fill_missing: bool = False,
    fill_strings_with: str = "const",  # "const" | "mode" | custom string
    fill_strings_const: str = "Unknown",
    fill_numbers_with: str
    | int
    | float = "median",  # "median" | "mean" | "zero" | number
    fill_overrides: dict[str, Any] | None = None,  # NEW: per-column overrides
    add_missing_flags: bool = False,
    drop_empty: bool = True,
    drop_constants: bool = False,
    drop_every_value_is_unique: bool = False,
    big_row_threshold: int = 5_000_000,
    numeric_sample_rows: int = 500_000,
    verbose: bool = False,
) -> dd.DataFrame:
    """
    Memory- and compute-efficient preprocess with step logging.
    """
    debug = verbose
    _log_step(f"[preprocess] start: rows≈unknown, cols={len(ddf.columns)}")

    _log_step("[preprocess] step 1/5: renaming columns to snake_case...")
    out = rename_columns_snake_ddf(ddf)
    _log_step(f"[preprocess] rename done → {len(out.columns)} columns")

    # Drop configured columns early (after rename) to avoid work on soon-to-be-dropped cols
    pre_drop = list(drop_columns) if drop_columns else []
    if pre_drop:
        to_drop = []
        for col in pre_drop:
            snake = _to_snake(col)
            if col in out.columns:
                to_drop.append(col)
            elif snake in out.columns:
                to_drop.append(snake)
        if to_drop:
            out = out.drop(columns=sorted(set(to_drop)))
            _log_step(
                f"[preprocess] early drop_columns → {len(set(to_drop))} column(s): {sorted(set(to_drop))}"
            )

    # OPTIMIZATION: Fuse bool-fix and sanitize into a single map_partitions call
    # to reduce task overhead when both are enabled
    boolfix_obj_cols: list[str] = []
    sanitize_obj_cols: list[str] = []

    if apply_boolfix:
        obj_like = [
            c for c in out.columns if str(out.dtypes[c]) in {"object", "category"}
        ]
        if boolfix_cols is not None:
            boolfix_obj_cols = [c for c in obj_like if c in boolfix_cols]
        else:
            boolfix_obj_cols = obj_like
        _log_step(
            f"[preprocess] step 2/5: bool-fix on {len(boolfix_obj_cols)} object/category cols"
        )
    else:
        _log_step("[preprocess] step 2/5: bool-fix disabled")

    if sanitize_values:
        obj_all = [c for c in out.columns if str(out.dtypes[c]) == "object"]
        if sanitize_cols is not None:
            sanitize_obj_cols = [c for c in obj_all if c in sanitize_cols]
        else:
            sanitize_obj_cols = obj_all
        _log_step(
            f"[preprocess] step 3/5: sanitizing {len(sanitize_obj_cols)} object cols"
        )
        if len(sanitize_obj_cols) > 100:
            logger.warning(
                "[preprocess] WARNING: %s object cols; consider restricting sanitize_cols=[...]",
                len(sanitize_obj_cols),
            )
    else:
        _log_step("[preprocess] step 3/5: sanitize disabled")

    # Apply fused partition function if both operations are needed
    if boolfix_obj_cols and sanitize_obj_cols:

        def _fused_boolfix_and_sanitize(
            pdf: pd.DataFrame, bf_cols: list[str], san_cols: list[str]
        ) -> pd.DataFrame:
            pdf = _boolfix_partition(pdf, bf_cols)
            pdf = _clean_value_partition(pdf, san_cols)
            return pdf

        out = out.map_partitions(
            _fused_boolfix_and_sanitize, boolfix_obj_cols, sanitize_obj_cols, meta=out
        )
    elif boolfix_obj_cols:
        out = out.map_partitions(_boolfix_partition, boolfix_obj_cols, meta=out)
    elif sanitize_obj_cols:
        out = out.map_partitions(_clean_value_partition, sanitize_obj_cols, meta=out)

    _log_step("[preprocess] computing non-null counts (for nulls & drop_empty)...")
    counts = out.count().compute()
    if not isinstance(counts, pd.Series):
        counts = pd.Series(counts, index=out.columns)
    row_count = int(counts.max()) if len(counts) else 0
    null_counts = pd.Series(
        {c: int(row_count - counts.get(c, 0)) for c in out.columns}, index=out.columns
    )
    dtypes = out.dtypes

    _log_step(f"[preprocess] inferred rows≈{row_count:,}, cols={len(out.columns)}")

    _log_step("[preprocess] step 4/5: missing-value report and fills...")
    out, _, fill_map = missing_report_and_fill_dask(
        out,
        row_count=row_count,
        null_counts=null_counts,
        dtypes=dtypes,
        fill=fill_missing,
        fill_strings_with=fill_strings_with,
        fill_strings_const=fill_strings_const,
        fill_numbers_with=fill_numbers_with,
        fill_overrides=fill_overrides,
        add_flags=add_missing_flags,
        verbose=debug,
        big_row_threshold=big_row_threshold,
        numeric_sample_rows=numeric_sample_rows,
    )
    _log_step(
        f"[preprocess] missing-value phase done; filled {len(fill_map)} column(s)"
    )

    dropped: List[str] = []

    if drop_empty:
        _log_step("[preprocess] step 5/5a: dropping empty columns...")
        empty_cols = counts[counts == 0].index.tolist()
        dropped += empty_cols
        _log_step(f"[preprocess] drop_empty → {len(empty_cols)} column(s)")
    else:
        _log_step("[preprocess] step 5/5a: drop_empty disabled")

    if drop_constants or drop_every_value_is_unique:
        if drop_constants and row_count <= 1:
            _log_step("[preprocess] step 5/5b: drop_constants skipped (row_count<=1)")
            drop_constants = False

        _log_step("[preprocess] step 5/5b: computing approx uniques for drop rules...")
        uniq_obj = out.nunique_approx()
        unique_counts = uniq_obj.compute()
        if not isinstance(unique_counts, pd.Series):
            unique_counts = pd.Series(unique_counts, index=out.columns)
        unique_counts = unique_counts.astype(int)

        if drop_constants:
            const_cols = unique_counts[unique_counts == 1].index.tolist()
            dropped += const_cols
            _log_step(f"[preprocess] drop_constants → {len(const_cols)} column(s)")

        if drop_every_value_is_unique and row_count > 0:
            uniq_cols = unique_counts[unique_counts >= row_count].index.tolist()
            dropped += uniq_cols
            _log_step(
                f"[preprocess] drop_every_value_is_unique → {len(uniq_cols)} column(s)"
            )
    else:
        _log_step("[preprocess] step 5/5b: constants/unique drops disabled")

    dropped = sorted(set(dropped))
    if dropped:
        out = out.drop(columns=dropped)
        _log_step(f"[preprocess] total dropped columns: {len(dropped)} → {dropped}")
    else:
        _log_step("[preprocess] no columns dropped in step 5")

    _log_step("[preprocess] ✅ complete.")
    return out


# ---------------------------------------------------------------------------
# ZSML helpers
# ---------------------------------------------------------------------------
RangeStyle = str  # kept loose; values "math" | "text"
Bounds = Tuple[Optional[int], Optional[int]]

LABEL_PREFIXES = {"zero": "0. ", "small": "1. ", "medium": "2. ", "large": "3. "}
_ZSML_PAT = re.compile(r"^(zero|small|medium|large)\b", re.IGNORECASE)


def _fmt_int_commas(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def _fmt_text_numbers_commas(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"{int(match.group(0)):,}"

    return re.sub(r"(?<!\d)(-?\d+)(?!\d)", repl, str(text))


def _format_range(
    lo: Optional[int], hi: Optional[int], *, unit: str = "", style: RangeStyle = "math"
) -> str:
    if style not in ("math", "text"):
        raise ValueError("`style` must be either 'math' or 'text'.")
    unit = (unit or "").strip()
    prefix = unit if unit in {"$", "£", "€"} else ""
    suffix = "" if prefix else unit
    if suffix and not suffix.startswith(" "):
        suffix = " " + suffix

    def fmt(val: Optional[int]) -> str:
        if val is None:
            return ""
        return f"{prefix}{_fmt_int_commas(val)}{suffix}"

    if style == "text":
        if lo is None and hi is None:
            return "—"
        if lo is None:
            return f"Up to {fmt(hi)}".strip()
        if hi is None:
            return f"{fmt(lo)} or more".strip()
        if hi < lo:
            hi = lo
        return f"{fmt(lo)} to {fmt(hi)}".strip()

    if lo is None and hi is None:
        return "—"
    if lo is None:
        return f"≤{fmt(hi)}"
    lower_bound = f">{fmt(lo - 1)}"
    if hi is None:
        return lower_bound
    if hi < lo:
        hi = lo
    upper_bound = f"≤{fmt(hi)}"
    return f"{lower_bound} – {upper_bound}"


def _to_num_dd(series: dd.Series) -> dd.Series:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.astype("float64")
    return series.map_partitions(pd.to_numeric, errors="coerce", meta=("x", "float64"))


def _zsml_boundaries(model: Dict[str, Any]) -> Dict[str, Bounds]:
    zero_threshold = model.get("zero_threshold", 0.0) or 0.0
    if not model.get("has_positive"):
        zero_hi = int(math.floor(zero_threshold))
        return {
            "zero": (None, zero_hi),
            "small": (zero_hi + 1, None),
            "medium": (None, None),
            "large": (None, None),
        }
    zero_hi = int(math.floor(zero_threshold))
    q1 = model["q1"]
    q2 = model["q2"]
    min_pos = model["min_pos"]

    small_lo = max(int(math.ceil(min_pos)), zero_hi + 1)
    small_hi = max(int(math.floor(q1)), small_lo)
    medium_lo = small_hi + 1
    medium_hi = max(int(math.floor(q2)), medium_lo)
    large_lo = medium_hi + 1

    return {
        "zero": (None, zero_hi),
        "small": (small_lo, small_hi),
        "medium": (medium_lo, medium_hi),
        "large": (large_lo, None),
    }


def _fit_zsml_edges_pandas(
    s: pd.Series,
    *,
    zero_threshold: float = 0.0,
    quantiles: tuple[float, float] = (1 / 3, 2 / 3),
    clip_high_quantile: Optional[float] = None,
) -> Dict[str, Any]:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    positive = s[s > zero_threshold].dropna()

    count_val = int(positive.shape[0])
    if count_val == 0:
        return {
            "zero_threshold": zero_threshold,
            "has_positive": False,
            "q1": None,
            "q2": None,
            "min_pos": None,
            "max_pos": None,
        }

    if clip_high_quantile is not None:
        cap_val = float(positive.quantile(clip_high_quantile))
        positive = positive.clip(upper=cap_val)

    min_pos_f = float(positive.min())
    max_pos_f = float(positive.max())
    q_vals = positive.quantile(list(quantiles))
    q1_f, q2_f = float(q_vals.iloc[0]), float(q_vals.iloc[1])

    if not (min_pos_f < q1_f < q2_f < max_pos_f):
        edges = np.linspace(min_pos_f, max(max_pos_f, min_pos_f + 1e-9), 4)
        q1_f, q2_f = float(edges[1]), float(edges[2])

    return {
        "zero_threshold": zero_threshold,
        "has_positive": True,
        "q1": q1_f,
        "q2": q2_f,
        "min_pos": min_pos_f,
        "max_pos": max_pos_f,
    }


def _fit_zsml_edges_numeric_dask(
    numeric: dd.Series,
    *,
    zero_threshold: float = 0.0,
    quantiles: tuple[float, float] = (1 / 3, 2 / 3),
    clip_high_quantile: Optional[float] = None,
    strict_clip: bool = False,
) -> Dict[str, Any]:
    positive = numeric[numeric > zero_threshold].dropna()

    # OPTIMIZATION: Batch all quantile computations into a single dask_compute call
    # This reduces graph traversal overhead significantly
    all_quantiles = list(quantiles)
    if clip_high_quantile is not None:
        all_quantiles.append(clip_high_quantile)

    min_val, max_val, all_q_vals, count_pos = dask_compute(
        positive.min(),
        positive.max(),
        positive.quantile(all_quantiles),
        positive.count(),
    )

    # Extract individual quantile values from the batched result
    q_vals = all_q_vals.iloc[: len(quantiles)]
    cap_val = float(all_q_vals.iloc[-1]) if clip_high_quantile is not None else None

    if count_pos == 0:
        return {
            "zero_threshold": zero_threshold,
            "has_positive": False,
            "q1": None,
            "q2": None,
            "min_pos": None,
            "max_pos": None,
        }

    min_pos_f = float(min_val)
    max_pos_f = float(max_val)
    q1_f, q2_f = float(q_vals.iloc[0]), float(q_vals.iloc[1])

    if cap_val is not None and strict_clip:
        cap = cap_val
        positive_clipped = positive.clip(upper=cap)
        min_val2, max_val2, q_vals2 = dask_compute(
            positive_clipped.min(),
            positive_clipped.max(),
            positive_clipped.quantile(list(quantiles)),
        )
        min_pos_f = float(min_val2)
        max_pos_f = float(max_val2)
        q1_f, q2_f = float(q_vals2.iloc[0]), float(q_vals2.iloc[1])
    elif cap_val is not None and not strict_clip:
        cap = cap_val
        if cap < max_pos_f:
            max_pos_f = cap
        if q2_f > max_pos_f:
            q2_f = max_pos_f
        if q1_f > q2_f:
            mid = zero_threshold + (q2_f - zero_threshold) / 2.0
            q1_f = max(zero_threshold, mid)

    if not (min_pos_f < q1_f < q2_f < max_pos_f):
        edges = np.linspace(min_pos_f, max(max_pos_f, min_pos_f + 1e-9), 4)
        q1_f, q2_f = float(edges[1]), float(edges[2])

    return {
        "zero_threshold": zero_threshold,
        "has_positive": True,
        "q1": q1_f,
        "q2": q2_f,
        "min_pos": min_pos_f,
        "max_pos": max_pos_f,
    }


def fit_zsml_edges_dask(
    values: dd.Series,
    *,
    zero_threshold: float = 0.0,
    quantiles: tuple[float, float] = (1 / 3, 2 / 3),
    clip_high_quantile: Optional[float] = None,
    small_n_threshold: int = 200_000,
    strict_clip: bool = False,
) -> Dict[str, Any]:
    numeric = _to_num_dd(values).replace([np.inf, -np.inf], np.nan)

    try:
        total_count = int(numeric.map_partitions(len).compute().sum())
    except Exception:
        total_count = None

    if total_count is not None and total_count <= small_n_threshold:
        s = numeric.compute()
        return _fit_zsml_edges_pandas(
            s,
            zero_threshold=zero_threshold,
            quantiles=quantiles,
            clip_high_quantile=clip_high_quantile,
        )

    return _fit_zsml_edges_numeric_dask(
        numeric,
        zero_threshold=zero_threshold,
        quantiles=quantiles,
        clip_high_quantile=clip_high_quantile,
        strict_clip=strict_clip,
    )


def apply_zsml_dask(
    values: dd.Series,
    model: Dict[str, Any],
    *,
    label_ranges: bool = True,
    add_prefix: bool = False,
    unit: str = "",
    range_style: RangeStyle = "math",
) -> dd.Series:
    bounds = _zsml_boundaries(model)
    tiers = {}
    for key, name in (
        ("zero", "Zero"),
        ("small", "Small"),
        ("medium", "Medium"),
        ("large", "Large"),
    ):
        prefix = LABEL_PREFIXES.get(key, "") if add_prefix else ""
        if label_ranges:
            body = (
                f"{name} ({_format_range(*bounds[key], unit=unit, style=range_style)})"
            )
        else:
            body = name
        tiers[key] = prefix + body

    numeric = _to_num_dd(values).fillna(0)

    if not model.get("has_positive"):
        return numeric.map_partitions(
            lambda pdf: pd.Series(tiers["zero"], index=pdf.index),
            meta=("tier", "object"),
        )

    z = float(model["zero_threshold"])
    q1 = float(model["q1"])
    q2 = float(model["q2"])

    def _label_partition(pdf: pd.Series) -> pd.Series:
        out = pd.Series(index=pdf.index, dtype="object")
        out[pdf <= z] = tiers["zero"]
        out[(pdf > z) & (pdf <= q1)] = tiers["small"]
        out[(pdf > q1) & (pdf <= q2)] = tiers["medium"]
        out[pdf > q2] = tiers["large"]
        return out

    return numeric.map_partitions(_label_partition, meta=("tier", "object"))


def zsml_report_dask(
    ddf: dd.DataFrame,
    tier_col: str,
    *,
    format_commas: bool = True,
    add_prefix: bool = False,
) -> pd.DataFrame:
    counts = ddf[tier_col].value_counts(dropna=False).compute()
    total = int(counts.sum()) or 1
    report = counts.rename_axis("tier").reset_index(name="count")
    report["share"] = (report["count"] / total * 100).round(1).astype(str) + "%"

    tier_text = report["tier"].astype(str)
    base = tier_text.str.extract(_ZSML_PAT, expand=False).str.lower()
    paren = tier_text.str.extract(r"\(([^)]*)\)", expand=False)
    report["range"] = np.where(paren.notna(), paren.str.strip(), "—")
    report.loc[base == "zero", "range"] = report.loc[base == "zero", "range"].replace(
        "—", "≤0"
    )

    order_map = {"zero": 0, "small": 1, "medium": 2, "large": 3}
    report = report.assign(
        _order=base.map(order_map).fillna(99).astype(int), _base=base
    )
    report = report.sort_values(["_order", "tier"]).drop(columns="_order")

    if format_commas:
        report["count"] = report["count"].map(_fmt_int_commas)
        report["range"] = report["range"].map(_fmt_text_numbers_commas)
        report["tier"] = report["tier"].map(_fmt_text_numbers_commas)

    if add_prefix:

        def _ensure_prefix(row):
            base_key = row["_base"]
            prefix = LABEL_PREFIXES.get(base_key)
            tier_value = row["tier"]
            if prefix and not tier_value.startswith(prefix):
                return prefix + tier_value
            return tier_value

        report["tier"] = report.apply(_ensure_prefix, axis=1)

    report = report.drop(columns="_base")
    return report[["tier", "range", "count", "share"]]


def ensure_zsml_kpi_dask(
    ddf: dd.DataFrame,
    source_col: str,
    *,
    out_col: Optional[str] = None,
    zero_threshold: float = 0.0,
    quantiles: tuple[float, float] = (1 / 3, 2 / 3),
    clip_high_quantile: Optional[float] = None,
    label_ranges: bool = True,
    add_prefix: bool = False,
    unit: str = "",
    range_style: RangeStyle = "math",
    small_n_threshold: int = 200_000,
    strict_clip: bool = False,
) -> dict:
    series = ddf[source_col]
    model = fit_zsml_edges_dask(
        series,
        zero_threshold=zero_threshold,
        quantiles=quantiles,
        clip_high_quantile=clip_high_quantile,
        small_n_threshold=small_n_threshold,
        strict_clip=strict_clip,
    )
    target_col = out_col or f"{source_col}_TIER"
    ddf[target_col] = apply_zsml_dask(
        series,
        model,
        label_ranges=label_ranges,
        add_prefix=add_prefix,
        unit=unit,
        range_style=range_style,
    )
    return {"model": model, "out_col": target_col}


# ---------------------------------------------------------------------------
# Public wrapper with core preprocessing only
# ---------------------------------------------------------------------------
def preprocess(ddf, cfg):
    """
    Apply core preprocessing only (rename, bool-fix, fill, drop empty/constant).
    Functions and drop-columns are applied in the pipeline stage.
    """
verbose = getattr(cfg.logging, "level", "info") == "debug"
    debug = verbose  # alias for compatibility
    pre_cfg = getattr(cfg, "preprocessing", None) or getattr(cfg, "cleaning", None)

    # Determine whether fill was explicitly configured
    fields_set = getattr(pre_cfg, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(pre_cfg, "__pydantic_fields_set__", set())
    fill_explicit = "fill" in (fields_set or set())

    fill_cfg = getattr(pre_cfg, "fill", None)

    # Determine fill settings from new FillConfig or legacy missing_fill
    fill_enabled = True
    fill_categorical = "Unknown"
    fill_continuous: str | int | float = "median"
    fill_overrides: dict = {}

if fill_explicit and fill_cfg is not None:
        fill_categorical = getattr(fill_cfg, "categorical", "Unknown") or "Unknown"
        fill_continuous = getattr(fill_cfg, "continuous", "median") or "median"
        fill_overrides = dict(getattr(fill_cfg, "overrides", {}) or {})
    else:
        legacy_fill = getattr(pre_cfg, "missing_fill", "auto")
        fill_enabled = legacy_fill == "auto"

    ddf = preprocess_dask_scaled(
        ddf,
        drop_columns=None,
        fill_missing=fill_enabled,
        fill_strings_with="const" if fill_categorical not in ("mode",) else "mode",
        fill_strings_const=fill_categorical
        if fill_categorical != "mode"
        else "Unknown",
        fill_numbers_with=fill_continuous,
        fill_overrides=fill_overrides,
        drop_constants=getattr(pre_cfg, "drop_constant", True),
        drop_every_value_is_unique=False,
        drop_empty=getattr(pre_cfg, "drop_empty", True),
        sanitize_values=False,
        apply_boolfix=getattr(pre_cfg, "bool_fix", True),
        verbose=debug,
    )

    return ddf


def drop_configured_columns(
    ddf: dd.DataFrame,
    drop_columns: Iterable[str] | None,
    *,
    verbose: bool = False,
) -> dd.DataFrame:
    """Drop configured columns after functions have run."""
    drops = list(drop_columns or [])
    if not drops:
        return ddf

    to_drop = []
    for col in drops:
        snake = _to_snake(col)
        if col in ddf.columns:
            to_drop.append(col)
        elif snake in ddf.columns:
            to_drop.append(snake)
    if to_drop:
        ddf = ddf.drop(columns=sorted(set(to_drop)))
        _log_step(
            f"[drop] drop_columns → {len(set(to_drop))} column(s): {sorted(set(to_drop))}",
            verbose,
        )
    return ddf


__all__ = [
    "rename_columns_snake_ddf",
    "suggest_null_fills_dask",
    "preprocess_dask_scaled",
    "fit_zsml_edges_dask",
    "apply_zsml_dask",
    "zsml_report_dask",
    "ensure_zsml_kpi_dask",
    "preprocess",
    "drop_configured_columns",
]


# Keep name for compatibility with earlier imports
def suggest_null_fills_dask(ddf, **kwargs):
    return missing_report_and_fill_dask(ddf, **kwargs)
