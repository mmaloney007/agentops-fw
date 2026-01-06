#!/usr/bin/env python3
"""
preprocess.py (Dask)
--------------------
Preprocessing + ZSML helpers

Purpose:
    - Apply optional feature_functions (partition-wise), then KPI functions, then core preprocessing
      (rename→bool-fix→missing→drop rules), and finally drop configured columns.
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

import math
import re
import unicodedata
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import compute as dask_compute
from dask.dataframe.utils import meta_nonempty

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_step(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def _log_head(ddf: dd.DataFrame, cols: list[str], verbose: bool, label: str) -> None:
    if not verbose or not cols:
        return
    try:
        sample = ddf[cols].head(5, compute=True)
        _log_step(f"[debug] {label} sample (first 5 rows):\n{sample}", True)
    except Exception:
        _log_step(f"[debug] {label} sample unavailable (head() failed)", True)


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
    verbose: bool = False,
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
            f"batched quantile() for {len(num_cols)} numeric cols",
            verbose,
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
        f"(sample_rows={sample_rows:,})",
        verbose,
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
    fill_numbers_with: str = "median",  # "median" | "zero"
    add_flags: bool = False,
    verbose: bool = True,
    big_row_threshold: int = 5_000_000,
    numeric_sample_rows: int = 500_000,
    progress_every: int = 25,
) -> tuple[dd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Memory-efficient missing report + optional fills with progress logging.
    """
    if row_count is None:
        try:
            row_count = int(ddf.map_partitions(len).sum().compute())
        except Exception:
            row_count = int(ddf.shape[0].compute())

    if dtypes is None:
        dtypes = ddf.dtypes

    if null_counts is None:
        _log_step("[missing] computing null counts via isna().sum()...", verbose)
        null_counts = ddf.isna().sum().compute()

    total_rows = int(row_count)
    if total_rows == 0:
        report_df = pd.DataFrame(
            columns=["column", "dtype", "null_count", "pct_null", "suggest_fill"]
        )
        _log_step("⚠️ DataFrame is empty; nothing to fill.", verbose)
        return ddf, report_df, {}

    _log_step(f"[missing] start: rows≈{total_rows:,}, cols={len(ddf.columns)}", verbose)

    cols_with_nulls = [c for c, n in null_counts.items() if n > 0]
    _log_step(f"[missing] {len(cols_with_nulls)} column(s) with ≥1 null", verbose)

    if not cols_with_nulls:
        report_df = pd.DataFrame(
            columns=["column", "dtype", "null_count", "pct_null", "suggest_fill"]
        )
        _log_step("✅ No missing values detected.", verbose)
        return ddf, report_df, {}

    num_cols = [c for c in cols_with_nulls if pd.api.types.is_numeric_dtype(dtypes[c])]
    dt_cols = [
        c for c in cols_with_nulls if pd.api.types.is_datetime64_any_dtype(dtypes[c])
    ]
    str_cols = [c for c in cols_with_nulls if c not in num_cols and c not in dt_cols]

    _log_step(
        f"[missing] split by type: numeric={len(num_cols)}, "
        f"datetime={len(dt_cols)}, string/other={len(str_cols)}",
        verbose,
    )

    medians: Dict[str, float] = {}
    if fill and num_cols and fill_numbers_with == "median":
        medians = _compute_numeric_medians_bigaware(
            ddf,
            num_cols,
            total_rows=total_rows,
            big_row_threshold=big_row_threshold,
            sample_rows=numeric_sample_rows,
            verbose=verbose,
        )

    string_modes: dict[str, Any] = {}
    if fill and str_cols and fill_strings_with == "mode":
        _log_step(
            f"[missing/modes] computing string modes for {len(str_cols)} columns...",
            verbose,
        )
        for i, col in enumerate(str_cols, start=1):
            if verbose and (i == 1 or i % progress_every == 0 or i == len(str_cols)):
                _log_step(f"[missing/modes] {i}/{len(str_cols)}: {col}", verbose)
            vc = ddf[col].dropna().value_counts().head(1).compute()
            if len(vc):
                string_modes[col] = vc.index[0]
            else:
                string_modes[col] = fill_strings_const

    suggestions: list[dict[str, Any]] = []
    fill_map: dict[str, Any] = {}
    flag_cols: list[str] = []

    _log_step(
        f"[missing] building fill_map over {len(cols_with_nulls)} column(s)...",
        verbose,
    )

    for i, col in enumerate(cols_with_nulls, start=1):
        nulls = int(null_counts[col])
        pct = round((nulls / total_rows) * 100, 2)
        dtype = dtypes[col]

        suggest_fill: Any = None

        if fill:
            if pd.api.types.is_numeric_dtype(dtype):
                if fill_numbers_with == "zero" or nulls == total_rows:
                    fill_val = 0
                else:
                    fill_val = medians.get(col, 0.0)
                fill_map[col] = fill_val
                suggest_fill = fill_val

            elif pd.api.types.is_datetime64_any_dtype(dtype):
                fill_val = pd.NaT

            else:
                if fill_strings_with == "const":
                    fill_val = fill_strings_const
                else:
                    fill_val = string_modes.get(col, fill_strings_const)
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

        if verbose and (i == 1 or i % progress_every == 0 or i == len(cols_with_nulls)):
            _log_step(
                f"[missing] processed {i}/{len(cols_with_nulls)} (last={col}, "
                f"nulls={nulls}, pct={pct}%)",
                verbose,
            )

    out = ddf

    if fill and fill_map:
        _log_step(
            f"[missing] applying fills for {len(fill_map)} column(s) via .fillna(...)",
            verbose,
        )
        out = out.fillna(fill_map)

    if add_flags and flag_cols:
        _log_step(
            f"[missing] adding missing flags for {len(flag_cols)} column(s)...",
            verbose,
        )
        flag_exprs = {f"{c}_is_missing": out[c].isna() for c in flag_cols}
        out = out.assign(**flag_exprs)

    report_df = pd.DataFrame(suggestions).sort_values("pct_null", ascending=False)
    if verbose:
        with pd.option_context("display.max_colwidth", None, "display.width", 200):
            print(report_df.to_markdown(index=False))

    _log_step("[missing] done.", verbose)

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
    fill_strings_with: str = "const",
    fill_strings_const: str = "Unknown",
    fill_numbers_with: str = "median",
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
    _log_step(
        f"[preprocess] start: rows≈unknown, cols={len(ddf.columns)}",
        verbose,
    )

    _log_step("[preprocess] step 1/5: renaming columns to snake_case...", verbose)
    out = rename_columns_snake_ddf(ddf)
    _log_step(f"[preprocess] rename done → {len(out.columns)} columns", verbose)

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
                f"[preprocess] early drop_columns → {len(set(to_drop))} column(s): {sorted(set(to_drop))}",
                verbose,
            )

    if apply_boolfix:
        obj_like = [
            c for c in out.columns if str(out.dtypes[c]) in {"object", "category"}
        ]
        if boolfix_cols is not None:
            obj_cols = [c for c in obj_like if c in boolfix_cols]
        else:
            obj_cols = obj_like

        _log_step(
            f"[preprocess] step 2/5: bool-fix on {len(obj_cols)} object/category cols",
            verbose,
        )
        if obj_cols:
            out = out.map_partitions(_boolfix_partition, obj_cols, meta=out)
    else:
        _log_step("[preprocess] step 2/5: bool-fix disabled", verbose)

    if sanitize_values:
        obj_all = [c for c in out.columns if str(out.dtypes[c]) == "object"]
        if sanitize_cols is not None:
            obj_cols = [c for c in obj_all if c in sanitize_cols]
        else:
            obj_cols = obj_all

        _log_step(
            f"[preprocess] step 3/5: sanitizing {len(obj_cols)} object cols",
            verbose,
        )
        if verbose and len(obj_cols) > 100:
            _log_step(
                f"[preprocess] WARNING: {len(obj_cols)} object cols; "
                f"consider restricting sanitize_cols=[...]",
                verbose,
            )
        if obj_cols:
            out = out.map_partitions(_clean_value_partition, obj_cols, meta=out)
    else:
        _log_step("[preprocess] step 3/5: sanitize disabled", verbose)

    _log_step(
        "[preprocess] computing non-null counts (for nulls & drop_empty)...", verbose
    )
    counts = out.count().compute()
    if not isinstance(counts, pd.Series):
        counts = pd.Series(counts, index=out.columns)
    row_count = int(counts.max()) if len(counts) else 0
    null_counts = pd.Series(
        {c: int(row_count - counts.get(c, 0)) for c in out.columns}, index=out.columns
    )
    dtypes = out.dtypes

    _log_step(
        f"[preprocess] inferred rows≈{row_count:,}, cols={len(out.columns)}",
        verbose,
    )

    _log_step("[preprocess] step 4/5: missing-value report and fills...", verbose)
    out, _, fill_map = missing_report_and_fill_dask(
        out,
        row_count=row_count,
        null_counts=null_counts,
        dtypes=dtypes,
        fill=fill_missing,
        fill_strings_with=fill_strings_with,
        fill_strings_const=fill_strings_const,
        fill_numbers_with=fill_numbers_with,
        add_flags=add_missing_flags,
        verbose=verbose,
        big_row_threshold=big_row_threshold,
        numeric_sample_rows=numeric_sample_rows,
    )
    _log_step(
        f"[preprocess] missing-value phase done; filled {len(fill_map)} column(s)",
        verbose,
    )

    dropped: List[str] = []

    if drop_empty:
        _log_step("[preprocess] step 5/5a: dropping empty columns...", verbose)
        empty_cols = counts[counts == 0].index.tolist()
        dropped += empty_cols
        _log_step(f"[preprocess] drop_empty → {len(empty_cols)} column(s)", verbose)
    else:
        _log_step("[preprocess] step 5/5a: drop_empty disabled", verbose)

    if drop_constants or drop_every_value_is_unique:
        if drop_constants and row_count <= 1:
            _log_step(
                "[preprocess] step 5/5b: drop_constants skipped (row_count<=1)",
                verbose,
            )
            drop_constants = False

        _log_step(
            "[preprocess] step 5/5b: computing approx uniques for drop rules...",
            verbose,
        )
        uniq_obj = out.nunique_approx()
        unique_counts = uniq_obj.compute()
        if not isinstance(unique_counts, pd.Series):
            unique_counts = pd.Series(unique_counts, index=out.columns)
        unique_counts = unique_counts.astype(int)

        if drop_constants:
            const_cols = unique_counts[unique_counts == 1].index.tolist()
            dropped += const_cols
            _log_step(
                f"[preprocess] drop_constants → {len(const_cols)} column(s)",
                verbose,
            )

        if drop_every_value_is_unique and row_count > 0:
            uniq_cols = unique_counts[unique_counts >= row_count].index.tolist()
            dropped += uniq_cols
            _log_step(
                f"[preprocess] drop_every_value_is_unique → {len(uniq_cols)} column(s)",
                verbose,
            )
    else:
        _log_step("[preprocess] step 5/5b: constants/unique drops disabled", verbose)

    dropped = sorted(set(dropped))
    if dropped:
        out = out.drop(columns=dropped)
        _log_step(
            f"[preprocess] total dropped columns: {len(dropped)} → {dropped}",
            verbose,
        )
    else:
        _log_step("[preprocess] no columns dropped in step 5", verbose)

    _log_step("[preprocess] ✅ complete.", verbose)
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
    if np.issubdtype(series.dtype, np.number):
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

    if clip_high_quantile is not None:
        min_val, max_val, q_vals, cap_val, count_pos = dask_compute(
            positive.min(),
            positive.max(),
            positive.quantile(list(quantiles)),
            positive.quantile(clip_high_quantile),
            positive.count(),
        )
    else:
        min_val, max_val, q_vals, count_pos = dask_compute(
            positive.min(),
            positive.max(),
            positive.quantile(list(quantiles)),
            positive.count(),
        )
        cap_val = None

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
        cap = float(cap_val)
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
        cap = float(cap_val)
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
# Public wrapper with feature_functions + KPI functions + preprocessing
# ---------------------------------------------------------------------------
def preprocess(ddf, cfg):
    """
    Apply feature_functions -> KPI functions -> preprocessing -> drop_columns.
    """
    hooks = getattr(cfg, "feature_functions", []) or []
    import importlib

    verbose = getattr(cfg.logging, "level", "info") == "debug"

    orig_cols = list(ddf.columns)

    if hooks:
        funcs = []
        for hook in hooks:
            mod_name, fn_name = hook.split(":", 1)
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            funcs.append(fn)

        def _run_all(pdf: pd.DataFrame) -> pd.DataFrame:
            out = pdf.copy()
            for fn in funcs:
                out = fn(out)
            return out

        sample_meta = meta_nonempty(ddf._meta)
        sample_meta = _run_all(sample_meta)
        ddf = ddf.map_partitions(_run_all, meta=sample_meta.head(0))

        before_cols = set(orig_cols)
        after_cols = set(ddf.columns)
        new_cols = sorted(after_cols - before_cols)
        _log_step(
            f"[feature_functions] applied {len(funcs)} hook(s); new columns: {new_cols or 'none'}",
            verbose,
        )
        # Show id cols and newly added columns, if present
        id_candidates = getattr(cfg.input, "id_cols", []) or []
        id_snake = [_to_snake(c) for c in id_candidates]
        cols_for_head = [
            c for c in id_candidates + id_snake + new_cols if c in ddf.columns
        ][:10]
        _log_head(ddf, cols_for_head, verbose, "after feature_functions")

    pre_cfg = getattr(cfg, "preprocessing", None) or getattr(cfg, "cleaning", None)

    # KPI functions (supports multiple ZSML definitions or cleaning.zsml)
    kpi_funcs = []
    # Back-compat: preprocessing/cleaning.zsml enabled
    zsml_cfg = getattr(pre_cfg, "zsml", None)
    if getattr(zsml_cfg, "enabled", False) and getattr(zsml_cfg, "source_col", None):
        kpi_funcs.append(
            {
                "type": "zsml",
                "source_col": zsml_cfg.source_col,
                "out_col": zsml_cfg.out_col,
                "zero_threshold": zsml_cfg.zero_threshold,
                "quantiles": zsml_cfg.quantiles,
                "clip_high_quantile": zsml_cfg.clip_high_quantile,
                "unit": zsml_cfg.unit,
                "range_style": zsml_cfg.range_style,
                "add_prefix": zsml_cfg.add_prefix,
            }
        )
    # New: kpi_functions list
    for k in getattr(cfg, "kpi_functions", []) or []:
        if getattr(k, "type", "zsml") == "zsml":
            kpi_funcs.append(
                {
                    "type": "zsml",
                    "source_col": k.source_col,
                    "out_col": k.out_col,
                    "zero_threshold": k.zero_threshold,
                    "quantiles": k.quantiles,
                    "clip_high_quantile": k.clip_high_quantile,
                    "unit": k.unit,
                    "range_style": k.range_style,
                    "add_prefix": k.add_prefix,
                }
            )

    for kpi in kpi_funcs:
        source_col = kpi["source_col"]
        source_col_snake = _to_snake(source_col)
        # Preprocessing (rename to snake_case) happens later; accept either form here.
        if source_col not in ddf.columns and source_col_snake in ddf.columns:
            source_col = source_col_snake
        if source_col not in ddf.columns:
            raise KeyError(
                f"ZSML source_col '{kpi['source_col']}' not found "
                f"(tried '{kpi['source_col']}' and '{source_col_snake}')"
            )

        tier_col = kpi["out_col"] or f"{source_col}_tier"
        tier_col = _to_snake(tier_col)

        model = fit_zsml_edges_dask(
            ddf[source_col],
            zero_threshold=kpi["zero_threshold"],
            quantiles=kpi["quantiles"],
            clip_high_quantile=kpi["clip_high_quantile"],
        )
        tier_series = apply_zsml_dask(
            ddf[source_col],
            model,
            label_ranges=kpi["range_style"] in {"math", "text"},
            add_prefix=kpi["add_prefix"],
            unit=kpi["unit"],
            range_style=kpi["range_style"],
        )
        ddf = ddf.assign(**{tier_col: tier_series})
        _log_step(
            f"[kpi] zsml applied on '{source_col}' -> '{tier_col}' "
            f"(quantiles={kpi['quantiles']}, zero_threshold={kpi['zero_threshold']})",
            verbose,
        )
        # Show id + source + tier columns
        id_candidates = getattr(cfg.input, "id_cols", []) or []
        id_snake = [_to_snake(c) for c in id_candidates]
        cols_for_head = [
            c
            for c in id_candidates + id_snake + [source_col, tier_col]
            if c in ddf.columns
        ][:10]
        _log_head(ddf, cols_for_head, verbose, f"kpi '{tier_col}'")

    # Core preprocessing
    ddf = preprocess_dask_scaled(
        ddf,
        drop_columns=getattr(cfg, "drop_columns", []),
        fill_missing=getattr(pre_cfg, "missing_fill", "auto") == "auto",
        drop_constants=getattr(pre_cfg, "drop_constant", True),
        drop_every_value_is_unique=False,
        drop_empty=getattr(pre_cfg, "drop_empty", True),
        sanitize_values=False,
        apply_boolfix=getattr(pre_cfg, "bool_fix", True),
        verbose=verbose,
    )

    # Drop configured columns after preprocessing (support original and snake_case forms)
    drops = getattr(cfg, "drop_columns", []) or []
    if drops:
        to_drop = []
        for col in drops:
            snake = _to_snake(col)
            if col in ddf.columns:
                to_drop.append(col)
            elif snake in ddf.columns:
                to_drop.append(snake)
        if to_drop:
            ddf = ddf.drop(columns=sorted(set(to_drop)))
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
]


# Keep name for compatibility with earlier imports
def suggest_null_fills_dask(ddf, **kwargs):
    return missing_report_and_fill_dask(ddf, **kwargs)
