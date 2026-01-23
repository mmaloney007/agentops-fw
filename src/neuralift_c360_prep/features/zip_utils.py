#!/usr/bin/env python3
"""
Dask-friendly ZIP cleaning helpers.

Provides:
    - clean_zip_codes_dask: add ZIP5 column with standardized 5-digit ZIPs.
    - add_distance_indicators_dask: merge distance/duration metrics produced offline.
Intended for use as pre-hooks (map_partitions).
"""

from __future__ import annotations

import pandas as pd


def clean_zip_codes_dask(
    df: pd.DataFrame, zip_column: str = "ZIP_Code"
) -> pd.DataFrame:
    """
    Add ZIP5 column with standardized 5-digit ZIPs.

    Handles both original column names (ZIP_Code) and snake_case (zip_code)
    since preprocessing may run before this function.
    """
    df = df.copy()
    # Try the specified column first, then common snake_case variants
    zip_col_candidates = [zip_column, zip_column.lower(), "zip_code", "zipcode", "zip"]
    actual_col = None
    for candidate in zip_col_candidates:
        if candidate in df.columns:
            actual_col = candidate
            break

    if actual_col is not None:
        df["zip5"] = (
            df[actual_col]
            .astype(str)
            .str.extract(r"(\d{1,5})", expand=False)
            .str.zfill(5)
        )
        # Convert to bool explicitly to avoid pandas FutureWarning about fillna downcasting
        valid = df["zip5"].str.fullmatch(r"[0-9]{5}").astype("boolean").fillna(False)
        df.loc[~valid | (df["zip5"] == "00000"), "zip5"] = pd.NA
    return df


def add_distance_indicators_dask(
    df: pd.DataFrame,
    distance_df: pd.DataFrame,
    on: str = "zip5",
    keep_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Merge precomputed distance metrics by zip5; optional column subset."""
    merged = df.merge(distance_df, on=on, how="left")
    if keep_cols:
        cols = [c for c in keep_cols if c in merged.columns]
        return merged[cols]
    return merged


__all__ = ["clean_zip_codes_dask", "add_distance_indicators_dask"]
