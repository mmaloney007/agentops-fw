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
    df = df.copy()
    if zip_column in df.columns:
        df["ZIP5"] = (
            df[zip_column]
            .astype(str)
            .str.extract(r"(\d{1,5})", expand=False)
            .str.zfill(5)
        )
        # Convert to bool explicitly to avoid pandas FutureWarning about fillna downcasting
        valid = df["ZIP5"].str.fullmatch(r"[0-9]{5}").astype("boolean").fillna(False)
        df.loc[~valid | (df["ZIP5"] == "00000"), "ZIP5"] = pd.NA
    return df


def add_distance_indicators_dask(
    df: pd.DataFrame,
    distance_df: pd.DataFrame,
    on: str = "ZIP5",
    keep_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Merge precomputed distance metrics by ZIP5; optional column subset."""
    merged = df.merge(distance_df, on=on, how="left")
    if keep_cols:
        cols = [c for c in keep_cols if c in merged.columns]
        return merged[cols]
    return merged


__all__ = ["clean_zip_codes_dask", "add_distance_indicators_dask"]
