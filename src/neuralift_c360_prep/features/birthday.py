#!/usr/bin/env python3
"""
Dask-friendly birthday helpers.

Purpose:
    Add simple birth month/day columns to a pandas partition; intended to run
    via map_partitions in the Dask pipeline pre-hooks.
"""

from __future__ import annotations

import pandas as pd


def add_birth_month(df: pd.DataFrame, date_col: str = "birth_date") -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["birth_month"] = out[date_col].dt.month
    out["birth_day"] = out[date_col].dt.day
    return out


__all__ = ["add_birth_month"]


def add_age_from_year(
    df: pd.DataFrame,
    year_col: str = "year_birth",
    *,
    current_year: int | None = None,
    fill_value: str = "Unknown",
) -> pd.DataFrame:
    """
    Compute age, age_range, and generation from a birth year column.
    Intended for Dask pre-hooks (map_partitions).
    """
    col = year_col
    if col not in df.columns:
        # Try case-insensitive match (handles "Year_Birth" before snake-case rename).
        col = next((c for c in df.columns if str(c).lower() == year_col.lower()), None)
        if col is None:
            return df

    out = df.copy()
    cy = current_year or pd.Timestamp.utcnow().year

    def _age_from_year(val):
        try:
            y = int(val)
            if y <= 0 or y > cy:
                return pd.NA
            return cy - y
        except Exception:
            return pd.NA

    out["age"] = out[col].apply(_age_from_year)

    def _age_range(a):
        if pd.isna(a):
            return fill_value
        a = int(a)
        if a < 18:
            return "Under 18"
        if a <= 24:
            return "18-24"
        if a <= 34:
            return "25-34"
        if a <= 44:
            return "35-44"
        if a <= 54:
            return "45-54"
        if a <= 64:
            return "55-64"
        return "65+"

    out["age_range"] = out["age"].apply(_age_range)

    def _generation(a):
        if pd.isna(a):
            return fill_value
        try:
            yr = cy - int(a)
        except Exception:
            return fill_value
        if yr >= 1997:
            return "Gen Z"
        if 1981 <= yr <= 1996:
            return "Millennial"
        if 1965 <= yr <= 1980:
            return "Gen X"
        if 1946 <= yr <= 1964:
            return "Boomer"
        return "Traditionalist"

    out["generation"] = out["age"].apply(_generation)
    return out


__all__.append("add_age_from_year")
