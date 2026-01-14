#!/usr/bin/env python3
"""
stats_cache.py
--------------
Session-scoped cache for expensive Dask DataFrame statistics.

Purpose:
    - Cache row counts, null counts, and unique counts to avoid recomputation
    - Support both exact and approximate unique counts
    - Automatically invalidate cache when DataFrame identity changes
    - Optimize for both local and Coiled distributed execution

Usage:
    from neuralift_c360_prep.stats_cache import StatsCache

    cache = StatsCache()
    row_count = cache.get_row_count(ddf)
    null_counts = cache.get_null_counts(ddf, columns)
    unique_counts = cache.get_unique_counts(ddf, columns, approx=True)

Author: Mike Maloney - Neuralift, Inc.
Updated: 2026-01-13
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import compute as dask_compute
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger(__name__)


class StatsCache:
    """Session-scoped cache for Dask DataFrame statistics.

    Caches expensive computations like row counts, null counts, and unique counts
    to avoid redundant Dask task graph executions. The cache automatically
    invalidates when a different DataFrame is passed.

    Thread-safe for single-writer scenarios (typical pipeline usage).

    Attributes:
        show_progress: Whether to log cache hits/misses.
    """

    def __init__(self, *, show_progress: bool = False):
        """Initialize the cache.

        Args:
            show_progress: If True, log cache hits and computation starts.
        """
        self._row_count: Optional[int] = None
        self._null_counts: Optional[pd.Series] = None
        self._unique_counts: Optional[pd.Series] = None
        self._unique_approx: Optional[bool] = None
        self._numeric_stats: Optional[pd.DataFrame] = None
        self._numeric_percentiles: Optional[List[float]] = None
        self._value_counts: Optional[Dict[str, pd.Series]] = None
        self._value_counts_top_k: Optional[int] = None
        self._ddf_id: Optional[int] = None
        self._column_set: Optional[frozenset] = None
        self.show_progress = show_progress

    def _check_identity(self, ddf: dd.DataFrame) -> bool:
        """Check if DataFrame identity matches cached data.

        Uses id() for fast identity check. This works because Dask DataFrames
        are immutable - operations return new DataFrames.

        Args:
            ddf: The Dask DataFrame to check.

        Returns:
            True if the DataFrame matches the cached identity.
        """
        return self._ddf_id == id(ddf)

    def _update_identity(self, ddf: dd.DataFrame) -> None:
        """Update cached DataFrame identity."""
        self._ddf_id = id(ddf)
        self._column_set = frozenset(ddf.columns)

    def invalidate(self) -> None:
        """Clear all cached values.

        Call this when you know the underlying data has changed or when
        you want to force recomputation.
        """
        self._row_count = None
        self._null_counts = None
        self._unique_counts = None
        self._unique_approx = None
        self._numeric_stats = None
        self._numeric_percentiles = None
        self._value_counts = None
        self._value_counts_top_k = None
        self._ddf_id = None
        self._column_set = None
        if self.show_progress:
            logger.info("[stats_cache] cache invalidated")

    def get_row_count(self, ddf: dd.DataFrame) -> int:
        """Get row count, computing only if not cached.

        Args:
            ddf: The Dask DataFrame to count rows for.

        Returns:
            Total number of rows in the DataFrame.
        """
        if self._check_identity(ddf) and self._row_count is not None:
            if self.show_progress:
                logger.info("[stats_cache] row_count cache hit: %d", self._row_count)
            return self._row_count

        if self.show_progress:
            logger.info("[stats_cache] computing row count...")

        # Efficient row count using len() which Dask optimizes
        try:
            row_count = int(len(ddf))
        except Exception:
            # Fallback for edge cases
            row_count = int(ddf.map_partitions(len).compute().sum())

        self._update_identity(ddf)
        self._row_count = row_count

        if self.show_progress:
            logger.info("[stats_cache] row_count computed: %d", row_count)

        return row_count

    def get_null_counts(
        self,
        ddf: dd.DataFrame,
        cols: Optional[List[str]] = None,
    ) -> pd.Series:
        """Get null counts for columns, computing only if not cached.

        Computes (row_count - non_null_count) for each column.

        Args:
            ddf: The Dask DataFrame.
            cols: Columns to compute null counts for. If None, uses all columns.

        Returns:
            pandas Series with null counts indexed by column name.
        """
        cols = list(cols) if cols is not None else list(ddf.columns)

        # Check if we can use cached data
        if (
            self._check_identity(ddf)
            and self._null_counts is not None
            and set(cols).issubset(set(self._null_counts.index))
        ):
            if self.show_progress:
                logger.info(
                    "[stats_cache] null_counts cache hit for %d cols", len(cols)
                )
            return self._null_counts.reindex(cols)

        if self.show_progress:
            logger.info("[stats_cache] computing null counts for %d cols...", len(cols))

        # Get row count (may be cached)
        row_count = self.get_row_count(ddf)

        # Compute non-null counts efficiently
        counts = ddf[cols].count().compute()
        if not isinstance(counts, pd.Series):
            counts = pd.Series(counts, index=cols)

        # Null counts = total rows - non-null counts
        null_counts = row_count - counts
        null_counts = null_counts.astype(int)

        self._update_identity(ddf)
        self._null_counts = null_counts

        if self.show_progress:
            logger.info("[stats_cache] null_counts computed for %d cols", len(cols))

        return null_counts

    def get_unique_counts(
        self,
        ddf: dd.DataFrame,
        cols: Optional[List[str]] = None,
        *,
        approx: bool = True,
    ) -> pd.Series:
        """Get unique counts, using approximate by default.

        Args:
            ddf: The Dask DataFrame.
            cols: Columns to compute unique counts for. If None, uses all columns.
            approx: If True (default), use nunique_approx for better performance
                on large datasets. If False, use exact nunique.

        Returns:
            pandas Series with unique counts indexed by column name.
        """
        cols = list(cols) if cols is not None else list(ddf.columns)

        # Check if we can use cached data (must match approx setting)
        if (
            self._check_identity(ddf)
            and self._unique_counts is not None
            and self._unique_approx == approx
            and set(cols).issubset(set(self._unique_counts.index))
        ):
            if self.show_progress:
                logger.info(
                    "[stats_cache] unique_counts cache hit for %d cols (approx=%s)",
                    len(cols),
                    approx,
                )
            return self._unique_counts.reindex(cols)

        if self.show_progress:
            logger.info(
                "[stats_cache] computing unique counts for %d cols (approx=%s)...",
                len(cols),
                approx,
            )

        if approx:
            # Use approximate uniques - much faster for large datasets
            # Compute per-column to avoid DataFrame.nunique_approx() scalar issues
            delayed_vals = [ddf[c].nunique_approx() for c in cols]
            computed = dask_compute(*delayed_vals)
            unique_counts = pd.Series(computed, index=cols).astype(int)
        else:
            # Exact uniques - slower but precise
            unique_counts = ddf[cols].nunique(dropna=True).compute()
            if not isinstance(unique_counts, pd.Series):
                unique_counts = pd.Series(unique_counts, index=cols)
            unique_counts = unique_counts.astype(int)

        self._update_identity(ddf)
        self._unique_counts = unique_counts
        self._unique_approx = approx

        if self.show_progress:
            logger.info("[stats_cache] unique_counts computed for %d cols", len(cols))

        return unique_counts

    def compute_all_stats(
        self,
        ddf: dd.DataFrame,
        cols: Optional[List[str]] = None,
        *,
        approx_unique: bool = True,
    ) -> tuple[int, pd.Series, pd.Series]:
        """Compute all stats in a single optimized pass.

        This is more efficient than calling individual methods when you need
        all statistics, as it can batch Dask computations.

        Args:
            ddf: The Dask DataFrame.
            cols: Columns to compute stats for. If None, uses all columns.
            approx_unique: If True, use approximate unique counts.

        Returns:
            Tuple of (row_count, null_counts, unique_counts).
        """
        cols = list(cols) if cols is not None else list(ddf.columns)

        if self.show_progress:
            logger.info("[stats_cache] computing all stats for %d cols...", len(cols))

        # Prepare delayed computations
        counts_delayed = ddf[cols].count()

        if approx_unique:
            unique_delayed = [ddf[c].nunique_approx() for c in cols]
            stats_to_compute = [counts_delayed] + unique_delayed
        else:
            unique_delayed = ddf[cols].nunique(dropna=True)
            stats_to_compute = [counts_delayed, unique_delayed]

        # Single compute call for efficiency on Coiled
        computed = dask_compute(*stats_to_compute)

        # Parse results
        counts = computed[0]
        if not isinstance(counts, pd.Series):
            counts = pd.Series(counts, index=cols)

        row_count = int(counts.max()) if len(counts) else 0
        null_counts = (row_count - counts).astype(int)

        if approx_unique:
            unique_counts = pd.Series(computed[1:], index=cols).astype(int)
        else:
            unique_counts = computed[1]
            if not isinstance(unique_counts, pd.Series):
                unique_counts = pd.Series(unique_counts, index=cols)
            unique_counts = unique_counts.astype(int)

        # Cache all results
        self._update_identity(ddf)
        self._row_count = row_count
        self._null_counts = null_counts
        self._unique_counts = unique_counts
        self._unique_approx = approx_unique

        if self.show_progress:
            logger.info(
                "[stats_cache] all stats computed: %d rows, %d cols",
                row_count,
                len(cols),
            )

        return row_count, null_counts, unique_counts

    def get_numeric_stats(
        self,
        ddf: dd.DataFrame,
        cols: Optional[List[str]] = None,
        *,
        percentiles: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Get numeric statistics for columns.

        Computes min, max, mean, std, and percentiles for numeric columns.
        Non-numeric columns in the list are silently skipped.

        Args:
            ddf: The Dask DataFrame.
            cols: Columns to compute stats for. If None, uses all numeric columns.
            percentiles: Percentiles to compute (0-1 scale). Default is
                [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99].

        Returns:
            DataFrame with stats as rows (min, max, mean, std, percentiles)
            and columns as columns. Includes 'skew' for skewness detection.
        """
        if percentiles is None:
            percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

        # Filter to numeric columns only
        if cols is None:
            cols = [c for c in ddf.columns if is_numeric_dtype(ddf[c].dtype)]
        else:
            cols = [
                c for c in cols if c in ddf.columns and is_numeric_dtype(ddf[c].dtype)
            ]

        if not cols:
            return pd.DataFrame()

        # Check cache
        if (
            self._check_identity(ddf)
            and self._numeric_stats is not None
            and self._numeric_percentiles == percentiles
            and set(cols).issubset(set(self._numeric_stats.columns))
        ):
            if self.show_progress:
                logger.info(
                    "[stats_cache] numeric_stats cache hit for %d cols", len(cols)
                )
            return self._numeric_stats[cols]

        if self.show_progress:
            logger.info(
                "[stats_cache] computing numeric stats for %d cols...", len(cols)
            )

        # Build delayed computations for efficiency
        ddf_cols = ddf[cols]

        # Basic stats
        min_delayed = ddf_cols.min()
        max_delayed = ddf_cols.max()
        mean_delayed = ddf_cols.mean()
        std_delayed = ddf_cols.std()

        # Percentiles - compute via quantile
        percentile_delayed = [ddf_cols.quantile(q) for q in percentiles]

        # Skewness (approximated via moment calculation)
        # skew = E[(X - mu)^3] / sigma^3
        # We'll use a simpler approach: (mean - median) / std as a proxy
        median_delayed = ddf_cols.quantile(0.5)

        # Compute all at once
        to_compute = [
            min_delayed,
            max_delayed,
            mean_delayed,
            std_delayed,
            median_delayed,
        ] + percentile_delayed
        computed = dask_compute(*to_compute)

        min_vals = computed[0]
        max_vals = computed[1]
        mean_vals = computed[2]
        std_vals = computed[3]
        median_vals = computed[4]
        percentile_vals = computed[5:]

        # Convert to Series if needed
        def to_series(val, cols):
            if isinstance(val, pd.Series):
                return val
            return pd.Series(val, index=cols)

        min_s = to_series(min_vals, cols)
        max_s = to_series(max_vals, cols)
        mean_s = to_series(mean_vals, cols)
        std_s = to_series(std_vals, cols)
        median_s = to_series(median_vals, cols)

        # Compute skewness proxy: (mean - median) / std
        # Positive = right-skewed, negative = left-skewed
        with np.errstate(divide="ignore", invalid="ignore"):
            skew_s = (mean_s - median_s) / std_s
            skew_s = skew_s.replace([np.inf, -np.inf], np.nan)

        # Build result DataFrame
        result_data = {
            "min": min_s,
            "max": max_s,
            "mean": mean_s,
            "std": std_s,
            "median": median_s,
            "skew": skew_s,
        }

        # Add percentiles
        for i, p in enumerate(percentiles):
            p_s = to_series(percentile_vals[i], cols)
            result_data[f"p{int(p * 100):02d}"] = p_s

        result = pd.DataFrame(result_data).T
        result.columns = cols

        # Cache
        self._update_identity(ddf)
        self._numeric_stats = result
        self._numeric_percentiles = percentiles

        if self.show_progress:
            logger.info("[stats_cache] numeric_stats computed for %d cols", len(cols))

        return result

    def get_value_counts(
        self,
        ddf: dd.DataFrame,
        cols: Optional[List[str]] = None,
        *,
        top_k: int = 10,
    ) -> Dict[str, pd.Series]:
        """Get top-k value counts for columns.

        Useful for categorical analysis, mode detection, and bucketing.

        Args:
            ddf: The Dask DataFrame.
            cols: Columns to compute value counts for. If None, uses all
                non-numeric columns.
            top_k: Number of top values to return per column.

        Returns:
            Dict mapping column name to Series of value counts (sorted descending).
            Each Series has at most top_k entries.
        """
        # Default to non-numeric columns
        if cols is None:
            cols = [c for c in ddf.columns if not is_numeric_dtype(ddf[c].dtype)]
        else:
            cols = [c for c in cols if c in ddf.columns]

        if not cols:
            return {}

        # Check cache
        if (
            self._check_identity(ddf)
            and self._value_counts is not None
            and self._value_counts_top_k == top_k
            and set(cols).issubset(set(self._value_counts.keys()))
        ):
            if self.show_progress:
                logger.info(
                    "[stats_cache] value_counts cache hit for %d cols", len(cols)
                )
            return {c: self._value_counts[c] for c in cols}

        if self.show_progress:
            logger.info(
                "[stats_cache] computing value counts (top %d) for %d cols...",
                top_k,
                len(cols),
            )

        # Compute value counts for each column
        # We compute more than top_k in case of ties, then truncate
        delayed_counts = [ddf[c].value_counts().head(top_k * 2) for c in cols]
        computed = dask_compute(*delayed_counts)

        result: Dict[str, pd.Series] = {}
        for col, counts in zip(cols, computed):
            if isinstance(counts, pd.Series):
                result[col] = counts.head(top_k)
            else:
                result[col] = pd.Series(dtype=object)

        # Cache
        self._update_identity(ddf)
        self._value_counts = result
        self._value_counts_top_k = top_k

        if self.show_progress:
            logger.info("[stats_cache] value_counts computed for %d cols", len(cols))

        return result

    def get_column_stats_summary(
        self,
        ddf: dd.DataFrame,
        col: str,
    ) -> Dict[str, Any]:
        """Get a comprehensive stats summary for a single column.

        Convenience method that returns all relevant stats for one column,
        useful for EDA and data_doctor analysis.

        Args:
            ddf: The Dask DataFrame.
            col: Column name to summarize.

        Returns:
            Dict with keys: null_count, unique_count, dtype, and either
            numeric_stats (for numeric cols) or value_counts (for categorical).
        """
        if col not in ddf.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        dtype = ddf[col].dtype
        is_numeric = is_numeric_dtype(dtype)

        # Get basic stats
        null_counts = self.get_null_counts(ddf, [col])
        unique_counts = self.get_unique_counts(ddf, [col])

        summary: Dict[str, Any] = {
            "column": col,
            "dtype": str(dtype),
            "is_numeric": is_numeric,
            "null_count": int(null_counts.get(col, 0)),
            "unique_count": int(unique_counts.get(col, 0)),
        }

        if is_numeric:
            numeric_stats = self.get_numeric_stats(ddf, [col])
            if col in numeric_stats.columns:
                stats_dict = numeric_stats[col].to_dict()
                summary["min"] = stats_dict.get("min")
                summary["max"] = stats_dict.get("max")
                summary["mean"] = stats_dict.get("mean")
                summary["std"] = stats_dict.get("std")
                summary["median"] = stats_dict.get("median")
                summary["skew"] = stats_dict.get("skew")
                summary["p01"] = stats_dict.get("p01")
                summary["p99"] = stats_dict.get("p99")
        else:
            value_counts = self.get_value_counts(ddf, [col])
            if col in value_counts:
                vc = value_counts[col]
                summary["top_values"] = vc.head(5).to_dict()
                summary["mode"] = vc.index[0] if len(vc) > 0 else None
                summary["mode_count"] = int(vc.iloc[0]) if len(vc) > 0 else 0

        return summary


# Module-level singleton for convenience
_default_cache: Optional[StatsCache] = None


def get_default_cache(*, show_progress: bool = False) -> StatsCache:
    """Get the module-level default cache instance.

    Args:
        show_progress: Whether to enable progress logging.

    Returns:
        The default StatsCache instance.
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = StatsCache(show_progress=show_progress)
    elif show_progress and not _default_cache.show_progress:
        _default_cache.show_progress = True
    return _default_cache


def reset_default_cache() -> None:
    """Reset the module-level default cache."""
    global _default_cache
    if _default_cache is not None:
        _default_cache.invalidate()
    _default_cache = None


__all__ = [
    "StatsCache",
    "get_default_cache",
    "reset_default_cache",
]
