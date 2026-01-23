#!/usr/bin/env python3
"""
tests/neuralift_c360_prep/test_stats_cache.py
---------------------------------------------
Tests for the stats_cache module.

Author: Mike Maloney - Neuralift, Inc.
Updated: 2026-01-13
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import pytest
import pandas as pd
import dask.dataframe as dd

from neuralift_c360_prep.stats_cache import (
    StatsCache,
    get_default_cache,
    reset_default_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_ddf():
    """Create a sample Dask DataFrame for testing."""
    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10.0, 20.0, None, 40.0, 50.0],
            "category": ["A", "B", "A", "C", "B"],
        }
    )
    return dd.from_pandas(pdf, npartitions=2)


@pytest.fixture
def large_ddf():
    """Create a larger Dask DataFrame for testing."""
    pdf = pd.DataFrame(
        {
            "id": list(range(1000)),
            "value": [float(i) if i % 10 != 0 else None for i in range(1000)],
            "category": [f"cat_{i % 50}" for i in range(1000)],
        }
    )
    return dd.from_pandas(pdf, npartitions=4)


@pytest.fixture
def fresh_cache():
    """Create a fresh StatsCache for each test."""
    return StatsCache(show_progress=False)


# ---------------------------------------------------------------------------
# Tests for StatsCache class
# ---------------------------------------------------------------------------
class TestStatsCache:
    """Tests for the StatsCache class."""

    def test_init(self):
        """Test cache initialization."""
        cache = StatsCache()
        assert cache._row_count is None
        assert cache._null_counts is None
        assert cache._unique_counts is None
        assert cache._ddf_id is None
        assert cache.show_progress is False

    def test_init_with_progress(self):
        """Test cache initialization with progress logging."""
        cache = StatsCache(show_progress=True)
        assert cache.show_progress is True


class TestGetRowCount:
    """Tests for get_row_count method."""

    def test_get_row_count(self, sample_ddf, fresh_cache):
        """Test getting row count."""
        row_count = fresh_cache.get_row_count(sample_ddf)
        assert row_count == 5

    def test_get_row_count_cached(self, sample_ddf, fresh_cache):
        """Test that row count is cached."""
        row_count1 = fresh_cache.get_row_count(sample_ddf)
        row_count2 = fresh_cache.get_row_count(sample_ddf)
        assert row_count1 == row_count2
        assert fresh_cache._row_count == 5

    def test_get_row_count_different_ddf(self, sample_ddf, large_ddf, fresh_cache):
        """Test that cache is invalidated for different DataFrame."""
        row_count1 = fresh_cache.get_row_count(sample_ddf)
        assert row_count1 == 5
        row_count2 = fresh_cache.get_row_count(large_ddf)
        assert row_count2 == 1000


class TestGetNullCounts:
    """Tests for get_null_counts method."""

    def test_get_null_counts(self, sample_ddf, fresh_cache):
        """Test getting null counts."""
        null_counts = fresh_cache.get_null_counts(sample_ddf)
        assert isinstance(null_counts, pd.Series)
        assert null_counts["value"] == 1  # One null in value column
        assert null_counts["id"] == 0

    def test_get_null_counts_cached(self, sample_ddf, fresh_cache):
        """Test that null counts are cached."""
        null_counts1 = fresh_cache.get_null_counts(sample_ddf)
        null_counts2 = fresh_cache.get_null_counts(sample_ddf)
        pd.testing.assert_series_equal(null_counts1, null_counts2)

    def test_get_null_counts_subset(self, sample_ddf, fresh_cache):
        """Test getting null counts for a subset of columns."""
        null_counts = fresh_cache.get_null_counts(sample_ddf, cols=["value"])
        assert len(null_counts) == 1
        assert null_counts["value"] == 1


class TestGetUniqueCounts:
    """Tests for get_unique_counts method."""

    def test_get_unique_counts_approx(self, sample_ddf, fresh_cache):
        """Test getting approximate unique counts."""
        unique_counts = fresh_cache.get_unique_counts(sample_ddf, approx=True)
        assert isinstance(unique_counts, pd.Series)
        assert unique_counts["id"] == 5
        assert unique_counts["category"] == 3

    def test_get_unique_counts_exact(self, sample_ddf, fresh_cache):
        """Test getting exact unique counts."""
        unique_counts = fresh_cache.get_unique_counts(sample_ddf, approx=False)
        assert isinstance(unique_counts, pd.Series)
        assert unique_counts["id"] == 5
        assert unique_counts["category"] == 3

    def test_get_unique_counts_cached(self, sample_ddf, fresh_cache):
        """Test that unique counts are cached."""
        unique_counts1 = fresh_cache.get_unique_counts(sample_ddf, approx=True)
        unique_counts2 = fresh_cache.get_unique_counts(sample_ddf, approx=True)
        pd.testing.assert_series_equal(unique_counts1, unique_counts2)

    def test_get_unique_counts_approx_change(self, sample_ddf, fresh_cache):
        """Test that changing approx flag recomputes."""
        fresh_cache.get_unique_counts(sample_ddf, approx=True)
        # Changing approx flag should recompute
        fresh_cache.get_unique_counts(sample_ddf, approx=False)
        # Values should be similar but cache state should update
        assert fresh_cache._unique_approx is False


class TestInvalidate:
    """Tests for invalidate method."""

    def test_invalidate(self, sample_ddf, fresh_cache):
        """Test cache invalidation."""
        fresh_cache.get_row_count(sample_ddf)
        fresh_cache.get_null_counts(sample_ddf)
        fresh_cache.get_unique_counts(sample_ddf)

        assert fresh_cache._row_count is not None
        assert fresh_cache._null_counts is not None
        assert fresh_cache._unique_counts is not None

        fresh_cache.invalidate()

        assert fresh_cache._row_count is None
        assert fresh_cache._null_counts is None
        assert fresh_cache._unique_counts is None
        assert fresh_cache._ddf_id is None


class TestComputeAllStats:
    """Tests for compute_all_stats method."""

    def test_compute_all_stats(self, sample_ddf, fresh_cache):
        """Test computing all stats at once."""
        row_count, null_counts, unique_counts = fresh_cache.compute_all_stats(
            sample_ddf, approx_unique=True
        )
        assert row_count == 5
        assert isinstance(null_counts, pd.Series)
        assert isinstance(unique_counts, pd.Series)
        assert null_counts["value"] == 1
        assert unique_counts["id"] == 5

    def test_compute_all_stats_exact(self, sample_ddf, fresh_cache):
        """Test computing all stats with exact uniques."""
        row_count, null_counts, unique_counts = fresh_cache.compute_all_stats(
            sample_ddf, approx_unique=False
        )
        assert row_count == 5
        assert unique_counts["category"] == 3

    def test_compute_all_stats_caches(self, sample_ddf, fresh_cache):
        """Test that compute_all_stats populates cache."""
        fresh_cache.compute_all_stats(sample_ddf)
        assert fresh_cache._row_count == 5
        assert fresh_cache._null_counts is not None
        assert fresh_cache._unique_counts is not None


class TestIdentityTracking:
    """Tests for DataFrame identity tracking."""

    def test_check_identity_same_ddf(self, sample_ddf, fresh_cache):
        """Test identity check with same DataFrame."""
        fresh_cache.get_row_count(sample_ddf)
        assert fresh_cache._check_identity(sample_ddf) is True

    def test_check_identity_different_ddf(self, sample_ddf, large_ddf, fresh_cache):
        """Test identity check with different DataFrame."""
        fresh_cache.get_row_count(sample_ddf)
        assert fresh_cache._check_identity(large_ddf) is False


# ---------------------------------------------------------------------------
# Tests for module-level functions
# ---------------------------------------------------------------------------
class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_default_cache(self):
        """Test getting the default cache."""
        reset_default_cache()  # Start fresh
        cache1 = get_default_cache()
        cache2 = get_default_cache()
        assert cache1 is cache2

    def test_get_default_cache_with_progress(self):
        """Test getting the default cache with progress enabled."""
        reset_default_cache()
        cache = get_default_cache(show_progress=True)
        assert cache.show_progress is True

    def test_reset_default_cache(self):
        """Test resetting the default cache."""
        cache1 = get_default_cache()
        reset_default_cache()
        cache2 = get_default_cache()
        assert cache1 is not cache2


# ---------------------------------------------------------------------------
# Tests for edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self, fresh_cache):
        """Test with an empty DataFrame."""
        pdf = pd.DataFrame()
        ddf = dd.from_pandas(pdf, npartitions=1)
        row_count = fresh_cache.get_row_count(ddf)
        assert row_count == 0

    def test_single_partition(self, fresh_cache):
        """Test with a single partition DataFrame."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        ddf = dd.from_pandas(pdf, npartitions=1)
        row_count = fresh_cache.get_row_count(ddf)
        assert row_count == 3

    def test_all_nulls(self, fresh_cache):
        """Test with a column that has all nulls."""
        pdf = pd.DataFrame({"a": [None, None, None]})
        ddf = dd.from_pandas(pdf, npartitions=1)
        null_counts = fresh_cache.get_null_counts(ddf)
        assert null_counts["a"] == 3

    def test_no_nulls(self, fresh_cache):
        """Test with no null values."""
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        ddf = dd.from_pandas(pdf, npartitions=1)
        null_counts = fresh_cache.get_null_counts(ddf)
        assert null_counts["a"] == 0
        assert null_counts["b"] == 0

    def test_high_cardinality(self, large_ddf, fresh_cache):
        """Test unique counts with high cardinality."""
        unique_counts = fresh_cache.get_unique_counts(large_ddf, approx=True)
        # category has 50 unique values
        assert unique_counts["category"] >= 40  # Allow some approximation error


# ---------------------------------------------------------------------------
# Tests for caching behavior with column subsets
# ---------------------------------------------------------------------------
class TestColumnSubsets:
    """Tests for caching behavior with column subsets."""

    def test_null_counts_column_subset_uses_cache(self, sample_ddf, fresh_cache):
        """Test that requesting a subset of columns uses cached data."""
        # First, get all columns
        all_nulls = fresh_cache.get_null_counts(sample_ddf)
        # Then request a subset
        subset_nulls = fresh_cache.get_null_counts(sample_ddf, cols=["id", "value"])
        assert len(subset_nulls) == 2
        assert subset_nulls["id"] == all_nulls["id"]
        assert subset_nulls["value"] == all_nulls["value"]

    def test_unique_counts_column_subset_uses_cache(self, sample_ddf, fresh_cache):
        """Test that requesting a subset of columns uses cached data."""
        # First, get all columns
        all_uniques = fresh_cache.get_unique_counts(sample_ddf)
        # Then request a subset
        subset_uniques = fresh_cache.get_unique_counts(
            sample_ddf, cols=["id", "category"]
        )
        assert len(subset_uniques) == 2
        assert subset_uniques["id"] == all_uniques["id"]
        assert subset_uniques["category"] == all_uniques["category"]


# ---------------------------------------------------------------------------
# Tests for get_numeric_stats method
# ---------------------------------------------------------------------------
class TestGetNumericStats:
    """Tests for get_numeric_stats method."""

    def test_get_numeric_stats_basic(self, sample_ddf, fresh_cache):
        """Test getting basic numeric stats."""
        stats = fresh_cache.get_numeric_stats(sample_ddf)
        assert isinstance(stats, pd.DataFrame)
        # Should have numeric columns (id, value)
        assert "id" in stats.columns
        assert "value" in stats.columns
        # Should not have categorical column
        assert "category" not in stats.columns

    def test_get_numeric_stats_rows(self, sample_ddf, fresh_cache):
        """Test that numeric stats has expected rows."""
        stats = fresh_cache.get_numeric_stats(sample_ddf)
        # Check for expected stat rows
        assert "min" in stats.index
        assert "max" in stats.index
        assert "mean" in stats.index
        assert "std" in stats.index
        assert "median" in stats.index
        assert "skew" in stats.index

    def test_get_numeric_stats_values(self, fresh_cache):
        """Test numeric stats values are correct."""
        pdf = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        ddf = dd.from_pandas(pdf, npartitions=1)
        stats = fresh_cache.get_numeric_stats(ddf)
        assert stats.loc["min", "a"] == 1.0
        assert stats.loc["max", "a"] == 5.0
        assert stats.loc["mean", "a"] == 3.0
        assert stats.loc["median", "a"] == 3.0

    def test_get_numeric_stats_percentiles(self, fresh_cache):
        """Test that percentiles are included."""
        pdf = pd.DataFrame({"a": list(range(100))})
        ddf = dd.from_pandas(pdf, npartitions=2)
        stats = fresh_cache.get_numeric_stats(ddf)
        # Default percentiles
        assert "p01" in stats.index
        assert "p05" in stats.index
        assert "p25" in stats.index
        assert "p50" in stats.index
        assert "p75" in stats.index
        assert "p95" in stats.index
        assert "p99" in stats.index

    def test_get_numeric_stats_custom_percentiles(self, fresh_cache):
        """Test custom percentiles."""
        pdf = pd.DataFrame({"a": list(range(100))})
        ddf = dd.from_pandas(pdf, npartitions=2)
        stats = fresh_cache.get_numeric_stats(ddf, percentiles=[0.1, 0.9])
        assert "p10" in stats.index
        assert "p90" in stats.index

    def test_get_numeric_stats_cached(self, sample_ddf, fresh_cache):
        """Test that numeric stats are cached."""
        stats1 = fresh_cache.get_numeric_stats(sample_ddf)
        stats2 = fresh_cache.get_numeric_stats(sample_ddf)
        pd.testing.assert_frame_equal(stats1, stats2)
        assert fresh_cache._numeric_stats is not None

    def test_get_numeric_stats_subset(self, sample_ddf, fresh_cache):
        """Test getting stats for a subset of columns."""
        stats = fresh_cache.get_numeric_stats(sample_ddf, cols=["id"])
        assert len(stats.columns) == 1
        assert "id" in stats.columns

    def test_get_numeric_stats_empty_for_no_numeric(self, fresh_cache):
        """Test that empty DataFrame is returned when no numeric columns."""
        pdf = pd.DataFrame({"a": ["x", "y", "z"]})
        ddf = dd.from_pandas(pdf, npartitions=1)
        stats = fresh_cache.get_numeric_stats(ddf)
        assert isinstance(stats, pd.DataFrame)
        assert len(stats.columns) == 0


# ---------------------------------------------------------------------------
# Tests for get_value_counts method
# ---------------------------------------------------------------------------
class TestGetValueCounts:
    """Tests for get_value_counts method."""

    def test_get_value_counts_basic(self, sample_ddf, fresh_cache):
        """Test getting basic value counts."""
        vc = fresh_cache.get_value_counts(sample_ddf)
        assert isinstance(vc, dict)
        # Should have categorical column
        assert "category" in vc
        # Should not have numeric columns by default
        assert "id" not in vc
        assert "value" not in vc

    def test_get_value_counts_values(self, fresh_cache):
        """Test value counts are correct."""
        # Use single partition to avoid partition-related aggregation issues
        pdf = pd.DataFrame({"category": ["A", "A", "B", "B", "C"]})
        ddf = dd.from_pandas(pdf, npartitions=1)
        vc = fresh_cache.get_value_counts(ddf, cols=["category"])
        cat_counts = vc["category"]
        assert isinstance(cat_counts, pd.Series)
        # "A" and "B" appear twice each, "C" once
        assert cat_counts["A"] == 2
        assert cat_counts["B"] == 2
        assert cat_counts["C"] == 1

    def test_get_value_counts_top_k(self, large_ddf, fresh_cache):
        """Test top_k parameter limits results."""
        vc = fresh_cache.get_value_counts(large_ddf, top_k=5)
        cat_counts = vc["category"]
        assert len(cat_counts) <= 5

    def test_get_value_counts_cached(self, sample_ddf, fresh_cache):
        """Test that value counts are cached."""
        vc1 = fresh_cache.get_value_counts(sample_ddf)
        vc2 = fresh_cache.get_value_counts(sample_ddf)
        pd.testing.assert_series_equal(vc1["category"], vc2["category"])
        assert fresh_cache._value_counts is not None

    def test_get_value_counts_specific_cols(self, sample_ddf, fresh_cache):
        """Test getting value counts for specific columns."""
        # Can request numeric columns explicitly
        vc = fresh_cache.get_value_counts(sample_ddf, cols=["id", "category"])
        assert "id" in vc
        assert "category" in vc

    def test_get_value_counts_empty(self, fresh_cache):
        """Test empty dict for numeric-only DataFrame."""
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        ddf = dd.from_pandas(pdf, npartitions=1)
        vc = fresh_cache.get_value_counts(ddf)
        assert vc == {}


# ---------------------------------------------------------------------------
# Tests for get_column_stats_summary method
# ---------------------------------------------------------------------------
class TestGetColumnStatsSummary:
    """Tests for get_column_stats_summary method."""

    def test_summary_numeric_column(self, sample_ddf, fresh_cache):
        """Test summary for a numeric column."""
        summary = fresh_cache.get_column_stats_summary(sample_ddf, "value")
        assert summary["column"] == "value"
        assert summary["is_numeric"] is True
        assert "null_count" in summary
        assert "unique_count" in summary
        assert "min" in summary
        assert "max" in summary
        assert "mean" in summary
        assert "median" in summary

    def test_summary_categorical_column(self, sample_ddf, fresh_cache):
        """Test summary for a categorical column."""
        summary = fresh_cache.get_column_stats_summary(sample_ddf, "category")
        assert summary["column"] == "category"
        assert summary["is_numeric"] is False
        assert "null_count" in summary
        assert "unique_count" in summary
        assert "top_values" in summary
        assert "mode" in summary
        assert "mode_count" in summary

    def test_summary_values_correct(self, sample_ddf, fresh_cache):
        """Test that summary values are correct."""
        summary = fresh_cache.get_column_stats_summary(sample_ddf, "value")
        assert summary["null_count"] == 1
        # 4 unique non-null values: 10, 20, 40, 50
        assert summary["unique_count"] >= 4

    def test_summary_mode_correct(self, fresh_cache):
        """Test that mode is computed correctly."""
        pdf = pd.DataFrame({"cat": ["a", "a", "a", "b", "b", "c"]})
        ddf = dd.from_pandas(pdf, npartitions=1)
        summary = fresh_cache.get_column_stats_summary(ddf, "cat")
        assert summary["mode"] == "a"
        assert summary["mode_count"] == 3

    def test_summary_invalid_column(self, sample_ddf, fresh_cache):
        """Test that invalid column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            fresh_cache.get_column_stats_summary(sample_ddf, "nonexistent")

    def test_summary_dtype_included(self, sample_ddf, fresh_cache):
        """Test that dtype is included in summary."""
        summary = fresh_cache.get_column_stats_summary(sample_ddf, "id")
        assert "dtype" in summary
        assert "int" in summary["dtype"].lower()
