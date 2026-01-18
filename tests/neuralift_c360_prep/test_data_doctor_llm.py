#!/usr/bin/env python3
"""
tests/neuralift_c360_prep/test_data_doctor_llm.py
-------------------------------------------------
Tests for the data_doctor_llm module - LLM-enhanced Data Doctor analysis.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
import pandas as pd
import dask.dataframe as dd

from neuralift_c360_prep.data_doctor_llm import (
    _build_column_profile,
    _build_analysis_prompt,
    analyze_with_llm,
    generate_executive_summary,
    convert_llm_result_to_suggestions,
    format_executive_summary,
    _create_fallback_result,
    _generate_transform_yaml,
)
from neuralift_c360_prep.data_doctor_models import (
    TransformationType,
    FillStrategy,
    Priority,
    ColumnAnalysis,
    ColumnRelationship,
    DataQualitySummary,
    ExecutiveSummary,
    LLMDataDoctorResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_ddf():
    """Create a sample Dask DataFrame for testing."""
    pdf = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "total_spend": [100.0, 200.0, None, 400.0, 500.0],
            "order_count": [1, 2, 3, 4, 5],
            "category_name": ["A", "B", "A", "C", "B"],
            "signup_date": pd.to_datetime(
                ["2024-01-01", "2024-02-15", "2024-03-10", "2024-04-20", "2024-05-05"]
            ),
            "revenue": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        }
    )
    return dd.from_pandas(pdf, npartitions=1)


@pytest.fixture
def sample_data_dict():
    """Create a sample data dictionary."""
    return {
        "columns": {
            "customer_id": {
                "column_name": "customer_id",
                "column_type": "id",
                "data_type": "INTEGER",
                "null_count": 0,
                "unique_count": 5,
            },
            "total_spend": {
                "column_name": "total_spend",
                "column_type": "continuous",
                "data_type": "DOUBLE",
                "null_count": 1,
                "unique_count": 4,
            },
            "order_count": {
                "column_name": "order_count",
                "column_type": "continuous",
                "data_type": "INTEGER",
                "null_count": 0,
                "unique_count": 5,
            },
            "category_name": {
                "column_name": "category_name",
                "column_type": "categorical",
                "data_type": "STRING",
                "null_count": 0,
                "unique_count": 3,
            },
            "signup_date": {
                "column_name": "signup_date",
                "column_type": "datetime",
                "data_type": "DATETIME64",
                "null_count": 0,
                "unique_count": 5,
            },
            "revenue": {
                "column_name": "revenue",
                "column_type": "continuous",
                "data_type": "DOUBLE",
                "null_count": 0,
                "unique_count": 5,
            },
        },
        "row_count": 5,
    }


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.name = "mock_provider"
    provider.is_available.return_value = True
    return provider


@pytest.fixture
def sample_llm_result():
    """Create a sample LLMDataDoctorResult for testing."""
    return LLMDataDoctorResult(
        executive_summary=ExecutiveSummary(
            table_description="Customer transaction data table",
            key_findings=["High null rate in spend column", "Revenue is right-skewed"],
            immediate_actions=["Fix missing values in total_spend"],
            data_quality_summary="Good overall quality with some null issues",
            feature_engineering_opportunities="Date parts extraction recommended",
            cross_column_insights="Revenue/orders can form AOV ratio",
        ),
        data_quality=DataQualitySummary(
            total_columns=6,
            columns_with_nulls=1,
            high_null_columns=["total_spend"],
            kpi_candidates=["revenue"],
            overall_quality_score=0.85,
        ),
        column_analyses=[
            ColumnAnalysis(
                column_name="total_spend",
                business_purpose="Customer spending metric",
                data_quality_issues=["Has null values"],
                suggested_transformation=TransformationType.LOG_TRANSFORM,
                transformation_rationale="Right-skewed distribution",
                fill_strategy=FillStrategy.MEDIAN,
                fill_rationale="Median is robust to outliers",
                is_kpi_candidate=True,
                kpi_rationale="Key business metric",
                priority=Priority.HIGH,
            ),
            ColumnAnalysis(
                column_name="signup_date",
                business_purpose="Customer registration date",
                suggested_transformation=TransformationType.DATE_PARTS,
                transformation_rationale="Extract temporal patterns",
                priority=Priority.MEDIUM,
            ),
        ],
        column_relationships=[
            ColumnRelationship(
                columns=["revenue", "order_count"],
                relationship_type="ratio",
                description="Average order value calculation",
                suggested_feature="aov",
                yaml_snippet="- type: ratio\n  numerator: revenue\n  denominator: order_count",
                business_value="Key metric for customer value",
                priority=Priority.HIGH,
            ),
        ],
        table_context="E-commerce customer data",
    )


# ---------------------------------------------------------------------------
# Tests for _build_column_profile
# ---------------------------------------------------------------------------
class TestBuildColumnProfile:
    """Tests for _build_column_profile function."""

    def test_builds_profile_from_dict_format(self, sample_ddf, sample_data_dict):
        """Test building profile with dict-format columns."""
        profile = _build_column_profile("total_spend", sample_ddf, sample_data_dict)
        assert profile["name"] == "total_spend"
        assert profile["dtype"] == "float64"
        assert profile["null_count"] == 1
        assert profile["unique_count"] == 4
        assert profile["column_type"] == "continuous"

    def test_builds_profile_from_list_format(self, sample_ddf):
        """Test building profile with list-format columns."""
        data_dict = {
            "columns": [
                {
                    "column_name": "revenue",
                    "column_type": "continuous",
                    "data_type": "DOUBLE",
                    "null_count": 0,
                    "unique_count": 5,
                },
            ],
            "row_count": 5,
        }
        profile = _build_column_profile("revenue", sample_ddf, data_dict)
        assert profile["name"] == "revenue"
        assert profile["null_count"] == 0
        assert profile["unique_count"] == 5

    def test_handles_missing_column_metadata(self, sample_ddf):
        """Test that missing column metadata returns defaults."""
        data_dict = {"columns": {}, "row_count": 5}
        profile = _build_column_profile("revenue", sample_ddf, data_dict)
        assert profile["name"] == "revenue"
        assert profile["null_count"] == 0
        assert profile["column_type"] == "unknown"

    def test_calculates_percentages(self, sample_ddf, sample_data_dict):
        """Test that null and unique percentages are calculated."""
        profile = _build_column_profile("total_spend", sample_ddf, sample_data_dict)
        assert profile["null_pct"] == 20.0  # 1/5 = 20%
        assert profile["unique_pct"] == 80.0  # 4/5 = 80%

    def test_includes_sample_values(self, sample_ddf, sample_data_dict):
        """Test that sample values are included."""
        profile = _build_column_profile("category_name", sample_ddf, sample_data_dict)
        assert "samples" in profile
        assert isinstance(profile["samples"], list)


# ---------------------------------------------------------------------------
# Tests for _build_analysis_prompt
# ---------------------------------------------------------------------------
class TestBuildAnalysisPrompt:
    """Tests for _build_analysis_prompt function."""

    def test_builds_prompt_with_profiles(self):
        """Test building prompt with column profiles."""
        profiles = [
            {
                "name": "revenue",
                "dtype": "float64",
                "null_pct": 0.0,
                "unique_count": 100,
                "column_type": "continuous",
                "samples": ["100.0", "200.0", "300.0"],
            }
        ]
        prompt = _build_analysis_prompt(profiles)
        assert "revenue" in prompt
        assert "float64" in prompt
        assert "continuous" in prompt

    def test_includes_organization_context(self):
        """Test that organization context is included."""
        profiles = [{"name": "col", "dtype": "int", "null_pct": 0, "unique_count": 5, "column_type": "id", "samples": []}]
        prompt = _build_analysis_prompt(
            profiles,
            organization_context="Retail Company",
        )
        assert "Retail Company" in prompt

    def test_includes_table_context(self):
        """Test that table context is included."""
        profiles = [{"name": "col", "dtype": "int", "null_pct": 0, "unique_count": 5, "column_type": "id", "samples": []}]
        prompt = _build_analysis_prompt(
            profiles,
            table_context="Customer transactions",
        )
        assert "Customer transactions" in prompt

    def test_handles_empty_profiles(self):
        """Test building prompt with no profiles."""
        prompt = _build_analysis_prompt([])
        assert "0 total" in prompt


# ---------------------------------------------------------------------------
# Tests for analyze_with_llm
# ---------------------------------------------------------------------------
class TestAnalyzeWithLlm:
    """Tests for analyze_with_llm function."""

    def test_calls_llm_provider(self, sample_ddf, sample_data_dict, mock_llm_provider, sample_llm_result):
        """Test that LLM provider is called correctly."""
        mock_llm_provider.complete_structured.return_value = sample_llm_result

        result = analyze_with_llm(
            sample_ddf,
            sample_data_dict,
            mock_llm_provider,
        )

        assert mock_llm_provider.complete_structured.called
        assert isinstance(result, LLMDataDoctorResult)
        assert result.executive_summary.table_description == "Customer transaction data table"

    def test_uses_cache_when_available(self, sample_ddf, sample_data_dict, mock_llm_provider, sample_llm_result):
        """Test that cache is used when available."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = sample_llm_result

        result = analyze_with_llm(
            sample_ddf,
            sample_data_dict,
            mock_llm_provider,
            cache=mock_cache,
        )

        # Should use cache and not call LLM
        assert mock_cache.get.called
        assert not mock_llm_provider.complete_structured.called
        assert result == sample_llm_result

    def test_caches_result(self, sample_ddf, sample_data_dict, mock_llm_provider, sample_llm_result):
        """Test that result is cached."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Cache miss
        mock_llm_provider.complete_structured.return_value = sample_llm_result

        analyze_with_llm(
            sample_ddf,
            sample_data_dict,
            mock_llm_provider,
            cache=mock_cache,
        )

        assert mock_cache.set.called

    def test_fallback_on_error(self, sample_ddf, sample_data_dict, mock_llm_provider):
        """Test that fallback result is returned on error."""
        mock_llm_provider.complete_structured.side_effect = Exception("LLM error")

        result = analyze_with_llm(
            sample_ddf,
            sample_data_dict,
            mock_llm_provider,
        )

        # Should return fallback result
        assert isinstance(result, LLMDataDoctorResult)
        assert "LLM analysis unavailable" in result.executive_summary.table_description

    def test_respects_max_columns(self, sample_ddf, sample_data_dict, mock_llm_provider, sample_llm_result):
        """Test that max_columns limits analysis."""
        mock_llm_provider.complete_structured.return_value = sample_llm_result

        analyze_with_llm(
            sample_ddf,
            sample_data_dict,
            mock_llm_provider,
            max_columns=2,  # Only analyze 2 columns
        )

        # Should have called with limited columns
        assert mock_llm_provider.complete_structured.called


# ---------------------------------------------------------------------------
# Tests for generate_executive_summary
# ---------------------------------------------------------------------------
class TestGenerateExecutiveSummary:
    """Tests for generate_executive_summary function."""

    def test_generates_summary(self, sample_ddf, sample_data_dict, mock_llm_provider):
        """Test generating executive summary."""
        expected = ExecutiveSummary(
            table_description="Test data",
            key_findings=["Finding 1"],
            immediate_actions=["Action 1"],
        )
        mock_llm_provider.complete_structured.return_value = expected

        result = generate_executive_summary(
            sample_ddf,
            sample_data_dict,
            mock_llm_provider,
        )

        assert isinstance(result, ExecutiveSummary)
        assert result.table_description == "Test data"

    def test_uses_cache(self, sample_ddf, sample_data_dict, mock_llm_provider):
        """Test that cache is used."""
        mock_cache = MagicMock()
        cached_summary = ExecutiveSummary(
            table_description="Cached summary",
        )
        mock_cache.get.return_value = cached_summary

        result = generate_executive_summary(
            sample_ddf,
            sample_data_dict,
            mock_llm_provider,
            cache=mock_cache,
        )

        assert result.table_description == "Cached summary"
        assert not mock_llm_provider.complete_structured.called

    def test_fallback_on_error(self, sample_ddf, sample_data_dict, mock_llm_provider):
        """Test fallback on error."""
        mock_llm_provider.complete_structured.side_effect = Exception("Error")

        result = generate_executive_summary(
            sample_ddf,
            sample_data_dict,
            mock_llm_provider,
        )

        assert "failed" in result.table_description.lower()


# ---------------------------------------------------------------------------
# Tests for convert_llm_result_to_suggestions
# ---------------------------------------------------------------------------
class TestConvertLlmResultToSuggestions:
    """Tests for convert_llm_result_to_suggestions function."""

    def test_converts_column_analyses(self, sample_llm_result):
        """Test converting column analyses to suggestions."""
        suggestions = convert_llm_result_to_suggestions(sample_llm_result)

        # Should have suggestions from column analyses
        transform_suggestions = [s for s in suggestions if s.category == "transform"]
        assert len(transform_suggestions) >= 1

        # Check transform suggestion content
        log_transform = next(
            (s for s in transform_suggestions if "log_transform" in s.message.lower()),
            None,
        )
        assert log_transform is not None
        assert log_transform.column == "total_spend"

    def test_converts_fill_strategies(self, sample_llm_result):
        """Test converting fill strategies to suggestions."""
        suggestions = convert_llm_result_to_suggestions(sample_llm_result)

        fill_suggestions = [s for s in suggestions if s.category == "fill"]
        assert len(fill_suggestions) >= 1

        median_fill = next(
            (s for s in fill_suggestions if "median" in s.message.lower()),
            None,
        )
        assert median_fill is not None

    def test_converts_kpi_candidates(self, sample_llm_result):
        """Test converting KPI candidates to suggestions."""
        suggestions = convert_llm_result_to_suggestions(sample_llm_result)

        kpi_suggestions = [s for s in suggestions if s.category == "kpi"]
        assert len(kpi_suggestions) >= 1

    def test_converts_relationships(self, sample_llm_result):
        """Test converting relationships to suggestions."""
        suggestions = convert_llm_result_to_suggestions(sample_llm_result)

        ratio_suggestions = [s for s in suggestions if s.category == "ratio"]
        assert len(ratio_suggestions) >= 1

        # Check that relationship columns are combined
        ratio = ratio_suggestions[0]
        assert "revenue" in ratio.column or "order_count" in ratio.column

    def test_sets_correct_source(self, sample_llm_result):
        """Test that source is set to 'llm'."""
        suggestions = convert_llm_result_to_suggestions(sample_llm_result)

        for s in suggestions:
            assert s.source == "llm"

    def test_includes_yaml_snippets(self, sample_llm_result):
        """Test that YAML snippets are included."""
        suggestions = convert_llm_result_to_suggestions(sample_llm_result)

        transform_suggestions = [s for s in suggestions if s.category == "transform"]
        for s in transform_suggestions:
            assert s.yaml_snippet != ""


# ---------------------------------------------------------------------------
# Tests for format_executive_summary
# ---------------------------------------------------------------------------
class TestFormatExecutiveSummary:
    """Tests for format_executive_summary function."""

    def test_formats_summary(self):
        """Test formatting executive summary."""
        summary = ExecutiveSummary(
            table_description="Customer data table",
            key_findings=["Finding 1", "Finding 2"],
            immediate_actions=["Action 1"],
            data_quality_summary="Good quality",
            feature_engineering_opportunities="Date parts",
            cross_column_insights="Ratios available",
        )

        formatted = format_executive_summary(summary)

        assert "EXECUTIVE SUMMARY" in formatted
        assert "Customer data table" in formatted
        assert "KEY FINDINGS:" in formatted
        assert "Finding 1" in formatted
        assert "IMMEDIATE ACTIONS:" in formatted
        assert "Action 1" in formatted
        assert "Good quality" in formatted

    def test_handles_empty_fields(self):
        """Test formatting with empty fields."""
        summary = ExecutiveSummary(
            table_description="Test",
        )

        formatted = format_executive_summary(summary)
        assert "EXECUTIVE SUMMARY" in formatted


# ---------------------------------------------------------------------------
# Tests for _create_fallback_result
# ---------------------------------------------------------------------------
class TestCreateFallbackResult:
    """Tests for _create_fallback_result function."""

    def test_creates_fallback(self, sample_ddf, sample_data_dict):
        """Test creating fallback result."""
        profiles = [
            {"name": "col1", "null_count": 5, "null_pct": 10},
            {"name": "col2", "null_count": 0, "null_pct": 0},
        ]
        result = _create_fallback_result(sample_ddf, sample_data_dict, profiles)

        assert isinstance(result, LLMDataDoctorResult)
        assert "unavailable" in result.executive_summary.table_description.lower()
        assert result.data_quality.total_columns == len(sample_ddf.columns)

    def test_identifies_high_null_columns(self, sample_ddf, sample_data_dict):
        """Test that high null columns are identified in fallback."""
        profiles = [
            {"name": "high_null", "null_count": 50, "null_pct": 50},
            {"name": "low_null", "null_count": 5, "null_pct": 5},
        ]
        result = _create_fallback_result(sample_ddf, sample_data_dict, profiles)

        assert "high_null" in result.data_quality.high_null_columns


# ---------------------------------------------------------------------------
# Tests for _generate_transform_yaml
# ---------------------------------------------------------------------------
class TestGenerateTransformYaml:
    """Tests for _generate_transform_yaml function."""

    def test_log_transform_yaml(self):
        """Test generating log transform YAML."""
        yaml = _generate_transform_yaml("revenue", TransformationType.LOG_TRANSFORM)
        assert "type: log_transform" in yaml
        assert "source_col: revenue" in yaml
        assert "revenue_log" in yaml

    def test_winsorize_yaml(self):
        """Test generating winsorize YAML."""
        yaml = _generate_transform_yaml("amount", TransformationType.WINSORIZE)
        assert "type: winsorize" in yaml
        assert "lower_quantile" in yaml
        assert "upper_quantile" in yaml

    def test_binning_yaml(self):
        """Test generating binning YAML."""
        yaml = _generate_transform_yaml("score", TransformationType.BINNING)
        assert "type: binning" in yaml
        assert "quantiles" in yaml

    def test_zsml_yaml(self):
        """Test generating ZSML YAML."""
        yaml = _generate_transform_yaml("ltv", TransformationType.ZSML)
        assert "type: zsml" in yaml
        assert "kpi: true" in yaml

    def test_date_parts_yaml(self):
        """Test generating date_parts YAML."""
        yaml = _generate_transform_yaml("created_at", TransformationType.DATE_PARTS)
        assert "type: date_parts" in yaml
        assert "year" in yaml
        assert "month" in yaml

    def test_categorical_bucket_yaml(self):
        """Test generating categorical_bucket YAML."""
        yaml = _generate_transform_yaml("category", TransformationType.CATEGORICAL_BUCKET)
        assert "type: categorical_bucket" in yaml
        assert "top_k" in yaml
        assert "other_label" in yaml

    def test_frequency_encode_yaml(self):
        """Test generating frequency_encode YAML."""
        yaml = _generate_transform_yaml("product", TransformationType.FREQUENCY_ENCODE)
        assert "type: frequency_encode" in yaml
        assert "normalize" in yaml

    def test_unknown_transform(self):
        """Test handling unknown transform type."""
        yaml = _generate_transform_yaml("col", TransformationType.NONE)
        assert "# Transform" in yaml


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------
class TestDataDoctorLlmIntegration:
    """Integration tests for data_doctor_llm module."""

    def test_full_analysis_flow(self, sample_ddf, sample_data_dict, mock_llm_provider, sample_llm_result):
        """Test full analysis flow from ddf to suggestions."""
        mock_llm_provider.complete_structured.return_value = sample_llm_result

        # Run analysis
        result = analyze_with_llm(
            sample_ddf,
            sample_data_dict,
            mock_llm_provider,
            table_context="Test table",
            organization_context="Test org",
        )

        # Convert to suggestions
        suggestions = convert_llm_result_to_suggestions(result)

        # Should have suggestions
        assert len(suggestions) > 0

        # All suggestions should be valid
        for s in suggestions:
            assert s.column != ""
            assert s.message != ""
            assert s.category in ["fill", "transform", "kpi", "ratio", "derived", "quality"]
