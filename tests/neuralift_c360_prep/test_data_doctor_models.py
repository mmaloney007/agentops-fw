#!/usr/bin/env python3
"""
tests/neuralift_c360_prep/test_data_doctor_models.py
----------------------------------------------------
Tests for the data_doctor_models module - Pydantic models for LLM Data Doctor.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralift_c360_prep.data_doctor_models import (
    TransformationType,
    FillStrategy,
    Priority,
    ColumnAnalysis,
    ColumnRelationship,
    DataQualitySummary,
    ExecutiveSummary,
    LLMDataDoctorResult,
    LLMSuggestion,
)


# ---------------------------------------------------------------------------
# Tests for Enums
# ---------------------------------------------------------------------------
class TestTransformationType:
    """Tests for TransformationType enum."""

    def test_all_values_exist(self):
        """Test that all expected transformation types are defined."""
        expected = [
            "log_transform",
            "winsorize",
            "binning",
            "zsml",
            "date_parts",
            "categorical_bucket",
            "frequency_encode",
            "string_normalize",
            "ratio",
            "days_since",
            "derived",
            "none",
        ]
        actual = [t.value for t in TransformationType]
        for exp in expected:
            assert exp in actual, f"Missing transformation type: {exp}"

    def test_enum_string_behavior(self):
        """Test that enum values work as strings."""
        assert TransformationType.LOG_TRANSFORM.value == "log_transform"
        assert str(TransformationType.WINSORIZE) == "TransformationType.WINSORIZE"
        assert TransformationType.NONE.value == "none"


class TestFillStrategy:
    """Tests for FillStrategy enum."""

    def test_all_values_exist(self):
        """Test that all expected fill strategies are defined."""
        expected = [
            "median",
            "mean",
            "mode",
            "zero",
            "forward_fill",
            "backward_fill",
            "interpolate",
            "Unknown",
            "group_median",
            "group_mode",
            "none",
        ]
        actual = [f.value for f in FillStrategy]
        for exp in expected:
            assert exp in actual, f"Missing fill strategy: {exp}"


class TestPriority:
    """Tests for Priority enum."""

    def test_priority_values(self):
        """Test that priority levels are correctly defined."""
        assert Priority.HIGH.value == "HIGH"
        assert Priority.MEDIUM.value == "MEDIUM"
        assert Priority.LOW.value == "LOW"


# ---------------------------------------------------------------------------
# Tests for ColumnAnalysis
# ---------------------------------------------------------------------------
class TestColumnAnalysis:
    """Tests for ColumnAnalysis model."""

    def test_minimal_creation(self):
        """Test creating ColumnAnalysis with required fields only."""
        ca = ColumnAnalysis(
            column_name="revenue",
            business_purpose="Total customer spend",
        )
        assert ca.column_name == "revenue"
        assert ca.business_purpose == "Total customer spend"
        assert ca.suggested_transformation == TransformationType.NONE
        assert ca.fill_strategy == FillStrategy.NONE
        assert ca.is_kpi_candidate is False
        assert ca.priority == Priority.LOW

    def test_full_creation(self):
        """Test creating ColumnAnalysis with all fields."""
        ca = ColumnAnalysis(
            column_name="total_spend",
            business_purpose="Customer lifetime value metric",
            data_quality_issues=["5% null values", "Slight right skew"],
            suggested_transformation=TransformationType.LOG_TRANSFORM,
            transformation_rationale="Right-skewed distribution benefits from log",
            fill_strategy=FillStrategy.MEDIAN,
            fill_rationale="Median is robust to outliers",
            is_kpi_candidate=True,
            kpi_rationale="Key business metric for segmentation",
            priority=Priority.HIGH,
        )
        assert ca.suggested_transformation == TransformationType.LOG_TRANSFORM
        assert ca.is_kpi_candidate is True
        assert ca.priority == Priority.HIGH
        assert len(ca.data_quality_issues) == 2

    def test_business_purpose_max_length(self):
        """Test that business_purpose respects max_length."""
        long_purpose = "x" * 201  # Over 200 limit
        with pytest.raises(ValidationError):
            ColumnAnalysis(
                column_name="test",
                business_purpose=long_purpose,
            )

    def test_enum_values_in_dict(self):
        """Test that enums serialize to values when use_enum_values is True."""
        ca = ColumnAnalysis(
            column_name="test",
            business_purpose="test purpose",
            suggested_transformation=TransformationType.WINSORIZE,
        )
        # With use_enum_values=True in Config, dict() should use enum values
        d = ca.model_dump()
        assert d["suggested_transformation"] == "winsorize"


# ---------------------------------------------------------------------------
# Tests for ColumnRelationship
# ---------------------------------------------------------------------------
class TestColumnRelationship:
    """Tests for ColumnRelationship model."""

    def test_minimal_creation(self):
        """Test creating ColumnRelationship with required fields."""
        cr = ColumnRelationship(
            columns=["revenue", "orders"],
            relationship_type="ratio",
            description="Average order value calculation",
        )
        assert cr.columns == ["revenue", "orders"]
        assert cr.relationship_type == "ratio"
        assert cr.priority == Priority.MEDIUM  # default

    def test_requires_minimum_columns(self):
        """Test that at least 2 columns are required."""
        with pytest.raises(ValidationError):
            ColumnRelationship(
                columns=["single_col"],
                relationship_type="ratio",
                description="Not enough columns",
            )

    def test_all_relationship_types(self):
        """Test that all relationship types are valid."""
        valid_types = ["ratio", "derived", "correlated", "grouped", "temporal"]
        for rel_type in valid_types:
            cr = ColumnRelationship(
                columns=["col1", "col2"],
                relationship_type=rel_type,
                description=f"Testing {rel_type}",
            )
            assert cr.relationship_type == rel_type

    def test_invalid_relationship_type(self):
        """Test that invalid relationship types are rejected."""
        with pytest.raises(ValidationError):
            ColumnRelationship(
                columns=["col1", "col2"],
                relationship_type="invalid_type",
                description="Should fail",
            )

    def test_yaml_snippet_field(self):
        """Test that yaml_snippet is properly stored."""
        snippet = """- type: ratio
  numerator: revenue
  denominator: orders"""
        cr = ColumnRelationship(
            columns=["revenue", "orders"],
            relationship_type="ratio",
            description="AOV",
            yaml_snippet=snippet,
        )
        assert "- type: ratio" in cr.yaml_snippet


# ---------------------------------------------------------------------------
# Tests for DataQualitySummary
# ---------------------------------------------------------------------------
class TestDataQualitySummary:
    """Tests for DataQualitySummary model."""

    def test_creation(self):
        """Test creating DataQualitySummary."""
        dqs = DataQualitySummary(
            total_columns=50,
            columns_with_nulls=10,
            high_null_columns=["col1", "col2"],
            high_cardinality_columns=["cat_col"],
            potential_id_columns=["customer_id"],
            kpi_candidates=["revenue", "orders"],
            overall_quality_score=0.85,
        )
        assert dqs.total_columns == 50
        assert dqs.columns_with_nulls == 10
        assert len(dqs.high_null_columns) == 2
        assert dqs.overall_quality_score == 0.85

    def test_quality_score_bounds(self):
        """Test that quality score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            DataQualitySummary(
                total_columns=10,
                overall_quality_score=1.5,  # Over 1.0
            )

        with pytest.raises(ValidationError):
            DataQualitySummary(
                total_columns=10,
                overall_quality_score=-0.1,  # Below 0.0
            )


# ---------------------------------------------------------------------------
# Tests for ExecutiveSummary
# ---------------------------------------------------------------------------
class TestExecutiveSummary:
    """Tests for ExecutiveSummary model."""

    def test_creation(self):
        """Test creating ExecutiveSummary."""
        es = ExecutiveSummary(
            table_description="Customer transaction data with purchase history",
            key_findings=["High null rate in email column", "Revenue is right-skewed"],
            immediate_actions=["Fix email data quality", "Log transform revenue"],
            data_quality_summary="Overall good quality with some null issues",
            feature_engineering_opportunities="Date parts extraction possible",
            cross_column_insights="Revenue/orders forms AOV ratio",
        )
        assert "Customer transaction" in es.table_description
        assert len(es.key_findings) == 2
        assert len(es.immediate_actions) == 2

    def test_table_description_required(self):
        """Test that table_description is required."""
        with pytest.raises(ValidationError):
            ExecutiveSummary()  # Missing required field

    def test_defaults(self):
        """Test default values."""
        es = ExecutiveSummary(
            table_description="Test table",
        )
        assert es.key_findings == []
        assert es.immediate_actions == []
        assert es.data_quality_summary == ""


# ---------------------------------------------------------------------------
# Tests for LLMDataDoctorResult
# ---------------------------------------------------------------------------
class TestLLMDataDoctorResult:
    """Tests for LLMDataDoctorResult model."""

    def test_creation(self):
        """Test creating complete LLM result."""
        result = LLMDataDoctorResult(
            executive_summary=ExecutiveSummary(
                table_description="Test data",
            ),
            data_quality=DataQualitySummary(
                total_columns=10,
            ),
        )
        assert result.executive_summary.table_description == "Test data"
        assert result.data_quality.total_columns == 10
        assert result.column_analyses == []
        assert result.column_relationships == []

    def test_with_analyses(self):
        """Test result with column analyses."""
        analysis = ColumnAnalysis(
            column_name="revenue",
            business_purpose="Sales metric",
        )
        result = LLMDataDoctorResult(
            executive_summary=ExecutiveSummary(table_description="Sales data"),
            data_quality=DataQualitySummary(total_columns=5),
            column_analyses=[analysis],
        )
        assert len(result.column_analyses) == 1
        assert result.column_analyses[0].column_name == "revenue"


# ---------------------------------------------------------------------------
# Tests for LLMSuggestion
# ---------------------------------------------------------------------------
class TestLLMSuggestion:
    """Tests for LLMSuggestion model."""

    def test_creation(self):
        """Test creating LLMSuggestion."""
        s = LLMSuggestion(
            priority=Priority.HIGH,
            category="transform",
            column="revenue",
            message="Apply log transform for better distribution",
        )
        assert s.priority == Priority.HIGH
        assert s.category == "transform"
        assert s.confidence == 0.8  # default
        assert s.source == "rule"  # default

    def test_all_categories(self):
        """Test that all valid categories work."""
        valid_categories = ["fill", "transform", "kpi", "ratio", "derived", "quality"]
        for cat in valid_categories:
            s = LLMSuggestion(
                priority=Priority.LOW,
                category=cat,
                column="test",
                message="Test message",
            )
            assert s.category == cat

    def test_invalid_category(self):
        """Test that invalid categories are rejected."""
        with pytest.raises(ValidationError):
            LLMSuggestion(
                priority=Priority.LOW,
                category="invalid",
                column="test",
                message="Test",
            )

    def test_to_dict(self):
        """Test to_dict serialization."""
        s = LLMSuggestion(
            priority=Priority.HIGH,
            category="kpi",
            column="revenue",
            message="KPI candidate",
            rationale="Key business metric",
            yaml_snippet="- type: zsml\n  source_col: revenue",
            confidence=0.95,
            source="llm",
        )
        d = s.to_dict()
        assert d["priority"] == "HIGH"
        assert d["category"] == "kpi"
        assert d["column"] == "revenue"
        assert d["confidence"] == 0.95
        assert d["source"] == "llm"
        assert d["yaml_snippet"].startswith("- type: zsml")

    def test_confidence_bounds(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            LLMSuggestion(
                priority=Priority.LOW,
                category="fill",
                column="test",
                message="Test",
                confidence=1.5,
            )

    def test_source_values(self):
        """Test valid source values."""
        for source in ["rule", "llm", "hybrid"]:
            s = LLMSuggestion(
                priority=Priority.LOW,
                category="fill",
                column="test",
                message="Test",
                source=source,
            )
            assert s.source == source
