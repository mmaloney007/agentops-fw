"""
Pydantic models for LLM-enhanced Data Doctor analysis.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TransformationType(str, Enum):
    """Types of data transformations."""

    LOG_TRANSFORM = "log_transform"
    WINSORIZE = "winsorize"
    BINNING = "binning"
    ZSML = "zsml"
    DATE_PARTS = "date_parts"
    CATEGORICAL_BUCKET = "categorical_bucket"
    FREQUENCY_ENCODE = "frequency_encode"
    STRING_NORMALIZE = "string_normalize"
    RATIO = "ratio"
    DAYS_SINCE = "days_since"
    DERIVED = "derived"
    NONE = "none"


class FillStrategy(str, Enum):
    """Missing value fill strategies."""

    MEDIAN = "median"
    MEAN = "mean"
    MODE = "mode"
    ZERO = "zero"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    UNKNOWN = "Unknown"
    GROUP_MEDIAN = "group_median"
    GROUP_MODE = "group_mode"
    NONE = "none"


class Priority(str, Enum):
    """Suggestion priority levels."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ColumnAnalysis(BaseModel):
    """LLM analysis of a single column."""

    column_name: str = Field(..., description="Name of the column")
    business_purpose: str = Field(
        ...,
        max_length=200,
        description="Inferred business purpose of this column",
    )
    data_quality_issues: list[str] = Field(
        default_factory=list,
        description="List of data quality issues detected",
    )
    suggested_transformation: TransformationType = Field(
        default=TransformationType.NONE,
        description="Recommended transformation type",
    )
    transformation_rationale: str = Field(
        default="",
        max_length=300,
        description="Why this transformation would help",
    )
    fill_strategy: FillStrategy = Field(
        default=FillStrategy.NONE,
        description="Recommended fill strategy for nulls",
    )
    fill_rationale: str = Field(
        default="",
        max_length=200,
        description="Why this fill strategy is appropriate",
    )
    is_kpi_candidate: bool = Field(
        default=False,
        description="Whether this column is a good KPI candidate",
    )
    kpi_rationale: str = Field(
        default="",
        max_length=200,
        description="Why this is/isn't a KPI candidate",
    )
    priority: Priority = Field(
        default=Priority.LOW,
        description="Priority level for addressing this column",
    )

    class Config:
        use_enum_values = True


class ColumnRelationship(BaseModel):
    """A detected relationship between columns."""

    columns: list[str] = Field(
        ...,
        min_length=2,
        description="Columns involved in this relationship",
    )
    relationship_type: Literal[
        "ratio", "derived", "correlated", "grouped", "temporal"
    ] = Field(
        ...,
        description="Type of relationship",
    )
    description: str = Field(
        ...,
        max_length=300,
        description="Description of the relationship",
    )
    suggested_feature: str = Field(
        default="",
        max_length=100,
        description="Suggested derived feature name",
    )
    yaml_snippet: str = Field(
        default="",
        description="Ready-to-use YAML config",
    )
    business_value: str = Field(
        default="",
        max_length=200,
        description="Business value of this derived feature",
    )
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Priority level",
    )

    class Config:
        use_enum_values = True


class DataQualitySummary(BaseModel):
    """Summary of data quality issues."""

    total_columns: int = Field(..., description="Total number of columns")
    columns_with_nulls: int = Field(
        default=0,
        description="Number of columns with missing values",
    )
    high_null_columns: list[str] = Field(
        default_factory=list,
        description="Columns with >20% nulls",
    )
    high_cardinality_columns: list[str] = Field(
        default_factory=list,
        description="Categorical columns with excessive unique values",
    )
    potential_id_columns: list[str] = Field(
        default_factory=list,
        description="Columns that might be IDs",
    )
    kpi_candidates: list[str] = Field(
        default_factory=list,
        description="Columns suitable for KPI analysis",
    )
    overall_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall data quality score (0-1)",
    )


class ExecutiveSummary(BaseModel):
    """Executive summary of data analysis."""

    table_description: str = Field(
        ...,
        max_length=500,
        description="Plain-English description of what this data represents",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Top 3-5 key findings about the data",
    )
    immediate_actions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Top 1-3 immediate actions recommended",
    )
    data_quality_summary: str = Field(
        default="",
        max_length=300,
        description="Brief summary of data quality status",
    )
    feature_engineering_opportunities: str = Field(
        default="",
        max_length=300,
        description="Summary of feature engineering opportunities",
    )
    cross_column_insights: str = Field(
        default="",
        max_length=300,
        description="Insights from multi-column analysis",
    )


class LLMDataDoctorResult(BaseModel):
    """Complete LLM analysis result for Data Doctor."""

    executive_summary: ExecutiveSummary = Field(
        ...,
        description="Executive summary of the analysis",
    )
    data_quality: DataQualitySummary = Field(
        ...,
        description="Data quality summary",
    )
    column_analyses: list[ColumnAnalysis] = Field(
        default_factory=list,
        description="Per-column analysis results",
    )
    column_relationships: list[ColumnRelationship] = Field(
        default_factory=list,
        description="Detected cross-column relationships",
    )
    table_context: str = Field(
        default="",
        max_length=200,
        description="Inferred context about the table",
    )


class LLMSuggestion(BaseModel):
    """Enhanced suggestion with LLM-derived insights."""

    priority: Priority = Field(..., description="Priority level")
    category: Literal["fill", "transform", "kpi", "ratio", "derived", "quality"] = (
        Field(
            ...,
            description="Category of suggestion",
        )
    )
    column: str = Field(..., description="Column(s) this applies to")
    message: str = Field(..., max_length=500, description="Suggestion message")
    rationale: str = Field(
        default="",
        max_length=300,
        description="Why this suggestion would help",
    )
    yaml_snippet: str = Field(default="", description="Ready-to-use YAML config")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in this suggestion",
    )
    source: Literal["rule", "llm", "hybrid"] = Field(
        default="rule",
        description="Source of this suggestion",
    )

    class Config:
        use_enum_values = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "priority": self.priority
            if isinstance(self.priority, str)
            else self.priority.value,
            "category": self.category,
            "column": self.column,
            "message": self.message,
            "rationale": self.rationale,
            "yaml_snippet": self.yaml_snippet,
            "confidence": self.confidence,
            "source": self.source,
        }


__all__ = [
    "TransformationType",
    "FillStrategy",
    "Priority",
    "ColumnAnalysis",
    "ColumnRelationship",
    "DataQualitySummary",
    "ExecutiveSummary",
    "LLMDataDoctorResult",
    "LLMSuggestion",
]
