"""
LLM-enhanced Data Doctor analysis.

Provides semantic analysis of data columns and cross-column relationships
using LLM providers.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any, Dict, List

import dask.dataframe as dd

from .data_doctor_models import (
    DataQualitySummary,
    ExecutiveSummary,
    LLMDataDoctorResult,
    LLMSuggestion,
    Priority,
    TransformationType,
)
from .llm import LLMProvider, LLMResponseCache

logger = logging.getLogger(__name__)


def _build_column_profile(
    col: str,
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a profile dict for a single column.

    Args:
        col: Column name.
        ddf: Dask DataFrame.
        data_dict: Data dictionary with column metadata.

    Returns:
        Profile dict with stats.
    """
    # Get metadata from data_dict
    columns = data_dict.get("columns", {})
    if isinstance(columns, dict):
        col_meta = columns.get(col, {})
    elif isinstance(columns, list):
        col_meta = next(
            (c for c in columns if c.get("column_name") == col or c.get("name") == col),
            {},
        )
    else:
        col_meta = {}

    dtype = str(ddf[col].dtype)
    null_count = int(col_meta.get("null_count", 0) or 0)
    unique_count = int(col_meta.get("unique_count", 0) or 0)
    row_count = int(data_dict.get("row_count", 0) or len(ddf))

    # Get sample values
    try:
        samples = ddf[col].head(10).dropna().tolist()
        samples = [str(v)[:50] for v in samples[:5]]  # Limit sample size
    except Exception:
        samples = []

    return {
        "name": col,
        "dtype": dtype,
        "null_count": null_count,
        "null_pct": (null_count / row_count * 100) if row_count > 0 else 0,
        "unique_count": unique_count,
        "unique_pct": (unique_count / row_count * 100) if row_count > 0 else 0,
        "samples": samples,
        "column_type": col_meta.get("column_type", "unknown"),
        "definition": col_meta.get("definition", ""),
    }


def _build_analysis_prompt(
    column_profiles: List[Dict[str, Any]],
    table_context: str | None = None,
    organization_context: str | None = None,
) -> str:
    """Build the LLM prompt for data analysis.

    Args:
        column_profiles: List of column profile dicts.
        table_context: Optional table description.
        organization_context: Optional organization context.

    Returns:
        Formatted prompt string.
    """
    profiles_text = "\n".join(
        [
            f"- {p['name']}: dtype={p['dtype']}, nulls={p['null_pct']:.1f}%, "
            f"unique={p['unique_count']}, type={p['column_type']}, samples={p['samples'][:3]}"
            for p in column_profiles
        ]
    )

    return textwrap.dedent(f"""\
        Analyze this dataset and provide recommendations for data preparation and feature engineering.

        Organization: {organization_context or "Not specified"}
        Table Context: {table_context or "Not specified"}

        Columns ({len(column_profiles)} total):
        {profiles_text}

        Please provide:

        1. EXECUTIVE SUMMARY:
           - What does this data represent?
           - Top 3-5 key findings
           - Top 1-3 immediate actions needed
           - Data quality summary (one sentence)
           - Feature engineering opportunities (one sentence)
           - Cross-column insights (one sentence)

        2. DATA QUALITY ASSESSMENT:
           - Overall quality score (0-1)
           - Columns with significant null issues (>20%)
           - High cardinality categorical columns
           - Potential ID columns
           - KPI candidates

        3. COLUMN ANALYSIS (for columns needing attention):
           For each column with issues or opportunities:
           - Business purpose (inferred)
           - Data quality issues
           - Recommended transformation (if any)
           - Why this transformation helps
           - Fill strategy for nulls (if applicable)
           - Is it a KPI candidate?

        4. CROSS-COLUMN RELATIONSHIPS:
           Identify column pairs/groups that could form:
           - Meaningful ratios (e.g., revenue/orders = AOV)
           - Derived features (e.g., end_date - start_date = duration)
           - Correlated columns (potential redundancy)
           - Temporal sequences

        Focus on actionable insights with clear business value.
        Prioritize suggestions by impact.
    """)


SYSTEM_PROMPT = """You are an expert data scientist analyzing datasets for machine learning preparation.
Provide practical, actionable recommendations for data quality improvement and feature engineering.
Focus on business value and explain WHY each recommendation would help.
Be concise but thorough. Prioritize by impact."""


def analyze_with_llm(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
    provider: LLMProvider,
    *,
    table_context: str | None = None,
    organization_context: str | None = None,
    cache: LLMResponseCache | None = None,
    max_columns: int = 30,
) -> LLMDataDoctorResult:
    """Analyze data using LLM for enhanced insights.

    Args:
        ddf: Dask DataFrame to analyze.
        data_dict: Data dictionary with column metadata.
        provider: LLM provider to use.
        table_context: Optional table description.
        organization_context: Optional organization context.
        cache: Optional response cache.
        max_columns: Maximum columns to include in prompt.

    Returns:
        LLMDataDoctorResult with analysis.
    """
    # Build column profiles
    columns = list(ddf.columns)[:max_columns]
    profiles = [_build_column_profile(col, ddf, data_dict) for col in columns]

    # Build prompt
    prompt = _build_analysis_prompt(profiles, table_context, organization_context)

    # Check cache
    if cache:
        cached = cache.get(prompt, provider.name, LLMDataDoctorResult)
        if cached:
            logger.debug("[data-doctor-llm] Using cached analysis")
            return cached

    # Call LLM
    logger.info("[data-doctor-llm] Analyzing %d columns with LLM...", len(columns))
    try:
        result = provider.complete_structured(
            prompt=prompt,
            response_model=LLMDataDoctorResult,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,  # Slight variation for creativity
            max_tokens=4096,
        )

        # Cache result
        if cache:
            cache.set(prompt, provider.name, result)

        return result

    except Exception as e:
        logger.warning("[data-doctor-llm] LLM analysis failed: %s", e)
        # Return minimal result on failure
        return _create_fallback_result(ddf, data_dict, profiles)


def _create_fallback_result(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
    profiles: List[Dict[str, Any]],
) -> LLMDataDoctorResult:
    """Create a fallback result when LLM fails.

    Args:
        ddf: Dask DataFrame.
        data_dict: Data dictionary.
        profiles: Column profiles.

    Returns:
        Basic LLMDataDoctorResult.
    """
    columns_with_nulls = [p["name"] for p in profiles if p["null_count"] > 0]
    high_null = [p["name"] for p in profiles if p["null_pct"] > 20]

    return LLMDataDoctorResult(
        executive_summary=ExecutiveSummary(
            table_description="Data table (LLM analysis unavailable)",
            key_findings=["LLM analysis failed - using rule-based fallback"],
            immediate_actions=["Review columns with high null rates"],
            data_quality_summary=f"{len(columns_with_nulls)} columns have missing values",
            feature_engineering_opportunities="Run rule-based analysis for suggestions",
            cross_column_insights="Multi-column analysis requires LLM",
        ),
        data_quality=DataQualitySummary(
            total_columns=len(ddf.columns),
            columns_with_nulls=len(columns_with_nulls),
            high_null_columns=high_null,
            high_cardinality_columns=[],
            potential_id_columns=[],
            kpi_candidates=[],
            overall_quality_score=0.5,
        ),
        column_analyses=[],
        column_relationships=[],
        table_context="Unknown",
    )


def generate_executive_summary(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
    provider: LLMProvider,
    *,
    table_context: str | None = None,
    organization_context: str | None = None,
    cache: LLMResponseCache | None = None,
) -> ExecutiveSummary:
    """Generate just the executive summary using LLM.

    This is a lighter-weight call when you only need the summary.

    Args:
        ddf: Dask DataFrame.
        data_dict: Data dictionary.
        provider: LLM provider.
        table_context: Optional table description.
        organization_context: Optional organization context.
        cache: Optional response cache.

    Returns:
        ExecutiveSummary.
    """
    # Build a simpler prompt for just the summary
    columns = list(ddf.columns)[:20]
    row_count = int(data_dict.get("row_count", 0) or len(ddf))

    # Get basic stats
    col_info = []
    for col in columns:
        dtype = str(ddf[col].dtype)
        col_info.append(f"{col} ({dtype})")

    prompt = textwrap.dedent(f"""\
        Provide an executive summary for this dataset:

        Organization: {organization_context or "Not specified"}
        Table: {table_context or "Data table"}
        Rows: {row_count:,}
        Columns: {len(ddf.columns)}

        Column names and types:
        {", ".join(col_info)}

        Provide:
        1. A plain-English description of what this data represents (1-2 sentences)
        2. Top 3-5 key findings or observations
        3. Top 1-3 immediate actions recommended
        4. One-sentence data quality summary
        5. One-sentence feature engineering opportunities
        6. One-sentence cross-column insights
    """)

    if cache:
        cached = cache.get(prompt, provider.name, ExecutiveSummary)
        if cached:
            return cached

    try:
        result = provider.complete_structured(
            prompt=prompt,
            response_model=ExecutiveSummary,
            system_prompt="You are a data analyst providing executive summaries. Be concise and actionable.",
            temperature=0.1,
            max_tokens=1024,
        )

        if cache:
            cache.set(prompt, provider.name, result)

        return result

    except Exception as e:
        logger.warning("[data-doctor-llm] Summary generation failed: %s", e)
        return ExecutiveSummary(
            table_description="Data table (summary generation failed)",
            key_findings=["Unable to generate LLM summary"],
            immediate_actions=["Review data manually"],
            data_quality_summary="Unknown",
            feature_engineering_opportunities="Unknown",
            cross_column_insights="Unknown",
        )


def convert_llm_result_to_suggestions(
    result: LLMDataDoctorResult,
) -> List[LLMSuggestion]:
    """Convert LLM result to list of suggestions.

    Args:
        result: LLM analysis result.

    Returns:
        List of LLMSuggestion objects.
    """
    suggestions: List[LLMSuggestion] = []

    # Convert column analyses to suggestions
    for analysis in result.column_analyses:
        # Get string values for enums (handles both enum and string due to use_enum_values)
        transform_val = (
            analysis.suggested_transformation.value
            if hasattr(analysis.suggested_transformation, "value")
            else str(analysis.suggested_transformation)
        )
        fill_val = (
            analysis.fill_strategy.value
            if hasattr(analysis.fill_strategy, "value")
            else str(analysis.fill_strategy)
        )

        # Transformation suggestion
        if transform_val != "none":
            yaml_snippet = _generate_transform_yaml(
                analysis.column_name,
                analysis.suggested_transformation,
            )
            suggestions.append(
                LLMSuggestion(
                    priority=analysis.priority,
                    category="transform",
                    column=analysis.column_name,
                    message=f"Apply {transform_val}: {analysis.transformation_rationale}",
                    rationale=analysis.transformation_rationale,
                    yaml_snippet=yaml_snippet,
                    confidence=0.85,
                    source="llm",
                )
            )

        # Fill strategy suggestion
        if fill_val != "none":
            yaml_snippet = f"""fill:
  overrides:
    {analysis.column_name}:
      strategy: {fill_val}"""
            suggestions.append(
                LLMSuggestion(
                    priority=analysis.priority,
                    category="fill",
                    column=analysis.column_name,
                    message=f"Fill nulls with {fill_val}: {analysis.fill_rationale}",
                    rationale=analysis.fill_rationale,
                    yaml_snippet=yaml_snippet,
                    confidence=0.8,
                    source="llm",
                )
            )

        # KPI suggestion
        if analysis.is_kpi_candidate:
            yaml_snippet = f"""- type: zsml
  source_col: {analysis.column_name}
  out_col: {analysis.column_name}_tier
  quantiles: [0.33, 0.66]
  kpi: true"""
            suggestions.append(
                LLMSuggestion(
                    priority=Priority.HIGH,
                    category="kpi",
                    column=analysis.column_name,
                    message=f"KPI candidate: {analysis.kpi_rationale}",
                    rationale=analysis.kpi_rationale,
                    yaml_snippet=yaml_snippet,
                    confidence=0.9,
                    source="llm",
                )
            )

    # Convert relationships to suggestions
    for rel in result.column_relationships:
        suggestions.append(
            LLMSuggestion(
                priority=rel.priority,
                category="derived" if rel.relationship_type == "derived" else "ratio",
                column=" + ".join(rel.columns),
                message=rel.description,
                rationale=rel.business_value,
                yaml_snippet=rel.yaml_snippet,
                confidence=0.75,
                source="llm",
            )
        )

    return suggestions


def _generate_transform_yaml(
    column: str,
    transform_type: TransformationType | str,
) -> str:
    """Generate YAML snippet for a transformation.

    Args:
        column: Column name.
        transform_type: Type of transformation (enum or string).

    Returns:
        YAML snippet string.
    """
    # Handle both enum and string values (due to use_enum_values=True in models)
    type_str = (
        transform_type.value
        if hasattr(transform_type, "value")
        else str(transform_type)
    )

    templates = {
        "log_transform": f"""- type: log_transform
  source_col: {column}
  out_col: {column}_log
  log_method: log1p""",
        "winsorize": f"""- type: winsorize
  source_col: {column}
  lower_quantile: 0.01
  upper_quantile: 0.99
  out_col: {column}_winsorized""",
        "binning": f"""- type: binning
  source_col: {column}
  out_col: {column}_bin
  quantiles: [0.2, 0.4, 0.6, 0.8]""",
        "zsml": f"""- type: zsml
  source_col: {column}
  out_col: {column}_tier
  quantiles: [0.33, 0.66]
  kpi: true""",
        "date_parts": f"""- type: date_parts
  source_col: {column}
  date_parts: [year, month, day_of_week, quarter]
  drop_source: true""",
        "categorical_bucket": f"""- type: categorical_bucket
  source_col: {column}
  top_k: 10
  out_col: {column}_bucket
  other_label: other""",
        "frequency_encode": f"""- type: frequency_encode
  source_col: {column}
  out_col: {column}_freq
  normalize: true""",
        "string_normalize": f"""- type: string_normalize
  source_col: {column}
  string_case: lower
  strip: true""",
        "days_since": f"""- type: days_since
  source_col: {column}
  out_col: days_since_{column}""",
    }

    return templates.get(type_str, f"# Transform {column} with {type_str}")


def format_executive_summary(summary: ExecutiveSummary) -> str:
    """Format executive summary for display.

    Args:
        summary: Executive summary to format.

    Returns:
        Formatted string.
    """
    lines = [
        "",
        "=" * 70,
        "  EXECUTIVE SUMMARY",
        "=" * 70,
        "",
        f"  {summary.table_description}",
        "",
        "  KEY FINDINGS:",
    ]

    for i, finding in enumerate(summary.key_findings, 1):
        lines.append(f"    {i}. {finding}")

    lines.extend(
        [
            "",
            "  IMMEDIATE ACTIONS:",
        ]
    )

    for i, action in enumerate(summary.immediate_actions, 1):
        lines.append(f"    {i}. {action}")

    lines.extend(
        [
            "",
            f"  Data Quality: {summary.data_quality_summary}",
            f"  Feature Engineering: {summary.feature_engineering_opportunities}",
            f"  Cross-Column Insights: {summary.cross_column_insights}",
            "",
            "=" * 70,
        ]
    )

    return "\n".join(lines)


__all__ = [
    "analyze_with_llm",
    "generate_executive_summary",
    "convert_llm_result_to_suggestions",
    "format_executive_summary",
    "LLMDataDoctorResult",
    "ExecutiveSummary",
    "LLMSuggestion",
]
