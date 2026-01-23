#!/usr/bin/env python3
"""
data_doctor.py
--------------
Post-data-dictionary analysis module for suggesting improvements.

Analyzes column metadata after data dictionary is built and provides
rule-based suggestions for:
    - Fill strategies for missing values
    - Feature engineering functions (log_transform, winsorize, date_parts, etc.)
    - KPI candidates for ZSML tiering
    - Ratio opportunities between related columns

Usage:
    from neuralift_c360_prep.data_doctor import analyze_data, print_report

    report = analyze_data(ddf, data_dict, cfg)
    print_report(report)

Author: Mike Maloney - Neuralift, Inc.
Updated: 2026-01-13
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import dask.dataframe as dd
import yaml
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity Signals (fun indicators!)
# ---------------------------------------------------------------------------
SEVERITY_HIGH = "\U0001f6a8"  # Police car light - urgent attention needed
SEVERITY_MEDIUM = "\U0001f9ea"  # Test tube - worth experimenting with
SEVERITY_LOW = "\U0001f4a1"  # Light bulb - nice-to-have idea

SEVERITY_LABELS = {
    SEVERITY_HIGH: "HIGH",
    SEVERITY_MEDIUM: "MEDIUM",
    SEVERITY_LOW: "LOW",
    # Legacy support
    "!": "HIGH",
    "~": "MEDIUM",
    "?": "LOW",
}

SEVERITY_DESCRIPTIONS = {
    SEVERITY_HIGH: "Action recommended - significant data quality issue",
    SEVERITY_MEDIUM: "Worth considering - could improve model performance",
    SEVERITY_LOW: "Optional enhancement - nice to have",
}


# ---------------------------------------------------------------------------
# Business-Friendly Fill Alternatives
# ---------------------------------------------------------------------------
BUSINESS_FILL_ALTERNATIVES = {
    "numeric": [
        {"strategy": "median", "description": "Use middle value (robust to outliers)"},
        {"strategy": "mean", "description": "Use average value"},
        {
            "strategy": "group_median",
            "description": "Median by customer segment",
            "requires": "segment column",
        },
        {"strategy": "forward_fill", "description": "Carry forward last known value"},
        {"strategy": "interpolate", "description": "Estimate from surrounding values"},
        {
            "strategy": "industry_benchmark",
            "description": "Use industry standard value",
            "requires": "benchmark data",
        },
        {"strategy": "zero", "description": "Fill with zero (for additive metrics)"},
    ],
    "categorical": [
        {"strategy": "mode", "description": "Use most common value"},
        {"strategy": "Unknown", "description": "Explicit 'Unknown' category"},
        {
            "strategy": "group_mode",
            "description": "Mode by customer segment",
            "requires": "segment column",
        },
        {"strategy": "Not Provided", "description": "Business-friendly missing label"},
        {"strategy": "Default", "description": "Use business default category"},
    ],
    "datetime": [
        {"strategy": "forward_fill", "description": "Carry forward last known date"},
        {"strategy": "backward_fill", "description": "Use next available date"},
        {
            "strategy": "reference_date",
            "description": "Use account creation date",
            "requires": "reference column",
        },
        {"strategy": "epoch", "description": "Use epoch start (1970-01-01)"},
    ],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FillAlternative:
    """A business-friendly fill alternative suggestion."""

    strategy: str
    description: str
    requires: Optional[str] = None


@dataclass
class Suggestion:
    """A single improvement suggestion."""

    priority: str  # SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW (or legacy !, ~, ?)
    category: str
    column: str
    message: str
    yaml_snippet: str
    alternatives: List[FillAlternative] = field(default_factory=list)


@dataclass
class DataDoctorReport:
    """Full report from data doctor analysis."""

    feature_suggestions: List[Suggestion] = field(default_factory=list)
    fill_suggestions: List[Suggestion] = field(default_factory=list)
    kpi_candidates: List[Suggestion] = field(default_factory=list)
    ratio_opportunities: List[Suggestion] = field(default_factory=list)
    yaml_text: str = ""
    # LLM-enhanced fields (optional)
    _llm_executive_summary: Optional[Any] = field(default=None, repr=False)
    _llm_suggestions: List[Any] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Pattern matchers for column names
# ---------------------------------------------------------------------------
_MONETARY_PATTERNS = [
    r"amount",
    r"spend",
    r"revenue",
    r"price",
    r"cost",
    r"fee",
    r"total",
    r"sales",
    r"income",
    r"payment",
    r"balance",
]
_KPI_PATTERNS = [
    r"revenue",
    r"spend",
    r"purchase",
    r"conversion",
    r"orders",
    r"sales",
    r"profit",
    r"margin",
    r"lifetime_value",
    r"ltv",
    r"clv",
]
_DATE_PATTERNS = [
    r"date",
    r"_dt$",
    r"_at$",
    r"timestamp",
    r"created",
    r"updated",
    r"time$",
    r"datetime",
]


def _matches_pattern(name: str, patterns: List[str]) -> bool:
    """Check if column name matches any of the patterns."""
    name_lower = name.lower()
    for pattern in patterns:
        if re.search(pattern, name_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Column metadata extraction
# ---------------------------------------------------------------------------
def build_minimal_data_dict(
    ddf: dd.DataFrame,
    *,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Build a minimal data dictionary from a Dask DataFrame using stats_cache.

    This is useful for standalone data doctor usage when no pre-built
    data dictionary is available.

    Args:
        ddf: Dask DataFrame to analyze.
        show_progress: Whether to log progress.

    Returns:
        Minimal data dictionary with column stats.
    """
    from .stats_cache import StatsCache

    if show_progress:
        logger.info("[data-doctor] Building minimal data dict from DataFrame...")

    cache = StatsCache(show_progress=show_progress)
    cols = list(ddf.columns)

    # Compute all stats in one optimized pass
    row_count, null_counts, unique_counts = cache.compute_all_stats(
        ddf, cols, approx_unique=True
    )

    columns: Dict[str, Dict[str, Any]] = {}
    for col in cols:
        dtype = ddf[col].dtype
        dtype_str = str(dtype).upper()

        # Determine column type based on dtype
        if "int" in dtype_str.lower() or "float" in dtype_str.lower():
            col_type = "continuous"
        elif "datetime" in dtype_str.lower():
            col_type = "datetime"
        elif "bool" in dtype_str.lower():
            col_type = "binary"
        else:
            col_type = "categorical"

        columns[col] = {
            "column_name": col,
            "column_type": col_type,
            "data_type": dtype_str,
            "null_count": int(null_counts.get(col, 0)),
            "unique_count": int(unique_counts.get(col, 0)),
        }

    if show_progress:
        logger.info(
            "[data-doctor] Minimal data dict built: %d columns, %d rows",
            len(cols),
            row_count,
        )

    return {"columns": columns, "row_count": row_count}


def _get_column_meta(col_name: str, data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata for a column from the data dictionary."""
    columns = data_dict.get("columns", {})
    if isinstance(columns, dict):
        return columns.get(col_name, {})
    elif isinstance(columns, list):
        for col in columns:
            if col.get("column_name") == col_name or col.get("name") == col_name:
                return col
    return {}


def _get_null_count(col_meta: Dict[str, Any]) -> int:
    """Extract null count from column metadata."""
    return int(col_meta.get("null_count", 0) or 0)


def _get_unique_count(col_meta: Dict[str, Any]) -> int:
    """Extract unique count from column metadata."""
    return int(col_meta.get("unique_count", 0) or 0)


def _get_column_type(col_meta: Dict[str, Any]) -> str:
    """Extract column type from metadata."""
    return str(
        col_meta.get("column_type", "") or col_meta.get("type", "") or ""
    ).lower()


def _get_data_type(col_meta: Dict[str, Any]) -> str:
    """Extract data type from metadata."""
    return str(col_meta.get("data_type", "") or col_meta.get("dtype", "") or "").upper()


# ---------------------------------------------------------------------------
# Rule-based analysis functions
# ---------------------------------------------------------------------------
def _analyze_null_columns(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
    *,
    threshold_pct: float = 0.0,
    high_null_threshold: int = 100,
    show_business_alternatives: bool = True,
) -> List[Suggestion]:
    """Analyze columns with missing values and suggest fill strategies."""
    suggestions: List[Suggestion] = []

    for col in ddf.columns:
        col_meta = _get_column_meta(col, data_dict)
        null_count = _get_null_count(col_meta)

        if null_count <= 0:
            continue

        # Determine fill strategy and alternatives based on dtype
        dtype = ddf[col].dtype
        col_type = _get_column_type(col_meta)
        alternatives: List[FillAlternative] = []

        if is_numeric_dtype(dtype) and col_type not in ("id", "kpi"):
            fill_method = "median"
            snippet = f"""fill:
  overrides:
    {col}:
      strategy: {fill_method}"""
            if show_business_alternatives:
                for alt in BUSINESS_FILL_ALTERNATIVES["numeric"]:
                    alternatives.append(
                        FillAlternative(
                            strategy=alt["strategy"],
                            description=alt["description"],
                            requires=alt.get("requires"),
                        )
                    )
        elif is_datetime64_any_dtype(dtype):
            fill_method = "forward_fill"
            snippet = f"""fill:
  overrides:
    {col}:
      strategy: {fill_method}"""
            if show_business_alternatives:
                for alt in BUSINESS_FILL_ALTERNATIVES["datetime"]:
                    alternatives.append(
                        FillAlternative(
                            strategy=alt["strategy"],
                            description=alt["description"],
                            requires=alt.get("requires"),
                        )
                    )
        else:
            fill_method = "mode"
            snippet = f"""fill:
  overrides:
    {col}:
      strategy: {fill_method}"""
            if show_business_alternatives:
                for alt in BUSINESS_FILL_ALTERNATIVES["categorical"]:
                    alternatives.append(
                        FillAlternative(
                            strategy=alt["strategy"],
                            description=alt["description"],
                            requires=alt.get("requires"),
                        )
                    )

        priority = (
            SEVERITY_HIGH if null_count > high_null_threshold else SEVERITY_MEDIUM
        )
        suggestions.append(
            Suggestion(
                priority=priority,
                category="fill",
                column=col,
                message=f"Column has {null_count} null values. Consider {fill_method} fill.",
                yaml_snippet=snippet,
                alternatives=alternatives,
            )
        )

    return suggestions


def _analyze_monetary_columns(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
) -> List[Suggestion]:
    """Analyze monetary columns and suggest log_transform for skewed distributions."""
    suggestions: List[Suggestion] = []

    for col in ddf.columns:
        if not _matches_pattern(col, _MONETARY_PATTERNS):
            continue

        dtype = ddf[col].dtype
        if not is_numeric_dtype(dtype):
            continue

        col_meta = _get_column_meta(col, data_dict)
        col_type = _get_column_type(col_meta)

        if col_type in ("id", "kpi"):
            continue

        snippet = f"""- type: log_transform
  source_col: {col}
  out_col: {col}_log
  log_method: log1p"""

        suggestions.append(
            Suggestion(
                priority=SEVERITY_MEDIUM,
                category="feature",
                column=col,
                message="Monetary/amount columns often have right-skewed distributions. Log transform helps normalize.",
                yaml_snippet=snippet,
            )
        )

    return suggestions


def _analyze_high_cardinality_categoricals(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
    *,
    high_card_threshold: int = 50,
    top_k: int = 10,
) -> List[Suggestion]:
    """Analyze high-cardinality categorical columns and suggest bucketing."""
    suggestions: List[Suggestion] = []

    for col in ddf.columns:
        col_meta = _get_column_meta(col, data_dict)
        unique_count = _get_unique_count(col_meta)
        col_type = _get_column_type(col_meta)
        data_type = _get_data_type(col_meta)

        if col_type in ("id", "kpi", "continuous"):
            continue

        if data_type not in ("STRING", "OBJECT"):
            continue

        if unique_count <= high_card_threshold:
            continue

        snippet = f"""- type: categorical_bucket
  source_col: {col}
  top_k: {top_k}
  out_col: {col}_bucket
  other_label: other"""

        suggestions.append(
            Suggestion(
                priority=SEVERITY_MEDIUM,
                category="feature",
                column=col,
                message=f"High cardinality ({unique_count} uniques). Bucketing keeps top {top_k}, groups rest as 'other'.",
                yaml_snippet=snippet,
            )
        )

    return suggestions


def _analyze_date_columns(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
) -> List[Suggestion]:
    """Analyze date columns and suggest date_parts extraction."""
    suggestions: List[Suggestion] = []

    for col in ddf.columns:
        dtype = ddf[col].dtype

        # Check if datetime type or matches date patterns
        is_datetime = is_datetime64_any_dtype(dtype)
        matches_date_name = _matches_pattern(col, _DATE_PATTERNS)

        if not (is_datetime or matches_date_name):
            continue

        col_meta = _get_column_meta(col, data_dict)
        col_type = _get_column_type(col_meta)

        if col_type in ("id", "kpi"):
            continue

        snippet = f"""- type: date_parts
  source_col: {col}
  date_parts: [year, month, day_of_week, quarter]
  drop_source: true"""

        suggestions.append(
            Suggestion(
                priority=SEVERITY_LOW,
                category="feature",
                column=col,
                message="Date column detected. Extract parts (year, month, day_of_week) for seasonality patterns.",
                yaml_snippet=snippet,
            )
        )

    return suggestions


def _analyze_kpi_candidates(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
) -> List[Suggestion]:
    """Identify potential KPI columns for ZSML tiering or binning."""
    suggestions: List[Suggestion] = []

    for col in ddf.columns:
        if not _matches_pattern(col, _KPI_PATTERNS):
            continue

        dtype = ddf[col].dtype
        if not is_numeric_dtype(dtype):
            continue

        col_meta = _get_column_meta(col, data_dict)
        col_type = _get_column_type(col_meta)

        # Skip if already marked as KPI
        if col_type == "kpi":
            continue

        # Suggest ZSML tiering (new flat format)
        zsml_snippet = f"""- type: zsml
  source_col: {col}
  out_col: {col}_tier
  quantiles: [0.33, 0.66]
  kpi: true"""

        suggestions.append(
            Suggestion(
                priority=SEVERITY_HIGH,
                category="kpi",
                column=col,
                message="Key business metric detected! ZSML creates Low/Medium/High customer segments.",
                yaml_snippet=zsml_snippet,
            )
        )

        # Also suggest binning as an alternative (new flat format)
        binning_snippet = f"""- type: binning
  source_col: {col}
  out_col: {col}_bin
  quantiles: [0.2, 0.4, 0.6, 0.8]
  kpi: true"""

        suggestions.append(
            Suggestion(
                priority=SEVERITY_MEDIUM,
                category="kpi",
                column=col,
                message="Alternative: Fixed-width bins give equal ranges (vs ZSML's equal counts).",
                yaml_snippet=binning_snippet,
            )
        )

    return suggestions


def _analyze_ratio_opportunities(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
) -> List[Suggestion]:
    """Identify pairs of columns that could form meaningful ratios."""
    suggestions: List[Suggestion] = []

    numeric_cols = [col for col in ddf.columns if is_numeric_dtype(ddf[col].dtype)]

    # Common ratio pairs with business context (numerator_pattern, denominator_pattern, description)
    ratio_pairs = [
        (
            r"spend|cost|amount",
            r"income|revenue|budget",
            "Spend-to-income ratio reveals financial behavior",
        ),
        (
            r"orders|purchases",
            r"visits|sessions",
            "Conversion rate shows purchase intent",
        ),
        (r"clicks", r"impressions|views", "Click-through rate measures engagement"),
        (r"conversions", r"visits|sessions|clicks", "Conversion funnel efficiency"),
        (r"revenue|sales", r"orders|transactions", "Average order value (AOV)"),
    ]

    for num_pattern, denom_pattern, description in ratio_pairs:
        num_cols = [c for c in numeric_cols if re.search(num_pattern, c.lower())]
        denom_cols = [c for c in numeric_cols if re.search(denom_pattern, c.lower())]

        for num_col in num_cols:
            for denom_col in denom_cols:
                if num_col == denom_col:
                    continue

                # Generate a clean ratio name
                ratio_name = f"{num_col}_per_{denom_col}"

                snippet = f"""- type: ratio
  numerator_col: {num_col}
  denominator_col: {denom_col}
  out_col: {ratio_name}
  on_zero: zero"""

                suggestions.append(
                    Suggestion(
                        priority=SEVERITY_LOW,
                        category="ratio",
                        column=f"{num_col}/{denom_col}",
                        message=f"{description}: {ratio_name}",
                        yaml_snippet=snippet,
                    )
                )

    return suggestions


# ---------------------------------------------------------------------------
# Scoring and ranking
# ---------------------------------------------------------------------------
# Scoring weights for impact calculation
CATEGORY_WEIGHTS = {
    "kpi": 100,  # KPI candidates are most impactful
    "fill": 85,  # Data quality issues are critical
    "feature": 60,  # Feature engineering is valuable
    "ratio": 40,  # Ratios are nice-to-have
    "transform": 60,  # Same as feature
    "derived": 50,  # Derived features
    "quality": 75,  # General quality issues
}

PRIORITY_MULTIPLIERS = {
    SEVERITY_HIGH: 1.0,
    SEVERITY_MEDIUM: 0.7,
    SEVERITY_LOW: 0.4,
    # Legacy support
    "!": 1.0,
    "~": 0.7,
    "?": 0.4,
    "HIGH": 1.0,
    "MEDIUM": 0.7,
    "LOW": 0.4,
}


def _score_suggestion(suggestion: Suggestion) -> int:
    """Calculate impact score (0-100) for a suggestion.

    Scoring factors:
    - Category base score (KPI=100, fill=85, feature=60, ratio=40)
    - Priority multiplier (HIGH=1.0, MEDIUM=0.7, LOW=0.4)
    - Message-based adjustments for context-specific boosts

    Args:
        suggestion: The Suggestion to score.

    Returns:
        Integer score from 0-100.
    """
    # Get base score from category
    base_score = CATEGORY_WEIGHTS.get(suggestion.category, 50)

    # Apply priority multiplier
    priority_mult = PRIORITY_MULTIPLIERS.get(suggestion.priority, 0.5)
    score = base_score * priority_mult

    # Context-specific boosts from message content
    msg_lower = suggestion.message.lower()

    # Boost for high null counts
    if "null" in msg_lower:
        null_match = re.search(r"(\d+)\s*null", msg_lower)
        if null_match:
            null_count = int(null_match.group(1))
            # Boost proportionally: 100+ nulls = +10, 1000+ = +15, 10000+ = +20
            if null_count >= 10000:
                score += 20
            elif null_count >= 1000:
                score += 15
            elif null_count >= 100:
                score += 10

    # Boost for key business metrics
    if any(
        kw in msg_lower
        for kw in ["key business metric", "kpi", "revenue", "conversion"]
    ):
        score += 10

    # Boost for high cardinality (complex issue)
    if "high cardinality" in msg_lower:
        score += 5

    # Cap at 100
    return min(100, max(0, int(score)))


def _rank_and_limit_suggestions(
    report: "DataDoctorReport",
    *,
    max_suggestions: Optional[int] = None,
    ranking_enabled: bool = True,
) -> "DataDoctorReport":
    """Rank suggestions by score and optionally limit to top N.

    Args:
        report: DataDoctorReport with suggestions.
        max_suggestions: Optional limit on total suggestions (None = no limit).
        ranking_enabled: Whether to sort by score.

    Returns:
        Modified DataDoctorReport with scored and potentially limited suggestions.
    """
    if not ranking_enabled and max_suggestions is None:
        return report

    # Collect all suggestions with their scores and original list reference
    all_suggestions: List[tuple] = []  # (score, list_name, index, suggestion)

    for list_name in [
        "kpi_candidates",
        "fill_suggestions",
        "feature_suggestions",
        "ratio_opportunities",
    ]:
        suggestions = getattr(report, list_name, [])
        for i, s in enumerate(suggestions):
            score = _score_suggestion(s)
            all_suggestions.append((score, list_name, i, s))

    # Sort by score descending
    if ranking_enabled:
        all_suggestions.sort(key=lambda x: x[0], reverse=True)

    # Apply limit if specified
    if max_suggestions is not None and len(all_suggestions) > max_suggestions:
        # Track original total before limiting
        report._original_total = len(all_suggestions)
        all_suggestions = all_suggestions[:max_suggestions]

    # Rebuild lists from sorted/limited results, preserving scores
    new_lists: Dict[str, List[Suggestion]] = {
        "kpi_candidates": [],
        "fill_suggestions": [],
        "feature_suggestions": [],
        "ratio_opportunities": [],
    }

    for score, list_name, _, suggestion in all_suggestions:
        # Store score in suggestion for YAML output (add as attribute if not present)
        suggestion._score = score
        new_lists[list_name].append(suggestion)

    # Update report
    report.kpi_candidates = new_lists["kpi_candidates"]
    report.fill_suggestions = new_lists["fill_suggestions"]
    report.feature_suggestions = new_lists["feature_suggestions"]
    report.ratio_opportunities = new_lists["ratio_opportunities"]

    return report


def _analyze_numeric_outliers(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
    *,
    unique_threshold: int = 100,
) -> List[Suggestion]:
    """Analyze numeric columns with high unique counts and suggest winsorization."""
    suggestions: List[Suggestion] = []

    for col in ddf.columns:
        dtype = ddf[col].dtype
        if not is_numeric_dtype(dtype):
            continue

        col_meta = _get_column_meta(col, data_dict)
        unique_count = _get_unique_count(col_meta)
        col_type = _get_column_type(col_meta)

        if col_type in ("id", "kpi"):
            continue

        if unique_count < unique_threshold:
            continue

        # Check if column name suggests it might have outliers
        outlier_indicators = [
            "amount",
            "spend",
            "revenue",
            "price",
            "age",
            "income",
            "score",
        ]
        if not _matches_pattern(col, outlier_indicators):
            continue

        snippet = f"""- type: winsorize
  source_col: {col}
  lower_quantile: 0.01
  upper_quantile: 0.99
  out_col: {col}_winsorized"""

        suggestions.append(
            Suggestion(
                priority=SEVERITY_MEDIUM,
                category="feature",
                column=col,
                message=f"Continuous data ({unique_count} values). Winsorize caps outliers at 1st/99th percentile.",
                yaml_snippet=snippet,
            )
        )

    return suggestions


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------
def analyze_data(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
    cfg: Optional[Any] = None,
    *,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    show_progress: bool = True,
    show_business_alternatives: bool = True,
    high_null_threshold: int = 100,
    high_cardinality_threshold: int = 50,
    top_k_bucket: int = 10,
) -> DataDoctorReport:
    """Analyze column metadata and generate improvement suggestions.

    Args:
        ddf: Dask DataFrame to analyze.
        data_dict: Data dictionary with column metadata.
        cfg: Optional configuration object (BundleConfig).
        use_llm: Whether to use LLM for additional analysis.
        llm_model: Model to use for LLM analysis.
        show_progress: Whether to log progress.
        show_business_alternatives: Include business-friendly fill alternatives.
        high_null_threshold: Null count threshold for HIGH severity.
        high_cardinality_threshold: Unique count threshold for high-cardinality warnings.
        top_k_bucket: Default top_k for categorical bucketing suggestions.

    Returns:
        DataDoctorReport with categorized suggestions.
    """
    # Extract config values if BundleConfig provided
    llm_enabled = use_llm
    llm_provider_config = None
    llm_cache_dir = ".nl_doctor_cache"
    max_llm_columns = 30
    generate_exec_summary = True
    organization_context = None
    # Ranking and limiting config
    max_suggestions: Optional[int] = None
    ranking_enabled: bool = True

    if cfg is not None:
        dd_cfg = getattr(cfg, "data_doctor", None)
        if dd_cfg is not None:
            show_business_alternatives = getattr(
                dd_cfg, "show_business_alternatives", show_business_alternatives
            )
            high_null_threshold = getattr(
                dd_cfg, "high_null_threshold", high_null_threshold
            )
            high_cardinality_threshold = getattr(
                dd_cfg, "high_cardinality_threshold", high_cardinality_threshold
            )
            top_k_bucket = getattr(dd_cfg, "top_k_bucket", top_k_bucket)
            # LLM settings
            llm_enabled = getattr(dd_cfg, "llm_enabled", llm_enabled)
            llm_provider_config = getattr(dd_cfg, "llm_provider", None)
            llm_cache_dir = getattr(dd_cfg, "llm_cache_dir", llm_cache_dir)
            max_llm_columns = getattr(dd_cfg, "max_llm_columns", max_llm_columns)
            generate_exec_summary = getattr(dd_cfg, "generate_executive_summary", True)
            # Ranking and limiting
            max_suggestions = getattr(dd_cfg, "max_suggestions", None)
            ranking_enabled = getattr(dd_cfg, "ranking_enabled", True)

        # Get organization context from metadata config
        meta_cfg = getattr(cfg, "metadata", None)
        if meta_cfg is not None:
            organization_context = getattr(meta_cfg, "context", None)

    if show_progress:
        logger.info(
            "[data-doctor] Starting analysis for %d columns...", len(ddf.columns)
        )

    report = DataDoctorReport()

    # Run all rule-based analyses
    if show_progress:
        logger.info("[data-doctor] Analyzing missing values...")
    report.fill_suggestions = _analyze_null_columns(
        ddf,
        data_dict,
        high_null_threshold=high_null_threshold,
        show_business_alternatives=show_business_alternatives,
    )

    if show_progress:
        logger.info("[data-doctor] Analyzing monetary columns...")
    monetary = _analyze_monetary_columns(ddf, data_dict)

    if show_progress:
        logger.info("[data-doctor] Analyzing high-cardinality categoricals...")
    high_card = _analyze_high_cardinality_categoricals(
        ddf,
        data_dict,
        high_card_threshold=high_cardinality_threshold,
        top_k=top_k_bucket,
    )

    if show_progress:
        logger.info("[data-doctor] Analyzing date columns...")
    dates = _analyze_date_columns(ddf, data_dict)

    if show_progress:
        logger.info("[data-doctor] Analyzing numeric outliers...")
    outliers = _analyze_numeric_outliers(ddf, data_dict)

    # Combine feature suggestions
    report.feature_suggestions = monetary + high_card + dates + outliers

    if show_progress:
        logger.info("[data-doctor] Identifying KPI candidates...")
    report.kpi_candidates = _analyze_kpi_candidates(ddf, data_dict)

    if show_progress:
        logger.info("[data-doctor] Finding ratio opportunities...")
    report.ratio_opportunities = _analyze_ratio_opportunities(ddf, data_dict)

    # LLM-enhanced analysis (optional)
    llm_executive_summary = None
    llm_suggestions = []

    if llm_enabled:
        if show_progress:
            logger.info("[data-doctor] Running LLM-enhanced analysis...")

        try:
            from .data_doctor_llm import (
                analyze_with_llm,
                generate_executive_summary,
                convert_llm_result_to_suggestions,
            )
            from .llm import get_llm_provider, LLMResponseCache

            # Get LLM provider
            provider_type = "auto"
            provider_model = None
            if llm_provider_config is not None:
                provider_type = getattr(llm_provider_config, "provider", "auto")
                provider_model = getattr(llm_provider_config, "model", None)

            provider = get_llm_provider(
                provider=provider_type,
                model=provider_model,
            )

            # Initialize cache if enabled
            cache = None
            if llm_cache_dir:
                cache = LLMResponseCache(llm_cache_dir)

            # Get table context from data_dict
            table_context = data_dict.get("table_comment", "")

            if generate_exec_summary:
                # Generate executive summary
                llm_executive_summary = generate_executive_summary(
                    ddf,
                    data_dict,
                    provider,
                    table_context=table_context,
                    organization_context=organization_context,
                    cache=cache,
                )
                if show_progress:
                    logger.info("[data-doctor] LLM executive summary generated")

            # Full LLM analysis
            llm_result = analyze_with_llm(
                ddf,
                data_dict,
                provider,
                table_context=table_context,
                organization_context=organization_context,
                cache=cache,
                max_columns=max_llm_columns,
            )

            # Convert to suggestions
            llm_suggestions = convert_llm_result_to_suggestions(llm_result)

            if show_progress:
                logger.info(
                    "[data-doctor] LLM analysis complete. %d additional suggestions.",
                    len(llm_suggestions),
                )

        except Exception as e:
            logger.warning("[data-doctor] LLM analysis failed: %s", e)

    # Store LLM results in report (using a simple attribute for now)
    report._llm_executive_summary = llm_executive_summary
    report._llm_suggestions = llm_suggestions

    # Rank and optionally limit suggestions
    if ranking_enabled or max_suggestions is not None:
        report = _rank_and_limit_suggestions(
            report,
            max_suggestions=max_suggestions,
            ranking_enabled=ranking_enabled,
        )
        if show_progress and max_suggestions is not None:
            original_total = getattr(report, "_original_total", None)
            if original_total is not None:
                logger.info(
                    "[data-doctor] Ranked and limited suggestions: %d/%d shown (top by impact score)",
                    max_suggestions,
                    original_total,
                )

    # Generate combined YAML
    report.yaml_text = _generate_suggestions_yaml(
        report, llm_executive_summary, ranking_enabled=ranking_enabled
    )

    if show_progress:
        total = (
            len(report.feature_suggestions)
            + len(report.fill_suggestions)
            + len(report.kpi_candidates)
            + len(report.ratio_opportunities)
        )
        logger.info("[data-doctor] Analysis complete. %d suggestions generated.", total)

    return report


def _generate_suggestions_yaml(
    report: DataDoctorReport,
    executive_summary: Optional[Any] = None,
    *,
    ranking_enabled: bool = True,
) -> str:
    """Generate YAML representation of all suggestions with legend header.

    Args:
        report: DataDoctorReport with suggestions.
        executive_summary: Optional LLM-generated executive summary.
        ranking_enabled: Whether to include impact scores in output.

    Returns:
        YAML formatted string.
    """
    # Build the YAML content with a header/legend
    lines = [
        "# ============================================================================",
        "# DATA DOCTOR SUGGESTIONS",
        "# ============================================================================",
        "#",
        "# Priority Legend:",
        "#   HIGH   - Action recommended (significant issue or opportunity)",
        "#   MEDIUM - Worth experimenting with (potential improvement)",
        "#   LOW    - Nice-to-have enhancement (minor optimization)",
        "#",
        "# Each suggestion includes:",
        "#   - column: The column name to apply the transformation to",
        "#   - message: Explanation of why this is suggested",
        "#   - yaml_snippet: Ready-to-use YAML config for the functions: section",
    ]

    if ranking_enabled:
        lines.extend(
            [
                "#   - score: Impact score (0-100) for prioritization",
                "#",
                "# Suggestions are ranked by impact score (highest first)",
            ]
        )

    lines.extend(
        [
            "#",
            "# ============================================================================",
            "",
        ]
    )

    # Add executive summary if available
    if executive_summary is not None:
        lines.extend(
            [
                "# EXECUTIVE SUMMARY",
                "# -----------------",
                f"# {getattr(executive_summary, 'table_description', 'N/A')}",
                "#",
                "# Key Findings:",
            ]
        )
        for finding in getattr(executive_summary, "key_findings", [])[:5]:
            lines.append(f"#   - {finding}")
        lines.extend(
            [
                "#",
                "# Immediate Actions:",
            ]
        )
        for action in getattr(executive_summary, "immediate_actions", [])[:3]:
            lines.append(f"#   - {action}")
        lines.extend(
            [
                "#",
                f"# Data Quality: {getattr(executive_summary, 'data_quality_summary', 'N/A')}",
                f"# Cross-Column Insights: {getattr(executive_summary, 'cross_column_insights', 'N/A')}",
                "#",
                "# ============================================================================",
                "",
            ]
        )

    suggestions_dict: Dict[str, List[Dict[str, Any]]] = {}

    def _make_entry(s: Suggestion) -> Dict[str, Any]:
        """Create a dict entry for a suggestion, optionally with score."""
        entry = {
            "priority": SEVERITY_LABELS.get(s.priority, "INFO"),
            "column": s.column,
            "message": s.message,
            "yaml_snippet": s.yaml_snippet,
        }
        if ranking_enabled:
            # Get score from _score attribute (set by ranking) or compute
            score = getattr(s, "_score", None)
            if score is None:
                score = _score_suggestion(s)
            entry["score"] = score
        return entry

    # KPI candidates first (most important)
    if report.kpi_candidates:
        suggestions_dict["kpi_candidates"] = []
        for s in report.kpi_candidates:
            suggestions_dict["kpi_candidates"].append(_make_entry(s))

    # Fill suggestions (data quality)
    if report.fill_suggestions:
        suggestions_dict["fill_suggestions"] = []
        for s in report.fill_suggestions:
            entry = _make_entry(s)
            if s.alternatives:
                entry["alternatives"] = [
                    {"strategy": a.strategy, "description": a.description}
                    for a in s.alternatives
                ]
            suggestions_dict["fill_suggestions"].append(entry)

    # Feature engineering
    if report.feature_suggestions:
        suggestions_dict["feature_suggestions"] = []
        for s in report.feature_suggestions:
            suggestions_dict["feature_suggestions"].append(_make_entry(s))

    # Ratio opportunities
    if report.ratio_opportunities:
        suggestions_dict["ratio_opportunities"] = []
        for s in report.ratio_opportunities:
            suggestions_dict["ratio_opportunities"].append(_make_entry(s))

    # Add summary
    total = (
        len(report.kpi_candidates)
        + len(report.fill_suggestions)
        + len(report.feature_suggestions)
        + len(report.ratio_opportunities)
    )
    summary: Dict[str, Any] = {
        "total_suggestions": total,
        "kpi_candidates": len(report.kpi_candidates),
        "fill_suggestions": len(report.fill_suggestions),
        "feature_suggestions": len(report.feature_suggestions),
        "ratio_opportunities": len(report.ratio_opportunities),
    }

    # Add info about limiting if applicable
    original_total = getattr(report, "_original_total", None)
    if original_total is not None and original_total > total:
        summary["original_total"] = original_total
        summary["limited_to_top"] = total
        summary["ranking_note"] = "Showing top suggestions by impact score"

    suggestions_dict["_summary"] = summary

    # Generate YAML with nice formatting
    yaml_content = yaml.safe_dump(
        suggestions_dict,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=100,
    )

    return "\n".join(lines) + yaml_content


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------
def _format_severity(priority: str) -> str:
    """Format severity with emoji and label."""
    label = SEVERITY_LABELS.get(priority, "INFO")
    return f"{priority} [{label}]"


def print_report(
    report: DataDoctorReport,
    *,
    output_path: Optional[Path] = None,
    show_alternatives: bool = True,
    use_print: bool = False,
) -> None:
    """Print a formatted report.

    Args:
        report: The DataDoctorReport to print.
        output_path: Optional path to write suggestions.yaml.
        show_alternatives: Show business alternatives for fill strategies.
        use_print: Use print() instead of logger (for CLI).
    """
    out = print if use_print else logger.info

    out("")
    out("=" * 70)
    out("  DATA DOCTOR - Your prescription for better data!")
    out("=" * 70)

    # Print executive summary if available
    exec_summary = getattr(report, "_llm_executive_summary", None)
    if exec_summary is not None:
        out("")
        out("-" * 70)
        out("  EXECUTIVE SUMMARY")
        out("-" * 70)
        out("")
        out(f"  {getattr(exec_summary, 'table_description', 'N/A')}")
        out("")
        out("  KEY FINDINGS:")
        for i, finding in enumerate(getattr(exec_summary, "key_findings", [])[:5], 1):
            out(f"    {i}. {finding}")
        out("")
        out("  IMMEDIATE ACTIONS:")
        for i, action in enumerate(
            getattr(exec_summary, "immediate_actions", [])[:3], 1
        ):
            out(f"    {i}. {action}")
        out("")
        dq_summary = getattr(exec_summary, "data_quality_summary", "")
        if dq_summary:
            out(f"  Data Quality: {dq_summary}")
        fe_opps = getattr(exec_summary, "feature_engineering_opportunities", "")
        if fe_opps:
            out(f"  Feature Engineering: {fe_opps}")
        cross_col = getattr(exec_summary, "cross_column_insights", "")
        if cross_col:
            out(f"  Cross-Column Insights: {cross_col}")
        out("")

    out("")
    out("  Legend:")
    out(f"    {SEVERITY_HIGH}  HIGH   - Action recommended (significant issue)")
    out(f"    {SEVERITY_MEDIUM}  MEDIUM - Worth experimenting with")
    out(f"    {SEVERITY_LOW}  LOW    - Nice-to-have enhancement")
    out("")

    total = (
        len(report.feature_suggestions)
        + len(report.fill_suggestions)
        + len(report.kpi_candidates)
        + len(report.ratio_opportunities)
    )

    if total == 0:
        out("  No suggestions - your data looks healthy!")
        out("")
        return

    if report.kpi_candidates:
        out("-" * 70)
        out(f"  KPI CANDIDATES ({len(report.kpi_candidates)})")
        out("-" * 70)
        for s in report.kpi_candidates:
            out(f"  {_format_severity(s.priority)} {s.column}")
            out(f"      {s.message}")
        out("")

    if report.fill_suggestions:
        out("-" * 70)
        out(f"  MISSING VALUE STRATEGIES ({len(report.fill_suggestions)})")
        out("-" * 70)
        for s in report.fill_suggestions:
            out(f"  {_format_severity(s.priority)} {s.column}")
            out(f"      {s.message}")
            if show_alternatives and s.alternatives:
                out("      Business alternatives:")
                for alt in s.alternatives[:4]:  # Show top 4 alternatives
                    req = f" (requires: {alt.requires})" if alt.requires else ""
                    out(f"        - {alt.strategy}: {alt.description}{req}")
        out("")

    if report.feature_suggestions:
        out("-" * 70)
        out(f"  FEATURE ENGINEERING ({len(report.feature_suggestions)})")
        out("-" * 70)
        for s in report.feature_suggestions:
            out(f"  {_format_severity(s.priority)} {s.column}")
            out(f"      {s.message}")
        out("")

    if report.ratio_opportunities:
        out("-" * 70)
        out(f"  RATIO OPPORTUNITIES ({len(report.ratio_opportunities)})")
        out("-" * 70)
        for s in report.ratio_opportunities:
            out(f"  {_format_severity(s.priority)} {s.column}")
            out(f"      {s.message}")
        out("")

    out("=" * 70)
    out(f"  Total suggestions: {total}")
    out("=" * 70)
    out("")

    if output_path:
        output_path.write_text(report.yaml_text)
        out(f"Suggestions written to: {output_path}")


def save_suggestions_yaml(report: DataDoctorReport, path: Path) -> None:
    """Save suggestions to a YAML file.

    Args:
        report: The DataDoctorReport to save.
        path: Path to write the YAML file.
    """
    path.write_text(report.yaml_text)
    logger.info("[data-doctor] Suggestions saved to: %s", path)


def run_standalone(
    ddf: dd.DataFrame,
    data_dict: Dict[str, Any],
    cfg: Optional[Any] = None,
    *,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
    show_alternatives: bool = True,
) -> DataDoctorReport:
    """Run Data Doctor as a standalone tool and print the report.

    This is the main entry point for standalone usage.

    Args:
        ddf: Dask DataFrame to analyze.
        data_dict: Data dictionary with column metadata.
        cfg: Optional BundleConfig for additional settings.
        output_path: Path to save suggestions.yaml (optional).
        show_progress: Show progress logs during analysis.
        show_alternatives: Show business alternatives in report.

    Returns:
        DataDoctorReport with all suggestions.
    """
    report = analyze_data(ddf, data_dict, cfg, show_progress=show_progress)
    print_report(
        report,
        output_path=output_path,
        show_alternatives=show_alternatives,
        use_print=True,
    )
    return report


def cli_main() -> None:
    """Command-line entry point for Data Doctor.

    Usage:
        python -m neuralift_c360_prep.data_doctor <config.yaml>
        python -m neuralift_c360_prep.data_doctor --parquet /path/to/data.parquet
        python -m neuralift_c360_prep.data_doctor --csv /path/to/data.csv

    Runs Data Doctor analysis on the data specified in the config file or path.
    """
    import argparse
    import json as json_mod
    import sys

    parser = argparse.ArgumentParser(
        prog="data-doctor",
        description="Analyze your data and get improvement suggestions!",
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to config YAML file (optional if --parquet or --csv is used)",
    )
    parser.add_argument(
        "--parquet",
        "-p",
        help="Path to parquet file (standalone mode, no config needed)",
    )
    parser.add_argument(
        "--csv",
        "-c",
        help="Path to CSV file (standalone mode, no config needed)",
    )
    parser.add_argument(
        "--data-dict",
        "-d",
        help="Path to existing data_dictionary.json (skip stats computation)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save suggestions.yaml",
    )
    parser.add_argument(
        "--no-alternatives",
        action="store_true",
        help="Hide business alternative suggestions",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress logs",
    )
    args = parser.parse_args()

    # Need either config OR direct path
    if not args.config and not args.parquet and not args.csv:
        parser.print_help()
        print("\nError: Must provide config file OR --parquet/--csv path")
        sys.exit(1)

    # Import here to avoid circular imports
    from .ingest import load_lazy_dask

    print("")
    print("Loading data...")

    cfg = None
    ddf = None

    # Direct path mode (standalone without config)
    if args.parquet or args.csv:
        if args.parquet:
            fmt, uri = "parquet", args.parquet
        else:
            fmt, uri = "csv", args.csv

        ddf = load_lazy_dask(
            fmt=fmt,
            uri=uri,
            id_cols=[],
            columns=None,
            dtype_overrides=None,
            show_progress=not args.quiet,
        )
    else:
        # Config mode
        from .config import load_config

        try:
            cfg = load_config(args.config)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

        # Determine input format and URI from config
        if cfg.input.source == "uc_table":
            fmt, uri = "databricks_table", cfg.input.uc_table
        elif cfg.input.source == "delta_path":
            fmt, uri = "delta", cfg.input.delta_path
        elif cfg.input.source == "parquet":
            fmt, uri = "parquet", cfg.input.parquet_path
        elif cfg.input.source == "csv":
            fmt, uri = "csv", cfg.input.csv_path
        else:
            print(f"Unsupported source for standalone: {cfg.input.source}")
            sys.exit(1)

        ddf = load_lazy_dask(
            fmt=fmt,
            uri=uri,
            id_cols=cfg.input.id_cols,
            columns=cfg.input.columns,
            dtype_overrides=cfg.input.dtype_overrides,
            show_progress=not args.quiet,
        )

    # Load or build data dictionary
    if args.data_dict:
        print(f"Loading data dictionary from {args.data_dict}...")
        with open(args.data_dict) as f:
            data_dict = json_mod.load(f)
    elif cfg is not None:
        # Use full metadata generation when config is available
        from .metadata import build_metadata

        print("Building metadata (this may take a moment)...")
        _, meta_text = build_metadata(ddf, cfg)
        data_dict = json_mod.loads(meta_text)
    else:
        # Standalone mode - build minimal data dict using stats_cache
        print("Computing column statistics...")
        data_dict = build_minimal_data_dict(ddf, show_progress=not args.quiet)

    # Run analysis
    output_path = Path(args.output) if args.output else None
    run_standalone(
        ddf,
        data_dict,
        cfg,
        output_path=output_path,
        show_progress=not args.quiet,
        show_alternatives=not args.no_alternatives,
    )


__all__ = [
    "Suggestion",
    "FillAlternative",
    "DataDoctorReport",
    "analyze_data",
    "print_report",
    "save_suggestions_yaml",
    "run_standalone",
    "cli_main",
    "build_minimal_data_dict",
    # Severity constants
    "SEVERITY_HIGH",
    "SEVERITY_MEDIUM",
    "SEVERITY_LOW",
    "SEVERITY_LABELS",
    "SEVERITY_DESCRIPTIONS",
    "BUSINESS_FILL_ALTERNATIVES",
]


if __name__ == "__main__":
    cli_main()
