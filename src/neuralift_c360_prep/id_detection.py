#!/usr/bin/env python3
"""
ID column detection and suggestion for agentic workflows.

Purpose:
    - Suggest ID columns based on naming conventions and uniqueness
    - Support explicit ID configuration with auto-detection fallback
    - Print suggestions for agentic use (agent can accept/reject)

Author: Mike Maloney - Neuralift, Inc.
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

import dask.dataframe as dd

logger = logging.getLogger(__name__)

# Common ID column naming patterns (case-insensitive)
ID_NAMING_PATTERNS = [
    r"^id$",  # exact "id"
    r"_id$",  # suffix "_id" (customer_id, player_id, etc.)
    r"^customer_id$",
    r"^player_id$",
    r"^user_id$",
    r"^account_id$",
    r"^member_id$",
    r"^subscriber_id$",
    r"^contact_id$",
    r"^profile_id$",
    r"^transaction_id$",
    r"^order_id$",
    r"^session_id$",
    r"^visitor_id$",
    r"^device_id$",
    r"^entity_id$",
    r"^record_id$",
    r"^row_id$",
    r"^pk$",  # primary key
    r"^primary_key$",
]


@dataclass
class IdSuggestion:
    """A suggested ID column with reasoning."""

    column: str
    reason: str  # "naming" | "uniqueness" | "both"
    confidence: str  # "high" | "medium" | "low"
    uniqueness_ratio: float | None = None

    def to_dict(self) -> dict:
        result = {
            "column": self.column,
            "reason": self.reason,
            "confidence": self.confidence,
        }
        if self.uniqueness_ratio is not None:
            result["uniqueness_ratio"] = self.uniqueness_ratio
        return result


def _matches_id_pattern(column_name: str) -> bool:
    """Check if column name matches any ID naming pattern."""
    col_lower = column_name.lower()
    for pattern in ID_NAMING_PATTERNS:
        if re.match(pattern, col_lower):
            return True
    return False


def suggest_id_columns(
    ddf: dd.DataFrame,
    *,
    row_count: int | None = None,
    uniqueness_threshold: float = 0.95,
    check_uniqueness: bool = True,
    exclude_columns: Sequence[str] | None = None,
) -> list[IdSuggestion]:
    """
    Suggest ID columns based on naming conventions and uniqueness.

    Args:
        ddf: Dask DataFrame to analyze
        row_count: Pre-computed row count (avoids recomputation)
        uniqueness_threshold: Ratio of unique values to consider as ID (default 0.95)
        check_uniqueness: Whether to check uniqueness (can be slow for large data)
        exclude_columns: Columns to exclude from suggestions

    Returns:
        List of IdSuggestion objects sorted by confidence (high first)
    """
    exclude_set = {c.lower() for c in (exclude_columns or [])}
    suggestions: list[IdSuggestion] = []
    columns_with_naming_match: set[str] = set()

    # Phase 1: Check naming patterns (fast)
    for col in ddf.columns:
        if col.lower() in exclude_set:
            continue

        if _matches_id_pattern(col):
            columns_with_naming_match.add(col)
            suggestions.append(
                IdSuggestion(
                    column=col,
                    reason="naming",
                    confidence="high",
                )
            )

    # Phase 2: Check uniqueness (slower, optional)
    if check_uniqueness:
        if row_count is None:
            try:
                row_count = len(ddf)
            except Exception:
                row_count = int(ddf.shape[0].compute())

        if row_count > 0:
            for col in ddf.columns:
                if col.lower() in exclude_set:
                    continue
                if col in columns_with_naming_match:
                    # Already suggested by naming, upgrade if also unique
                    continue

                try:
                    # Use approximate unique count for speed
                    nunique = ddf[col].nunique_approx().compute()
                    ratio = nunique / row_count

                    if ratio >= uniqueness_threshold:
                        suggestions.append(
                            IdSuggestion(
                                column=col,
                                reason="uniqueness",
                                confidence="medium",
                                uniqueness_ratio=ratio,
                            )
                        )
                except Exception:
                    # Skip columns that fail uniqueness check
                    continue

    # Update naming matches that are also unique
    if check_uniqueness and row_count and row_count > 0:
        for suggestion in suggestions:
            if suggestion.reason == "naming" and suggestion.column in ddf.columns:
                try:
                    nunique = ddf[suggestion.column].nunique_approx().compute()
                    ratio = nunique / row_count
                    if ratio >= uniqueness_threshold:
                        suggestion.reason = "both"
                        suggestion.uniqueness_ratio = ratio
                except Exception:
                    pass

    # Sort by confidence: both > naming (high) > uniqueness (medium)
    confidence_order = {"both": 0, "naming": 1, "uniqueness": 2}
    suggestions.sort(key=lambda s: (confidence_order.get(s.reason, 99), s.column))

    return suggestions


def print_id_suggestions(
    suggestions: list[IdSuggestion],
    *,
    explicit_ids: Sequence[str] | None = None,
) -> None:
    """
    Print ID suggestions to log for agentic use.

    Args:
        suggestions: List of ID suggestions
        explicit_ids: Explicitly configured IDs (shown for context)
    """
    if explicit_ids:
        logger.info("[ids] Configured ID columns: %s", list(explicit_ids))

    if not suggestions:
        if not explicit_ids:
            logger.info("[ids] No ID column suggestions found")
        return

    logger.info("[ids] Suggested ID columns:")
    for s in suggestions:
        if s.reason == "both":
            logger.info(
                "  - %s (naming convention + %.1f%% unique, confidence: high)",
                s.column,
                (s.uniqueness_ratio or 0) * 100,
            )
        elif s.reason == "naming":
            logger.info(
                "  - %s (naming convention, confidence: %s)",
                s.column,
                s.confidence,
            )
        else:
            logger.info(
                "  - %s (%.1f%% unique values, confidence: %s)",
                s.column,
                (s.uniqueness_ratio or 0) * 100,
                s.confidence,
            )


def resolve_id_columns(
    ddf: dd.DataFrame,
    *,
    explicit_ids: Sequence[str] | None = None,
    auto_detect: bool = True,
    row_count: int | None = None,
) -> list[str]:
    """
    Resolve final ID columns from explicit config and/or auto-detection.

    Args:
        ddf: Dask DataFrame
        explicit_ids: Explicitly configured ID columns
        auto_detect: Whether to run auto-detection and print suggestions
        row_count: Pre-computed row count

    Returns:
        List of ID column names (explicit IDs take precedence)
    """
    explicit = list(explicit_ids or [])

    if auto_detect:
        suggestions = suggest_id_columns(
            ddf,
            row_count=row_count,
            exclude_columns=explicit,
        )
        print_id_suggestions(suggestions, explicit_ids=explicit)

    # Return explicit IDs (auto-detected are just suggestions for now)
    return explicit


__all__ = [
    "IdSuggestion",
    "suggest_id_columns",
    "print_id_suggestions",
    "resolve_id_columns",
]
