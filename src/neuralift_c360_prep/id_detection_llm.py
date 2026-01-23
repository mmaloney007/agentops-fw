"""
LLM-enhanced ID column detection.

Extends the base id_detection module with semantic analysis using LLMs.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
import re
import textwrap
from typing import Sequence

import dask.dataframe as dd

from .id_detection_models import (
    IdDetectionResult,
    IdFormat,
    KeyType,
    LLMIdSuggestion,
)
from .llm import LLMProvider, LLMResponseCache

logger = logging.getLogger(__name__)

# UUID regex patterns for format detection
UUID_V4_PATTERN = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$",
    re.IGNORECASE,
)
UUID_GENERIC_PATTERN = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
    re.IGNORECASE,
)
GUID_PATTERN = re.compile(
    r"^\{?[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\}?$",
    re.IGNORECASE,
)
# Hash patterns (MD5, SHA1, SHA256)
MD5_PATTERN = re.compile(r"^[a-f0-9]{32}$", re.IGNORECASE)
SHA1_PATTERN = re.compile(r"^[a-f0-9]{40}$", re.IGNORECASE)
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$", re.IGNORECASE)

# ID-like name fragments for ambiguity detection
ID_FRAGMENTS = {"ref", "num", "code", "key", "identifier", "guid", "uuid", "pk"}


def detect_uuid_format(values: Sequence) -> IdFormat:
    """Detect UUID/GUID/hash format from sample values.

    Args:
        values: Sample values from the column.

    Returns:
        Detected format or UNKNOWN.
    """
    if not values:
        return IdFormat.UNKNOWN

    # Convert to strings and filter nulls
    sample = [str(v) for v in values[:100] if v is not None and str(v).strip()]
    if not sample:
        return IdFormat.UNKNOWN

    threshold = len(sample) * 0.9  # 90% must match

    # Check patterns in order of specificity
    v4_matches = sum(1 for v in sample if UUID_V4_PATTERN.match(v))
    if v4_matches >= threshold:
        return IdFormat.UUID_V4

    guid_matches = sum(1 for v in sample if GUID_PATTERN.match(v))
    if guid_matches >= threshold:
        return IdFormat.GUID

    generic_uuid_matches = sum(1 for v in sample if UUID_GENERIC_PATTERN.match(v))
    if generic_uuid_matches >= threshold:
        return IdFormat.UUID_OTHER

    # Check hash formats
    sha256_matches = sum(1 for v in sample if SHA256_PATTERN.match(v))
    if sha256_matches >= threshold:
        return IdFormat.HASH

    sha1_matches = sum(1 for v in sample if SHA1_PATTERN.match(v))
    if sha1_matches >= threshold:
        return IdFormat.HASH

    md5_matches = sum(1 for v in sample if MD5_PATTERN.match(v))
    if md5_matches >= threshold:
        return IdFormat.HASH

    return IdFormat.UNKNOWN


def detect_sequential_int(values: Sequence) -> bool:
    """Check if values appear to be sequential integers.

    Args:
        values: Sample values from the column.

    Returns:
        True if values appear sequential.
    """
    if not values:
        return False

    try:
        # Convert to integers
        ints = sorted([int(v) for v in values[:100] if v is not None])
        if len(ints) < 10:
            return False

        # Check if mostly sequential (allow some gaps)
        diffs = [ints[i + 1] - ints[i] for i in range(len(ints) - 1)]
        if not diffs:
            return False

        # Most differences should be small positive integers
        small_diffs = sum(1 for d in diffs if 0 < d <= 10)
        return small_diffs / len(diffs) > 0.8
    except (ValueError, TypeError):
        return False


def _build_id_detection_prompt(
    column_profiles: list[dict],
    table_context: str | None = None,
) -> str:
    """Build prompt for LLM ID detection.

    Args:
        column_profiles: List of column profile dicts.
        table_context: Optional context about the table.

    Returns:
        Formatted prompt string.
    """
    profiles_text = "\n".join(
        [
            f"- {p['name']}: dtype={p['dtype']}, unique_ratio={p['unique_ratio']:.2%}, "
            f"nulls={p['null_pct']:.1%}, samples={p['samples'][:5]}"
            for p in column_profiles
        ]
    )

    return textwrap.dedent(f"""\
        Analyze these columns from a data table to identify primary key candidates.

        Context: {table_context or "Unknown table"}

        Columns:
        {profiles_text}

        For each column, determine:
        1. Is it likely a primary identifier (ID/key)?
        2. Is it a surrogate key (auto-generated) or business key (domain-meaningful)?
        3. What format does it use (UUID, sequential int, hash, etc.)?

        IMPORTANT: Only suggest composite keys if the individual columns are NOT already unique.
        If a column has >95% unique values on its own, do NOT include it in composite suggestions.

        Focus on:
        - Semantic patterns (cust_ref, acct_num, player_code are IDs even without "_id" suffix)
        - Value patterns (UUIDs, sequential numbers, hashes)
        - Uniqueness combined with naming conventions
    """)


SYSTEM_PROMPT = """You are an expert data engineer analyzing database schemas.
Identify primary key and unique identifier columns based on naming conventions,
data patterns, and uniqueness characteristics. Be precise and provide reasoning.
Only suggest composite keys when individual columns lack sufficient uniqueness."""


def analyze_columns_with_llm(
    column_profiles: list[dict],
    provider: LLMProvider,
    *,
    table_context: str | None = None,
    cache: LLMResponseCache | None = None,
) -> IdDetectionResult:
    """Use LLM to analyze columns for ID detection.

    Args:
        column_profiles: List of column profile dicts with name, dtype, samples, etc.
        provider: LLM provider to use.
        table_context: Optional context about the table.
        cache: Optional response cache.

    Returns:
        IdDetectionResult with analysis.
    """
    prompt = _build_id_detection_prompt(column_profiles, table_context)

    # Check cache first
    if cache:
        cached = cache.get(prompt, provider.name, IdDetectionResult)
        if cached:
            return cached

    result = provider.complete_structured(
        prompt=prompt,
        response_model=IdDetectionResult,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
    )

    # Cache result
    if cache:
        cache.set(prompt, provider.name, result)

    return result


def identify_ambiguous_columns(
    ddf: dd.DataFrame,
    naming_matches: set[str],
    uniqueness_matches: set[str],
    *,
    uniqueness_threshold: float = 0.95,
    gray_zone_lower: float = 0.80,
    row_count: int | None = None,
) -> list[str]:
    """Identify columns that are ambiguous and would benefit from LLM analysis.

    Ambiguous columns are those that:
    - Have high but not definitive uniqueness (80-95%)
    - Have ID-like name fragments but don't match exact patterns
    - Have very high uniqueness but unclear naming

    Args:
        ddf: Dask DataFrame to analyze.
        naming_matches: Columns already matched by naming patterns.
        uniqueness_matches: Columns already matched by uniqueness.
        uniqueness_threshold: Upper threshold for uniqueness.
        gray_zone_lower: Lower bound for gray zone uniqueness.
        row_count: Pre-computed row count.

    Returns:
        List of ambiguous column names.
    """
    ambiguous = []
    already_identified = naming_matches | uniqueness_matches

    # Get row count if not provided
    if row_count is None:
        try:
            row_count = len(ddf)
        except Exception:
            row_count = int(ddf.shape[0].compute())

    if row_count == 0:
        return []

    for col in ddf.columns:
        if col in already_identified:
            continue

        col_lower = col.lower()

        # Check for partial ID naming
        has_id_fragment = any(frag in col_lower for frag in ID_FRAGMENTS)

        # Check uniqueness ratio (approximate)
        try:
            nunique = ddf[col].nunique_approx().compute()
            ratio = nunique / row_count if row_count > 0 else 0
        except Exception:
            continue

        # Conditions for ambiguity
        is_gray_zone = gray_zone_lower <= ratio < uniqueness_threshold
        is_high_unique_unclear = ratio >= uniqueness_threshold and not has_id_fragment

        if is_gray_zone or (has_id_fragment and ratio > 0.5) or is_high_unique_unclear:
            ambiguous.append(col)

    return ambiguous


def suggest_id_columns_with_llm(
    ddf: dd.DataFrame,
    *,
    row_count: int | None = None,
    uniqueness_threshold: float = 0.95,
    check_uniqueness: bool = True,
    exclude_columns: Sequence[str] | None = None,
    llm_provider: LLMProvider | None = None,
    llm_enabled: bool = True,
    table_context: str | None = None,
    max_llm_columns: int = 20,
    cache_dir: str | None = None,
) -> list[LLMIdSuggestion]:
    """Enhanced ID column suggestion with LLM semantic analysis.

    Backward compatible with suggest_id_columns() but adds LLM capabilities.

    Args:
        ddf: Dask DataFrame to analyze.
        row_count: Pre-computed row count.
        uniqueness_threshold: Ratio to consider unique (default 0.95).
        check_uniqueness: Whether to check uniqueness.
        exclude_columns: Columns to exclude.
        llm_provider: LLM provider to use (optional).
        llm_enabled: Whether to use LLM for ambiguous columns.
        table_context: Optional table context for LLM.
        max_llm_columns: Maximum columns to analyze with LLM.
        cache_dir: Optional cache directory for LLM responses.

    Returns:
        List of LLMIdSuggestion objects.
    """
    from .id_detection import suggest_id_columns

    # Phase 1 & 2: Use existing detection
    base_suggestions = suggest_id_columns(
        ddf,
        row_count=row_count,
        uniqueness_threshold=uniqueness_threshold,
        check_uniqueness=check_uniqueness,
        exclude_columns=exclude_columns,
    )

    # Get row count for later phases
    if row_count is None:
        try:
            row_count = len(ddf)
        except Exception:
            row_count = int(ddf.shape[0].compute())

    # Convert to enhanced suggestions
    suggestions: list[LLMIdSuggestion] = []
    naming_matches: set[str] = set()
    uniqueness_matches: set[str] = set()

    for s in base_suggestions:
        confidence_score = (
            1.0 if s.confidence == "high" else 0.7 if s.confidence == "medium" else 0.4
        )
        enhanced = LLMIdSuggestion(
            column=s.column,
            reason=s.reason,
            confidence=s.confidence,
            confidence_score=confidence_score,
            uniqueness_ratio=s.uniqueness_ratio,
        )
        suggestions.append(enhanced)

        if s.reason in ("naming", "both"):
            naming_matches.add(s.column)
        if s.reason in ("uniqueness", "both"):
            uniqueness_matches.add(s.column)

    # Phase 3: UUID/GUID format detection (fast, no LLM)
    exclude_set = set(exclude_columns or [])
    for col in ddf.columns:
        if col in naming_matches or col in uniqueness_matches or col in exclude_set:
            continue

        try:
            sample = ddf[col].head(100).dropna().tolist()
            uuid_format = detect_uuid_format(sample)

            if uuid_format != IdFormat.UNKNOWN:
                suggestions.append(
                    LLMIdSuggestion(
                        column=col,
                        reason="semantic",
                        confidence="high",
                        confidence_score=0.95,
                        id_format=uuid_format,
                        key_type=KeyType.SURROGATE,
                        explanation=f"Detected {uuid_format.value} format in values",
                    )
                )
                uniqueness_matches.add(col)
        except Exception:
            continue

    # Phase 4: LLM analysis for ambiguous columns
    if llm_enabled and llm_provider is not None:
        ambiguous = identify_ambiguous_columns(
            ddf,
            naming_matches,
            uniqueness_matches,
            uniqueness_threshold=uniqueness_threshold,
            row_count=row_count,
        )

        if ambiguous:
            ambiguous = ambiguous[:max_llm_columns]  # Cost control
            logger.info(
                "[id-llm] Analyzing %d ambiguous columns with LLM", len(ambiguous)
            )

            # Build column profiles for LLM
            profiles = []
            for col in ambiguous:
                try:
                    sample = ddf[col].head(20).dropna().tolist()
                    nunique = ddf[col].nunique_approx().compute()
                    null_count = ddf[col].isna().sum().compute()

                    profiles.append(
                        {
                            "name": col,
                            "dtype": str(ddf[col].dtype),
                            "unique_ratio": nunique / row_count if row_count else 0,
                            "null_pct": null_count / row_count if row_count else 0,
                            "samples": [str(v) for v in sample[:10]],
                        }
                    )
                except Exception:
                    continue

            if profiles:
                # Initialize cache if directory provided
                cache = LLMResponseCache(cache_dir) if cache_dir else None

                try:
                    result = analyze_columns_with_llm(
                        profiles, llm_provider, table_context=table_context, cache=cache
                    )

                    # Add LLM suggestions
                    for analysis in result.primary_candidates:
                        if analysis.is_likely_id and analysis.confidence > 0.6:
                            suggestions.append(
                                LLMIdSuggestion(
                                    column=analysis.column_name,
                                    reason="llm",
                                    confidence="high"
                                    if analysis.confidence > 0.85
                                    else "medium",
                                    confidence_score=analysis.confidence,
                                    key_type=analysis.key_type,
                                    id_format=analysis.id_format,
                                    explanation=analysis.reasoning,
                                )
                            )

                    # Log cache stats if using cache
                    if cache:
                        stats = cache.stats
                        logger.debug("[id-llm] Cache stats: %s", stats)

                except Exception as e:
                    logger.warning("[id-llm] LLM analysis failed: %s", e)

    # Deduplicate by column name, keeping highest confidence
    seen: dict[str, LLMIdSuggestion] = {}
    for s in suggestions:
        if s.column not in seen or s.confidence_score > seen[s.column].confidence_score:
            seen[s.column] = s

    # Sort by confidence score
    final = sorted(seen.values(), key=lambda s: -s.confidence_score)
    return final


def print_llm_id_suggestions(
    suggestions: list[LLMIdSuggestion],
    *,
    explicit_ids: Sequence[str] | None = None,
) -> None:
    """Print enhanced ID suggestions to log.

    Args:
        suggestions: List of LLM ID suggestions.
        explicit_ids: Explicitly configured IDs (shown for context).
    """
    if explicit_ids:
        logger.info("[ids] Configured ID columns: %s", list(explicit_ids))

    if not suggestions:
        if not explicit_ids:
            logger.info("[ids] No ID column suggestions found")
        return

    logger.info("[ids] Suggested ID columns:")
    for s in suggestions:
        parts = [f"{s.column}"]

        if s.reason == "both":
            parts.append(f"naming + {(s.uniqueness_ratio or 0) * 100:.1f}% unique")
        elif s.reason == "naming":
            parts.append("naming convention")
        elif s.reason == "uniqueness":
            parts.append(f"{(s.uniqueness_ratio or 0) * 100:.1f}% unique")
        elif s.reason == "semantic":
            parts.append(f"format: {s.id_format.value}")
        elif s.reason == "llm":
            parts.append("LLM semantic analysis")

        parts.append(f"confidence: {s.confidence} ({s.confidence_score:.0%})")

        if s.key_type != KeyType.UNKNOWN:
            parts.append(f"type: {s.key_type.value}")

        if s.explanation:
            parts.append(f"- {s.explanation}")

        logger.info("  - %s", ", ".join(parts))


__all__ = [
    "detect_uuid_format",
    "detect_sequential_int",
    "analyze_columns_with_llm",
    "identify_ambiguous_columns",
    "suggest_id_columns_with_llm",
    "print_llm_id_suggestions",
]
