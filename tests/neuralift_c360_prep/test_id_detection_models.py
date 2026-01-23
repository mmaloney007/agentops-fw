#!/usr/bin/env python3
"""
tests/neuralift_c360_prep/test_id_detection_models.py
-----------------------------------------------------
Tests for the id_detection_models module - Pydantic models for LLM ID detection.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralift_c360_prep.id_detection_models import (
    KeyType,
    IdFormat,
    IdColumnAnalysis,
    CompositeKeyCandidate,
    IdDetectionResult,
    LLMIdSuggestion,
)


# ---------------------------------------------------------------------------
# Tests for KeyType enum
# ---------------------------------------------------------------------------
class TestKeyType:
    """Tests for KeyType enum."""

    def test_all_values_exist(self):
        """Test that all expected key types are defined."""
        expected = ["surrogate", "business", "composite", "unknown"]
        actual = [kt.value for kt in KeyType]
        for exp in expected:
            assert exp in actual, f"Missing key type: {exp}"

    def test_string_behavior(self):
        """Test enum string behavior."""
        assert KeyType.SURROGATE.value == "surrogate"
        assert KeyType.BUSINESS.value == "business"
        assert KeyType.COMPOSITE.value == "composite"
        assert KeyType.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# Tests for IdFormat enum
# ---------------------------------------------------------------------------
class TestIdFormat:
    """Tests for IdFormat enum."""

    def test_all_values_exist(self):
        """Test that all expected ID formats are defined."""
        expected = [
            "uuid_v4",
            "uuid_v1",
            "uuid_other",
            "guid",
            "sequential_int",
            "hash",
            "alphanumeric",
            "email",
            "phone",
            "custom",
            "unknown",
        ]
        actual = [idf.value for idf in IdFormat]
        for exp in expected:
            assert exp in actual, f"Missing ID format: {exp}"

    def test_uuid_formats(self):
        """Test UUID format values."""
        assert IdFormat.UUID_V4.value == "uuid_v4"
        assert IdFormat.UUID_V1.value == "uuid_v1"
        assert IdFormat.UUID_OTHER.value == "uuid_other"
        assert IdFormat.GUID.value == "guid"


# ---------------------------------------------------------------------------
# Tests for IdColumnAnalysis
# ---------------------------------------------------------------------------
class TestIdColumnAnalysis:
    """Tests for IdColumnAnalysis model."""

    def test_creation(self):
        """Test creating IdColumnAnalysis."""
        analysis = IdColumnAnalysis(
            column_name="customer_id",
            is_likely_id=True,
            confidence=0.95,
            key_type=KeyType.SURROGATE,
            id_format=IdFormat.SEQUENTIAL_INT,
            reasoning="Sequential integers with high uniqueness",
        )
        assert analysis.column_name == "customer_id"
        assert analysis.is_likely_id is True
        assert analysis.confidence == 0.95
        assert analysis.key_type == KeyType.SURROGATE
        assert analysis.id_format == IdFormat.SEQUENTIAL_INT

    def test_confidence_bounds(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            IdColumnAnalysis(
                column_name="test",
                is_likely_id=True,
                confidence=1.5,  # Over 1.0
                key_type=KeyType.UNKNOWN,
                id_format=IdFormat.UNKNOWN,
                reasoning="Test",
            )

        with pytest.raises(ValidationError):
            IdColumnAnalysis(
                column_name="test",
                is_likely_id=True,
                confidence=-0.1,  # Below 0.0
                key_type=KeyType.UNKNOWN,
                id_format=IdFormat.UNKNOWN,
                reasoning="Test",
            )

    def test_reasoning_max_length(self):
        """Test reasoning max length constraint."""
        long_reasoning = "x" * 301  # Over 300 limit
        with pytest.raises(ValidationError):
            IdColumnAnalysis(
                column_name="test",
                is_likely_id=True,
                confidence=0.5,
                key_type=KeyType.UNKNOWN,
                id_format=IdFormat.UNKNOWN,
                reasoning=long_reasoning,
            )

    def test_enum_values_serialization(self):
        """Test that enums serialize to values."""
        analysis = IdColumnAnalysis(
            column_name="test",
            is_likely_id=False,
            confidence=0.3,
            key_type=KeyType.BUSINESS,
            id_format=IdFormat.EMAIL,
            reasoning="Email format detected",
        )
        d = analysis.model_dump()
        assert d["key_type"] == "business"
        assert d["id_format"] == "email"

    def test_all_key_types(self):
        """Test that all key types can be used."""
        for kt in KeyType:
            analysis = IdColumnAnalysis(
                column_name="test",
                is_likely_id=True,
                confidence=0.5,
                key_type=kt,
                id_format=IdFormat.UNKNOWN,
                reasoning="Test",
            )
            assert analysis.key_type == kt

    def test_all_id_formats(self):
        """Test that all ID formats can be used."""
        for idf in IdFormat:
            analysis = IdColumnAnalysis(
                column_name="test",
                is_likely_id=True,
                confidence=0.5,
                key_type=KeyType.UNKNOWN,
                id_format=idf,
                reasoning="Test",
            )
            assert analysis.id_format == idf


# ---------------------------------------------------------------------------
# Tests for CompositeKeyCandidate
# ---------------------------------------------------------------------------
class TestCompositeKeyCandidate:
    """Tests for CompositeKeyCandidate model."""

    def test_creation(self):
        """Test creating CompositeKeyCandidate."""
        candidate = CompositeKeyCandidate(
            columns=["order_id", "product_id"],
            confidence=0.85,
            reasoning="Together they form unique combination",
        )
        assert candidate.columns == ["order_id", "product_id"]
        assert candidate.confidence == 0.85

    def test_requires_minimum_columns(self):
        """Test that at least 2 columns are required."""
        with pytest.raises(ValidationError):
            CompositeKeyCandidate(
                columns=["single_col"],  # Only one column
                confidence=0.5,
                reasoning="Test",
            )

    def test_multiple_columns(self):
        """Test composite key with multiple columns."""
        candidate = CompositeKeyCandidate(
            columns=["col1", "col2", "col3"],
            confidence=0.9,
            reasoning="Three-column composite key",
        )
        assert len(candidate.columns) == 3

    def test_confidence_bounds(self):
        """Test confidence bounds."""
        with pytest.raises(ValidationError):
            CompositeKeyCandidate(
                columns=["a", "b"],
                confidence=1.1,
                reasoning="Test",
            )


# ---------------------------------------------------------------------------
# Tests for IdDetectionResult
# ---------------------------------------------------------------------------
class TestIdDetectionResult:
    """Tests for IdDetectionResult model."""

    def test_creation(self):
        """Test creating IdDetectionResult."""
        result = IdDetectionResult(
            primary_candidates=[
                IdColumnAnalysis(
                    column_name="id",
                    is_likely_id=True,
                    confidence=0.99,
                    key_type=KeyType.SURROGATE,
                    id_format=IdFormat.SEQUENTIAL_INT,
                    reasoning="Primary key",
                )
            ],
            composite_candidates=[],
            table_context="User account data",
        )
        assert len(result.primary_candidates) == 1
        assert result.table_context == "User account data"

    def test_empty_candidates(self):
        """Test result with empty candidates."""
        result = IdDetectionResult(
            primary_candidates=[],
            composite_candidates=[],
            table_context="Unknown table",
        )
        assert result.primary_candidates == []
        assert result.composite_candidates == []

    def test_with_composite_candidates(self):
        """Test result with composite candidates."""
        result = IdDetectionResult(
            primary_candidates=[],
            composite_candidates=[
                CompositeKeyCandidate(
                    columns=["a", "b"],
                    confidence=0.8,
                    reasoning="Composite key",
                )
            ],
            table_context="Transaction data",
        )
        assert len(result.composite_candidates) == 1

    def test_table_context_max_length(self):
        """Test table_context max length."""
        long_context = "x" * 201
        with pytest.raises(ValidationError):
            IdDetectionResult(
                primary_candidates=[],
                composite_candidates=[],
                table_context=long_context,
            )


# ---------------------------------------------------------------------------
# Tests for LLMIdSuggestion
# ---------------------------------------------------------------------------
class TestLLMIdSuggestion:
    """Tests for LLMIdSuggestion model."""

    def test_creation(self):
        """Test creating LLMIdSuggestion."""
        suggestion = LLMIdSuggestion(
            column="customer_id",
            reason="naming",
            confidence="high",
            confidence_score=0.95,
        )
        assert suggestion.column == "customer_id"
        assert suggestion.reason == "naming"
        assert suggestion.confidence == "high"
        assert suggestion.confidence_score == 0.95
        assert suggestion.key_type == KeyType.UNKNOWN  # default
        assert suggestion.id_format == IdFormat.UNKNOWN  # default

    def test_all_reason_values(self):
        """Test all valid reason values."""
        valid_reasons = ["naming", "uniqueness", "both", "semantic", "llm"]
        for reason in valid_reasons:
            suggestion = LLMIdSuggestion(
                column="test",
                reason=reason,
                confidence="high",
                confidence_score=0.9,
            )
            assert suggestion.reason == reason

    def test_invalid_reason(self):
        """Test invalid reason is rejected."""
        with pytest.raises(ValidationError):
            LLMIdSuggestion(
                column="test",
                reason="invalid_reason",
                confidence="high",
                confidence_score=0.9,
            )

    def test_all_confidence_levels(self):
        """Test all valid confidence levels."""
        for conf in ["high", "medium", "low"]:
            suggestion = LLMIdSuggestion(
                column="test",
                reason="naming",
                confidence=conf,
                confidence_score=0.5,
            )
            assert suggestion.confidence == conf

    def test_invalid_confidence_level(self):
        """Test invalid confidence level is rejected."""
        with pytest.raises(ValidationError):
            LLMIdSuggestion(
                column="test",
                reason="naming",
                confidence="very_high",  # Invalid
                confidence_score=0.9,
            )

    def test_full_creation(self):
        """Test creation with all fields."""
        suggestion = LLMIdSuggestion(
            column="user_uuid",
            reason="semantic",
            confidence="high",
            confidence_score=0.98,
            uniqueness_ratio=0.9999,
            key_type=KeyType.SURROGATE,
            id_format=IdFormat.UUID_V4,
            explanation="Detected UUID v4 format in values",
        )
        assert suggestion.uniqueness_ratio == 0.9999
        assert suggestion.key_type == KeyType.SURROGATE
        assert suggestion.id_format == IdFormat.UUID_V4
        assert "UUID" in suggestion.explanation

    def test_to_dict(self):
        """Test to_dict serialization."""
        suggestion = LLMIdSuggestion(
            column="id",
            reason="both",
            confidence="high",
            confidence_score=0.95,
            uniqueness_ratio=1.0,
            key_type=KeyType.SURROGATE,
            id_format=IdFormat.SEQUENTIAL_INT,
            explanation="Primary key",
        )
        d = suggestion.to_dict()

        assert d["column"] == "id"
        assert d["reason"] == "both"
        assert d["confidence"] == "high"
        assert d["confidence_score"] == 0.95
        assert d["uniqueness_ratio"] == 1.0
        assert d["key_type"] == "surrogate"
        assert d["id_format"] == "sequential_int"
        assert d["explanation"] == "Primary key"

    def test_to_dict_without_optional_fields(self):
        """Test to_dict without optional fields."""
        suggestion = LLMIdSuggestion(
            column="test",
            reason="naming",
            confidence="low",
            confidence_score=0.3,
        )
        d = suggestion.to_dict()

        assert d["column"] == "test"
        assert "uniqueness_ratio" not in d  # None should be excluded
        assert "explanation" not in d  # Empty should be excluded

    def test_confidence_score_bounds(self):
        """Test confidence_score bounds."""
        with pytest.raises(ValidationError):
            LLMIdSuggestion(
                column="test",
                reason="naming",
                confidence="high",
                confidence_score=1.5,  # Over 1.0
            )

        with pytest.raises(ValidationError):
            LLMIdSuggestion(
                column="test",
                reason="naming",
                confidence="high",
                confidence_score=-0.1,  # Below 0.0
            )

    def test_explanation_max_length(self):
        """Test explanation max length."""
        long_explanation = "x" * 301
        with pytest.raises(ValidationError):
            LLMIdSuggestion(
                column="test",
                reason="naming",
                confidence="high",
                confidence_score=0.9,
                explanation=long_explanation,
            )
