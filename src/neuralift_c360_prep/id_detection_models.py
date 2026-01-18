"""
Pydantic models for LLM-enhanced ID detection.

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ASCII pattern for structured output validation
ASCII = r"^[\x20-\x7E]*$"


class KeyType(str, Enum):
    """Classification of key type."""

    SURROGATE = "surrogate"  # System-generated (auto-increment, UUID)
    BUSINESS = "business"  # Domain-meaningful (SSN, email, product_code)
    COMPOSITE = "composite"  # Multiple columns together
    UNKNOWN = "unknown"


class IdFormat(str, Enum):
    """Detected format of ID values."""

    UUID_V4 = "uuid_v4"
    UUID_V1 = "uuid_v1"
    UUID_OTHER = "uuid_other"
    GUID = "guid"
    SEQUENTIAL_INT = "sequential_int"
    HASH = "hash"
    ALPHANUMERIC = "alphanumeric"
    EMAIL = "email"
    PHONE = "phone"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class IdColumnAnalysis(BaseModel):
    """LLM analysis of a single column as potential ID."""

    column_name: str = Field(..., description="Name of the analyzed column")
    is_likely_id: bool = Field(
        ..., description="Whether this appears to be an ID column"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    key_type: KeyType = Field(..., description="Classification of the key type")
    id_format: IdFormat = Field(..., description="Detected format of ID values")
    reasoning: str = Field(
        ...,
        max_length=300,
        description="Brief explanation of the classification",
    )

    class Config:
        use_enum_values = True


class CompositeKeyCandidate(BaseModel):
    """Suggestion for a composite primary key.

    Note: Only suggested when individual columns are NOT unique on their own.
    """

    columns: list[str] = Field(..., min_length=2, description="Columns forming the key")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(
        ...,
        max_length=300,
        description="Why these columns together form a unique identifier",
    )


class IdDetectionResult(BaseModel):
    """Complete LLM analysis result for ID detection."""

    primary_candidates: list[IdColumnAnalysis] = Field(
        default_factory=list,
        description="Columns likely to be primary identifiers",
    )
    composite_candidates: list[CompositeKeyCandidate] = Field(
        default_factory=list,
        description="Suggested composite key combinations (only when singles are not unique)",
    )
    table_context: str = Field(
        ...,
        max_length=200,
        description="Inferred context about the table (e.g., 'customer transactions')",
    )


class LLMIdSuggestion(BaseModel):
    """Enhanced ID suggestion with LLM-derived insights.

    Extends the base IdSuggestion concept with semantic analysis.
    """

    column: str
    reason: Literal["naming", "uniqueness", "both", "semantic", "llm"]
    confidence: Literal["high", "medium", "low"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    uniqueness_ratio: float | None = None
    key_type: KeyType = KeyType.UNKNOWN
    id_format: IdFormat = IdFormat.UNKNOWN
    explanation: str = Field(default="", max_length=300)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        result = {
            "column": self.column,
            "reason": self.reason,
            "confidence": self.confidence,
            "confidence_score": self.confidence_score,
            "key_type": self.key_type.value
            if isinstance(self.key_type, KeyType)
            else self.key_type,
            "id_format": self.id_format.value
            if isinstance(self.id_format, IdFormat)
            else self.id_format,
        }
        if self.uniqueness_ratio is not None:
            result["uniqueness_ratio"] = self.uniqueness_ratio
        if self.explanation:
            result["explanation"] = self.explanation
        return result


__all__ = [
    "KeyType",
    "IdFormat",
    "IdColumnAnalysis",
    "CompositeKeyCandidate",
    "IdDetectionResult",
    "LLMIdSuggestion",
]
