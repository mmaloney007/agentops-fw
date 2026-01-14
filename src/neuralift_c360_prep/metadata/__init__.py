#!/usr/bin/env python3
"""
metadata/__init__.py
--------------------
Metadata package for Neuralift C360 Prep.

This package provides:
    - LLM-based column definitions and table comments
    - Column tagging and classification
    - Segmenter configuration generation with ML heuristics
    - Schema alignment inspection utilities

All public APIs are re-exported here for backward compatibility:
    from neuralift_c360_prep.metadata import build_metadata, SEGMENTER_CONFIG_DEFAULTS

Author: Mike Maloney - Neuralift, Inc.
Updated: 2026-01-13
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

# LLM-based column definitions and table comments
from .llm import (
    ColumnDescription,
    CommentModel,
    DEFAULT_TABLE_COMMENT_MODEL,
    build_prompt,
    create_intelligent_data_dictionary,
    create_table_comment,
)

# Column tagging and data dictionary JSON building
from .tagging import (
    build_column_tags_yaml_dask,
    build_data_dictionary_json,
    inspect_schema_alignment,
)

# Segmenter config generation with ML heuristics
from .config_builder import (
    SEGMENTER_CONFIG_DEFAULTS,
    build_pretty_config_from_data_dict,
    infer_column_roles_from_data_dict,
    print_config_yaml,
    render_config_yaml_with_comments,
    save_config_yaml,
    suggest_autoencoder_dims,
    suggest_backbone_type,
    suggest_batch_size,
    suggest_compute_stats_plan,
    suggest_corruption_probs,
    suggest_explainability_hparams,
    suggest_feature_embed_dim,
    suggest_learning_rate,
    suggest_scheduler,
    suggest_segmenter_hparams,
    suggest_target_segments,
)

# Shared helpers (exposed for advanced use cases)
from .helpers import (
    ASCII,
    _ascii7,
    _ascii_deep,
    _cache_load,
    _cache_save,
    _current_rate_limit_delay,
    _ensure_instructor_client,
    _gc_collect,
    _hash_key,
    _profile_single_column,
    _rate_limit_before_sleep,
    _relax_rate_limit,
    _wrap_instructor,
)

# High-level orchestration functions
from .orchestrator import (
    build_metadata,
    build_table_comment,
    resolve_output_table_name,
)

__all__ = [
    # LLM module
    "ColumnDescription",
    "CommentModel",
    "DEFAULT_TABLE_COMMENT_MODEL",
    "build_prompt",
    "create_intelligent_data_dictionary",
    "create_table_comment",
    # Tagging module
    "build_column_tags_yaml_dask",
    "build_data_dictionary_json",
    "inspect_schema_alignment",
    # Config builder module
    "SEGMENTER_CONFIG_DEFAULTS",
    "build_pretty_config_from_data_dict",
    "infer_column_roles_from_data_dict",
    "print_config_yaml",
    "render_config_yaml_with_comments",
    "save_config_yaml",
    "suggest_autoencoder_dims",
    "suggest_backbone_type",
    "suggest_batch_size",
    "suggest_compute_stats_plan",
    "suggest_corruption_probs",
    "suggest_explainability_hparams",
    "suggest_feature_embed_dim",
    "suggest_learning_rate",
    "suggest_scheduler",
    "suggest_segmenter_hparams",
    "suggest_target_segments",
    # Helpers (exposed for advanced use)
    "ASCII",
    "_ascii7",
    "_ascii_deep",
    "_cache_load",
    "_cache_save",
    "_current_rate_limit_delay",
    "_ensure_instructor_client",
    "_gc_collect",
    "_hash_key",
    "_profile_single_column",
    "_rate_limit_before_sleep",
    "_relax_rate_limit",
    "_wrap_instructor",
    # Orchestration functions
    "build_metadata",
    "build_table_comment",
    "resolve_output_table_name",
]
