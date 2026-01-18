#!/usr/bin/env python3
"""
Structured configuration loader for the Dask/Coiled bundle pipeline.

Purpose:
    - Define logging, runtime/cluster, input, feature_functions, KPI functions,
      preprocessing, drop-columns, metadata, and output options for the CLI.
    - Enforce validation (required creds, unsupported sources, volume existence guard flag).
    - Provide a single entry point `load_config(path)` that reads YAML, applies defaults,
      and returns a validated `BundleConfig`.

Policy note:
    - Output partitioning defaults are now:
        * target_mb_per_part = 512
        * minimum partitions = 2
      which are applied at read/write time unless force_npartitions is set.

Usage:
    from neuralift_c360_prep.config import load_config
    cfg = load_config("configs/data_prep.yaml")

Dependencies:
    - pydantic
    - pyyaml

Author: Mike Maloney - Neuralift, Inc.
Updated: 2025-12-09
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Union

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
    ConfigDict,
)

from .env import load_dotenv_file

logger = logging.getLogger(__name__)

SourceType = Literal[
    "uc_table",
    "delta_path",
    "parquet",
    "csv",
    "dbsql_query",
    "bigquery",
    "azure_parquet",
    "azure_csv",
]


class CoiledConfig(BaseModel):
    name: str = "neuralift_c360_prep"
    software_env: str = "neuralift_c360_prep"
    n_workers: int = 2
    worker_cpu: int | None = 2
    worker_memory: str | None = "16GiB"
    worker_vm_types: Optional[List[str]] = None
    scheduler_cpu: int | None = None
    scheduler_memory: str | None = None
    scheduler_vm_types: Optional[List[str]] = None
    idle_timeout: str = "2h"
    no_client_timeout: str = "2h"
    create_timeout: str = "15m"
    op_timeout: str = "45m"
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_resources(self) -> "CoiledConfig":
        if self.worker_vm_types:
            if not all(self.worker_vm_types):
                raise ValueError("worker_vm_types must be a non-empty list")
            if (
                self.scheduler_vm_types is None
                and self.scheduler_cpu is None
                and self.scheduler_memory is None
            ):
                self.scheduler_vm_types = list(self.worker_vm_types)
        else:
            if self.worker_cpu is None or self.worker_cpu <= 0:
                raise ValueError(
                    "worker_cpu must be positive when worker_vm_types is not set"
                )
            if not self.worker_memory:
                raise ValueError(
                    "worker_memory must be set when worker_vm_types is not set"
                )

        if self.scheduler_vm_types:
            if not all(self.scheduler_vm_types):
                raise ValueError("scheduler_vm_types must be a non-empty list")
        else:
            if self.scheduler_cpu is None:
                self.scheduler_cpu = self.worker_cpu
            if self.scheduler_memory is None:
                self.scheduler_memory = self.worker_memory
            if self.scheduler_cpu is None or self.scheduler_cpu <= 0:
                raise ValueError(
                    "scheduler_cpu must be positive when scheduler_vm_types is not set"
                )
            if not self.scheduler_memory:
                raise ValueError(
                    "scheduler_memory must be set when scheduler_vm_types is not set"
                )

        return self


class RuntimeConfig(BaseModel):
    engine: Literal["local", "coiled"] = "local"
    coiled: CoiledConfig = Field(default_factory=CoiledConfig)


class BigQueryConfig(BaseModel):
    project: str = ""
    dataset: str = ""
    table: str = ""
    query: Optional[str] = None
    temp_gcs_bucket: str = ""


class AzureConfig(BaseModel):
    account_name: str = ""
    container: str = ""
    path: str = ""


# ---------------------------------------------------------------------------
#  New unified config classes (agentic experience)
# ---------------------------------------------------------------------------


class LLMProviderConfig(BaseModel):
    """Configuration for LLM provider."""

    provider: Literal["openai", "anthropic", "auto"] = "auto"
    model: Optional[str] = None  # Provider-specific model name
    timeout_seconds: int = 60
    max_retries: int = 4


class IdDetectionConfig(BaseModel):
    """Configuration for LLM-enhanced ID/Primary Key detection."""

    # LLM enhancement options
    llm_enabled: bool = False  # Opt-in by default
    llm_provider: LLMProviderConfig = Field(default_factory=LLMProviderConfig)

    # Cost control
    max_llm_columns: int = 20  # Max columns to analyze with LLM
    llm_cache_enabled: bool = True
    llm_cache_dir: str = ".nl_id_cache"

    # Detection tuning
    uniqueness_threshold: float = 0.95  # Ratio to consider unique
    gray_zone_lower: float = 0.80  # Lower bound for ambiguous uniqueness
    detect_uuid_format: bool = True  # Enable UUID/GUID format detection


class IdsConfig(BaseModel):
    """ID column configuration with auto-detection support."""

    columns: List[str] = Field(default_factory=list)
    auto_detect: bool = True
    detection: IdDetectionConfig = Field(default_factory=IdDetectionConfig)


class FillConfig(BaseModel):
    """Granular missing value fill strategies."""

    categorical: Union[str, None] = "Unknown"  # "Unknown" | "mode" | custom string
    continuous: Union[str, float, int, None] = "median"  # "median" | "mean" | number
    overrides: dict[str, Union[str, float, int, None]] = Field(default_factory=dict)


class LiftConfig(BaseModel):
    """Lift/tag metadata for KPIs (value and event sums)."""

    value_sum_column: Optional[str] = None
    value_sum_unit: Optional[str] = None
    event_sum_column: Optional[str] = None
    event_sum_unit: Optional[str] = None


class IdentityKPIConfig(BaseModel):
    """Identity KPI: tag existing column as KPI without transformation.

    DEPRECATED: Use FunctionConfig with type='identity' instead.
    """

    column: str
    lift: LiftConfig = Field(default_factory=LiftConfig)


class InputConfig(BaseModel):
    source: SourceType = "parquet"
    uc_table: Optional[str] = None
    delta_path: Optional[str] = None
    parquet_path: Optional[str] = None
    csv_path: Optional[str] = None
    dbsql_query: Optional[str] = None
    bigquery: BigQueryConfig = Field(default_factory=BigQueryConfig)
    azure: AzureConfig = Field(default_factory=AzureConfig)
    columns: Optional[List[str]] = None
    dtype_overrides: Optional[dict[str, str]] = None
    require_logical_names: bool = True
    id_cols: List[str] = Field(default_factory=list)
    # Performance options
    use_pyarrow_strings: bool = (
        True  # Use PyArrow-backed strings for ~50% memory reduction
    )
    read_blocksize_mb: int = 128  # Blocksize for reads; smaller = more parallelism

    @model_validator(mode="before")
    def _reject_snapshot(cls, data):
        if isinstance(data, dict) and "snapshot" in data:
            raise ValueError("snapshot is no longer supported; remove input.snapshot")
        return data

    @model_validator(mode="after")
    def _validate_source(self) -> "InputConfig":
        src = self.source
        if src == "uc_table" and not self.uc_table:
            raise ValueError("uc_table source requires uc_table=<catalog.schema.table>")
        if src == "delta_path" and not self.delta_path:
            raise ValueError("delta_path source requires delta_path")
        if src == "parquet" and not self.parquet_path:
            raise ValueError("parquet source requires parquet_path")
        if src == "csv" and not self.csv_path:
            raise ValueError("csv source requires csv_path")
        if src == "dbsql_query" and not self.dbsql_query:
            raise ValueError("dbsql_query source requires dbsql_query text")
        if src == "bigquery":
            if not (self.bigquery.table or self.bigquery.query):
                raise ValueError(
                    "bigquery source requires bigquery.table or bigquery.query"
                )
            if not self.bigquery.project:
                raise ValueError("bigquery source requires bigquery.project")
        if src in {"azure_parquet", "azure_csv"}:
            # We intentionally keep this configurable but unimplemented for now.
            pass
        return self


class ZSMLConfig(BaseModel):
    enabled: bool = False
    source_col: Optional[str] = None
    out_col: Optional[str] = None
    zero_threshold: float = 0.0
    quantiles: Sequence[float] = (0.33, 0.66)
    clip_high_quantile: float = 0.95
    unit: str = ""
    range_style: Literal["text", "math"] = "text"
    add_prefix: bool = True


class PreprocessingConfig(BaseModel):
    rename_to_snake: bool = True
    bool_fix: bool = True
    drop_empty: bool = True
    drop_constant: bool = True
    # Legacy field (deprecated, use fill instead)
    missing_fill: Literal["auto", "none"] = "auto"
    # NEW: Granular fill options
    fill: FillConfig = Field(default_factory=FillConfig)
    zsml: ZSMLConfig = Field(default_factory=ZSMLConfig)

    @model_validator(mode="after")
    def _warn_legacy_fill(self) -> "PreprocessingConfig":
        # Note: We can't easily detect if missing_fill was explicitly set vs default
        # The deprecation warning will be emitted in BundleConfig validator instead
        return self


class FunctionConfig(BaseModel):
    """Unified function configuration - all function types in one model.

    This is the NEW simplified config format where all functions live in a single
    flat list. Any function can be marked as a KPI with `kpi: true` and can have
    lift metadata with the `lift:` block.

    Example YAML:
        functions:
          - type: zsml
            source_col: Revenue
            out_col: revenue_tier
            kpi: true
            lift:
              value_sum_column: Revenue
              value_sum_unit: USD
              event_sum_column: Purchases
              event_sum_unit: events

          - type: identity
            column: CustomerComplainedRecently
            kpi: true

          - type: binning
            source_col: Age
            out_col: age_bin
    """

    type: Literal[
        # KPI/Identity types
        "zsml",  # ZSML KPI tiering
        "identity",  # Tag existing column (no transformation)
        # Feature engineering types
        "callable",
        "binning",
        "winsorize",
        "log_transform",
        "date_parts",
        "categorical_bucket",
        "ratio",
        "string_normalize",
        "frequency_encode",
        "days_since",
    ] = "callable"

    # =========================================================================
    # Universal fields (available on ALL function types)
    # =========================================================================
    kpi: bool = False  # Mark output as KPI column
    kpi_columns: List[str] = Field(default_factory=list)  # For multi-output functions
    lift: LiftConfig = Field(default_factory=LiftConfig)  # Lift metadata
    return_mode: Literal["all", "new_only", "list"] = "all"
    return_columns: List[str] = Field(default_factory=list)

    # =========================================================================
    # Identity type - tag existing column
    # =========================================================================
    column: Optional[str] = None  # Required for type=identity

    # =========================================================================
    # ZSML type - KPI tiering
    # =========================================================================
    zero_threshold: float = 0.0
    clip_high_quantile: float = 0.95
    unit: str = ""
    range_style: Literal["text", "math"] = "text"
    add_prefix: bool = True

    # =========================================================================
    # Common source/output fields
    # =========================================================================
    source_col: Optional[str] = None
    out_col: Optional[str] = None
    output_suffix: Optional[str] = None
    callable: Optional[str] = None  # "module:function" for type=callable
    inputs: List[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

    # =========================================================================
    # Binning fields
    # =========================================================================
    bins: Optional[Sequence[float]] = None
    quantiles: Optional[Sequence[float]] = (0.33, 0.66)  # Shared with ZSML
    labels: Optional[Sequence[str]] = None
    right: bool = True
    include_lowest: bool = True

    # =========================================================================
    # Winsorize fields
    # =========================================================================
    lower_quantile: Optional[float] = 0.01
    upper_quantile: Optional[float] = 0.99
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    # =========================================================================
    # Log transform fields
    # =========================================================================
    log_method: Literal["log1p", "log"] = "log1p"
    log_offset: float = 0.0
    log_clip_min: Optional[float] = 0.0
    log_clip_max: Optional[float] = None
    log_on_nonpositive: Literal["nan", "zero"] = "nan"

    # =========================================================================
    # String normalize fields
    # =========================================================================
    string_case: Literal["lower", "upper", "title", "none"] = "lower"
    strip: bool = True
    collapse_whitespace: bool = True
    replace_regex: Optional[str] = None
    replace_with: str = ""

    # =========================================================================
    # Frequency encode fields
    # =========================================================================
    normalize: bool = True

    # =========================================================================
    # Date parts fields
    # =========================================================================
    date_parts: Optional[Sequence[str]] = None
    drop_source: bool = True
    timestamp_unit: Optional[Literal["auto", "s", "ms", "us", "ns"]] = "auto"
    timezone: Optional[str] = None
    auto_daypart: bool = True
    daypart_bins: Optional[dict[str, Sequence[int]]] = None

    # =========================================================================
    # Days since fields
    # =========================================================================
    reference_date: Optional[str] = None

    # =========================================================================
    # Categorical bucket fields
    # =========================================================================
    top_k: Optional[int] = None
    min_count: Optional[int] = None
    other_label: str = "other"

    # =========================================================================
    # Ratio fields
    # =========================================================================
    numerator_col: Optional[str] = None
    denominator_col: Optional[str] = None
    on_zero: Literal["nan", "zero", "epsilon"] = "zero"
    epsilon: float = 1.0e-9

    @model_validator(mode="before")
    def _coerce_aliases(cls, data):
        if not isinstance(data, dict):
            return data
        # Alias mappings
        if "out_suffix" in data and "output_suffix" not in data:
            data["output_suffix"] = data.pop("out_suffix")
        if "return" in data and "return_mode" not in data:
            data["return_mode"] = data.pop("return")
        if "kpi_cols" in data and "kpi_columns" not in data:
            data["kpi_columns"] = data.pop("kpi_cols")
        if "method" in data and "log_method" not in data:
            data["log_method"] = data.pop("method")
        if "offset" in data and "log_offset" not in data:
            data["log_offset"] = data.pop("offset")
        if "clip_min" in data and "log_clip_min" not in data:
            data["log_clip_min"] = data.pop("clip_min")
        if "clip_max" in data and "log_clip_max" not in data:
            data["log_clip_max"] = data.pop("clip_max")
        if "on_nonpositive" in data and "log_on_nonpositive" not in data:
            data["log_on_nonpositive"] = data.pop("on_nonpositive")
        if "parts" in data and "date_parts" not in data:
            data["date_parts"] = data.pop("parts")
        # Only map unit -> timestamp_unit for date_parts functions
        # (zsml uses `unit` for currency formatting like "$")
        if (
            "unit" in data
            and "timestamp_unit" not in data
            and data.get("type") == "date_parts"
        ):
            data["timestamp_unit"] = data.pop("unit")
        if "tz" in data and "timezone" not in data:
            data["timezone"] = data.pop("tz")
        if "timezone_col" in data or "tz_col" in data:
            raise ValueError("timezone_col is not supported; use timezone")
        if "dayparts" in data and "daypart_bins" not in data:
            data["daypart_bins"] = data.pop("dayparts")
        if "numerator" in data and "numerator_col" not in data:
            data["numerator_col"] = data.pop("numerator")
        if "denominator" in data and "denominator_col" not in data:
            data["denominator_col"] = data.pop("denominator")
        return data

    @model_validator(mode="after")
    def _validate_type_fields(self) -> "FunctionConfig":
        """Validate required fields per function type."""
        needs_source = {
            "binning",
            "winsorize",
            "log_transform",
            "date_parts",
            "categorical_bucket",
            "string_normalize",
            "frequency_encode",
            "days_since",
        }
        # Identity type validation
        if self.type == "identity":
            if not self.column:
                raise ValueError("type=identity requires 'column'")
        # ZSML type validation
        elif self.type == "zsml":
            if not self.source_col:
                raise ValueError("type=zsml requires 'source_col'")
        # Callable type validation
        elif self.type == "callable":
            if not self.callable:
                raise ValueError("type=callable requires 'callable'")
        # Source column types
        elif self.type in needs_source:
            if not (self.source_col or self.inputs):
                raise ValueError(f"type={self.type} requires source_col or inputs")
        # Ratio type validation
        if self.type == "ratio":
            has_inputs = len(self.inputs or []) >= 2
            if not (self.numerator_col and self.denominator_col) and not has_inputs:
                raise ValueError(
                    "type=ratio requires numerator_col/denominator_col or inputs[0:2]"
                )
        # Categorical bucket validation
        if (
            self.type == "categorical_bucket"
            and self.top_k is None
            and self.min_count is None
        ):
            raise ValueError("categorical_bucket requires top_k or min_count")
        # Winsorize validation
        if self.type == "winsorize":
            if self.lower_bound is None and self.lower_quantile is None:
                raise ValueError("winsorize requires lower_bound or lower_quantile")
            if self.upper_bound is None and self.upper_quantile is None:
                raise ValueError("winsorize requires upper_bound or upper_quantile")
        # Date parts validation
        if (
            self.type == "date_parts"
            and self.date_parts is not None
            and len(self.date_parts) == 0
        ):
            raise ValueError("date_parts must be non-empty if provided")
        if self.daypart_bins is not None:
            for name, bounds in self.daypart_bins.items():
                if len(bounds) != 2:
                    raise ValueError(f"daypart_bins '{name}' must have 2 bounds")
                if not all(isinstance(v, (int, float)) for v in bounds):
                    raise ValueError(f"daypart_bins '{name}' must be numeric")
        return self


# Legacy aliases for backward compatibility
KPIFunctionConfig = FunctionConfig  # DEPRECATED
FeatureFunctionConfig = FunctionConfig  # DEPRECATED


class FunctionsConfig(BaseModel):
    """Unified function configuration - single flat list.

    NEW FORMAT (recommended):
        functions:
          - type: zsml
            source_col: Revenue
            kpi: true
          - type: binning
            source_col: Age

    OLD FORMAT (deprecated but still works):
        functions:
          kpis:
            - type: zsml
              source_col: Revenue
          features:
            - type: binning
              source_col: Age
    """

    # Single flat list of functions (new format)
    functions: List[FunctionConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    def _migrate_legacy_format(cls, data):
        """Migrate old 3-section format to new flat list.

        Also handles new format where functions: is a flat list in YAML.
        """
        # NEW FORMAT: functions: [list] - wrap in dict
        if isinstance(data, list):
            return {"functions": data}

        if not isinstance(data, dict):
            return data

        # Check for old format (has kpis, identity_kpis, or features keys)
        has_old_format = any(k in data for k in ("kpis", "identity_kpis", "features"))
        if not has_old_format:
            return data

        # Migrate to flat list
        flat: List[dict] = []

        # Migrate kpis → type=zsml with kpi=True
        for kpi in data.pop("kpis", []) or []:
            if isinstance(kpi, dict):
                kpi = dict(kpi)  # Copy to avoid mutating original
                kpi["kpi"] = True
                if "type" not in kpi:
                    kpi["type"] = "zsml"
                flat.append(kpi)

        # Migrate identity_kpis → type=identity with kpi=True
        for ident in data.pop("identity_kpis", []) or []:
            if isinstance(ident, dict):
                flat.append(
                    {
                        "type": "identity",
                        "column": ident.get("column"),
                        "kpi": True,
                        "lift": ident.get("lift", {}),
                    }
                )

        # Migrate features (keep as-is)
        for feat in data.pop("features", []) or []:
            if isinstance(feat, str):
                flat.append({"type": "callable", "callable": feat})
            elif isinstance(feat, dict):
                flat.append(feat)

        warnings.warn(
            "functions.{kpis,identity_kpis,features} format is deprecated. "
            "Use a flat list: functions: [{type: zsml, kpi: true, ...}]",
            DeprecationWarning,
            stacklevel=4,
        )

        return {"functions": flat}


class TagConfig(BaseModel):
    id_cols: List[str] = Field(default_factory=list)
    kpi_cols: List[str] = Field(default_factory=list)
    missing_indicator_cols: List[str] = Field(default_factory=list)
    extra_tags_all: Optional[dict[str, str]] = None
    extra_tags_by_column: dict[str, dict[str, str]] = Field(default_factory=dict)
    max_card: int = 25
    use_approx_unique: bool = True
    approx_row_threshold: int = 2_000_000
    approx_gray_band: int = 5  # Set to -1 to skip gray-band exact uniques.
    skip_unique_counts: bool = (
        False  # Skip unique count computation entirely (fast mode)
    )


class DataDoctorConfig(BaseModel):
    """Configuration for the Data Doctor analysis module."""

    enabled: bool = True  # Run after metadata by default
    save_yaml: bool = True  # Save suggestions.yaml alongside output
    show_business_alternatives: bool = True  # Show business-friendly fill options
    high_null_threshold: int = 100  # Null count threshold for high priority
    high_cardinality_threshold: int = 50  # Unique count for high-card categoricals
    top_k_bucket: int = 10  # Default top_k for categorical bucketing suggestions
    # LLM enhancement settings
    llm_enabled: bool = False  # Opt-in for LLM analysis
    llm_provider: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    llm_cache_enabled: bool = True
    llm_cache_dir: str = ".nl_doctor_cache"
    max_llm_columns: int = 30  # Max columns to send to LLM
    generate_executive_summary: bool = True  # Generate LLM executive summary


class MetadataConfig(BaseModel):
    model: str = "gpt-5-nano"
    sample_rows: int = 5_000
    max_concurrency: int = 75
    tags: TagConfig = Field(default_factory=TagConfig)
    schema_alignment: bool = True
    context: str = ""
    op_timeout: str = "45m"
    lift_strict: bool = False
    use_wandb: bool = True
    use_gpu: bool = True
    config_debug: bool = False
    wandb_project: str = "project_name_here"
    # LLM optimization options
    use_llm_cache: bool = False
    llm_cache_dir: str = ".nl_dd_cache"
    skip_table_comment: bool = False
    max_columns_for_comment: Optional[int] = None


class OutputConfig(BaseModel):
    uc_catalog: str
    uc_schema: str
    uc_table: str | None = None
    run_name: str | None = None
    uc_volume_name: str | None = None
    s3_base: str | None = None
    partitions: List[str] = Field(default_factory=list)
    target_mb_per_part: int = (
        512  # Target write size; tune with force_npartitions when needed.
    )
    force_npartitions: Optional[int] = None
    write_index: bool = False
    include_c360_tag: bool = True
    # Performance toggles
    persist_after_preprocess: bool = (
        False  # Avoid pre-LLM persist to reduce memory pressure.
    )
    shuffle_before_partition_on: bool = True
    persist_before_write: bool = False
    rebalance_before_write: bool = False
    count_output_files: bool = True
    # Dtype casting toggles
    cast_bool_to_int8: bool = True
    cast_category_to_string: bool = True
    cast_object_to_string: bool = True

    @model_validator(mode="before")
    def _promote_legacy_partition_keys(cls, data):
        if not isinstance(data, dict):
            return data
        if "target_mb_per_part" not in data and "min_partition_mb" in data:
            data["target_mb_per_part"] = data.get("min_partition_mb")
        return data

    @model_validator(mode="after")
    def _validate_partitions(self) -> "OutputConfig":
        if self.target_mb_per_part <= 0:
            raise ValueError("target_mb_per_part must be positive")
        if self.force_npartitions is not None and self.force_npartitions <= 0:
            raise ValueError("force_npartitions must be positive when set")
        return self


class LoggingConfig(BaseModel):
    level: Literal["debug", "info", "warning", "error"] = "info"
    dask_level: Literal["debug", "info", "warning", "error"] = "info"
    llm_level: Literal["debug", "info", "warning", "error"] = "info"
    debug_head_rows: int = 5
    show_progress: bool = True  # Show progress bars for long-running operations


class BundleConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    # NEW: Unified ID configuration
    ids: IdsConfig = Field(default_factory=IdsConfig)
    # NEW: Unified functions configuration
    functions: FunctionsConfig = Field(default_factory=FunctionsConfig)
    # Legacy fields (deprecated, use ids and functions instead)
    feature_functions: List[str] = Field(default_factory=list)
    kpi_functions: List[KPIFunctionConfig] = Field(default_factory=list)
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig, alias="cleaning"
    )
    drop_columns: List[str] = Field(default_factory=list)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    # Data Doctor: post-metadata analysis and suggestions
    data_doctor: DataDoctorConfig = Field(default_factory=DataDoctorConfig)
    output: OutputConfig

    @model_validator(mode="before")
    def _promote_legacy_keys(cls, data):
        if not isinstance(data, dict):
            return data

        # Track deprecation warnings to emit
        deprecations: list[str] = []

        # Accept legacy "cleaning" key as alias for preprocessing.
        if "cleaning" in data and "preprocessing" not in data:
            data["preprocessing"] = data.get("cleaning") or {}

        pre = data.get("preprocessing")
        # Lift legacy drop_columns nested under cleaning/preprocessing to top-level.
        if isinstance(pre, dict) and "drop_columns" in pre:
            data.setdefault("drop_columns", pre.get("drop_columns", []))
            pre = {k: v for k, v in pre.items() if k != "drop_columns"}
            data["preprocessing"] = pre
        if data.get("drop_columns") is None:
            data["drop_columns"] = []
        if isinstance(pre, dict) and "missing_fill" in pre and "fill" not in pre:
            deprecations.append(
                "preprocessing.missing_fill is deprecated, use preprocessing.fill instead"
            )

        # Promote input.id_cols → ids.columns
        inp = data.get("input", {})
        if isinstance(inp, dict) and inp.get("id_cols"):
            ids_config = data.setdefault("ids", {})
            if isinstance(ids_config, dict) and not ids_config.get("columns"):
                ids_config["columns"] = inp["id_cols"]
                deprecations.append(
                    "input.id_cols is deprecated, use ids.columns instead"
                )

        # Promote feature_functions → functions (flat list)
        if data.get("feature_functions"):
            funcs = data.setdefault("functions", {})
            if isinstance(funcs, dict) and not funcs.get("functions"):
                # Convert string callables to function configs
                flat = []
                for feat in data["feature_functions"]:
                    if isinstance(feat, str):
                        flat.append({"type": "callable", "callable": feat})
                    else:
                        flat.append(feat)
                funcs["functions"] = flat
                deprecations.append(
                    "feature_functions is deprecated, use functions list instead"
                )

        # Promote kpi_functions → functions (flat list with kpi=True)
        if data.get("kpi_functions"):
            funcs = data.setdefault("functions", {})
            if isinstance(funcs, dict):
                existing = funcs.get("functions", [])
                for kpi in data["kpi_functions"]:
                    if isinstance(kpi, dict):
                        kpi = dict(kpi)
                        kpi["kpi"] = True
                        if "type" not in kpi:
                            kpi["type"] = "zsml"
                        existing.append(kpi)
                funcs["functions"] = existing
                deprecations.append(
                    "kpi_functions is deprecated, use functions list instead"
                )

        # Emit deprecation warnings
        for msg in deprecations:
            warnings.warn(msg, DeprecationWarning, stacklevel=4)

        return data

    @model_validator(mode="after")
    def _sync_legacy_fields(self) -> "BundleConfig":
        """Sync new config fields to legacy fields for backward compatibility."""
        # Sync ids.columns → input.id_cols (so existing code still works)
        if self.ids.columns and not self.input.id_cols:
            object.__setattr__(self.input, "id_cols", list(self.ids.columns))

        # Sync functions.functions → feature_functions (for legacy code)
        if self.functions.functions and not self.feature_functions:
            legacy_features = [
                f.callable
                for f in self.functions.functions
                if getattr(f, "type", None) == "callable" and f.callable
            ]
            object.__setattr__(self, "feature_functions", legacy_features)

        # Sync functions.functions (kpi=True) → kpi_functions (for legacy code)
        if self.functions.functions and not self.kpi_functions:
            kpi_funcs = [
                f
                for f in self.functions.functions
                if getattr(f, "kpi", False) and getattr(f, "type", None) == "zsml"
            ]
            object.__setattr__(self, "kpi_functions", kpi_funcs)

        return self

    @model_validator(mode="after")
    def _require_creds(self) -> "BundleConfig":
        missing = []
        required_envs = [
            "OPENAI_API_KEY",
            "WANDB_API_KEY",
            "DATABRICKS_HOST",
            "DATABRICKS_CLIENT_ID",
            "DATABRICKS_CLIENT_SECRET",
            "DATABRICKS_WAREHOUSE_ID",
        ]
        for env in required_envs:
            if os.getenv(env) is None:
                missing.append(env)
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
        return self


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path) -> BundleConfig:
    """
    Load and validate a YAML config file into a BundleConfig.
    """
    load_dotenv_file()
    cfg_path = Path(path)
    data = _read_yaml(cfg_path)
    try:
        return BundleConfig.model_validate(data)
    except ValidationError as exc:
        raise SystemExit(f"Invalid config {cfg_path}:\n{exc}") from exc


CleaningConfig = PreprocessingConfig  # backwards compatibility

__all__ = [
    "BundleConfig",
    "load_config",
    "PreprocessingConfig",
    "CleaningConfig",
    # New config classes
    "IdsConfig",
    "IdDetectionConfig",
    "LLMProviderConfig",
    "FillConfig",
    "LiftConfig",
    "FunctionConfig",  # NEW: Unified function config
    "FunctionsConfig",
    "DataDoctorConfig",
    # Legacy aliases (deprecated)
    "KPIFunctionConfig",
    "FeatureFunctionConfig",
    "IdentityKPIConfig",
]
