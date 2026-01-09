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

import os
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator, ConfigDict

from .env import load_dotenv_file

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


class SnapshotConfig(BaseModel):
    enabled: bool = False
    repartition_target: Optional[int] = None


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
    snapshot: SnapshotConfig = Field(default_factory=SnapshotConfig)

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
    missing_fill: Literal["auto", "none"] = "auto"
    zsml: ZSMLConfig = Field(default_factory=ZSMLConfig)


class KPIFunctionConfig(BaseModel):
    type: Literal["zsml"] = "zsml"
    source_col: str
    out_col: Optional[str] = None
    zero_threshold: float = 0.0
    quantiles: Sequence[float] = (0.33, 0.66)
    clip_high_quantile: float = 0.95
    unit: str = ""
    range_style: Literal["text", "math"] = "text"
    add_prefix: bool = True


class TagConfig(BaseModel):
    id_cols: List[str] = Field(default_factory=list)
    kpi_cols: List[str] = Field(default_factory=list)
    missing_indicator_cols: List[str] = Field(default_factory=list)
    extra_tags_all: Optional[dict[str, str]] = None
    extra_tags_by_column: dict[str, dict[str, str]] = Field(default_factory=dict)
    max_card: int = 25
    use_approx_unique: bool = True


class MetadataConfig(BaseModel):
    model: str = "gpt-5-nano"
    sample_rows: int = 5_000
    max_concurrency: int = 15
    tags: TagConfig = Field(default_factory=TagConfig)
    schema_alignment: bool = True
    context: str = ""
    op_timeout: str = "45m"
    use_wandb: bool = True
    use_gpu: bool = True
    config_debug: bool = False
    wandb_project: str = "project_name_here"


class OutputConfig(BaseModel):
    uc_catalog: str
    uc_schema: str
    uc_table: str | None = None
    run_name: str | None = None
    uc_volume_name: str | None = None
    s3_base: str | None = None
    partitions: List[str] = Field(default_factory=list)
    target_mb_per_part: int = 512
    force_npartitions: Optional[int] = None
    write_index: bool = False
    include_c360_tag: bool = True

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
    debug_head_rows: int = 5


class BundleConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    feature_functions: List[str] = Field(default_factory=list)
    kpi_functions: List[KPIFunctionConfig] = Field(default_factory=list)
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig, alias="cleaning"
    )
    drop_columns: List[str] = Field(default_factory=list)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    output: OutputConfig

    @model_validator(mode="before")
    def _promote_legacy_keys(cls, data):
        if not isinstance(data, dict):
            return data
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
        return data

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

__all__ = ["BundleConfig", "load_config", "PreprocessingConfig", "CleaningConfig"]
