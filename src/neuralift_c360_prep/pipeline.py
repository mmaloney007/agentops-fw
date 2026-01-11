#!/usr/bin/env python3
"""
Orchestration for the Neuralift Dask/Coiled data prep pipeline.

Purpose:
    - Spin up a local or Coiled Dask client, then ingest UC/Parquet/CSV sources with DBSQL-based logical column fallback.
    - Run preprocessing hooks, build metadata (tags, LLM definitions, table comment), and serialize config + metadata YAML/JSON.
    - Write partitioned Parquet and sidecar YAML to S3 or a managed UC volume, creating the volume when s3_base is absent.

Usage:
    from neuralift_c360_prep.pipeline import run_from_config
    output_uri = run_from_config(cfg)

Dependencies:
    - dask.distributed (via cluster helpers)
    - PyYAML
    - databricks-sql-connector (for UC logical columns fallback)
    - neuralift_c360_prep.ingest/preprocess/write/metadata

Author: Mike Maloney - Neuralift, Inc.
Updated: 2025-12-08
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import json
import logging
import os
import re
import yaml

from .cluster import get_client
from .config import BundleConfig
from .databricks_oauth import get_databricks_access_token
from .ingest import load_lazy_dask
from .log_utils import setup_logging
from .preprocess import preprocess
from .write import (
    count_parquet_files,
    write_ddf_and_yaml_to_s3,
    create_managed_uc_volume_via_sql,
    tag_uc_volume_via_sql,
)
from .metadata import (
    SEGMENTER_CONFIG_DEFAULTS,
    build_metadata,
    build_pretty_config_from_data_dict,
    render_config_yaml_with_comments,
    resolve_output_table_name,
)

logger = logging.getLogger(__name__)


def _strip_scheme(host: str) -> str:
    return host.replace("https://", "").replace("http://", "").rstrip("/")


def _build_dbsql_conn_params(*, require: bool) -> dict | None:
    host = os.getenv("DATABRICKS_HOST", "")
    client_id = os.getenv("DATABRICKS_CLIENT_ID", "")
    client_secret = os.getenv("DATABRICKS_CLIENT_SECRET", "")
    wh = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
    if not (host and client_id and client_secret and wh):
        if require:
            raise ValueError(
                "Missing Databricks OAuth envs for DBSQL: "
                "DATABRICKS_HOST, DATABRICKS_CLIENT_ID, "
                "DATABRICKS_CLIENT_SECRET, DATABRICKS_WAREHOUSE_ID"
            )
        return None

    access_token = get_databricks_access_token(
        host=host,
        client_id=client_id,
        client_secret=client_secret,
    )
    return {
        "server_hostname": _strip_scheme(host),
        "http_path": f"/sql/1.0/warehouses/{wh}",
        "access_token": access_token,
    }


def _snake_case(value: str) -> str:
    cleaned = re.sub(r"[^\w]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "run"


def _resolve_wandb_project(cfg, fallback: str) -> str:
    project = getattr(cfg.metadata, "wandb_project", "") or ""
    if not project or project == "project_name_here":
        return _snake_case(fallback)
    return project


def _print_output_summary(
    *,
    base_uri: str,
    table_name: str,
    parquet_count: int | None,
    volume_info: dict | None,
) -> None:
    logger.info("=== Output Summary ===")
    logger.info("Base URI       : %s", base_uri)
    if parquet_count is None:
        logger.info("Parquet files  : unknown")
    else:
        logger.info("Parquet files  : %s", parquet_count)
    logger.info("Artifacts      : config.yaml, bundleconfig.yaml, data_dictionary.json")
    if volume_info:
        logger.info("=== UC Volume ===")
        logger.info("Name           : %s", volume_info["volume_name"])
        logger.info("Comment        : %s", volume_info["volume_comment"])
        logger.info("UC Path        : %s", volume_info["uc_path"])
        tags = volume_info.get("tags", {})
        if tags:
            tags_text = ", ".join(f"{k}={v}" for k, v in tags.items())
            logger.info("Tags           : %s", tags_text)


def run_from_config(cfg: BundleConfig) -> str:
    """
    Execute the pipeline and return the output base URI.
    """
    # Re-setup logging in case run_from_config() is called directly (e.g., from tests)
    # This is idempotent, so it's safe even if cli.py already called it
    setup_logging(
        cfg.logging.level,
        dask_level=cfg.logging.dask_level,
        llm_level=cfg.logging.llm_level,
    )

    with get_client(cfg):
        logger.info("Starting pipeline (engine=%s)", cfg.runtime.engine)
        if cfg.logging.level in {"info", "debug"}:
            logger.info(
                "Resolved config:\n%s",
                yaml.safe_dump(cfg.model_dump(exclude_none=True), sort_keys=False),
            )
        if cfg.input.source == "uc_table":
            fmt, uri = "databricks_table", cfg.input.uc_table
        elif cfg.input.source == "delta_path":
            fmt, uri = "delta", cfg.input.delta_path
        elif cfg.input.source == "parquet":
            fmt, uri = "parquet", cfg.input.parquet_path
        elif cfg.input.source == "csv":
            fmt, uri = "csv", cfg.input.csv_path
        else:
            raise ValueError(f"Unsupported source for pipeline: {cfg.input.source}")

        # Build Databricks conn params for UC schema lookup / DBSQL fallback.
        conn_params = None
        if cfg.input.source == "uc_table":
            conn_params = _build_dbsql_conn_params(require=True)

        ddf = load_lazy_dask(
            fmt=fmt,
            uri=uri,
            id_cols=cfg.input.id_cols,
            columns=cfg.input.columns,
            dtype_overrides=cfg.input.dtype_overrides,
            read_blocksize_mb=cfg.output.target_mb_per_part,
            snapshot_mode="off",
            conn_params=conn_params,
            allow_dbsql_fallback=cfg.input.source == "uc_table",
            require_logical_names=cfg.input.require_logical_names,
            debug_rename_map=cfg.logging.level == "debug",
            debug_head_rows=cfg.logging.debug_head_rows
            if cfg.logging.level == "debug"
            else 0,
        )

        ddf = preprocess(ddf, cfg)

        # Persist after preprocessing to avoid re-computing during metadata generation
        if getattr(cfg.output, "persist_after_preprocess", True):
            try:
                from dask.distributed import get_client as get_dask_client, wait
                import time
                client = get_dask_client()
                logger.info(
                    "[persist] persisting preprocessed dataframe to cluster memory "
                    "(avoids re-computing preprocessing for every metadata stat)..."
                )
                t_persist = time.time()
                ddf = client.persist(ddf)
                # Wait for persist to actually complete (persist() is async)
                wait(ddf)
                logger.info(f"[persist] ✅ preprocessed dataframe persisted in {time.time() - t_persist:.2f}s")
            except ValueError:
                # No distributed client (local mode)
                logger.debug("[persist] no distributed client; skipping persist_after_preprocess")

        table_name = resolve_output_table_name(cfg)
        meta, meta_text = build_metadata(ddf, cfg, table_name_override=table_name)
        bundle_config_text = yaml.safe_dump(cfg.model_dump(), sort_keys=False)
        meta_json = json.loads(meta_text)
        wandb_project = _resolve_wandb_project(cfg, cfg.output.run_name or table_name)
        pretty_config = build_pretty_config_from_data_dict(
            data_dict=meta_json,
            ddf=ddf,
            use_gpu=cfg.metadata.use_gpu,
            use_wandb=cfg.metadata.use_wandb,
            config_debug=cfg.metadata.config_debug,
            wandb_project=wandb_project,
            max_card_for_cat=cfg.metadata.tags.max_card,
        )
        pretty_config_text = render_config_yaml_with_comments(
            pretty_config,
            header_comment_lines=[
                "Generated by neuralift_c360_prep.",
                "Inline '# default:' comments reflect neuralift_segmenter/config.py defaults.",
            ],
            defaults=SEGMENTER_CONFIG_DEFAULTS,
        )

        base_uri = cfg.output.s3_base
        volume_tags = None
        if base_uri is None:
            include_c360_tag = getattr(cfg.output, "include_c360_tag", True)
            # Create volume and use its storage location
            conn = conn_params or _build_dbsql_conn_params(require=True)
            volume_comment = (
                cfg.output.run_name
                or cfg.output.uc_table
                or cfg.output.uc_volume_name
                or "Managed UC volume"
            )
            vol, uc_path, s3_loc, created_date_utc, created_ts_utc = (
                create_managed_uc_volume_via_sql(
                    catalog=cfg.output.uc_catalog,
                    schema=cfg.output.uc_schema,
                    table_name=table_name,
                    conn_params=conn,
                    volume_comment=volume_comment,
                )
            )
            tag_map = {
                "table": table_name,
                "created_date_utc": created_date_utc,
                "created_ts_utc": created_ts_utc,
            }
            if include_c360_tag:
                tag_map["type"] = "c360"
            if cfg.output.uc_volume_name:
                tag_map["uc_volume_name"] = cfg.output.uc_volume_name
            volume_tags = {
                "volume_name": vol,
                "uc_path": uc_path,
                "s3_loc": s3_loc,
                "volume_comment": volume_comment,
                "created_date_utc": created_date_utc,
                "created_ts_utc": created_ts_utc,
                "conn_params": conn,
                "tags": tag_map,
                "include_c360_tag": include_c360_tag,
            }
            base_uri = s3_loc

        base_uri = write_ddf_and_yaml_to_s3(
            ddf=ddf,
            s3_base=base_uri,
            config_yaml_text=pretty_config_text,
            meta_json_text=meta_text,
            bundle_config_yaml_text=bundle_config_text,
            partition_on=cfg.output.partitions,
            target_mb_per_part=cfg.output.target_mb_per_part,
            force_npartitions=cfg.output.force_npartitions,
            write_index=cfg.output.write_index,
        )

        if volume_tags:
            tag_uc_volume_via_sql(
                catalog=cfg.output.uc_catalog,
                schema=cfg.output.uc_schema,
                volume_name=volume_tags["volume_name"],
                table_name=table_name,
                conn_params=volume_tags["conn_params"],
                created_date_utc=volume_tags["created_date_utc"],
                created_ts_utc=volume_tags["created_ts_utc"],
                uc_volume_name=cfg.output.uc_volume_name,
                include_c360_tag=volume_tags.get("include_c360_tag", True),
            )

        parquet_count = count_parquet_files(s3_base=base_uri)
        _print_output_summary(
            base_uri=base_uri,
            table_name=table_name,
            parquet_count=parquet_count,
            volume_info=volume_tags,
        )

    logger.info("Pipeline complete; output at %s", base_uri)
    return base_uri


__all__ = ["run_from_config"]
