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

import logging
import os
import yaml

from .cluster import get_client
from .config import BundleConfig
from .databricks_oauth import get_databricks_access_token
from .data_doctor import (
    analyze_data as data_doctor_analyze,
    print_report as data_doctor_print,
)
from .functions import apply_functions
from .id_detection import suggest_id_columns, print_id_suggestions
from .ingest import load_lazy_dask
from .log_utils import setup_logging
from .preprocess import preprocess, drop_configured_columns
from .write import (
    count_parquet_files,
    write_ddf_and_yaml_to_s3,
    create_managed_uc_volume_via_sql,
    tag_uc_volume_via_sql,
)
from .metadata import build_metadata, build_minimal_config, resolve_output_table_name

logger = logging.getLogger(__name__)


def _strip_scheme(host: str) -> str:
    return host.replace("https://", "").replace("http://", "").rstrip("/")


def _configure_third_party_logging(*, debug_mode: bool) -> None:
    if debug_mode:
        return
    logging.getLogger("distributed.shuffle").setLevel(logging.ERROR)
    logging.getLogger("distributed.shuffle._scheduler_plugin").setLevel(logging.ERROR)
    logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)
    os.environ.setdefault("RUST_LOG", "deltalake=error,delta_kernel=error")


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


def _print_output_summary(
    *,
    base_uri: str,
    table_name: str,
    parquet_count: int | None,
    volume_info: dict | None,
    has_suggestions: bool = False,
) -> None:
    logger.info("=== Output Summary ===")
    logger.info("Base URI       : %s", base_uri)
    if parquet_count is None:
        logger.info("Parquet files  : unknown")
    else:
logger.info("Parquet files  : %s", parquet_count)
    artifacts = "config.yaml, bundleconfig.yaml, data_dictionary.json"
    if has_suggestions:
        artifacts += ", suggestions.yaml"
    logger.info("Artifacts      : %s", artifacts)
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

    # If debug, force progress logs on and configure third party logging
    debug_mode = cfg.logging.level == "debug"
    _configure_third_party_logging(debug_mode=debug_mode)
    if debug_mode and not cfg.logging.show_progress:
        cfg.logging.show_progress = True

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
            conn_params=conn_params,
            allow_dbsql_fallback=cfg.input.source == "uc_table",
            require_logical_names=cfg.input.require_logical_names,
            debug_rename_map=cfg.logging.level == "debug",
            debug_head_rows=cfg.logging.debug_head_rows
            if cfg.logging.level == "debug"
            else 0,
        )

        # ID detection and suggestions (for agentic use)
        ids_cfg = getattr(cfg, "ids", None)
        if ids_cfg and getattr(ids_cfg, "auto_detect", True):
            explicit_ids = list(getattr(ids_cfg, "columns", []) or [])
            # Also include legacy input.id_cols
            explicit_ids = explicit_ids or list(cfg.input.id_cols or [])
            suggestions = suggest_id_columns(
                ddf,
                exclude_columns=explicit_ids,
                check_uniqueness=cfg.logging.show_progress,  # skip expensive uniqueness check if not showing progress
            )
            print_id_suggestions(suggestions, explicit_ids=explicit_ids)

        ddf = preprocess(ddf, cfg)
        ddf = apply_functions(ddf, cfg)
        ddf = drop_configured_columns(
            ddf, getattr(cfg, "drop_columns", []), verbose=cfg.logging.level == "debug"
        )

        # Persist after preprocessing to avoid re-computing during metadata generation
        if getattr(cfg.output, "persist_after_preprocess", True):
            logger.warning(
                "[persist] persist_after_preprocess=True can destabilize large Coiled runs; disable if the cluster dies post-LLM."
            )
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
                logger.info(
                    f"[persist] ✅ preprocessed dataframe persisted in {time.time() - t_persist:.2f}s"
                )
            except ValueError:
                # No distributed client (local mode)
                logger.debug(
                    "[persist] no distributed client; skipping persist_after_preprocess"
                )

        # Verify cluster is still alive before starting expensive metadata computation
        try:
            from dask.distributed import get_client as get_dask_client

            client = get_dask_client()
            n_workers = len(client.scheduler_info().get("workers", {}))
            if n_workers == 0:
                raise RuntimeError(
                    "Cluster has no workers available. Check Coiled dashboard for cluster status."
                )
            logger.debug(
                "[cluster] verified %d workers available before metadata computation",
                n_workers,
            )
        except ValueError:
            pass  # No distributed client (local mode)

        table_name = resolve_output_table_name(cfg)
        meta, meta_text = build_metadata(ddf, cfg, table_name_override=table_name)
        bundle_config_text = yaml.safe_dump(cfg.model_dump(), sort_keys=False)
meta_json = json.loads(meta_text)

        # Run Data Doctor analysis (if enabled)
        data_doctor_report = None
        if getattr(cfg.data_doctor, "enabled", True):
            logger.info("Running Data Doctor analysis...")
            data_doctor_report = data_doctor_analyze(
                ddf,
                meta_json,
                cfg,
                show_progress=cfg.logging.show_progress,
            )
            data_doctor_print(
                data_doctor_report,
                show_alternatives=getattr(
                    cfg.data_doctor, "show_business_alternatives", True
                ),
                use_print=cfg.logging.level in {"info", "debug"},
            )

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
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
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

        # Generate data doctor suggestions yaml if enabled
        suggestions_yaml_text = None
        if data_doctor_report and getattr(cfg.data_doctor, "save_yaml", True):
            suggestions_yaml_text = data_doctor_report.yaml_text

        base_uri = write_ddf_and_yaml_to_s3(
            ddf=ddf,
            s3_base=base_uri,
            config_yaml_text=pretty_config_text,
            meta_json_text=meta_text,
            bundle_config_yaml_text=bundle_config_text,
            suggestions_yaml_text=suggestions_yaml_text,
            partition_on=cfg.output.partitions,
            target_mb_per_part=cfg.output.target_mb_per_part,
            force_npartitions=cfg.output.force_npartitions,
            write_index=cfg.output.write_index,
            shuffle_before_partition_on=getattr(
                cfg.output, "shuffle_before_partition_on", True
            ),
            persist_before_write=getattr(cfg.output, "persist_before_write", True),
            rebalance_before_write=getattr(cfg.output, "rebalance_before_write", True),
            cast_bool_to_int8=getattr(cfg.output, "cast_bool_to_int8", True),
            cast_category_to_string=getattr(
                cfg.output, "cast_category_to_string", True
            ),
            cast_object_to_string=getattr(cfg.output, "cast_object_to_string", True),
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
            has_suggestions=suggestions_yaml_text is not None,
        )

    logger.info("Pipeline complete; output at %s", base_uri)
    return base_uri


__all__ = ["run_from_config"]
