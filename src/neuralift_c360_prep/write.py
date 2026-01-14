#!/usr/bin/env python3
"""
write.py (Dask)
---------------
Writers and UC volume helpers extracted from bundle_impl.py.

Purpose:
    - Create managed UC volumes via Databricks SQL (schema/volume creation, tagging, discovery of storage_location).
    - Write Dask DataFrames to S3 (or volume-backed paths) as Parquet with optional repartitioning and type casts, plus sidecar config/meta files.
    - Provide compatibility wrapper for legacy write_outputs signature.

Usage:
    from neuralift_c360_prep.write import (
        create_managed_uc_volume_via_sql,
        write_ddf_and_yaml_to_s3,
        write_outputs,
    )

Dependencies:
    - dask[dataframe] + dask.distributed
    - fsspec
    - pandas
    - PyYAML
    - databricks-sql-connector
    - uuid6 (fallbacks to uuid4)
    - pyarrow (Parquet engine)

Author: Mike Maloney - Neuralift, Inc.
Updated: 2025-12-08
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Iterable, Mapping, Optional

import dask
import dask.dataframe as dd
import fsspec
import yaml
from databricks import sql  # type: ignore
from dask.distributed import as_completed, get_client, wait

try:
    from uuid6 import uuid7  # type: ignore
except Exception:  # pragma: no cover

    def uuid7() -> uuid.UUID:  # type: ignore
        return uuid.uuid4()


logger = logging.getLogger(__name__)


def _quote_ident(name: str) -> str:
    """Quote an identifier for Databricks SQL (backticks, escape internal backticks)."""
    return f"`{name.replace('`', '``')}`"


def _escape_tag_value(val: str) -> str:
    """Escape a string literal for use in a SQL tag value (single quotes)."""
    return val.replace("'", "''")


def create_managed_uc_volume_via_sql(
    *,
    catalog: str,
    schema: str,
    table_name: str,
    conn_params: Mapping[
        str, str
    ],  # {server_hostname, http_path, access_token} or oauth fields
    volume_name: Optional[str] = None,
    volume_comment: Optional[str] = None,
    show_sql: bool = True,
):
    """
    Create a MANAGED UC volume using only Databricks SQL (no SparkSession).
    Returns (volume_name, uc_mount_path, s3_location, created_date_utc, created_ts_utc).
    """
    vol = volume_name or str(uuid7())
    now_utc = datetime.now(timezone.utc)
    created_ts_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    created_date_utc = now_utc.strftime("%Y-%m-%d")

    comment_raw = (volume_comment or "").strip() or table_name
    comment_sql = _escape_tag_value(comment_raw)

    qcat, qsch, qvol = map(_quote_ident, (catalog, schema, vol))

    create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {qcat}.{qsch};"
    create_volume_sql = f"""
    CREATE VOLUME IF NOT EXISTS {qcat}.{qsch}.{qvol}
    COMMENT '{comment_sql}';
    """
    describe_sql = f"DESCRIBE VOLUME {qcat}.{qsch}.{qvol};"

    if show_sql:
        logger.info("CREATE SCHEMA SQL:\n%s", create_schema_sql)
        logger.info("CREATE VOLUME SQL:\n%s", create_volume_sql)
        logger.info("DESCRIBE VOLUME SQL:\n%s", describe_sql)

    with _connect_dbsql(conn_params) as conn, conn.cursor() as cur:
        cur.execute(create_schema_sql)
        cur.execute(create_volume_sql)
        cur.execute(describe_sql)
        rows = cur.fetchall()
        if not rows:
            raise RuntimeError(
                f"No rows returned from DESCRIBE VOLUME for {catalog}.{schema}.{vol}"
            )
        row = rows[0]
        col_names = [d[0].lower() for d in cur.description]
        try:
            idx = col_names.index("storage_location")
        except ValueError:
            raise RuntimeError(
                f"'storage_location' column not found in DESCRIBE VOLUME output; "
                f"columns={col_names}, row={row}"
            )
        s3_location = row[idx]

    uc_mount_path = f"/Volumes/{catalog}/{schema}/{vol}"
    logger.info("Managed volume created:")
    logger.info("  Volume name       : %s", vol)
    logger.info("  UC mount path     : %s", uc_mount_path)
    logger.info("  S3 backing loc    : %s", s3_location)
    logger.info("  created_date_utc  : %s", created_date_utc)
    logger.info("  created_ts_utc    : %s", created_ts_utc)

    return vol, uc_mount_path, s3_location, created_date_utc, created_ts_utc


def tag_uc_volume_via_sql(
    *,
    catalog: str,
    schema: str,
    volume_name: str,
    table_name: str,
    conn_params: Mapping[
        str, str
    ],  # {server_hostname, http_path, auth_type, client_id, client_secret}
    created_date_utc: str,
    created_ts_utc: str,
    uc_volume_name: Optional[str] = None,
    include_c360_tag: bool = True,
    show_sql: bool = True,
) -> None:
    """
    Apply UC volume tags after successful writes.
    """
    qcat, qsch, qvol = map(_quote_ident, (catalog, schema, volume_name))
    tags = {
        "table": table_name,
        "created_date_utc": created_date_utc,
        "created_ts_utc": created_ts_utc,
    }
    if include_c360_tag:
        tags["type"] = "c360"
    if uc_volume_name:
        tags["uc_volume_name"] = uc_volume_name

    tag_pairs = ",\n      ".join(
        f"'{k}' = '{_escape_tag_value(str(v))}'" for k, v in tags.items()
    )
    tag_sql = f"""
    ALTER VOLUME {qcat}.{qsch}.{qvol}
    SET TAGS (
      {tag_pairs}
    );
    """

    if show_sql:
        logger.info("TAG SQL:\n%s", tag_sql)

    with _connect_dbsql(conn_params) as conn, conn.cursor() as cur:
        cur.execute(tag_sql)


def _connect_dbsql(conn_params: Mapping[str, str]) -> sql.Connection:  # type: ignore[name-defined]
    connect_kwargs = {
        "server_hostname": conn_params["server_hostname"],
        "http_path": conn_params["http_path"],
    }
    if "access_token" in conn_params:
        connect_kwargs["access_token"] = conn_params["access_token"]
    else:
        connect_kwargs.update(
            {
                "auth_type": conn_params["auth_type"],
                "client_id": conn_params["client_id"],
                "client_secret": conn_params["client_secret"],
            }
        )
    return sql.connect(**connect_kwargs)


def _storage_options_for_base(
    s3_base: str,
    *,
    aws_key_env: str,
    aws_secret_env: str,
    require_creds: bool,
) -> dict:
    storage_options = {}
    if s3_base.startswith("s3://"):
        aws_key = os.getenv(aws_key_env)
        aws_secret = os.getenv(aws_secret_env)
        if require_creds and (not aws_key or not aws_secret):
            raise RuntimeError(
                f"Missing AWS credentials in env: {aws_key_env}/{aws_secret_env}"
            )
        if aws_key and aws_secret:
            storage_options = {"key": aws_key, "secret": aws_secret}
    return storage_options


def count_parquet_files(
    *,
    s3_base: str,
    aws_key_env: str = "AWS_ACCESS_KEY_ID",
    aws_secret_env: str = "AWS_SECRET_ACCESS_KEY",
) -> int | None:
    """
    Count parquet files under <s3_base>/input_data/. Returns None if inaccessible.
    """
    s3_base = s3_base.rstrip("/") + "/"
    storage_options = _storage_options_for_base(
        s3_base,
        aws_key_env=aws_key_env,
        aws_secret_env=aws_secret_env,
        require_creds=False,
    )
    if s3_base.startswith("s3://") and not storage_options:
        return None

    try:
        fs, path = fsspec.core.url_to_fs(s3_base, **storage_options)
        base = path.rstrip("/") + "/input_data"
        try:
            matches = fs.glob(f"{base}/**/*.parquet")
        except Exception:
            matches = [p for p in fs.find(base) if p.endswith(".parquet")]
        return len(matches)
    except Exception:
        return None


def write_ddf_and_yaml_to_s3(
    *,
    ddf: dd.DataFrame,
    s3_base: str,  # e.g., "s3://bucket/prefix/"
    config_yaml_text: str,
    meta_json_text: str,  # this is already JSON text
    bundle_config_yaml_text: str | None = None,
    suggestions_yaml_text: str | None = None,  # Data Doctor suggestions
    partition_on: Iterable[str] | None = None,
    aws_key_env: str = "AWS_ACCESS_KEY_ID",
    aws_secret_env: str = "AWS_SECRET_ACCESS_KEY",
    force_npartitions: int | None = None,  # if set, force exact nparts (min 2)
    target_mb_per_part: int = 512,  # DEFAULT = 512MB (tune for downstream IO)
    write_index: bool = False,
    shuffle_before_partition_on: bool = True,  # default True to reduce file explosion
    # Performance toggles (configurable via OutputConfig)
    persist_before_write: bool = True,
    rebalance_before_write: bool = True,
    # Dtype casting toggles
    cast_bool_to_int8: bool = True,
    cast_category_to_string: bool = True,
    cast_object_to_string: bool = True,
) -> str:
    """
    Write Parquet under <s3_base>/input_data/ and config/metadata files at <s3_base>.

    Notes:
      - repartition(partition_size=...) triggers computation to determine partition sizes and can be expensive.  (https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.repartition.html)
      - partition_on can create >npartitions parquet files because each dask partition may write multiple files per key.  (https://docs.dask.org/en/stable/_modules/dask/dataframe/dask_expr/io/parquet.html)
      - shuffle(on=partition_on) is the standard way to reduce per-key file explosion (at cost of a shuffle).
    """

    s3_base = s3_base.rstrip("/") + "/"
    storage_options = _storage_options_for_base(
        s3_base,
        aws_key_env=aws_key_env,
        aws_secret_env=aws_secret_env,
        require_creds=s3_base.startswith("s3://"),
    )

    logger.info(f"[init] s3_base={s3_base}")
    if s3_base.startswith("s3://"):
        logger.info(f"[init] using AWS creds from {aws_key_env}/{aws_secret_env}")

    # Get client once (or None)
    try:
        client = get_client()
    except ValueError:
        client = None

    ddf_to_write = ddf

    # Prefer pyarrow-backed strings for cross-environment pickle stability.
    string_dtype = "string"
    try:
        string_dtype = "string[pyarrow]"
    except Exception:
        string_dtype = "string"

    # 1) BOOL -> int8 (for parquet / downstream systems that dislike bool)
    if cast_bool_to_int8:
        bool_cols = []
        for c, dt in ddf_to_write.dtypes.items():
            dt_l = str(dt).lower()
            # catches numpy bool, pandas BooleanDtype, and many backends
            if dt_l in {"bool", "boolean"}:
                bool_cols.append(c)

        if bool_cols:
            logger.info(f"[cast] converting boolean cols to int8: {bool_cols}")
            ddf_to_write = ddf_to_write.astype({c: "int8" for c in bool_cols})

    # 2) category -> string
    if cast_category_to_string:
        cat_cols = [
            c for c, dt in ddf_to_write.dtypes.items() if "category" in str(dt).lower()
        ]
        if cat_cols:
            logger.info(f"[cast] converting categoricals to {string_dtype}: {cat_cols}")
            ddf_to_write = ddf_to_write.astype({c: string_dtype for c in cat_cols})

    # 3) object -> string
    if cast_object_to_string:
        obj_cols = [
            c for c, dt in ddf_to_write.dtypes.items() if "object" in str(dt).lower()
        ]
        if obj_cols:
            logger.info(f"[cast] converting object cols to {string_dtype}: {obj_cols}")
            ddf_to_write = ddf_to_write.astype({c: string_dtype for c in obj_cols})

    current_parts = ddf_to_write.npartitions
    logger.info(f"[stats] current nparts={current_parts}")

    if target_mb_per_part <= 0:
        raise ValueError("target_mb_per_part must be positive")

    min_npartitions = 2

    # --- Decide partitioning policy (before shuffle) ---
    if force_npartitions is not None and force_npartitions > 0:
        desired = max(force_npartitions, min_npartitions)
        if desired == current_parts:
            logger.info(
                f"[repartition] force_npartitions={desired} matches current nparts; keeping existing partitions."
            )
        else:
            direction = "UP" if desired > current_parts else "DOWN"
            logger.info(
                f"[repartition] forcing nparts {direction} from {current_parts} → {desired}"
            )
            if desired > current_parts and current_parts == 1:
                ddf_to_write = ddf_to_write.clear_divisions()
            ddf_to_write = ddf_to_write.repartition(npartitions=desired)
            current_parts = desired
    else:
        part_size = f"{int(target_mb_per_part)}MB"
        logger.info(f"[repartition] target partition_size={part_size}")

        # repartition(partition_size=...) triggers a size measurement pass and can be expensive.  (https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.repartition.html)
        # Guard: on a distributed client with very low nparts, this is where people often OOM if one partition is huge.
        if client is not None and ddf_to_write.npartitions <= 4:
            logger.info(
                "[repartition] distributed cluster + low nparts; skipping repartition(partition_size=...) "
                "to avoid single-worker OOM. Fix partitioning at read time or use force_npartitions."
            )
        else:
            logger.info(
                f"[repartition] measuring partition sizes and repartitioning to {part_size} (this may take time)..."
            )
            t_repartition = time.time()
            ddf_to_write = ddf_to_write.repartition(partition_size=part_size)
            logger.info(
                f"[repartition] ✅ complete in {time.time() - t_repartition:.2f}s"
            )

        current_parts = ddf_to_write.npartitions
        if current_parts < min_npartitions:
            logger.info(
                f"[repartition] only {current_parts} partition(s); enforcing {min_npartitions}"
            )
            ddf_to_write = ddf_to_write.clear_divisions().repartition(
                npartitions=min_npartitions
            )
            current_parts = min_npartitions

    # --- Shuffle for partition_on to avoid file explosion ---
    part_cols = list(partition_on or [])
    parquet_path = f"{s3_base}input_data/"

    if part_cols:
        logger.info(
            f"[write] partition_on={part_cols} "
            "(note: can create >npartitions files because each dask partition can write multiple files per key)."
        )  #  (https://docs.dask.org/en/stable/_modules/dask/dataframe/dask_expr/io/parquet.html)

        if shuffle_before_partition_on:
            logger.info(
                "[shuffle] shuffling on partition_on columns to reduce per-key file explosion (this is a real shuffle, may take time)..."
            )
            t_shuffle = time.time()
            ddf_to_write = ddf_to_write.shuffle(on=part_cols)
            logger.info(f"[shuffle] ✅ complete in {time.time() - t_shuffle:.2f}s")

            # After shuffle, sizes may change; if you're in local mode or already have many partitions,
            # it can be helpful to re-apply partition_size gently. Keep guarded to avoid OOM.
            if client is None or ddf_to_write.npartitions > 8:
                part_size = f"{int(target_mb_per_part)}MB"
                logger.info(
                    f"[shuffle] post-shuffle repartition to {part_size} (measuring sizes)..."
                )
                t_post_shuffle = time.time()
                ddf_to_write = ddf_to_write.repartition(partition_size=part_size)
                logger.info(
                    f"[shuffle] ✅ post-shuffle repartition complete in {time.time() - t_post_shuffle:.2f}s"
                )

    # --- Persist + rebalance ONCE, after shuffle, right before writing (cluster-only) ---
    if client is not None:
        if persist_before_write:
            logger.warning(
                "[persist] persist_before_write=True materializes the full dataset; disable if memory pressure or cluster instability occurs."
            )
            logger.info(
                "[persist] materializing dataframe to cluster memory (all lazy operations execute now - this may take several minutes)..."
            )
            t_persist = time.time()
            ddf_to_write = client.persist(ddf_to_write)
            # Wait for persist to actually complete (persist() is async)
            wait(ddf_to_write)
            logger.info(
                f"[persist] ✅ dataframe materialized in {time.time() - t_persist:.2f}s"
            )

            if rebalance_before_write:
                logger.info("[rebalance] redistributing partitions across workers...")
                t_rebalance = time.time()
                try:
                    client.rebalance(ddf_to_write)
                    logger.info(
                        f"[rebalance] ✅ complete in {time.time() - t_rebalance:.2f}s"
                    )
                except Exception as e:
                    logger.warning(f"[rebalance] failed: {e}")
        else:
            logger.info(
                "[write] persist_before_write=False; skipping persist/rebalance"
            )

    logger.info(f"[s3] preparing Parquet write to {parquet_path} (overwrite=True)...")

    logger.info(f"[debug] nparts before to_parquet = {ddf_to_write.npartitions}")
    logger.info(f"[debug] partition_on = {part_cols}")

    write_delayed = ddf_to_write.to_parquet(
        parquet_path,
        engine="pyarrow",
        write_index=write_index,
        partition_on=part_cols,
        overwrite=True,
        storage_options=storage_options,
        compute=False,
    )

    delayed_tasks = (
        list(write_delayed)
        if isinstance(write_delayed, (list, tuple))
        else [write_delayed]
    )

    if client is None:
        logger.info("[s3] no global Client; using local scheduler to write parquet...")
        t0 = time.time()
        dask.compute(*delayed_tasks)
        logger.info(f"[s3] Parquet write complete in {time.time() - t0:.2f}s.")
    else:
        futures = client.compute(delayed_tasks)
        n_parts = len(futures)
        logger.info(f"[s3] starting Parquet write for ~{n_parts} task(s)...")
        t0 = time.time()
        for i, fut in enumerate(as_completed(futures), start=1):
            fut.result()
            logger.info(f"[s3] finished task {i}/{n_parts}")
        logger.info(f"[s3] Parquet write complete in {time.time() - t0:.2f}s.")

    extra_files = ["config.yaml"]
    if bundle_config_yaml_text is not None:
        extra_files.append("bundleconfig.yaml")
    extra_files.append("data_dictionary.json")
    if suggestions_yaml_text is not None:
        extra_files.append("suggestions.yaml")
    logger.info(f"[s3] writing {', '.join(extra_files)} ...")
    t1 = time.time()

    for fname, content in (
        ("config.yaml", config_yaml_text),
        ("bundleconfig.yaml", bundle_config_yaml_text),
        ("data_dictionary.json", meta_json_text),
        ("suggestions.yaml", suggestions_yaml_text),
    ):
        if content is None:
            continue
        text = (
            content
            if isinstance(content, str)
            else yaml.safe_dump(content, sort_keys=False, allow_unicode=False)
        )
        with fsspec.open(f"{s3_base}{fname}", "w", **storage_options) as fh:
            fh.write(text)

    logger.info(
        f"[done] wrote dataset and metadata to {s3_base} in {time.time() - t1:.2f}s"
    )
    return s3_base


def write_outputs(ddf, cfg, meta_json_text: str, config_text: str, partition_on=None):
    """Compatibility wrapper to match older signature."""
    return write_ddf_and_yaml_to_s3(
        ddf=ddf,
        s3_base=cfg.output.s3_base or "/tmp",
        config_yaml_text=config_text,
        meta_json_text=meta_json_text,
        partition_on=partition_on or cfg.output.partitions,
        target_mb_per_part=cfg.output.target_mb_per_part,
        force_npartitions=cfg.output.force_npartitions,
        write_index=cfg.output.write_index,
        # Performance toggles from config
        shuffle_before_partition_on=getattr(
            cfg.output, "shuffle_before_partition_on", True
        ),
        persist_before_write=getattr(cfg.output, "persist_before_write", True),
        rebalance_before_write=getattr(cfg.output, "rebalance_before_write", True),
        # Dtype casting toggles from config
        cast_bool_to_int8=getattr(cfg.output, "cast_bool_to_int8", True),
        cast_category_to_string=getattr(cfg.output, "cast_category_to_string", True),
        cast_object_to_string=getattr(cfg.output, "cast_object_to_string", True),
    )


def _assert_volume_absent(cfg) -> None:
    """
    Legacy hook: in the notebook this validated UC volumes. Here it is a no-op
    to allow tests/monkeypatching without bundle_impl.
    """
    return None


__all__ = [
    "count_parquet_files",
    "write_ddf_and_yaml_to_s3",
    "write_outputs",
    "create_managed_uc_volume_via_sql",
    "tag_uc_volume_via_sql",
    "_assert_volume_absent",
]
