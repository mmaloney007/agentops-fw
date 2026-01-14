#!/usr/bin/env python3
"""
ingest.py (Dask)
----------------
Lazy loaders for CSV/Parquet/UC Databricks tables with logical-column remapping.

Purpose:
    - Load Dask DataFrames lazily (CSV/Parquet/UC) with optional column projection, filters, dtype overrides, and row limits.
    - Resolve UC storage_location, apply delta-rs physical->logical renames or DBSQL schema fallback.

Usage:
    from neuralift_c360_prep.ingest import load_lazy_dask, load_ddf
    ddf = load_lazy_dask(fmt="parquet", uri="s3://bucket/path", id_cols=["id"])
    ddf = load_ddf(cfg)  # BundleConfig-driven wrapper

Dependencies:
    - dask[dataframe]
    - pandas
    - fsspec
    - databricks-sdk (for UC table metadata)
    - databricks-sql-connector (schema fallback)
    - deltalake (delta-rs; optional)
    - uuid6

Author: Mike Maloney - Neuralift, Inc.
Updated: 2025-12-08
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Mapping, Sequence

import dask
import dask.dataframe as dd
import fsspec
import pandas as pd

try:
    from databricks.sdk import WorkspaceClient
except Exception:  # pragma: no cover
    WorkspaceClient = None  # type: ignore

try:
    from deltalake import DeltaTable
except Exception:  # pragma: no cover
    DeltaTable = None  # type: ignore

_HAVE_DELTA = hasattr(dd, "read_delta")
_PHYSICAL_NAME_RE = re.compile(
    r"^col[-_][0-9a-f]{8}([-_][0-9a-f]{4}){3}[-_][0-9a-f]{12}$",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


def _looks_like_physical_name(name: str) -> bool:
    return bool(_PHYSICAL_NAME_RE.match(name))


def _schema_to_dict(schema: Any) -> dict[str, Any] | None:
    if hasattr(schema, "to_dict"):
        try:
            return schema.to_dict()
        except Exception:
            pass
    if hasattr(schema, "json"):
        try:
            return json.loads(schema.json())
        except Exception:
            pass
    if hasattr(schema, "to_json"):
        try:
            return json.loads(schema.to_json())
        except Exception:
            pass
    return None


def _apply_positional_rename(
    ddf: dd.DataFrame,
    logical_cols: Sequence[str],
    label: str,
    debug_rename_map: bool,
) -> tuple[dd.DataFrame, bool]:
    rename_map: dict[str, str] = {}
    for idx, col in enumerate(ddf.columns):
        target = logical_cols[idx]
        if col == target:
            continue
        if not _looks_like_physical_name(col):
            continue
        rename_map[col] = target

    if not rename_map:
        return ddf, False

    candidate = list(ddf.columns)
    for idx, col in enumerate(ddf.columns):
        if col in rename_map:
            candidate[idx] = rename_map[col]

    if len(set(candidate)) != len(candidate):
        logger.warning(
            "[rename] %s positional rename would create duplicate columns; skipping",
            label,
        )
        return ddf, False

    _log_rename_map(rename_map, debug_rename_map, label)
    ddf = ddf.rename(columns=rename_map)
    logger.info("[rename] %s positional rename for %s columns", label, len(rename_map))
    return ddf, True


def _log_rename_map(
    rename_map: Mapping[str, str],
    debug_rename_map: bool,
    label: str,
) -> None:
    if not debug_rename_map or not rename_map:
        return
    lines = "\n".join(f"  {src} -> {dst}" for src, dst in rename_map.items())
    logger.debug("[rename] %s mapping (%s):\n%s", label, len(rename_map), lines)


def _sanitize_table_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def _format_debug_head_table(pdf: pd.DataFrame, rows: int) -> tuple[str, int]:
    head_pdf = pdf.head(rows)
    row_count = len(head_pdf)
    if row_count == 0:
        return "", 0
    headers = ["column"] + [f"v{i + 1}" for i in range(row_count)]
    table_rows = []
    for col in head_pdf.columns:
        values = [
            _sanitize_table_cell(head_pdf.iloc[idx][col]) for idx in range(row_count)
        ]
        table_rows.append([str(col)] + values)
    widths = [len(h) for h in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _format_row(row: list[str]) -> str:
        padded = [cell.ljust(widths[idx]) for idx, cell in enumerate(row)]
        return "| " + " | ".join(padded) + " |"

    lines = [_format_row(headers), _format_row(["-" * width for width in widths])]
    for row in table_rows:
        lines.append(_format_row(row))
    return "\n".join(lines), row_count


def _log_debug_head(
    ddf: dd.DataFrame,
    rows: int,
    head_pdf: pd.DataFrame | None = None,
) -> None:
    if rows <= 0:
        return
    try:
        pdf = (
            head_pdf.head(rows)
            if head_pdf is not None
            else ddf.head(rows, compute=True)
        )
    except Exception as exc:
        logger.debug("[debug] head(%s) failed: %s", rows, exc)
        return
    if pdf.empty:
        logger.debug("[debug] head(%s) empty; columns=%s", rows, list(ddf.columns))
        return
    table, row_count = _format_debug_head_table(pdf, rows)
    if not table:
        logger.debug("[debug] head(%s) empty; columns=%s", rows, list(ddf.columns))
        return
    logger.debug(
        "[debug] head(%s) with %s cols:\n%s", row_count, len(ddf.columns), table
    )


def _drop_extra_physical_columns(
    ddf: dd.DataFrame,
    logical_cols: Sequence[str] | None,
    label: str,
) -> tuple[dd.DataFrame, bool]:
    if not logical_cols:
        return ddf, False
    physical_cols = [c for c in ddf.columns if _looks_like_physical_name(c)]
    if not physical_cols:
        return ddf, False
    if len(logical_cols) != len(ddf.columns) - len(physical_cols):
        return ddf, False

    non_physical = [c for c in ddf.columns if c not in physical_cols]
    if not set(non_physical).issubset(set(logical_cols)):
        return ddf, False

    ddf = ddf.drop(columns=physical_cols)
    logger.info(
        "[rename] dropped %s extra physical columns based on %s logical list",
        len(physical_cols),
        label,
    )
    return ddf, True


def _quote_fqdn(fqdn: str) -> str:
    parts = fqdn.strip("`").split(".")
    if len(parts) != 3:
        raise ValueError(f"Expected catalog.schema.table, got: {fqdn}")
    return ".".join(f"`{p}`" for p in parts)


def _make_workspace_client() -> "WorkspaceClient":
    """
    Create a WorkspaceClient that works both on-cluster and off-cluster.
    """
    if WorkspaceClient is None:
        raise RuntimeError("databricks-sdk not installed")

    host = os.getenv("DATABRICKS_HOST")
    client_id = os.getenv("DATABRICKS_CLIENT_ID")
    client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")

    if not host or not client_id or not client_secret:
        raise RuntimeError(
            "Databricks OAuth requires DATABRICKS_HOST, DATABRICKS_CLIENT_ID, and "
            "DATABRICKS_CLIENT_SECRET."
        )

    return WorkspaceClient(host=host, client_id=client_id, client_secret=client_secret)


def _lookup_storage_location(table_fqdn: str) -> str:
    """
    Look up UC table storage_location using databricks-sdk.
    Works both on-cluster and off-cluster.
    """
    w = _make_workspace_client()

    try:
        t = w.tables.get(full_name=table_fqdn)
    except TypeError:
        t = w.tables.get(table_fqdn)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch table metadata for {table_fqdn}") from exc

    loc = getattr(t, "storage_location", None)
    if not loc:
        raise RuntimeError(
            f"Table {table_fqdn} has no storage_location (not external?)"
        )
    return loc


def _split_fqdn(table_fqdn: str) -> tuple[str, str, str]:
    clean = table_fqdn.strip("`")
    parts = clean.split(".")
    if len(parts) != 3:
        raise ValueError(f"Expected catalog.schema.table, got: {table_fqdn}")
    return parts[0], parts[1], parts[2]


def _schema_string_to_mapping(schema_string: str) -> dict[str, str]:
    try:
        schema = json.loads(schema_string)
    except json.JSONDecodeError:
        return {}
    mapping: dict[str, str] = {}
    for f in schema.get("fields", []):
        log = f.get("name")
        phys = f.get("metadata", {}).get("delta.columnMapping.physicalName")
        if phys and log:
            mapping[phys] = log
    return mapping


def _read_delta_log_mapping(
    loc: str,
    storage_options: Mapping[str, Any] | None,
) -> dict[str, str]:
    log_uri = loc.rstrip("/") + "/_delta_log"
    try:
        fs, base = fsspec.core.url_to_fs(log_uri, **(storage_options or {}))
        try:
            candidates = fs.glob(f"{base}/*.json")
        except Exception:
            candidates = [p for p in fs.find(base) if p.endswith(".json")]
    except Exception as exc:
        logger.warning("[delta-log] failed to list %s: %s", log_uri, exc)
        return {}

    versioned: list[tuple[int, str]] = []
    for path in candidates:
        name = os.path.basename(path)
        if re.match(r"^\d{20}\.json$", name):
            versioned.append((int(name[:20]), path))

    if not versioned:
        logger.info("[delta-log] no json log files found at %s", log_uri)
        return {}

    versioned.sort(reverse=True)
    for version, path in versioned:
        try:
            with fs.open(path, "rt") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        action = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    meta = action.get("metaData")
                    if meta and "schemaString" in meta:
                        mapping = _schema_string_to_mapping(meta["schemaString"])
                        if mapping:
                            logger.info(
                                "[delta-log] extracted mapping from version %s (%s cols)",
                                version,
                                len(mapping),
                            )
                            return mapping
                        logger.info(
                            "[delta-log] metadata in version %s but no mapping entries",
                            version,
                        )
                        return {}
        except Exception as exc:
            logger.warning("[delta-log] failed reading %s: %s", path, exc)
            continue

    logger.info("[delta-log] no metadata found in delta log")
    return {}


def _phys_to_log_mapping(
    loc: str,
    storage_options: Mapping[str, Any] | None,
) -> dict[str, str]:
    """
    For Delta tables with column mapping, build physical->logical mapping.
    """
    mapping: dict[str, str] = {}
    if DeltaTable is None:
        logger.info("[delta-rs] deltalake not installed; skipping mapping")
        return _read_delta_log_mapping(loc, storage_options)

    try:
        if "RUST_LOG" not in os.environ:
            # Silence noisy delta_kernel warnings unless the user opts in.
            os.environ["RUST_LOG"] = "delta_kernel=error"

        if storage_options:
            try:
                dt = DeltaTable(loc, storage_options=storage_options)
            except TypeError:
                dt = DeltaTable(loc)
        else:
            dt = DeltaTable(loc)
        schema_obj = dt.schema()
        schema = _schema_to_dict(schema_obj)
        if not schema:
            raise RuntimeError("delta schema did not expose a dict/json representation")
        for f in schema.get("fields", []):
            log = f.get("name")
            phys = f.get("metadata", {}).get("delta.columnMapping.physicalName")
            if phys and log:
                mapping[phys] = log
    except Exception as exc:
        logger.warning("[delta-rs] mapping failed: %s", exc)
        mapping = {}

    if mapping:
        logger.info("[delta-rs] extracted mapping for %s columns", len(mapping))
        return mapping

    return _read_delta_log_mapping(loc, storage_options)


def _logical_cols_uc(table_fqdn: str) -> Sequence[str]:
    """
    Get the logical column names from UC table metadata.
    """
    if WorkspaceClient is None:
        return []

    try:
        cat, sch, tbl = _split_fqdn(table_fqdn)
        w = _make_workspace_client()
        cols = list(w.columns.list(catalog_name=cat, schema_name=sch, table_name=tbl))
        return [c.name for c in cols]
    except Exception:
        return []


def _read_via_dbsql_schema(
    table_fqdn: str,
    conn_params: Mapping[str, Any],
    columns: Sequence[str] | None,
) -> Sequence[str]:
    """
    Schema-only DBSQL call: get logical column names without reading data.
    """
    try:
        from databricks import sql  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"databricks-sql-connector not installed: {exc}") from exc

    qf = _quote_fqdn(table_fqdn)
    sel = ", ".join([f"`{c}`" for c in columns]) if columns else "*"
    query = f"SELECT {sel} FROM {qf} LIMIT 0"

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

    with sql.connect(**connect_kwargs) as conn, conn.cursor() as cur:
        cur.execute(query)
        if not cur.description:
            return []
        colnames = [d[0] for d in cur.description]
        return colnames


def load_lazy_dask(
    *,
    fmt: str,
    uri: str,
    id_cols: Sequence[str],
    storage_options: Mapping[str, Any] | None = None,
    dtype_overrides: Mapping[str, Any] | None = None,
    assume_missing: bool = False,
    row_limit: int | None = None,
    columns: Sequence[str] | None = None,
    prefer_delta: bool = True,
    read_blocksize_mb: int = 128,  # Reduced from 512 for better parallelism
    conn_params: Mapping[str, Any] | None = None,
    allow_dbsql_fallback: bool = True,
    require_logical_names: bool = True,
    debug_rename_map: bool = False,
    debug_head_rows: int = 0,
    max_expected_columns: int = 512,
    filters: Any | None = None,
    # --- performance options ---
    use_pyarrow_strings: bool = True,  # Use PyArrow-backed strings for ~50% memory reduction
) -> dd.DataFrame:
    """
    Lazy Dask loader for CSV / Parquet / UC Databricks tables.
    """
    storage_options = dict(storage_options or {})
    fmt_lc = fmt.lower()
    if read_blocksize_mb <= 0:
        raise ValueError("read_blocksize_mb must be positive")
    blocksize = f"{int(read_blocksize_mb)}MB"

    # OPTIMIZATION: Enable PyArrow-backed strings for ~50% memory reduction
    if use_pyarrow_strings:
        try:
            import pyarrow  # noqa: F401

            dask.config.set({"dataframe.convert-string": True})
            logger.info("[perf] PyArrow-backed strings enabled for memory optimization")
        except ImportError:
            logger.warning("[perf] PyArrow not available; using default string dtype")

    # Read from ORIGINAL source (UC / Parquet / CSV)
    logger.info("[init] fmt=%s, uri=%s", fmt_lc, uri)

    if fmt_lc == "csv":
        if columns is not None:
            logger.warning(
                "[warn] columns pruning not supported for CSV at read; will load all columns"
            )
        ddf = dd.read_csv(
            uri,
            storage_options=storage_options,
            blocksize=blocksize,
            dtype=dtype_overrides or None,
            assume_missing=assume_missing,
        )

    elif fmt_lc in {"parquet", "pq"}:
        ddf = dd.read_parquet(
            uri,
            storage_options=storage_options,
            columns=columns,
            filters=filters,
            blocksize=blocksize,
        )

    elif fmt_lc in {"delta", "delta_path"}:
        loc = uri
        mapping = _phys_to_log_mapping(loc, storage_options=storage_options)
        read_kwargs = {
            "storage_options": storage_options,
            "columns": columns,
            "filters": filters,
        }
        parquet_kwargs = {
            **read_kwargs,
            "blocksize": blocksize,
            "split_row_groups": "adaptive",
            "aggregate_files": True,
        }

        if prefer_delta and _HAVE_DELTA:
            try:
                ddf = dd.read_delta(loc, **read_kwargs)
                logger.info("[delta-rs] read_delta succeeded")
            except Exception as exc:
                logger.warning(
                    "[delta-rs] read_delta failed (%s); falling back to parquet", exc
                )
                ddf = dd.read_parquet(loc, **parquet_kwargs)
        else:
            ddf = dd.read_parquet(loc, **parquet_kwargs)
            logger.info("[delta-rs] delta not preferred/available; using parquet")

        renamed = False
        if mapping:
            rename_map = {c: mapping[c] for c in ddf.columns if c in mapping}
            if rename_map:
                _log_rename_map(rename_map, debug_rename_map, "delta metadata")
                ddf = ddf.rename(columns=rename_map)
                remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                logger.info(
                    "[rename] applied mapping for %s columns (delta metadata)",
                    len(rename_map),
                )
                if remaining:
                    sample = ", ".join(remaining[:5])
                    logger.warning(
                        "[rename] %s physical cols remain after delta mapping (sample: %s)",
                        len(remaining),
                        sample,
                    )
                else:
                    renamed = True
            else:
                logger.warning(
                    "[rename] delta mapping had %s entries; none matched Dask columns",
                    len(mapping),
                )

        physical_cols = [c for c in ddf.columns if _looks_like_physical_name(c)]
        if not renamed and physical_cols:
            logger.warning(
                "[warn] no logical-column mapping available; columns remain physical names"
            )
        if require_logical_names and physical_cols:
            sample = ", ".join(physical_cols[:5])
            raise RuntimeError(
                "Logical column mapping failed; physical column names remain. "
                f"Sample: {sample}. Ensure delta log access or enable delta reads."
            )

    elif fmt_lc in {"databricks_table", "table"}:
        loc = _lookup_storage_location(uri)
        logger.info("[storage_location] %s", loc)

        mapping = _phys_to_log_mapping(loc, storage_options=storage_options)
        uc_logical_cols = _logical_cols_uc(uri)
        if uc_logical_cols:
            logger.info(
                "[uc] got %s logical cols from UC; first 5: %s",
                len(uc_logical_cols),
                uc_logical_cols[:5],
            )

        read_kwargs = {
            "storage_options": storage_options,
            "columns": columns,
            "filters": filters,
        }
        parquet_kwargs = {
            **read_kwargs,
            "blocksize": blocksize,
            "split_row_groups": "adaptive",
            "aggregate_files": True,
        }

        if prefer_delta and _HAVE_DELTA:
            try:
                ddf = dd.read_delta(loc, **read_kwargs)
                logger.info("[delta-rs] read_delta succeeded")
            except Exception as exc:
                logger.warning(
                    "[delta-rs] read_delta failed (%s); falling back to parquet", exc
                )
                ddf = dd.read_parquet(loc, **parquet_kwargs)
        else:
            ddf = dd.read_parquet(loc, **parquet_kwargs)
            logger.info("[delta-rs] delta not preferred/available; using parquet")

        # Column renaming: physical -> logical
        renamed = False

        if mapping:
            rename_map = {c: mapping[c] for c in ddf.columns if c in mapping}
            if rename_map:
                _log_rename_map(rename_map, debug_rename_map, "delta metadata")
                ddf = ddf.rename(columns=rename_map)
                remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                logger.info(
                    "[rename] applied mapping for %s columns (delta metadata)",
                    len(rename_map),
                )
                if remaining:
                    sample = ", ".join(remaining[:5])
                    logger.warning(
                        "[rename] %s physical cols remain after delta mapping "
                        "(sample: %s); attempting logical list fallback",
                        len(remaining),
                        sample,
                    )
                else:
                    renamed = True
            else:
                logger.warning(
                    "[rename] delta mapping had %s entries; none matched Dask columns",
                    len(mapping),
                )

        if not renamed and uc_logical_cols and len(uc_logical_cols) == len(ddf.columns):
            ddf, applied = _apply_positional_rename(
                ddf, uc_logical_cols, "UC logical list", debug_rename_map
            )
            if applied:
                remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                if not remaining:
                    renamed = True

        if not renamed and uc_logical_cols and len(uc_logical_cols) != len(ddf.columns):
            logger.warning(
                "[rename] UC logical cols=%s != Dask cols=%s; skipping positional rename",
                len(uc_logical_cols),
                len(ddf.columns),
            )

        dbsql_logical_cols: Sequence[str] | None = None
        if not renamed and allow_dbsql_fallback and conn_params is not None:
            logger.info(
                "[fallback] using DBSQL to fetch logical column names (schema only)"
            )
            try:
                logical_names = _read_via_dbsql_schema(uri, conn_params, columns=None)
            except Exception as exc:
                logger.warning("[fallback] DBSQL schema fetch failed: %s", exc)
                logical_names = None

            if logical_names and len(logical_names) == len(ddf.columns):
                dbsql_logical_cols = logical_names
                ddf, applied = _apply_positional_rename(
                    ddf, logical_names, "DBSQL logical list", debug_rename_map
                )
                if applied:
                    remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                    if not remaining:
                        renamed = True
                        logger.info("[fallback] DBSQL schema-based rename succeeded")
            elif logical_names:
                dbsql_logical_cols = logical_names
                logger.warning(
                    "[fallback] DBSQL cols=%s != Dask cols=%s; skipping rename",
                    len(logical_names),
                    len(ddf.columns),
                )

        if not renamed:
            ddf, dropped = _drop_extra_physical_columns(ddf, uc_logical_cols, "UC")
            if dropped:
                remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                if not remaining:
                    renamed = True

        if not renamed:
            ddf, dropped = _drop_extra_physical_columns(
                ddf, dbsql_logical_cols, "DBSQL"
            )
            if dropped:
                remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                if not remaining:
                    renamed = True

        physical_cols = [c for c in ddf.columns if _looks_like_physical_name(c)]
        if not renamed and physical_cols:
            logger.warning(
                "[warn] no logical-column mapping available; columns remain physical names"
            )
        if require_logical_names and physical_cols:
            sample = ", ".join(physical_cols[:5])
            raise RuntimeError(
                "Logical column mapping failed; physical column names remain. "
                f"Sample: {sample}. Ensure delta log access or enable delta reads."
            )

    else:
        raise ValueError("fmt must be csv | parquet | delta | databricks_table")

    n_cols = len(ddf.columns)
    if n_cols > max_expected_columns and columns is None:
        logger.warning(
            "[warn] loading %s columns without projection; consider passing columns=... "
            "or increasing max_expected_columns",
            n_cols,
        )
    logger.info(
        "[cols] ddf has %s columns across %s partitions",
        n_cols,
        ddf.npartitions,
    )

    head_pdf = None

    # row_limit path: for quick local dev
    if row_limit:
        logger.info(
            "[row_limit] materializing first %s rows into in-memory Dask df", row_limit
        )
        head_pdf = ddf.head(row_limit, compute=True)
        ddf = dd.from_pandas(head_pdf, npartitions=max(1, min(4, row_limit)))

    if debug_head_rows > 0:
        _log_debug_head(ddf, debug_head_rows, head_pdf=head_pdf)

    return ddf


def load_ddf(cfg):
    """Compatibility wrapper to load a Dask DF from BundleConfig."""
    src = cfg.input.source
    if src == "uc_table":
        fmt, uri = "databricks_table", cfg.input.uc_table
    elif src == "parquet":
        fmt, uri = "parquet", cfg.input.parquet_path
    elif src == "csv":
        fmt, uri = "csv", cfg.input.csv_path
    elif src == "delta_path":
        fmt, uri = "delta", cfg.input.delta_path
    elif src in {"azure_parquet", "azure_csv", "bigquery"}:
        raise NotImplementedError(f"Source '{src}' is not implemented")
    else:
        raise ValueError(f"Unsupported source: {src}")
    return load_lazy_dask(
        fmt=fmt,
        uri=uri,
        id_cols=cfg.input.id_cols,
        columns=cfg.input.columns,
        dtype_overrides=cfg.input.dtype_overrides,
        read_blocksize_mb=getattr(
            cfg.input, "read_blocksize_mb", cfg.output.target_mb_per_part
        ),
        require_logical_names=cfg.input.require_logical_names,
        debug_rename_map=cfg.logging.level == "debug",
        debug_head_rows=cfg.logging.debug_head_rows
        if cfg.logging.level == "debug"
        else 0,
        use_pyarrow_strings=getattr(cfg.input, "use_pyarrow_strings", True),
    )


auto_repartition = None

__all__ = [
    "load_ddf",
    "load_lazy_dask",
    "auto_repartition",
]
