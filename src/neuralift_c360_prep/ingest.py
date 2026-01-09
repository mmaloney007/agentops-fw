#!/usr/bin/env python3
"""
ingest.py (Dask)
----------------
Lazy loaders for CSV/Parquet/UC Databricks tables with optional snapshotting and logical-column remapping.

Purpose:
    - Load Dask DataFrames lazily (CSV/Parquet/UC) with optional column projection, filters, dtype overrides, and row limits.
    - Resolve UC storage_location, apply delta-rs physical->logical renames or DBSQL schema fallback.
    - Build/read/refresh parquet snapshots with optional preprocessing/repartitioning hooks.

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
import os
import re
from typing import Any, Mapping, Sequence

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


def _log(msg: str, enabled: bool) -> None:
    if enabled:
        print(msg)


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
    show_progress: bool,
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
        _log(
            f"[rename] {label} positional rename would create duplicate columns; skipping",
            show_progress,
        )
        return ddf, False

    _log_rename_map(rename_map, show_progress, debug_rename_map, label)
    ddf = ddf.rename(columns=rename_map)
    _log(
        f"[rename] {label} positional rename for {len(rename_map)} columns",
        show_progress,
    )
    return ddf, True


def _log_rename_map(
    rename_map: Mapping[str, str],
    show_progress: bool,
    debug_rename_map: bool,
    label: str,
) -> None:
    if not debug_rename_map or not rename_map:
        return
    lines = "\n".join(f"  {src} -> {dst}" for src, dst in rename_map.items())
    _log(f"[rename] {label} mapping ({len(rename_map)}):\n{lines}", show_progress)


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
    show_progress: bool,
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
        _log(f"[debug] head({rows}) failed: {exc}", show_progress)
        return
    if pdf.empty:
        _log(f"[debug] head({rows}) empty; columns={list(ddf.columns)}", show_progress)
        return
    table, row_count = _format_debug_head_table(pdf, rows)
    if not table:
        _log(f"[debug] head({rows}) empty; columns={list(ddf.columns)}", show_progress)
        return
    _log(
        f"[debug] head({row_count}) with {len(ddf.columns)} cols:\n{table}",
        show_progress,
    )


def _drop_extra_physical_columns(
    ddf: dd.DataFrame,
    logical_cols: Sequence[str] | None,
    show_progress: bool,
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
    _log(
        f"[rename] dropped {len(physical_cols)} extra physical columns "
        f"based on {label} logical list",
        show_progress,
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
    show_progress: bool,
) -> dict[str, str]:
    log_uri = loc.rstrip("/") + "/_delta_log"
    try:
        fs, base = fsspec.core.url_to_fs(log_uri, **(storage_options or {}))
        try:
            candidates = fs.glob(f"{base}/*.json")
        except Exception:
            candidates = [p for p in fs.find(base) if p.endswith(".json")]
    except Exception as exc:
        _log(f"[delta-log] failed to list {log_uri}: {exc}", show_progress)
        return {}

    versioned: list[tuple[int, str]] = []
    for path in candidates:
        name = os.path.basename(path)
        if re.match(r"^\d{20}\.json$", name):
            versioned.append((int(name[:20]), path))

    if not versioned:
        _log(f"[delta-log] no json log files found at {log_uri}", show_progress)
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
                            _log(
                                f"[delta-log] extracted mapping from version {version} "
                                f"({len(mapping)} cols)",
                                show_progress,
                            )
                            return mapping
                        _log(
                            f"[delta-log] metadata in version {version} but no mapping entries",
                            show_progress,
                        )
                        return {}
        except Exception as exc:
            _log(f"[delta-log] failed reading {path}: {exc}", show_progress)
            continue

    _log("[delta-log] no metadata found in delta log", show_progress)
    return {}


def _phys_to_log_mapping(
    loc: str,
    storage_options: Mapping[str, Any] | None,
    show_progress: bool,
) -> dict[str, str]:
    """
    For Delta tables with column mapping, build physical->logical mapping.
    """
    mapping: dict[str, str] = {}
    if DeltaTable is None:
        _log("[delta-rs] deltalake not installed; skipping mapping", show_progress)
        return _read_delta_log_mapping(loc, storage_options, show_progress)

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
        _log(f"[delta-rs] mapping failed: {exc}", show_progress)
        mapping = {}

    if mapping:
        _log(f"[delta-rs] extracted mapping for {len(mapping)} columns", show_progress)
        return mapping

    return _read_delta_log_mapping(loc, storage_options, show_progress)


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


def _snapshot_exists(uri: str, storage_options: Mapping[str, Any] | None) -> bool:
    fs, path = fsspec.core.url_to_fs(uri, **(storage_options or {}))
    return fs.exists(path)


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
    read_blocksize_mb: int = 512,
    show_progress: bool = True,
    conn_params: Mapping[str, Any] | None = None,
    allow_dbsql_fallback: bool = True,
    require_logical_names: bool = True,
    debug_rename_map: bool = False,
    debug_head_rows: int = 0,
    max_expected_columns: int = 512,
    filters: Any | None = None,
    # --- snapshot options ---
    snapshot_uri: str | None = None,
    snapshot_mode: str = "off",  # "off" | "read" | "read_or_build" | "refresh"
    snapshot_storage_options: Mapping[str, Any] | None = None,
    snapshot_repartition_fn: Any | None = None,
    snapshot_preprocess_fn: Any | None = None,
) -> dd.DataFrame:
    """
    Lazy Dask loader for CSV / Parquet / UC Databricks tables with optional snapshotting.
    """
    storage_options = dict(storage_options or {})
    fmt_lc = fmt.lower()
    if read_blocksize_mb <= 0:
        raise ValueError("read_blocksize_mb must be positive")
    blocksize = f"{int(read_blocksize_mb)}MB"

    # Snapshot short-circuit: read existing snapshot if requested
    use_snapshot = snapshot_uri is not None and snapshot_mode in {
        "read",
        "read_or_build",
    }
    if use_snapshot:
        snap_opts = snapshot_storage_options or {}
        if _snapshot_exists(snapshot_uri, snap_opts):
            _log(
                f"[snapshot] using existing snapshot at {snapshot_uri} (mode={snapshot_mode})",
                show_progress,
            )
            return dd.read_parquet(
                snapshot_uri, storage_options=snap_opts, blocksize=blocksize
            )

        if snapshot_mode == "read":
            raise FileNotFoundError(
                f"Snapshot URI {snapshot_uri} not found and snapshot_mode='read'."
            )

    # Read from ORIGINAL source (UC / Parquet / CSV)
    _log(f"[init] fmt={fmt_lc}, uri={uri}", show_progress)

    if fmt_lc == "csv":
        if columns is not None:
            _log(
                "[warn] columns pruning not supported for CSV at read; will load all columns",
                show_progress,
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
        mapping = _phys_to_log_mapping(
            loc, storage_options=storage_options, show_progress=show_progress
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
                _log("[delta-rs] read_delta succeeded", show_progress)
            except Exception as exc:
                _log(
                    f"[delta-rs] read_delta failed ({exc}); falling back to parquet",
                    show_progress,
                )
                ddf = dd.read_parquet(loc, **parquet_kwargs)
        else:
            ddf = dd.read_parquet(loc, **parquet_kwargs)
            _log(
                "[delta-rs] delta not preferred/available; using parquet", show_progress
            )

        renamed = False
        if mapping:
            rename_map = {c: mapping[c] for c in ddf.columns if c in mapping}
            if rename_map:
                _log_rename_map(
                    rename_map, show_progress, debug_rename_map, "delta metadata"
                )
                ddf = ddf.rename(columns=rename_map)
                remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                _log(
                    f"[rename] applied mapping for {len(rename_map)} columns (delta metadata)",
                    show_progress,
                )
                if remaining:
                    sample = ", ".join(remaining[:5])
                    _log(
                        f"[rename] {len(remaining)} physical cols remain after delta mapping "
                        f"(sample: {sample})",
                        show_progress,
                    )
                else:
                    renamed = True
            else:
                _log(
                    f"[rename] delta mapping had {len(mapping)} entries; none matched Dask columns",
                    show_progress,
                )

        physical_cols = [c for c in ddf.columns if _looks_like_physical_name(c)]
        if not renamed and physical_cols:
            _log(
                "[warn] no logical-column mapping available; columns remain physical names",
                show_progress,
            )
        if require_logical_names and physical_cols:
            sample = ", ".join(physical_cols[:5])
            raise RuntimeError(
                "Logical column mapping failed; physical column names remain. "
                f"Sample: {sample}. Ensure delta log access or enable delta reads."
            )

    elif fmt_lc in {"databricks_table", "table"}:
        loc = _lookup_storage_location(uri)
        _log(f"[storage_location] {loc}", show_progress)

        mapping = _phys_to_log_mapping(
            loc, storage_options=storage_options, show_progress=show_progress
        )
        uc_logical_cols = _logical_cols_uc(uri)
        if uc_logical_cols:
            _log(
                f"[uc] got {len(uc_logical_cols)} logical cols from UC; first 5: {uc_logical_cols[:5]}",
                show_progress,
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
                _log("[delta-rs] read_delta succeeded", show_progress)
            except Exception as exc:
                _log(
                    f"[delta-rs] read_delta failed ({exc}); falling back to parquet",
                    show_progress,
                )
                ddf = dd.read_parquet(loc, **parquet_kwargs)
        else:
            ddf = dd.read_parquet(loc, **parquet_kwargs)
            _log(
                "[delta-rs] delta not preferred/available; using parquet", show_progress
            )

        # Column renaming: physical -> logical
        renamed = False

        if mapping:
            rename_map = {c: mapping[c] for c in ddf.columns if c in mapping}
            if rename_map:
                _log_rename_map(
                    rename_map, show_progress, debug_rename_map, "delta metadata"
                )
                ddf = ddf.rename(columns=rename_map)
                remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                _log(
                    f"[rename] applied mapping for {len(rename_map)} columns (delta metadata)",
                    show_progress,
                )
                if remaining:
                    sample = ", ".join(remaining[:5])
                    _log(
                        f"[rename] {len(remaining)} physical cols remain after delta mapping "
                        f"(sample: {sample}); attempting logical list fallback",
                        show_progress,
                    )
                else:
                    renamed = True
            else:
                _log(
                    f"[rename] delta mapping had {len(mapping)} entries; none matched Dask columns",
                    show_progress,
                )

        if not renamed and uc_logical_cols and len(uc_logical_cols) == len(ddf.columns):
            ddf, applied = _apply_positional_rename(
                ddf, uc_logical_cols, show_progress, "UC logical list", debug_rename_map
            )
            if applied:
                remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                if not remaining:
                    renamed = True

        if not renamed and uc_logical_cols and len(uc_logical_cols) != len(ddf.columns):
            _log(
                f"[rename] UC logical cols={len(uc_logical_cols)} "
                f"!= Dask cols={len(ddf.columns)}; skipping positional rename",
                show_progress,
            )

        dbsql_logical_cols: Sequence[str] | None = None
        if not renamed and allow_dbsql_fallback and conn_params is not None:
            _log(
                "[fallback] using DBSQL to fetch logical column names (schema only)",
                show_progress,
            )
            try:
                logical_names = _read_via_dbsql_schema(uri, conn_params, columns=None)
            except Exception as exc:
                _log(f"[fallback] DBSQL schema fetch failed: {exc}", show_progress)
                logical_names = None

            if logical_names and len(logical_names) == len(ddf.columns):
                dbsql_logical_cols = logical_names
                ddf, applied = _apply_positional_rename(
                    ddf,
                    logical_names,
                    show_progress,
                    "DBSQL logical list",
                    debug_rename_map,
                )
                if applied:
                    remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                    if not remaining:
                        renamed = True
                        _log(
                            "[fallback] DBSQL schema-based rename succeeded",
                            show_progress,
                        )
            elif logical_names:
                dbsql_logical_cols = logical_names
                _log(
                    f"[fallback] DBSQL cols={len(logical_names)} "
                    f"!= Dask cols={len(ddf.columns)}; skipping rename",
                    show_progress,
                )

        if not renamed:
            ddf, dropped = _drop_extra_physical_columns(
                ddf, uc_logical_cols, show_progress, "UC"
            )
            if dropped:
                remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                if not remaining:
                    renamed = True

        if not renamed:
            ddf, dropped = _drop_extra_physical_columns(
                ddf, dbsql_logical_cols, show_progress, "DBSQL"
            )
            if dropped:
                remaining = [c for c in ddf.columns if _looks_like_physical_name(c)]
                if not remaining:
                    renamed = True

        physical_cols = [c for c in ddf.columns if _looks_like_physical_name(c)]
        if not renamed and physical_cols:
            _log(
                "[warn] no logical-column mapping available; columns remain physical names",
                show_progress,
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
        _log(
            f"[warn] loading {n_cols} columns without projection; "
            f"consider passing columns=... or increasing max_expected_columns",
            show_progress,
        )
    _log(
        f"[cols] ddf has {n_cols} columns across {ddf.npartitions} partitions",
        show_progress,
    )

    head_pdf = None

    # row_limit path: for quick local dev
    if row_limit:
        _log(
            f"[row_limit] materializing first {row_limit} rows into in-memory Dask df",
            show_progress,
        )
        head_pdf = ddf.head(row_limit, compute=True)
        ddf = dd.from_pandas(head_pdf, npartitions=max(1, min(4, row_limit)))

    if debug_head_rows > 0:
        _log_debug_head(ddf, debug_head_rows, show_progress, head_pdf=head_pdf)

    # Snapshot build/refresh if requested
    build_snapshot = snapshot_uri is not None and snapshot_mode in {
        "read_or_build",
        "refresh",
    }
    if build_snapshot:
        snap_opts = snapshot_storage_options or {}
        _log(
            f"[snapshot] building snapshot at {snapshot_uri} (mode={snapshot_mode})",
            show_progress,
        )

        snapshot_ddf = ddf

        # optional preprocess for snapshot
        if snapshot_preprocess_fn is not None:
            snapshot_ddf = snapshot_preprocess_fn(snapshot_ddf)

        # optional repartition for snapshot
        if snapshot_repartition_fn is not None:
            snapshot_ddf = snapshot_repartition_fn(snapshot_ddf)

        snapshot_ddf.to_parquet(
            snapshot_uri,
            storage_options=snap_opts,
            engine="pyarrow",
            write_index=False,
            overwrite=True,
        )
        _log("[snapshot] parquet snapshot write complete", show_progress)

        # reload snapshot as new lazy Dask DataFrame
        del snapshot_ddf
        del ddf
        ddf = dd.read_parquet(
            snapshot_uri, storage_options=snap_opts, blocksize=blocksize
        )
        _log("[snapshot] reloaded snapshot as new lazy Dask DataFrame", show_progress)

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
        read_blocksize_mb=cfg.output.target_mb_per_part,
        snapshot_mode="off",
        require_logical_names=cfg.input.require_logical_names,
        debug_rename_map=cfg.logging.level == "debug",
        debug_head_rows=cfg.logging.debug_head_rows
        if cfg.logging.level == "debug"
        else 0,
        show_progress=getattr(cfg.logging, "show_progress", True),
    )


auto_repartition = None
snapshot_repartition = None
snapshot_preprocess = None

__all__ = [
    "load_ddf",
    "load_lazy_dask",
    "auto_repartition",
    "snapshot_repartition",
    "snapshot_preprocess",
]
