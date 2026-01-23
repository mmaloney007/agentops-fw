#!/usr/bin/env python3
"""
Generate a BundleConfig from an existing tagged Databricks UC table.

Purpose:
    - Read UC table schema and column tags
    - Extract ID columns, KPI columns, and lift metadata
    - Generate a starter YAML config for the pipeline

Usage:
    python -m neuralift_c360_prep.config_generator staging-c360.media.vivastream_27m
    python -m neuralift_c360_prep.config_generator staging-c360.media.vivastream_27m -o configs/generated.yaml

Expected UC column tags:
    - type: id | kpi | cat | continuous
    - value_sum_column: <column_name>  (for KPIs)
    - value_sum_unit: <unit>           (for KPIs)
    - event_sum_column: <column_name>  (for KPIs)
    - event_sum_unit: <unit>           (for KPIs)

Author: Mike Maloney - Neuralift, Inc.
Copyright 2025 Neuralift, Inc.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _parse_uc_table(uc_table: str) -> tuple[str, str, str]:
    """Parse UC table identifier into (catalog, schema, table)."""
    parts = uc_table.replace("-", "_temp_hyphen_").split(".")
    parts = [p.replace("_temp_hyphen_", "-") for p in parts]

    if len(parts) != 3:
        raise ValueError(
            f"Invalid UC table format: {uc_table}. Expected: catalog.schema.table"
        )

    return parts[0], parts[1], parts[2]


def _get_column_tags_via_sql(uc_table: str) -> dict[str, dict[str, str]]:
    """
    Fetch column tags from Unity Catalog via SQL.

    Returns dict mapping column_name -> {tag_key: tag_value}
    """
    import os

    from databricks import sql as dbsql

    catalog, schema, table = _parse_uc_table(uc_table)

    # Quote identifiers for SQL (backticks for Databricks)
    def quote_ident(name: str) -> str:
        return f"`{name.replace('`', '``')}`"

    # Query information_schema for column tags
    query = f"""
    SELECT
        column_name,
        tag_name,
        tag_value
    FROM {quote_ident(catalog)}.information_schema.column_tags
    WHERE schema_name = '{schema}'
      AND table_name = '{table}'
    """

    host = os.getenv("DATABRICKS_HOST", "").replace("https://", "")
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    access_token = os.getenv("DATABRICKS_TOKEN")

    if not host or not warehouse_id:
        logger.warning(
            "DATABRICKS_HOST or DATABRICKS_WAREHOUSE_ID not set, skipping tag fetch"
        )
        return {}

    column_tags: dict[str, dict[str, str]] = {}

    # Get access token - use PAT if available, else OAuth M2M
    if not access_token:
        client_id = os.getenv("DATABRICKS_CLIENT_ID")
        client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
        if client_id and client_secret:
            from .databricks_oauth import get_databricks_access_token

            try:
                access_token = get_databricks_access_token(
                    host=f"https://{host}",
                    client_id=client_id,
                    client_secret=client_secret,
                )
            except Exception as e:
                logger.warning(f"Failed to get OAuth token: {e}")
                return {}

    if not access_token:
        logger.warning("No access token available, skipping tag fetch")
        return {}

    connect_kwargs: dict[str, Any] = {
        "server_hostname": host,
        "http_path": f"/sql/1.0/warehouses/{warehouse_id}",
        "access_token": access_token,
    }

    try:
        with dbsql.connect(**connect_kwargs) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()

                for row in rows:
                    col_name, tag_name, tag_value = row
                    if col_name not in column_tags:
                        column_tags[col_name] = {}
                    column_tags[col_name][tag_name] = tag_value

        logger.info(f"Fetched tags for {len(column_tags)} columns")
    except Exception as e:
        logger.warning(f"Failed to fetch column tags via SQL: {e}")

    return column_tags


def _get_column_tags(uc_table: str) -> tuple[list[dict[str, Any]], str | None]:
    """
    Fetch column metadata and tags from a Unity Catalog table.

    Returns:
        Tuple of (columns, table_comment) where columns is a list of dicts
        with keys: name, data_type, tags (dict)
    """
    from databricks.sdk import WorkspaceClient

    catalog, schema, table = _parse_uc_table(uc_table)

    client = WorkspaceClient()

    # Get table info with columns
    table_info = client.tables.get(full_name=f"{catalog}.{schema}.{table}")

    # Extract table comment
    table_comment = table_info.comment if table_info.comment else None

    # Fetch column tags via SQL (SDK doesn't expose them directly)
    column_tags = _get_column_tags_via_sql(uc_table)

    columns = []
    for col in table_info.columns or []:
        col_name = col.name
        tags = column_tags.get(col_name, {})

        columns.append(
            {
                "name": col_name,
                "data_type": str(col.type_name) if col.type_name else "unknown",
                "tags": tags,
            }
        )

    return columns, table_comment


def generate_config_from_columns(
    uc_table: str,
    columns: list[dict[str, Any]],
    output_path: str | Path | None = None,
    workspace: str | None = None,
    software_env: str = "neuralift_c360_prep",
    table_comment: str | None = None,
) -> dict[str, Any]:
    """
    Generate a BundleConfig dict from column metadata.

    Args:
        uc_table: Fully qualified UC table name (catalog.schema.table)
        columns: List of column dicts with keys: name, data_type, tags
        output_path: Optional path to write YAML config
        workspace: Coiled workspace name (auto-detected from catalog if None)
        software_env: Coiled software environment name

    Returns:
        Config dict ready for YAML serialization or BundleConfig.model_validate()
    """

    # Parse table identifier for output config
    parts = uc_table.replace("-", "_temp_hyphen_").split(".")
    parts = [p.replace("_temp_hyphen_", "-") for p in parts]
    catalog, schema, table = parts

    # Auto-detect workspace from catalog if not specified
    # staging-c360 -> neuralift-dev, everything else -> neuralift-prod
    is_prod_workspace = False
    if workspace is None:
        if uc_table.startswith("staging-c360"):
            workspace = "neuralift-dev"
        else:
            workspace = "neuralift-prod"
            is_prod_workspace = True
            logger.info(
                f"Catalog '{catalog}' is not staging-c360, using neuralift-prod workspace"
            )

    # Categorize columns by type tag
    id_columns: list[str] = []
    kpi_columns: list[dict[str, Any]] = []
    categorical_columns: list[str] = []
    continuous_columns: list[str] = []

    for col in columns:
        tags = col["tags"]
        col_type = tags.get("type", "").lower()

        if col_type == "id":
            id_columns.append(col["name"])

        elif col_type == "kpi":
            # Build KPI function config with lift metadata
            kpi_config: dict[str, Any] = {
                "type": "identity",
                "column": col["name"],
                "kpi": True,
            }

            # Extract lift metadata if present
            lift: dict[str, str] = {}
            if tags.get("value_sum_column"):
                lift["value_sum_column"] = tags["value_sum_column"]
            if tags.get("value_sum_unit"):
                lift["value_sum_unit"] = tags["value_sum_unit"]
            if tags.get("event_sum_column"):
                lift["event_sum_column"] = tags["event_sum_column"]
            if tags.get("event_sum_unit"):
                lift["event_sum_unit"] = tags["event_sum_unit"]

            if lift:
                kpi_config["lift"] = lift

            kpi_columns.append(kpi_config)

        elif col_type == "cat":
            categorical_columns.append(col["name"])

        elif col_type == "continuous":
            continuous_columns.append(col["name"])

    # Log summary
    logger.info(f"Found {len(id_columns)} ID column(s): {id_columns}")
    logger.info(f"Found {len(kpi_columns)} KPI column(s)")
    logger.info(f"Found {len(categorical_columns)} categorical column(s)")
    logger.info(f"Found {len(continuous_columns)} continuous column(s)")

    # Build config
    config: dict[str, Any] = {
        "# Generated from": uc_table,
        "logging": {
            "level": "info",
            "show_progress": True,
        },
        "runtime": {
            "engine": "coiled",
            "coiled": {
                "workspace": workspace,
                "software_env": software_env,
                "n_workers": 4,
                "worker_vm_types": ["m5.2xlarge"],
                "scheduler_vm_types": ["c5d.xlarge"],
                "idle_timeout": "2h",
            },
        },
        "input": {
            "source": "uc_table",
            "uc_table": uc_table,
        },
        "ids": {
            "columns": id_columns,
            "auto_detect": False,
        },
        "functions": kpi_columns if kpi_columns else [],
        "preprocessing": {
            "rename_to_snake": True,
            "bool_fix": True,
            "drop_empty": True,
            "drop_constant": True,
            "fill": {
                "categorical": "Unknown",
                "continuous": "median",
            },
        },
        "drop_columns": [],
        "metadata": {
            "model": "gpt-5-nano",
            "sample_rows": 5000,
            "max_concurrency": 50,
            "context": table_comment
            if table_comment
            else f"Generated config from {uc_table}",
        },
        "data_doctor": {
            "enabled": False,
        },
        "output": {
            "uc_catalog": catalog,
            "uc_schema": schema,
            "uc_table": f"{table}_prepared",
            "uc_volume_name": f"{table}_prepared",
            "run_name": f"{table}_prepared",
            "partitions": [],
            "target_mb_per_part": 512,
        },
    }

    # Write to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            # Remove the comment key before dumping (it's just for reference)
            comment = config.pop("# Generated from", None)
            if comment:
                f.write(f"# Generated from: {comment}\n")
                f.write(
                    f"# Run: python -m neuralift_c360_prep.config_generator {uc_table}\n"
                )
                if is_prod_workspace:
                    f.write(
                        f"# NOTE: Using neuralift-prod workspace (catalog '{catalog}' is not staging-c360)\n"
                    )
                f.write("\n")
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

        logger.info(f"Config written to {output_path}")

    return config


def generate_config_from_uc_table(
    uc_table: str,
    output_path: str | Path | None = None,
    workspace: str | None = None,
    software_env: str = "neuralift_c360_prep",
) -> dict[str, Any]:
    """
    Generate a BundleConfig dict from an existing tagged UC table.

    Fetches column metadata from Unity Catalog and generates config.

    Args:
        uc_table: Fully qualified UC table name (catalog.schema.table)
        output_path: Optional path to write YAML config
        workspace: Coiled workspace name (auto-detected from catalog if None)
        software_env: Coiled software environment name

    Returns:
        Config dict ready for YAML serialization or BundleConfig.model_validate()
    """
    logger.info(f"Fetching column metadata from {uc_table}")
    columns, table_comment = _get_column_tags(uc_table)
    return generate_config_from_columns(
        uc_table=uc_table,
        columns=columns,
        output_path=output_path,
        workspace=workspace,
        software_env=software_env,
        table_comment=table_comment,
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a pipeline config from an existing tagged UC table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m neuralift_c360_prep.config_generator staging-c360.media.vivastream_27m
    python -m neuralift_c360_prep.config_generator staging-c360.media.vivastream_27m -o configs/generated.yaml
    python -m neuralift_c360_prep.config_generator staging-c360.media.vivastream_27m --workspace neuralift-prod
        """,
    )
    parser.add_argument(
        "uc_table",
        help="Fully qualified UC table name (catalog.schema.table)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for YAML config file",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Coiled workspace (auto-detected: neuralift-dev for staging-c360, neuralift-prod otherwise)",
    )
    parser.add_argument(
        "--software-env",
        default="neuralift_c360_prep",
        help="Coiled software environment (default: neuralift_c360_prep)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        config = generate_config_from_uc_table(
            uc_table=args.uc_table,
            output_path=args.output,
            workspace=args.workspace,
            software_env=args.software_env,
        )

        # If no output file, print to stdout
        if not args.output:
            print("\n# Generated config:\n")
            print(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))

        return 0

    except Exception as e:
        logger.error(f"Failed to generate config: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
