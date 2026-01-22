#!/usr/bin/env python3
"""
Generate labels.npy and precomputed_points.npy for Neuralift Segmenter.

Reads parquet files in deterministic part-file order, extracts segment labels,
and writes labels + 2D points aligned with row order. Optionally updates
config.yaml to include labels_file_name and ranked_points_file_name at the top.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import re
import uuid
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dask import delayed
from dask import compute as dask_compute
from dask.distributed import Client, LocalCluster

from .env import dotenv_env_vars
from .log_utils import configure_dask_logging, setup_logging

logger = logging.getLogger(__name__)


def _storage_options_for_uri(
    uri: str,
    *,
    aws_key_env: str = "AWS_ACCESS_KEY_ID",
    aws_secret_env: str = "AWS_SECRET_ACCESS_KEY",
    require_creds: bool = False,
) -> dict:
    if uri.startswith("s3://"):
        aws_key = os.getenv(aws_key_env)
        aws_secret = os.getenv(aws_secret_env)
        if require_creds and (not aws_key or not aws_secret):
            raise RuntimeError(
                f"Missing AWS credentials in env: {aws_key_env}/{aws_secret_env}"
            )
        if aws_key and aws_secret:
            return {"key": aws_key, "secret": aws_secret}
    return {}


def _format_uri(protocol: str | None, path: str) -> str:
    if protocol in (None, "file"):
        return path
    return f"{protocol}://{path}"


def _part_index(name: str) -> int | None:
    """Extract numeric index from parquet file name, supporting multiple naming conventions."""
    # Try multiple common patterns in order of specificity
    patterns = [
        r"part-(\d+)",  # Dask/Spark default: part-00000.parquet
        r"part\.(\d+)",  # Dask alternate: part.0.parquet
        r"part(\d+)",  # No separator: part00000.parquet
        r"_(\d+)\.parquet$",  # Trailing underscore: data_00000.parquet
        r"-(\d+)\.parquet$",  # Trailing hyphen: data-00000.parquet
        r"(\d+)\.parquet$",  # Just number: 00000.parquet
    ]
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _parquet_sort_key(path: str) -> tuple[int, int, str]:
    name = Path(path).name
    index = _part_index(name)
    if index is None:
        return (1, 0, path)
    return (0, index, path)


def _list_parquet_files(input_uri: str, storage_options: dict) -> list[str]:
    fs, base_path = fsspec.core.url_to_fs(input_uri, **storage_options)
    protocol = fs.protocol[0] if isinstance(fs.protocol, (tuple, list)) else fs.protocol

    if protocol in (None, "file"):
        base_path = str(Path(base_path).resolve())

    try:
        paths = fs.find(base_path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to list parquet files under {input_uri}: {exc}"
        ) from exc

    parquet_paths = [
        path
        for path in paths
        if path.endswith(".parquet")
        and not path.endswith("_metadata")
        and not path.endswith("_common_metadata")
    ]
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {input_uri}")

    # Check if file naming follows a recognized pattern
    sample_files = parquet_paths[: min(5, len(parquet_paths))]
    unrecognized = [p for p in sample_files if _part_index(Path(p).name) is None]
    if unrecognized:
        if len(unrecognized) == len(sample_files):
            # None of the sampled files have recognized naming
            logger.warning(
                "[order] Parquet files don't follow a recognized naming pattern "
                "(e.g., part-00000.parquet). Falling back to alphabetical sort. "
                "Row order may differ from other parquet readers. "
                "Sample files: %s",
                [Path(p).name for p in unrecognized[:3]],
            )
        else:
            # Mixed naming - some recognized, some not
            logger.warning(
                "[order] Mixed parquet file naming detected. "
                "Files without part numbers will sort after numbered files: %s",
                [Path(p).name for p in unrecognized],
            )

    parquet_paths.sort(key=_parquet_sort_key)
    return [_format_uri(protocol, path) for path in parquet_paths]


def _read_segment_column(
    uri: str, segment_col: str, storage_options: dict
) -> pd.Series:
    with fsspec.open(uri, "rb", **storage_options) as fh:
        table = pq.read_table(fh, columns=[segment_col])
    if segment_col not in table.column_names:
        raise ValueError(f"Column '{segment_col}' not found in {uri}")
    series = table[segment_col].to_pandas()
    series.name = segment_col
    return series


def _normalize_segment(value):
    if pd.isna(value):
        return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    value = value.lower()
    if value in {"none", "noise", ""}:
        return None
    return value


def _int_to_uuid(arr: np.ndarray) -> np.ndarray:
    uuids = {i: str(i) if i < 0 else str(uuid.uuid4()) for i in np.unique(arr)}
    keys = np.array(sorted(uuids))
    idx = np.searchsorted(keys, arr)
    return np.array([uuids[k] for k in keys[idx]])


def _generate_points(segment_id: np.ndarray, rng_seed: int) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    centres = {sid: (2 * i, 0.0) for i, sid in enumerate(pd.unique(segment_id))}

    def rand_point(sid: int) -> tuple[float, float]:
        cx, cy = centres[sid]
        ang = rng.uniform(0.0, 2.0 * np.pi)
        rad = rng.uniform(0.0, 1.0)
        return cx + rad * np.cos(ang), cy + rad * np.sin(ang)

    return np.array([rand_point(int(sid)) for sid in segment_id], dtype="float64")


def _ensure_dir(uri: str, storage_options: dict) -> None:
    fs, path = fsspec.core.url_to_fs(uri, **storage_options)
    protocol = fs.protocol[0] if isinstance(fs.protocol, (tuple, list)) else fs.protocol
    if protocol in (None, "file"):
        Path(path).mkdir(parents=True, exist_ok=True)
    else:
        fs.makedirs(path, exist_ok=True)


def _update_config_yaml(
    config_uri: str,
    *,
    labels_file_name: str,
    ranked_points_file_name: str,
    storage_options: dict,
) -> bool:
    fs, path = fsspec.core.url_to_fs(config_uri, **storage_options)
    if not fs.exists(path):
        logger.warning("[config] not found, skipping update: %s", config_uri)
        return False

    with fs.open(path, "r", **storage_options) as fh:
        text = fh.read()

    key_pattern = re.compile(r"^\s*(labels_file_name|ranked_points_file_name)\s*:")
    lines = [line for line in text.splitlines() if not key_pattern.match(line)]

    insert_at = 0
    for line in lines:
        if line.strip() == "" or line.lstrip().startswith("#"):
            insert_at += 1
        else:
            break

    insert_lines = [
        f"labels_file_name: {labels_file_name}",
        f"ranked_points_file_name: {ranked_points_file_name}",
        "",
    ]
    new_lines = lines[:insert_at] + insert_lines + lines[insert_at:]
    new_text = "\n".join(new_lines).rstrip() + "\n"

    with fs.open(path, "w", **storage_options) as fh:
        fh.write(new_text)

    logger.info(
        "[config] updated config.yaml with labels_file_name and ranked_points_file_name"
    )
    return True


@contextlib.contextmanager
def _client_context(
    *,
    runtime: str,
    coiled_kwargs: dict,
) -> Client | None:
    if runtime == "local":
        cluster = LocalCluster()
        client = Client(cluster)
        try:
            yield client
        finally:
            client.close()
            cluster.close()
    else:
        try:
            import coiled
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Coiled is required for --runtime coiled") from exc
        cluster = coiled.Cluster(**coiled_kwargs)
        client = Client(cluster)
        try:
            yield client
        finally:
            client.close()
            cluster.close()


def _build_coiled_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {
        "name": args.coiled_name,
        "software": args.coiled_software_env,
        "n_workers": args.n_workers,
        "idle_timeout": args.idle_timeout,
        "no_client_timeout": args.no_client_timeout,
        "shutdown_on_close": True,
        "environ": args.coiled_env,
    }
    if args.worker_vm_types:
        kwargs["worker_vm_types"] = args.worker_vm_types
    else:
        kwargs["worker_cpu"] = args.worker_cpu
        kwargs["worker_memory"] = args.worker_memory
    if args.scheduler_vm_types:
        kwargs["scheduler_vm_types"] = args.scheduler_vm_types
    return kwargs


def _parse_env_pairs(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for entry in values:
        if "=" not in entry:
            raise ValueError(f"Invalid --coiled-env value: {entry}")
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --coiled-env value: {entry}")
        env[key] = value
    return env


def _join_uri(base: str, *parts: str) -> str:
    uri = base.rstrip("/")
    for part in parts:
        if part:
            uri = f"{uri}/{part.strip('/')}"
    return uri


def _collect_segments(
    parquet_uris: list[str],
    *,
    segment_col: str,
    storage_options: dict,
    client: Client | None,
) -> pd.Series:
    tasks = [
        delayed(_read_segment_column)(uri, segment_col, storage_options)
        for uri in parquet_uris
    ]
    if client is None:
        chunks = list(dask_compute(*tasks))
    else:
        futures = client.compute(tasks)
        chunks = list(client.gather(futures))
    return pd.concat(chunks, ignore_index=True)


def generate_points_and_labels(
    *,
    input_uri: str,
    output_uri: str,
    segment_col: str,
    labels_file_name: str,
    ranked_points_file_name: str,
    rng_seed: int,
    input_storage_options: dict,
    output_storage_options: dict,
    client: Client | None,
) -> tuple[str, str]:
    parquet_uris = _list_parquet_files(input_uri, input_storage_options)
    logger.info("[scan] found %s parquet files under %s", len(parquet_uris), input_uri)

    segments = _collect_segments(
        parquet_uris,
        segment_col=segment_col,
        storage_options=input_storage_options,
        client=client,
    )
    logger.info("[read] collected %s rows of '%s'", len(segments), segment_col)

    seg_series = segments.astype("object").apply(_normalize_segment)
    unique = pd.unique(seg_series.dropna())
    seg_id_map = {s: i for i, s in enumerate(unique)}
    seg_id_map[None] = -1
    segment_id = seg_series.map(seg_id_map).fillna(-1).astype("int32").to_numpy()

    labels = _int_to_uuid(segment_id)
    points = _generate_points(segment_id, rng_seed)

    _ensure_dir(output_uri, output_storage_options)
    labels_uri = _join_uri(output_uri, labels_file_name)
    points_uri = _join_uri(output_uri, ranked_points_file_name)

    with fsspec.open(labels_uri, "wb", **output_storage_options) as fh:
        np.save(fh, labels)
    with fsspec.open(points_uri, "wb", **output_storage_options) as fh:
        np.save(fh, points)

    logger.info("[write] labels.npy -> %s", labels_uri)
    logger.info("[write] precomputed_points.npy -> %s", points_uri)
    logger.info(
        "[done] ordering guarantee: rows follow sorted parquet part-file order (part-00000, part-00001, ...)"
    )

    return labels_uri, points_uri


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate labels.npy and precomputed_points.npy from segmented parquet data."
    )
    parser.add_argument("--volume", required=True, help="Base URI for the dataset.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Override input parquet directory (defaults to <volume>/input_data).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (defaults to <volume>/segmented_data).",
    )
    parser.add_argument(
        "--input-subdir",
        default="input_data",
        help="Input subdirectory under --volume (default: input_data).",
    )
    parser.add_argument(
        "--output-subdir",
        default="",
        help="Output subdirectory under --volume (default: volume root).",
    )
    parser.add_argument(
        "--segment-col",
        default="segment",
        help="Column to read from parquet files (default: segment).",
    )
    parser.add_argument(
        "--labels-file-name",
        default="labels.npy",
        help="Labels file name to write (default: labels.npy).",
    )
    parser.add_argument(
        "--ranked-points-file-name",
        default="precomputed_points.npy",
        help="Points file name to write (default: precomputed_points.npy).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Random seed for point generation (default: 42).",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Override config.yaml path (defaults to <volume>/config.yaml).",
    )
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Skip updating config.yaml with labels/points file names.",
    )
    parser.add_argument(
        "--runtime",
        choices=["local", "coiled"],
        default="coiled",
        help="Execution runtime (default: coiled).",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info).",
    )
    parser.add_argument("--coiled-name", default="neuralift_c360_prep_pr-5")
    parser.add_argument("--coiled-software-env", default="neuralift_c360_prep_pr-5")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--worker-cpu", type=int, default=8)
    parser.add_argument("--worker-memory", default="32GiB")
    parser.add_argument("--worker-vm-types", default=None)
    parser.add_argument("--scheduler-vm-types", default=None)
    parser.add_argument("--idle-timeout", default="2h")
    parser.add_argument("--no-client-timeout", default="2h")
    parser.add_argument(
        "--coiled-env",
        action="append",
        default=[],
        help="Env vars to pass to Coiled workers (repeatable KEY=VALUE).",
    )
    return parser.parse_args()


def main() -> None:
    env_from_dotenv = dotenv_env_vars()
    args = _parse_args()
    setup_logging(args.log_level)
    args.coiled_env = _parse_env_pairs(args.coiled_env)
    args.coiled_env = {**env_from_dotenv, **args.coiled_env}
    args.worker_vm_types = (
        [v.strip() for v in args.worker_vm_types.split(",") if v.strip()]
        if args.worker_vm_types
        else None
    )
    args.scheduler_vm_types = (
        [v.strip() for v in args.scheduler_vm_types.split(",") if v.strip()]
        if args.scheduler_vm_types
        else None
    )

    volume = args.volume.rstrip("/")
    input_uri = args.input_dir or _join_uri(volume, args.input_subdir)
    output_uri = args.output_dir or _join_uri(volume, args.output_subdir)
    config_uri = args.config_path or _join_uri(volume, "config.yaml")

    input_protocol = fsspec.utils.infer_storage_options(input_uri).get("protocol")
    if args.runtime == "coiled" and input_protocol in (None, "file"):
        raise SystemExit("Coiled runtime requires s3:// or remote paths.")

    input_storage_options = _storage_options_for_uri(
        input_uri, require_creds=input_uri.startswith("s3://")
    )
    output_storage_options = _storage_options_for_uri(
        output_uri, require_creds=output_uri.startswith("s3://")
    )
    config_storage_options = _storage_options_for_uri(
        config_uri, require_creds=config_uri.startswith("s3://")
    )

    with _client_context(
        runtime=args.runtime, coiled_kwargs=_build_coiled_kwargs(args)
    ) as client:
        if client is not None:
            configure_dask_logging(
                client,
                level=args.log_level,
                forward_to_scheduler=args.runtime == "coiled",
            )
        logger.info("[init] input=%s", input_uri)
        logger.info("[init] output=%s", output_uri)
        logger.info("[init] runtime=%s", args.runtime)
        generate_points_and_labels(
            input_uri=input_uri,
            output_uri=output_uri,
            segment_col=args.segment_col,
            labels_file_name=args.labels_file_name,
            ranked_points_file_name=args.ranked_points_file_name,
            rng_seed=args.rng_seed,
            input_storage_options=input_storage_options,
            output_storage_options=output_storage_options,
            client=client,
        )

    if not args.no_update_config:
        _update_config_yaml(
            config_uri,
            labels_file_name=args.labels_file_name,
            ranked_points_file_name=args.ranked_points_file_name,
            storage_options=config_storage_options,
        )

    logger.info("[done] points_and_labels job complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
