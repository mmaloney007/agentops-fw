#!/usr/bin/env python3
"""
Console entrypoint for the neuralift_c360_prep Dask bundle.

Purpose:
    - Parse CLI arguments for config path, runtime override, and log level.
    - Load the bundle configuration and invoke the Dask/Coiled pipeline via run_from_config.
    - Configure logging based on config or CLI overrides.

Usage:
    python -m neuralift_c360_prep.cli --config configs/data_prep.yaml
    # or, if installed with console_scripts:
    neuralift_c360_prep --config configs/data_prep.yaml

Dependencies:
    - argparse
    - logging
    - neuralift_c360_prep.config
    - neuralift_c360_prep.pipeline

Author: Mike Maloney - Neuralift, Inc.
Updated: 2025-12-08
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import argparse
import base64
import os
import shlex
from pathlib import Path

from .cluster import build_coiled_cluster_kwargs
from .config import BundleConfig, load_config
from .env import BATCH_ENV_FLAG, CONFIG_B64_ENV_KEY, collect_coiled_env_vars
from .log_utils import setup_logging
from .pipeline import run_from_config


def _encode_config(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _should_submit_batch(cfg: BundleConfig) -> bool:
    if cfg.runtime.engine != "coiled":
        return False
    if os.getenv(BATCH_ENV_FLAG) == "1":
        return False
    return cfg.runtime.coiled.submit_batch


def _submit_coiled_batch(cfg: BundleConfig, *, config_path: Path) -> None:
    try:
        import coiled
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("Coiled is required for batch submission") from exc

    cluster_kwargs = build_coiled_cluster_kwargs(cfg, include_software=False)
    batch_env = collect_coiled_env_vars(cfg.runtime.coiled.env)
    batch_env[BATCH_ENV_FLAG] = "1"
    batch_env[CONFIG_B64_ENV_KEY] = _encode_config(config_path)

    cmd = [
        "/app/entrypoint.sh",
        "python",
        "-m",
        "neuralift_c360_prep.batch_entrypoint",
        "--runtime",
        "coiled",
    ]
    if cfg.logging.level:
        cmd += ["--log-level", cfg.logging.level]

    batch_kwargs: dict = {
        "software": cfg.runtime.coiled.software_env,
        "cluster_kwargs": cluster_kwargs,
        "task_on_scheduler": True,
        "secret_env": batch_env,
    }
    if cfg.runtime.coiled.batch_region:
        batch_kwargs["region"] = cfg.runtime.coiled.batch_region
    batch_vm_type = cfg.runtime.coiled.batch_vm_type
    if not batch_vm_type and cfg.runtime.coiled.worker_vm_types:
        batch_vm_type = cfg.runtime.coiled.worker_vm_types[0]
    if batch_vm_type:
        batch_kwargs["vm_type"] = batch_vm_type

    coiled.batch.run(shlex.join(cmd), **batch_kwargs)
    print(
        "Submitted Coiled batch job for config:",
        config_path,
        "| cluster:",
        cluster_kwargs["name"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Neuralift Dask bundle pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--runtime", choices=["local", "coiled"], help="Override runtime engine"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        help="Override log level",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable Coiled batch submission and run driver locally",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.runtime:
        cfg.runtime.engine = args.runtime
    if args.log_level:
        cfg.logging.level = args.log_level
    if args.no_batch:
        cfg.runtime.coiled.submit_batch = False

    if _should_submit_batch(cfg):
        _submit_coiled_batch(cfg, config_path=Path(args.config))
        return

    setup_logging(
        cfg.logging.level,
        dask_level=cfg.logging.dask_level,
        llm_level=cfg.logging.llm_level,
    )
    run_from_config(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
