#!/usr/bin/env python3
"""
Console entrypoint for the neuralift_c360_prep Dask bundle.

Purpose:
    - Parse CLI arguments for config path, runtime, and log level.
    - Support three execution modes:
        1. Local: --runtime local (default)
        2. Coiled non-batch: --runtime coiled (driver local, workers in Coiled)
        3. Coiled batch: --runtime coiled --batch (everything in Coiled)

Usage:
    neuralift_c360_prep --config configs/data_prep.yaml --runtime local
    neuralift_c360_prep --config configs/data_prep.yaml --runtime coiled
    neuralift_c360_prep --config configs/data_prep.yaml --runtime coiled --batch

Author: Mike Maloney - Neuralift, Inc.
Updated: 2025-01-12
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .log_utils import setup_logging
from .pipeline import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Neuralift Dask bundle pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--runtime",
        choices=["local", "coiled"],
        default="local",
        help="Execution runtime (default: local)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Submit as Coiled batch job (requires --runtime coiled)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        help="Override log level",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.log_level:
        cfg.logging.level = args.log_level

    # Batch submission mode
    if args.batch:
        if args.runtime != "coiled":
            raise SystemExit("--batch requires --runtime coiled")
        from .batch import submit_batch

        submit_batch(cfg, config_path=Path(args.config))
        return

    # Direct execution mode
    cfg.runtime.engine = args.runtime
    setup_logging(
        cfg.logging.level,
        dask_level=cfg.logging.dask_level,
        llm_level=cfg.logging.llm_level,
    )
    run_from_config(cfg)


if __name__ == "__main__":
    main()
