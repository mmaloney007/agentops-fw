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
import logging
from pathlib import Path

from .config import load_config
from .pipeline import run_from_config


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
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.runtime:
        cfg.runtime.engine = args.runtime
    if args.log_level:
        cfg.logging.level = args.log_level

    logging.basicConfig(level=getattr(logging, cfg.logging.level.upper(), logging.INFO))
    run_from_config(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
