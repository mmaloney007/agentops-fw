#!/usr/bin/env python3
"""
Batch entrypoint for running the pipeline inside a Coiled batch job.
"""

from __future__ import annotations

import argparse
import base64
import os

import yaml
from pydantic import ValidationError

from .config import BundleConfig
from .env import BATCH_ENV_FLAG, CONFIG_B64_ENV_KEY, load_dotenv_file
from .log_utils import setup_logging
from .pipeline import run_from_config


def _load_config_from_b64(config_b64: str) -> BundleConfig:
    load_dotenv_file()
    try:
        decoded = base64.b64decode(config_b64.encode("utf-8")).decode("utf-8")
    except Exception as exc:
        raise SystemExit("Invalid base64 config payload") from exc
    try:
        data = yaml.safe_load(decoded) or {}
    except yaml.YAMLError as exc:
        raise SystemExit("Invalid YAML config payload") from exc
    try:
        return BundleConfig.model_validate(data)
    except ValidationError as exc:
        raise SystemExit(f"Invalid config from batch payload:\n{exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Neuralift pipeline inside Coiled batch"
    )
    parser.add_argument(
        "--config-b64",
        help="Base64-encoded YAML config (defaults to env var)",
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
        help="Override log level",
    )
    args = parser.parse_args()

    config_b64 = args.config_b64 or os.getenv(CONFIG_B64_ENV_KEY, "")
    if not config_b64:
        raise SystemExit("Missing batch config payload")

    cfg = _load_config_from_b64(config_b64)
    cfg.runtime.engine = args.runtime
    if args.log_level:
        cfg.logging.level = args.log_level
    if os.getenv(BATCH_ENV_FLAG) == "1":
        cfg.runtime.coiled.use_existing = True

    setup_logging(
        cfg.logging.level,
        dask_level=cfg.logging.dask_level,
        llm_level=cfg.logging.llm_level,
    )
    run_from_config(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
