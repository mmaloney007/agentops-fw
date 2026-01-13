#!/usr/bin/env python3
"""
Coiled batch job submission.

Purpose:
    Submit the pipeline as a Coiled batch job where both driver and workers
    run in the cloud.

Usage:
    from neuralift_c360_prep.batch import submit_batch
    submit_batch(cfg, config_path=Path("configs/my_config.yaml"))

Author: Mike Maloney - Neuralift, Inc.
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from .cluster import build_coiled_cluster_kwargs
from .config import BundleConfig

logger = logging.getLogger(__name__)


def submit_batch(cfg: BundleConfig, *, config_path: Path) -> None:
    """
    Submit the pipeline as a Coiled batch job.

    The config file is uploaded to /config.yaml in the batch environment.
    The entrypoint.sh activates the pixi environment before running.
    """
    try:
        import coiled
    except ImportError as exc:
        raise RuntimeError("Coiled is required for batch submission") from exc

    cluster_kwargs = build_coiled_cluster_kwargs(cfg, include_software=False)
    cluster_name = cluster_kwargs["name"]

    # Build command using entrypoint to activate pixi environment
    # Config is uploaded to ./config.yaml (relative to working dir) via buffers_to_upload
    cmd_parts = [
        "/app/entrypoint.sh",
        "neuralift_c360_prep",
        "--config",
        "./config.yaml",
        "--runtime",
        "coiled",
    ]
    if cfg.logging.level:
        cmd_parts.extend(["--log-level", cfg.logging.level])

    cmd = " ".join(cmd_parts)

    # Upload config file to batch environment
    config_content = config_path.read_bytes()
    config_buffer = io.BytesIO(config_content)

    batch_kwargs: dict = {
        "software": cfg.runtime.coiled.software_env,
        "cluster_kwargs": cluster_kwargs,
        "task_on_scheduler": True,
        "buffers_to_upload": [
            {"relative_path": "config.yaml", "buffer": config_buffer}
        ],
        "command_as_script": True,  # Suppress file detection warnings
    }

    # Use worker VM type for batch if available
    if cfg.runtime.coiled.worker_vm_types:
        batch_kwargs["vm_type"] = cfg.runtime.coiled.worker_vm_types[0]

    logger.info("Submitting Coiled batch job: %s", cmd)
    coiled.batch.run(cmd, **batch_kwargs)
    print(
        f"Submitted Coiled batch job for config: {config_path} | cluster: {cluster_name}"
    )


__all__ = ["submit_batch"]
