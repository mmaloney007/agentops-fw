#!/usr/bin/env python3
"""
Cluster utilities for running the Dask bundle locally or on Coiled.

Purpose:
    - Provide context managers that yield a connected Dask Client.
    - Normalize Coiled configuration (name/env/timeouts) and environment propagation.
    - Keep logging simple and driver-safe.

Usage:
    from neuralift_c360_prep.cluster import get_client
    with get_client(cfg) as client:
        ...

Dependencies:
    - dask[complete]
    - coiled (for Coiled runtime)

Author: Mike Maloney - Neuralift, Inc.
Updated: 2025-12-08
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Iterator

import dask
from dask.distributed import Client, LocalCluster

from .config import BundleConfig
from .env import dotenv_env_vars
from .log_utils import configure_dask_logging, set_worker_shutdown_flag

logger = logging.getLogger(__name__)

# Default Dask performance settings applied to all clusters
DASK_PERF_DEFAULTS = {
    "dataframe.convert-string": True,  # Use PyArrow-backed strings
    "dataframe.query-planning": False,  # Disable query planning by default
    "distributed.scheduler.work-stealing": True,
    "distributed.worker.memory.target": 0.6,
    "distributed.worker.memory.spill": 0.7,
    "distributed.worker.memory.pause": 0.8,
    "distributed.worker.memory.terminate": 0.95,
}


def _apply_dask_perf_defaults() -> None:
    """Apply default Dask performance settings."""
    for key, value in DASK_PERF_DEFAULTS.items():
        try:
            dask.config.set({key: value})
        except Exception:
            pass  # Skip if config key doesn't exist in this Dask version


def _generate_unique_cluster_name(base_name: str) -> str:
    """Generate a unique cluster name with timestamp and short UUID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{base_name}-{timestamp}-{short_uuid}"


_DASK_SHUTDOWN_NOISE = "Failed to communicate with scheduler during heartbeat"
_shutdown_in_progress = False


class _DaskShutdownFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not _shutdown_in_progress:
            return True
        msg = record.getMessage()
        if _DASK_SHUTDOWN_NOISE in msg:
            return False
        return True


def _ensure_shutdown_filter() -> None:
    worker_logger = logging.getLogger("distributed.worker")
    for existing in worker_logger.filters:
        if isinstance(existing, _DaskShutdownFilter):
            return
    worker_logger.addFilter(_DaskShutdownFilter())


def _set_shutdown_flag(enabled: bool) -> None:
    global _shutdown_in_progress
    _shutdown_in_progress = enabled


@contextlib.contextmanager
def local_client(cfg: BundleConfig) -> Iterator[Client]:
    # Apply performance defaults before creating cluster
    _apply_dask_perf_defaults()

    cluster = LocalCluster()
    client = Client(cluster)
    configure_dask_logging(
        client,
        level=cfg.logging.level,
        dask_level=cfg.logging.dask_level,
        llm_level=cfg.logging.llm_level,
        forward_to_scheduler=False,
    )
    _ensure_shutdown_filter()
    logger.info("Started local Dask cluster at %s", client.scheduler_info()["address"])
    try:
        yield client
    finally:
        _set_shutdown_flag(True)
        try:
            client.run(set_worker_shutdown_flag, True)
        except Exception:
            pass
        client.close()
        cluster.close()
        _set_shutdown_flag(False)
        logger.info("Closed local Dask cluster")


@contextlib.contextmanager
def coiled_client(cfg: BundleConfig) -> Iterator[Client]:
    try:
        import coiled
    except Exception as exc:  # pragma: no cover - missing optional dep
        raise RuntimeError("Coiled is required for coiled runtime") from exc

    # Apply performance defaults before creating cluster
    _apply_dask_perf_defaults()

    c = cfg.runtime.coiled

    # Generate unique cluster name for better monitoring/debugging
    cluster_name = _generate_unique_cluster_name(c.name)

    # Merge performance env vars with user-specified env vars
    # User env vars take precedence (applied second)
    perf_env_vars = {
        "DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING": "true",
        "DASK_DISTRIBUTED__WORKER__MEMORY__TARGET": "0.6",
        "DASK_DISTRIBUTED__WORKER__MEMORY__SPILL": "0.7",
        "DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE": "0.8",
        "DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE": "0.95",
        "DASK_DATAFRAME__QUERY_PLANNING": "0",
    }
    env_vars = {**perf_env_vars, **dotenv_env_vars(), **c.env}

    cluster_kwargs: dict = {
        "name": cluster_name,
        "software": c.software_env,
        "n_workers": c.n_workers,
        "idle_timeout": c.idle_timeout,
        "no_client_timeout": c.no_client_timeout,
        "shutdown_on_close": True,
        "environ": env_vars,
    }
    if c.worker_vm_types:
        cluster_kwargs["worker_vm_types"] = c.worker_vm_types
    else:
        cluster_kwargs["worker_cpu"] = c.worker_cpu
        cluster_kwargs["worker_memory"] = c.worker_memory
    if c.scheduler_vm_types:
        cluster_kwargs["scheduler_vm_types"] = c.scheduler_vm_types
    else:
        cluster_kwargs["scheduler_cpu"] = c.scheduler_cpu
        cluster_kwargs["scheduler_memory"] = c.scheduler_memory

    logger.info("Creating Coiled cluster '%s' (base: '%s')...", cluster_name, c.name)
    coiled_cluster = coiled.Cluster(**cluster_kwargs)
    client = Client(coiled_cluster)
    configure_dask_logging(
        client,
        level=cfg.logging.level,
        dask_level=cfg.logging.dask_level,
        llm_level=cfg.logging.llm_level,
        forward_to_scheduler=True,
    )
    _ensure_shutdown_filter()
    logger.info(
        "Started Coiled cluster '%s' at %s",
        cluster_name,
        client.scheduler_info()["address"],
    )
    try:
        yield client
    finally:
        _set_shutdown_flag(True)
        try:
            client.run(set_worker_shutdown_flag, True)
        except Exception:
            pass
        client.close()
        coiled_cluster.close()
        _set_shutdown_flag(False)
        logger.info("Closed Coiled cluster '%s'", cluster_name)


def get_client(cfg: BundleConfig) -> contextlib.AbstractContextManager[Client]:
    """
    Pick the appropriate client context manager for the configured runtime.
    """
    if cfg.runtime.engine == "coiled":
        return coiled_client(cfg)
    return local_client(cfg)


__all__ = ["get_client"]
