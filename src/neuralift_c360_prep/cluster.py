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
from typing import Iterator
from dask.distributed import Client, LocalCluster

from .config import BundleConfig

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def local_client(cfg: BundleConfig) -> Iterator[Client]:
    cluster = LocalCluster()
    client = Client(cluster)
    logger.info("Started local Dask cluster at %s", client.scheduler_info()["address"])
    try:
        yield client
    finally:
        client.close()
        cluster.close()
        logger.info("Closed local Dask cluster")


@contextlib.contextmanager
def coiled_client(cfg: BundleConfig) -> Iterator[Client]:
    try:
        import coiled
    except Exception as exc:  # pragma: no cover - missing optional dep
        raise RuntimeError("Coiled is required for coiled runtime") from exc

    c = cfg.runtime.coiled
    cluster_kwargs: dict = {
        "name": c.name,
        "software": c.software_env,
        "n_workers": c.n_workers,
        "idle_timeout": c.idle_timeout,
        "no_client_timeout": c.no_client_timeout,
        "shutdown_on_close": True,
        "environ": c.env,
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

    coiled_cluster = coiled.Cluster(**cluster_kwargs)
    client = Client(coiled_cluster)
    logger.info(
        "Started Coiled cluster '%s' at %s", c.name, client.scheduler_info()["address"]
    )
    try:
        yield client
    finally:
        client.close()
        coiled_cluster.close()
        logger.info("Closed Coiled cluster '%s'", c.name)


def get_client(cfg: BundleConfig) -> contextlib.AbstractContextManager[Client]:
    """
    Pick the appropriate client context manager for the configured runtime.
    """
    if cfg.runtime.engine == "coiled":
        return coiled_client(cfg)
    return local_client(cfg)


__all__ = ["get_client"]
