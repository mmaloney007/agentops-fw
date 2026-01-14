#!/usr/bin/env python3
"""
utils.py (Dask)
---------------
Partition diagnostics and allocator/reporting helpers for Dask workflows.

Purpose:
    - Report allocator/ENV info for troubleshooting (jemalloc, CONDA_PREFIX, etc.).
    - Estimate bytes per row from a sample to guide partition sizing.
    - Inspect partition counts/skew and suggest or apply repartitioning heuristics.

Usage:
    from neuralift_c360_prep.utils import (
        report_allocator,
        estimate_bytes_per_row,
        diagnose_partitions,
        auto_repartition,
    )

Dependencies:
    - dask[dataframe]
    - pandas
    - platform/os

Author: Mike Maloney - Neuralift, Inc.
Updated: 2025-12-08
Copyright © 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
import math
import os
import platform

import dask.dataframe as dd
import pandas as pd

logger = logging.getLogger(__name__)


def report_allocator(label="process"):
    return {
        "label": label,
        "pid": os.getpid(),
        "platform": platform.system(),
        "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
        "DYLD_INSERT_LIBRARIES": os.environ.get("DYLD_INSERT_LIBRARIES"),
        "MALLOC_CONF": os.environ.get("MALLOC_CONF"),
        "jemalloc_env": (
            (os.environ.get("DYLD_INSERT_LIBRARIES") or "").lower().find("jemalloc")
            >= 0
        ),
    }


def estimate_bytes_per_row(ddf: dd.DataFrame, sample_rows: int = 10_000) -> int:
    """
    Estimate average bytes per row using a small sample.
    """
    pdf = ddf.head(sample_rows, compute=True)
    if len(pdf) == 0:
        return 0
    total_bytes = pdf.memory_usage(deep=True, index=False).sum()
    return int(total_bytes // len(pdf))


def diagnose_partitions(
    ddf: dd.DataFrame,
    *,
    label: str = "ddf",
    warn_only: bool = True,
    skew_threshold: float = 3.0,
) -> dict:
    """
    Inspect partition count and row skew.
    """
    nparts = ddf.npartitions
    rows_per_part = ddf.map_partitions(len).compute()
    if not isinstance(rows_per_part, pd.Series):
        rows_per_part = pd.Series(rows_per_part)

    min_rows = int(rows_per_part.min())
    max_rows = int(rows_per_part.max())
    mean_rows = float(rows_per_part.mean()) if len(rows_per_part) else 0.0
    skew_ratio = (max_rows / max(min_rows, 1)) if min_rows > 0 else math.inf

    logger.info("🔎 Partition diagnostics for %s:", label)
    logger.info("  partitions      : %s", nparts)
    logger.info(
        "  rows/partition  : min=%s, max=%s, mean=%s",
        f"{min_rows:,}",
        f"{max_rows:,}",
        f"{mean_rows:,.1f}",
    )
    logger.info("  skew ratio      : %s", f"{skew_ratio:,.2f}x (max / min)")

    if warn_only:
        if skew_ratio >= skew_threshold:
            logger.warning(
                "  ⚠️ Skew is high (>%sx). Consider repartitioning.", skew_threshold
            )
    return {
        "npartitions": nparts,
        "rows_per_part": rows_per_part,
        "min_rows": min_rows,
        "max_rows": max_rows,
        "mean_rows": mean_rows,
        "skew_ratio": skew_ratio,
    }


def auto_repartition(
    ddf: dd.DataFrame,
    *,
    target_min_parts: int = 200,
    target_max_parts: int = 400,
    max_skew: float = 3.0,
    per_worker_mem_gib: float | None = None,
    n_workers: int | None = None,
    label: str = "ddf",
    apply: bool = True,
    min_nparts: int = 2,
    target_fraction_per_part: float = 0.03,
) -> dd.DataFrame:
    """
    Diagnose current partitioning and optionally repartition.
    """
    diag = diagnose_partitions(
        ddf, label=label, warn_only=False, skew_threshold=max_skew
    )
    nparts = diag["npartitions"]
    skew_ratio = diag["skew_ratio"]
    rows_per_part = diag["rows_per_part"]
    row_count = int(rows_per_part.sum())

    need_repartition = False

    if nparts < target_min_parts or nparts > target_max_parts:
        logger.info(
            "  ℹ️ partition count outside target range [%s, %s]: %s",
            target_min_parts,
            target_max_parts,
            nparts,
        )
        need_repartition = True

    if skew_ratio >= max_skew:
        logger.info(
            "  ℹ️ skew %sx exceeds threshold %sx",
            f"{skew_ratio:,.2f}",
            max_skew,
        )
        need_repartition = True

    if not need_repartition:
        logger.info("  ✅ Partitioning looks fine; no repartition needed.")
        return ddf

    base_min_parts = max(min_nparts, 1)
    if n_workers is not None:
        base_min_parts = max(base_min_parts, n_workers)

    if row_count <= 10_000:
        row_based_nparts = max(base_min_parts, 2)
    elif row_count <= 100_000:
        row_based_nparts = max(base_min_parts, row_count // 10_000)
    elif row_count <= 1_000_000:
        row_based_nparts = max(base_min_parts, row_count // 50_000)
    else:
        row_based_nparts = max(base_min_parts, row_count // 200_000)

    mem_based_nparts = None
    if per_worker_mem_gib is not None and row_count > 0:
        bpr = estimate_bytes_per_row(ddf)
        if bpr <= 0:
            bpr = 50 * 1024

        per_worker_bytes = per_worker_mem_gib * (1024**3)
        target_bytes_per_part = per_worker_bytes * target_fraction_per_part
        mem_based_nparts = max(
            base_min_parts, int((row_count * bpr) / max(target_bytes_per_part, 1))
        )

        logger.info(
            "  ℹ️ memory-aware suggestion: bytes_per_row≈%s, "
            "per_worker≈%.2fGiB, target_bytes_per_part≈%s, mem_based_nparts≈%s",
            f"{bpr:,}",
            per_worker_mem_gib,
            f"{int(target_bytes_per_part):,}",
            mem_based_nparts,
        )
    else:
        logger.info(
            "  ℹ️ per_worker_mem_gib not provided; skipping memory-based nparts hint."
        )

    target_nparts = row_based_nparts
    if mem_based_nparts is not None:
        target_nparts = max(target_nparts, mem_based_nparts)

    target_nparts = max(target_nparts, base_min_parts)
    if row_count > 1_000_000:
        target_nparts = max(target_nparts, target_min_parts)
    target_nparts = min(target_nparts, target_max_parts * 2)

    logger.info(
        "  🔄 Repartitioning %s → %s partitions (rows≈%s, current=%s)...",
        label,
        target_nparts,
        f"{row_count:,}",
        nparts,
    )
    if not apply:
        logger.info("  (dry-run: apply=False, not actually repartitioning)")
        return ddf

    ddf2 = ddf.repartition(npartitions=target_nparts)
    logger.info("  ✅ Repartition complete.")
    return ddf2


__all__ = [
    "report_allocator",
    "estimate_bytes_per_row",
    "diagnose_partitions",
    "auto_repartition",
]
