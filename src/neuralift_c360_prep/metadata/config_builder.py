#!/usr/bin/env python3
"""
metadata/config_builder.py
--------------------------
Segmenter config generation with ML heuristics.

Contains:
    - SEGMENTER_CONFIG_DEFAULTS: Default configuration for segmenter
    - suggest_* functions: ML heuristics for hyperparameters
    - build_pretty_config_from_data_dict: Main config generator
    - render_config_yaml_with_comments: YAML rendering with annotations
    - print_config_yaml, save_config_yaml: Output utilities

Author: Mike Maloney - Neuralift, Inc.
Updated: 2026-01-13
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import logging
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union
from uuid import uuid4

import dask.dataframe as dd
import pandas as pd
import yaml
from pandas.api.types import is_bool_dtype, is_numeric_dtype

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT_PATH = str(Path(tempfile.gettempdir()) / str(uuid4()))

SEGMENTER_CONFIG_DEFAULTS: Dict[str, Any] = {
    "use_gpu": True,
    "is_container": False,
    "use_wandb": False,
    "input_uri": None,
    "input_path": None,
    "config_file_name": "config.yaml",
    "output_path": _DEFAULT_OUTPUT_PATH,
    "delete_existing_artifacts": False,
    "labels_file_name": None,
    "ranked_points_file_name": None,
    "verbose": 0,
    "headless": False,
    "resume": False,
    "json_logging": None,
    "data": {
        "sample_frac": None,
    },
    "dae": {
        "dataset_stats_file_name": "dataset_stats.joblib",
        "estimate_batch_size": True,
        "rmm_allocator": True,
        "compile": False,
        "scale_batch_size": False,
        "weight_averaging": True,
        "matmul_precision": "high",
        "data_module": {
            "val_split": 0.2,
            "batch_size": 256,
            "compute_stats_from": "full",
            "num_sample_partitions": 5,
            "optimize_memory": False,
        },
        "model": {
            "learning_rate": 4e-3,
            "backbone_type": "mlp",
            "encoder_hidden_dims": [256, 128],
            "decoder_hidden_dims": [128, 256],
            "latent_dim": 64,
            "feature_embed_dim": 32,
            "scheduler": "onecycle",
            "optimizer": "adam",
            "gradient_checkpointing": False,
            "use_sparse_categorical": False,
            "use_grouped_categorical_head": False,
            "boolean_cardinality_threshold": 2,
            "use_mixed_categorical": True,
            "max_onehot_cardinality": 4,
            "batch_norm_continuous": True,
            "batch_norm_embeddings": True,
            "embedding_dropout": 0.1,
            "robust_scaler": False,
            "num_swap_prob": 0.35,
            "cat_swap_prob": 0.35,
        },
        "trainer": {
            "max_epochs": 50,
            "accelerator": "auto",
            "precision": "bf16-mixed",
            "fast_dev_run": False,
            "gradient_clip_val": 1.0,
            "devices": -1,
            "enable_model_summary": True,
            "sync_batchnorm": False,
        },
        "distributed": {
            "enabled": False,
            "backend": "nccl",
        },
    },
    "segmenter": {
        "cluster_selection_method": "eom",
        "min_cluster_pct": None,
        "min_cluster_size": None,
        "min_cluster_threshold": None,
        "min_samples": 10,
        "min_samples_pct": None,
        "min_dist": 0.0,
        "soft_clustering_batch_size": None,
        "noise_threshold": None,
        "n_neighbors": 15,
        "n_components": None,
        "nnd_n_clusters": 4,
        "nnd_overlap_factor": 2,
        "knn_n_clusters": 4,
        "knn_overlap_factor": 2,
        "metric": "euclidean",
        "prediction_data": True,
    },
    "xgboost": {
        "scale_pos_weight": True,
        "max_bin": 256,
        "rmm_pool_frac": 0.8,
        "jit_unspill": True,
        "enable_cudf_spill": False,
        "protocol": None,
    },
    "explainability": {
        "top_n": 5,
        "num_features": 25,
    },
    "wandb": {
        "entity": "neuralift-ai",
        "project": None,
        "group": None,
        "mode": "online",
    },
}

_MISSING_DEFAULT = object()


# ---------------------------------------------------------------------------
# Config helper functions
# ---------------------------------------------------------------------------
def _flatten_defaults(values: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict into dotted paths."""
    flat: Dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_defaults(value, path))
        else:
            flat[path] = value
    return flat


def _format_default_value(value: Any) -> str:
    """Format a value for YAML comment."""
    text = yaml.safe_dump(
        value,
        sort_keys=False,
        default_flow_style=True,
        allow_unicode=False,
    ).strip()
    if text.endswith("..."):
        text = text[:-3].strip()
    return text.replace("\n", " ")


def _get_value_by_path(config: Mapping[str, Any], path: str) -> Any:
    """Get a value from nested dict by dotted path."""
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING_DEFAULT
        current = current[part]
    return current


def _strip_nulls_and_defaults(value: Any, defaults: Any = _MISSING_DEFAULT) -> Any:
    """Remove None values and values matching defaults."""
    if isinstance(value, Mapping):
        cleaned: Dict[str, Any] = {}
        defaults_map = defaults if isinstance(defaults, Mapping) else {}
        for key, val in value.items():
            default_val = defaults_map.get(key, _MISSING_DEFAULT)
            cleaned_val = _strip_nulls_and_defaults(val, default_val)
            if cleaned_val is None:
                continue
            if default_val is not _MISSING_DEFAULT and cleaned_val == default_val:
                continue
            cleaned[key] = cleaned_val
        return cleaned or None
    if isinstance(value, list):
        cleaned_list = [_strip_nulls_and_defaults(val) for val in value]
        cleaned_list = [val for val in cleaned_list if val is not None]
        if isinstance(defaults, list) and cleaned_list == defaults:
            return None
        return cleaned_list
    if value is None:
        return None
    if defaults is not _MISSING_DEFAULT and value == defaults:
        return None
    return value


def _filter_rationale(
    rationale: Mapping[str, str], config: Mapping[str, Any]
) -> Dict[str, str]:
    """Filter rationale to only include keys present in config."""
    filtered: Dict[str, str] = {}
    for path, msg in rationale.items():
        if _get_value_by_path(config, path) is not _MISSING_DEFAULT:
            filtered[path] = msg
    return filtered


def _annotate_yaml_with_defaults(
    yaml_text: str,
    *,
    config: Mapping[str, Any],
    defaults: Mapping[str, Any] | None,
) -> str:
    """Add inline default comments to YAML."""
    if not defaults:
        return yaml_text

    flat_defaults = _flatten_defaults(defaults)
    lines = yaml_text.splitlines()
    output: List[str] = []
    stack: List[Tuple[int, str]] = []

    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            output.append(line)
            continue

        match = re.match(r"^(\s*)([^:#]+):(?:\s*.*)?$", line)
        if not match:
            output.append(line)
            continue

        indent = len(match.group(1))
        key = match.group(2).strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        path = ".".join([item[1] for item in stack] + [key])
        value = _get_value_by_path(config, path)
        if isinstance(value, Mapping):
            output.append(line)
            stack.append((indent, key))
            continue

        default_value = flat_defaults.get(path, _MISSING_DEFAULT)
        if default_value is not _MISSING_DEFAULT and value != default_value:
            line = f"{line} # default: {_format_default_value(default_value)}"

        output.append(line)
    return "\n".join(output)


def _read_env_float(name: str) -> float | None:
    """Read float from environment variable."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _dtype_is_numeric(dtype: Any) -> bool:
    """Check if dtype is numeric (excluding bool)."""
    try:
        dt = pd.api.types.pandas_dtype(dtype)
        if is_bool_dtype(dt):
            return False
        return bool(is_numeric_dtype(dt))
    except Exception:
        s = str(dtype).lower()
        return s.startswith(("int", "uint", "float", "decimal"))


def _safe_row_count(ddf: dd.DataFrame) -> int:
    """Safely get row count from Dask DataFrame."""
    try:
        return int(ddf.shape[0].compute())
    except Exception:
        try:
            return int(ddf.map_partitions(len).compute().sum())
        except Exception:
            return 0


def _approx_nunique_dask(
    ddf: dd.DataFrame,
    col: str,
    *,
    n_rows: int,
    max_sample: int = 100_000,
) -> int:
    """Approximate nunique using sampling."""
    if col not in ddf.columns:
        return 0

    try:
        if n_rows <= 0:
            series = ddf[col]
        elif n_rows <= max_sample:
            series = ddf[col]
        else:
            frac = max_sample / float(n_rows)
            series = ddf[col].sample(frac=frac, random_state=42)

        try:
            return int(series.nunique().compute())
        except Exception:
            if hasattr(series, "nunique_approx"):
                return int(series.nunique_approx().compute())
            return int(series.astype(str).nunique().compute())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Column role inference
# ---------------------------------------------------------------------------
def infer_column_roles_from_data_dict(
    data_dict: Dict[str, Any],
    ddf: dd.DataFrame,
    *,
    max_card_for_cat: int = 20,
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, int]]:
    """Infer column roles and collect categorical cardinalities.

    Returns:
        (id_cols, kpi_cols, cat_cols, cont_cols, cat_cardinalities_by_col)
    """
    cols_meta = data_dict.get("columns") or []

    n_rows = _safe_row_count(ddf)

    id_cols: List[str] = []
    kpi_cols: List[str] = []
    cat_cols: List[str] = []
    cont_cols: List[str] = []
    cat_cards: Dict[str, int] = {}

    for col_meta in cols_meta:
        name = (
            col_meta.get("name")
            or col_meta.get("column_name")
            or col_meta.get("Column Name")
        )
        if not name:
            continue

        ctype = (col_meta.get("type") or col_meta.get("column_type") or "").lower()

        if ctype == "id":
            id_cols.append(name)
            continue
        if ctype == "kpi":
            kpi_cols.append(name)
            continue

        if ctype == "categorical":
            cat_cols.append(name)
        elif ctype == "continuous":
            cont_cols.append(name)
        else:
            if name in ddf.columns:
                dtype_val = ddf[name].dtype
            else:
                dtype_val = (
                    col_meta.get("dtype") or col_meta.get("data_type") or "STRING"
                )

            if not _dtype_is_numeric(dtype_val):
                cat_cols.append(name)
            else:
                uniq = col_meta.get("unique_count") or col_meta.get("nunique")
                try:
                    uniq_int = int(uniq) if uniq is not None else None
                except Exception:
                    uniq_int = None

                card = (
                    uniq_int
                    if uniq_int is not None
                    else _approx_nunique_dask(ddf, name, n_rows=n_rows)
                )
                if int(card) <= int(max_card_for_cat):
                    cat_cols.append(name)
                else:
                    cont_cols.append(name)

        if name in cat_cols:
            uniq = col_meta.get("unique_count") or col_meta.get("nunique")
            try:
                cat_cards[name] = int(uniq)
            except Exception:
                cat_cards[name] = _approx_nunique_dask(ddf, name, n_rows=n_rows)

    def dedupe_keep_order(xs: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return (
        dedupe_keep_order(id_cols),
        dedupe_keep_order(kpi_cols),
        dedupe_keep_order(cat_cols),
        dedupe_keep_order(cont_cols),
        cat_cards,
    )


# ---------------------------------------------------------------------------
# Suggest functions (ML heuristics)
# ---------------------------------------------------------------------------
def _next_pow2(x: int) -> int:
    """Get next power of 2."""
    x = int(max(1, x))
    p = 1
    while p < x:
        p *= 2
    return p


def suggest_autoencoder_dims(
    n_features: int,
    *,
    max_width: int = 1024,
    max_latent: int = 128,
    min_latent: int = 16,
    max_layers: int = 4,
) -> Tuple[List[int], List[int], int]:
    """Suggest encoder/decoder hidden dims + latent_dim for a DAE."""
    n_features = max(1, int(n_features))

    latent_target = int(round(4.0 * math.sqrt(n_features)))
    latent_dim = _next_pow2(latent_target)
    latent_dim = max(min_latent, min(max_latent, latent_dim))

    top_width = min(max_width, max(128, 4 * latent_dim))
    top_width = _next_pow2(top_width)

    encoder: List[int] = []
    w = top_width
    while w > latent_dim and len(encoder) < max_layers:
        encoder.append(int(w))
        next_w = int(w // 2)
        if next_w <= latent_dim:
            break
        w = next_w

    if not encoder:
        encoder = [int(top_width)]

    decoder = list(reversed(encoder))
    return encoder, decoder, int(latent_dim)


def suggest_feature_embed_dim(
    cat_cardinalities: Sequence[int],
    *,
    default_if_unknown: int = 8,
    max_embed_dim: int = 32,
) -> int:
    """Global embedding dim for categorical features."""
    cards = [int(c) for c in cat_cardinalities if c is not None and int(c) > 0]
    if not cards:
        return int(default_if_unknown)

    med = statistics.median(cards)

    if med <= 10:
        return 4
    if med <= 50:
        return 8
    if med <= 200:
        return 16
    return int(max_embed_dim)


def suggest_corruption_probs(
    *,
    n_features: int,
    cat_cardinalities: Sequence[int],
) -> Tuple[float, float]:
    """Swap corruption probabilities for numeric and categorical inputs."""
    n_features = max(1, int(n_features))

    num_swap_prob = 0.35 if n_features >= 50 else 0.30

    cards = [int(c) for c in cat_cardinalities if c is not None and int(c) > 0]
    med_card = statistics.median(cards) if cards else 0

    if med_card >= 1000:
        cat_swap_prob = 0.20
    elif med_card >= 100:
        cat_swap_prob = 0.30
    else:
        cat_swap_prob = 0.35

    return float(num_swap_prob), float(cat_swap_prob)


def suggest_backbone_type(
    *,
    n_rows: int,
    n_features: int,
    n_categorical: int,
) -> str:
    """Pick a backbone for the DAE."""
    n_rows = max(1, int(n_rows))
    n_features = max(1, int(n_features))
    n_categorical = max(0, int(n_categorical))
    cat_ratio = n_categorical / float(n_features)

    if n_rows >= 1_000_000 and n_features >= 200 and cat_ratio >= 0.25:
        return "ft_transformer"
    return "mlp"


def suggest_batch_size(
    *,
    use_gpu: bool,
    n_features: int,
    n_rows: int,
    device_mem_gb: float | None = None,
    target_steps_per_epoch: Tuple[int, int] = (10, 20),
) -> int:
    """Batch-size heuristic based on memory and steps per epoch."""
    n_rows = max(1, int(n_rows))
    n_features = max(1, int(n_features))

    min_steps, max_steps = sorted([int(x) for x in target_steps_per_epoch])
    min_steps = max(1, min_steps)
    max_steps = max(min_steps, max_steps)
    steps_mid = int(round((min_steps + max_steps) / 2.0))

    bs_target = max(1, int(round(n_rows / float(steps_mid))))
    bs_min = max(1, int(math.floor(n_rows / float(max_steps))))
    bs_max = max(1, int(math.ceil(n_rows / float(min_steps))))
    bs_by_steps = max(bs_min, min(bs_target, bs_max))

    if not use_gpu:
        base_cap = 256
    else:
        base_cap = 2048
        if device_mem_gb:
            base_cap = int(round(base_cap * (device_mem_gb / 16.0)))

    if n_features > 200:
        base_cap = int(base_cap * 0.5)
    if n_features > 500:
        base_cap = int(base_cap * 0.5)

    base_cap = max(1, int(base_cap))
    return int(min(bs_by_steps, base_cap))


def suggest_scheduler(
    *,
    use_gpu: bool,
    steps_per_epoch: int,
) -> str:
    """Choose a LR scheduler based on step count and hardware."""
    steps_per_epoch = max(1, int(steps_per_epoch))
    if not use_gpu:
        return "cosine"
    if steps_per_epoch < 10:
        return "cosine"
    return "onecycle"


def suggest_learning_rate(
    *,
    batch_size: int,
    use_gpu: bool,
    scheduler: str,
) -> float:
    """Learning-rate heuristic aligned with the chosen scheduler."""
    if not use_gpu:
        return 1e-3
    bs = max(1, int(batch_size))
    scheduler = (scheduler or "").lower()
    base = 4e-3 if scheduler == "onecycle" else 2e-3
    lr = base * math.sqrt(bs / 2048.0)
    return float(max(5e-4, min(8e-3, lr)))


def suggest_compute_stats_plan(n_rows: int) -> Tuple[str, int]:
    """How to compute dataset stats."""
    n_rows = max(1, int(n_rows))
    if n_rows <= 1_000_000:
        return "full", 5
    if n_rows <= 20_000_000:
        return "sample", 10
    return "sample", 10


def suggest_target_segments(n_rows: int, target_range: Tuple[int, int]) -> int:
    """Pick a midpoint target segment count based on dataset size."""
    low, high = target_range
    n_rows = max(1, int(n_rows))
    if n_rows < 200_000:
        return max(low, min(high, 8))
    if n_rows < 5_000_000:
        return max(low, min(high, 12))
    return max(low, min(high, 16))


def suggest_segmenter_hparams(
    n_rows: int,
    *,
    target_segments_range: Tuple[int, int] = (5, 20),
) -> Dict[str, Any]:
    """Heuristics for UMAP + HDBSCAN segmentation."""
    n_rows = max(1, int(n_rows))
    target_mid = suggest_target_segments(n_rows, target_segments_range)

    min_cluster_threshold = max(30, int(n_rows / float(target_mid)))
    min_cluster_size = max(30, int(n_rows / float(target_mid * 6)))
    min_samples = max(10, int(min_cluster_size / 3))

    if n_rows < 100_000:
        n_neighbors = 30
        n_components = 8
    elif n_rows < 1_000_000:
        n_neighbors = 40
        n_components = 10
    else:
        n_neighbors = 45
        n_components = 15

    sample_target = 2_000_000
    if n_rows <= sample_target:
        sample_hdbscan = 1.0
    else:
        sample_hdbscan = min(1.0, sample_target / float(n_rows))
        sample_hdbscan = max(sample_hdbscan, 0.002)

    if n_rows >= 10_000_000:
        soft_batch = 100_000
    elif n_rows >= 1_000_000:
        soft_batch = 50_000
    else:
        soft_batch = None

    return {
        "cluster_selection_method": "eom",
        "min_cluster_size": int(min_cluster_size),
        "min_cluster_pct": None,
        "min_cluster_threshold": int(min_cluster_threshold),
        "min_samples": int(min_samples),
        "min_samples_pct": None,
        "n_neighbors": int(n_neighbors),
        "n_components": int(n_components),
        "min_dist": 0.0,
        "metric": "euclidean",
        "sample_hdbscan": float(sample_hdbscan),
        "prediction_data": True,
        "soft_clustering_batch_size": soft_batch,
        "noise_threshold": 0.05 if n_rows >= 1_000_000 else None,
        "nnd_n_clusters": 4,
        "nnd_overlap_factor": 2,
        "knn_n_clusters": 4,
        "knn_overlap_factor": 2,
    }


def suggest_explainability_hparams(n_features: int) -> Dict[str, int]:
    """Heuristics for explainability module."""
    n_features = max(1, int(n_features))
    num_features = int(min(100, max(25, round(0.5 * n_features))))
    top_n = int(min(num_features, max(10, round(0.2 * num_features))))
    return {"num_features": num_features, "top_n": top_n}


# ---------------------------------------------------------------------------
# Main config builder
# ---------------------------------------------------------------------------
def build_pretty_config_from_data_dict(
    data_dict: Dict[str, Any],
    ddf: dd.DataFrame,
    *,
    use_gpu: bool = True,
    use_wandb: bool = False,
    config_debug: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_mode: str | None = None,
    delete_existing_artifacts: bool = False,
    robust_scaler: bool = False,
    max_card_for_cat: int = 20,
    max_epochs: int = 50,
    backbone_type: str | None = None,
    scheduler: str | None = None,
    optimizer: str = "adam",
    device_mem_gb: float | None = None,
    target_steps_per_epoch: Tuple[int, int] = (10, 20),
    target_segments_range: Tuple[int, int] = (5, 20),
    return_rationale: bool = False,
) -> Any:
    """Build a schema-aligned config for Neuralift segmentation."""
    n_rows = _safe_row_count(ddf)
    logger.info("[config] Building segmenter config for %d rows", n_rows)

    id_cols, kpi_cols, cat_cols, cont_cols, cat_cards_by_col = (
        infer_column_roles_from_data_dict(
            data_dict, ddf, max_card_for_cat=max_card_for_cat
        )
    )
    n_features = int(len(cat_cols) + len(cont_cols))
    logger.info(
        "[config] Inferred roles: %d id, %d kpi, %d cat, %d cont columns",
        len(id_cols),
        len(kpi_cols),
        len(cat_cols),
        len(cont_cols),
    )

    encoder_hidden_dims, decoder_hidden_dims, latent_dim = suggest_autoencoder_dims(
        n_features
    )

    feature_embed_dim = suggest_feature_embed_dim(list(cat_cards_by_col.values()))
    num_swap_prob, cat_swap_prob = suggest_corruption_probs(
        n_features=n_features, cat_cardinalities=list(cat_cards_by_col.values())
    )

    if device_mem_gb is None:
        device_mem_gb = _read_env_float("NL_DEVICE_MEM_GB") or _read_env_float(
            "NL_GPU_MEM_GB"
        )

    batch_size = suggest_batch_size(
        use_gpu=use_gpu,
        n_features=n_features,
        n_rows=n_rows,
        device_mem_gb=device_mem_gb,
        target_steps_per_epoch=target_steps_per_epoch,
    )
    steps_per_epoch = max(1, int(math.ceil(n_rows / float(batch_size))))
    if scheduler is None:
        scheduler = suggest_scheduler(use_gpu=use_gpu, steps_per_epoch=steps_per_epoch)

    learning_rate = suggest_learning_rate(
        batch_size=batch_size, use_gpu=use_gpu, scheduler=scheduler
    )

    compute_stats_from, num_sample_partitions = suggest_compute_stats_plan(n_rows)

    if backbone_type is None:
        backbone_type = suggest_backbone_type(
            n_rows=n_rows, n_features=n_features, n_categorical=len(cat_cols)
        )

    primary_width = encoder_hidden_dims[0] if encoder_hidden_dims else 0
    gradient_checkpointing = bool(
        use_gpu and (primary_width >= 512 or n_features > 300)
    )

    use_wandb_flag = bool(use_wandb)
    labels_file_name: str | None = "labels.npy"
    ranked_points_file_name: str | None = "precomputed_points.npy"
    if use_wandb_flag:
        labels_file_name = None
        ranked_points_file_name = None

    seg = suggest_segmenter_hparams(n_rows, target_segments_range=target_segments_range)

    explain = suggest_explainability_hparams(n_features)
    default_output_path = SEGMENTER_CONFIG_DEFAULTS["output_path"]

    logger.info(
        "[config] Suggested: batch_size=%d, lr=%.6f, latent_dim=%d, backbone=%s",
        batch_size,
        learning_rate,
        latent_dim,
        backbone_type,
    )

    config: Dict[str, Any] = {
        "use_gpu": bool(use_gpu),
        "is_container": False,
        "use_wandb": use_wandb_flag,
        "wandb": {
            "project": wandb_project,
        },
        "input_uri": None,
        "input_path": None,
        "config_file_name": "config.yaml",
        "output_path": default_output_path,
        "labels_file_name": labels_file_name,
        "ranked_points_file_name": ranked_points_file_name,
        "verbose": 0,
        "headless": False,
        "resume": False,
        "json_logging": None,
        "data": {
            "sample_frac": None,
        },
        "dae": {
            "matmul_precision": "high" if use_gpu else "medium",
            "dataset_stats_file_name": "dataset_stats.joblib",
            "estimate_batch_size": True,
            "rmm_allocator": bool(use_gpu),
            "compile": False,
            "scale_batch_size": False,
            "weight_averaging": True,
            "data_module": {
                "val_split": 0.2,
                "batch_size": int(batch_size),
                "compute_stats_from": str(compute_stats_from),
                "num_sample_partitions": int(num_sample_partitions),
                "optimize_memory": False,
            },
            "model": {
                "learning_rate": float(learning_rate),
                "backbone_type": str(backbone_type),
                "encoder_hidden_dims": [int(x) for x in encoder_hidden_dims],
                "decoder_hidden_dims": [int(x) for x in decoder_hidden_dims],
                "latent_dim": int(latent_dim),
                "feature_embed_dim": int(feature_embed_dim),
                "num_swap_prob": float(num_swap_prob),
                "cat_swap_prob": float(cat_swap_prob),
                "scheduler": str(scheduler),
                "optimizer": str(optimizer),
                "gradient_checkpointing": bool(gradient_checkpointing),
                "use_sparse_categorical": False,
                "use_grouped_categorical_head": False,
                "boolean_cardinality_threshold": 2,
                "use_mixed_categorical": True,
                "max_onehot_cardinality": 4,
                "batch_norm_continuous": True,
                "batch_norm_embeddings": True,
                "embedding_dropout": 0.1,
                "robust_scaler": bool(robust_scaler),
            },
            "trainer": {
                "max_epochs": int(max_epochs),
                "accelerator": "auto",
                "precision": "bf16-mixed" if use_gpu else "32",
                "fast_dev_run": False,
                "gradient_clip_val": 1.0,
                "devices": -1 if use_gpu else 1,
                "enable_model_summary": True,
                "sync_batchnorm": False,
            },
            "distributed": {
                "enabled": False,
                "backend": "nccl",
            },
        },
        "segmenter": {
            "cluster_selection_method": seg["cluster_selection_method"],
            "min_cluster_pct": seg["min_cluster_pct"],
            "min_cluster_size": seg["min_cluster_size"],
            "min_cluster_threshold": seg["min_cluster_threshold"],
            "min_samples": seg["min_samples"],
            "min_samples_pct": seg["min_samples_pct"],
            "min_dist": seg["min_dist"],
            "soft_clustering_batch_size": seg["soft_clustering_batch_size"],
            "noise_threshold": seg["noise_threshold"],
            "n_neighbors": seg["n_neighbors"],
            "n_components": seg["n_components"],
            "nnd_n_clusters": seg["nnd_n_clusters"],
            "nnd_overlap_factor": seg["nnd_overlap_factor"],
            "knn_n_clusters": seg["knn_n_clusters"],
            "knn_overlap_factor": seg["knn_overlap_factor"],
            "metric": seg["metric"],
            "prediction_data": seg["prediction_data"],
        },
        "explainability": {
            "num_features": int(explain["num_features"]),
            "top_n": int(explain["top_n"]),
        },
    }

    # Rationale for YAML comments
    rationale: Dict[str, str] = {
        "use_gpu": "Enable GPU acceleration (A10-class GPUs+ support fast bf16 mixed precision).",
        "use_wandb": "Track runs/metrics/artifacts; disable for air-gapped/offline runs.",
        "labels_file_name": "Standard output filename for final HDBSCAN labels; omitted when W&B is enabled.",
        "ranked_points_file_name": "Standard output filename for precomputed or ranked embedding points; omitted when W&B is enabled.",
        "data.sample_frac": "Optional subsample ratio; leave unset to use the full dataset.",
        "dae.matmul_precision": "Use high matmul precision on GPU for stable/faster tensor cores; medium on CPU.",
        "dae.dataset_stats_file_name": "Persist dataset scaling/statistics for reproducible transforms.",
        "dae.estimate_batch_size": "Let the trainer shrink batch size automatically if out-of-memory.",
        "dae.rmm_allocator": "Use RAPIDS RMM allocator on GPU runs to reduce fragmentation when cuDF/RAPIDS is in play.",
        "dae.compile": "torch.compile can speed training, but is kept off by default for maximum stability.",
        "dae.scale_batch_size": "Disable implicit batch scaling; use explicit batch size with memory + steps/epoch heuristics.",
        "dae.weight_averaging": "Weight averaging can smooth training; disable if you need strict checkpoint comparability.",
        "dae.data_module.val_split": "20% validation split is a robust default for unsupervised early sanity checks.",
        "dae.data_module.batch_size": "Batch size targets 10-20 steps/epoch and is capped by device memory and table width.",
        "dae.data_module.compute_stats_from": "Use full stats for small data; sample stats for large data to avoid expensive full scans.",
        "dae.data_module.num_sample_partitions": "Number of partitions sampled to estimate dataset stats on large tables.",
        "dae.data_module.optimize_memory": "Reduce memory pressure during data module setup when enabled.",
        "dae.model.learning_rate": "LR scales ~sqrt(batch_size) with scheduler-specific base (Smith 2017).",
        "dae.model.scheduler": "OneCycle for GPU runs with enough steps/epoch; cosine otherwise.",
        "dae.model.optimizer": "Adam is a stable baseline; switch to AdamW/SGD if you need different regularization.",
        "dae.model.backbone_type": "MLP default; switch to FT-Transformer for large, wide, categorical-heavy tables.",
        "dae.model.encoder_hidden_dims": "Geometric compression toward latent space; latent_dim is not part of hidden dims.",
        "dae.model.decoder_hidden_dims": "Symmetric decoder mirroring encoder hidden dims.",
        "dae.model.latent_dim": "Latent dimension scales with sqrt(feature_count) to balance compression and capacity.",
        "dae.model.feature_embed_dim": "Embedding dimension chosen from median categorical cardinality (Guo & Berkhahn 2016).",
        "dae.model.num_swap_prob": "Numeric corruption probability for denoising objective (Vincent et al. 2008).",
        "dae.model.cat_swap_prob": "Categorical corruption probability; reduced when categorical cardinalities are very high.",
        "dae.model.gradient_checkpointing": "Checkpointing trades compute for memory; useful for big batches/width (Chen et al. 2016).",
        "dae.model.use_sparse_categorical": "Disable sparse categorical path by default; enable only if your model supports it.",
        "dae.model.use_grouped_categorical_head": "Group categorical reconstruction heads to reduce parameters and overfitting risk.",
        "dae.model.boolean_cardinality_threshold": "Treat boolean-like features as categoricals with cardinality <= 2.",
        "dae.model.robust_scaler": "Robust scaling is resistant to heavy-tailed numeric features common in marketing/adtech fact tables.",
        "dae.trainer.max_epochs": "50 epochs is a stable default; reduce for fast iteration, increase for harder domains.",
        "dae.trainer.accelerator": "Auto accelerator chooses GPU when available.",
        "dae.trainer.precision": "bf16-mixed is a strong GPU default; BF16 has FP32-like exponent range (Kalamkar et al. 2019).",
        "dae.trainer.gradient_clip_val": "Clip gradients to stabilize training under noise and large LR schedules.",
        "dae.trainer.devices": "Use all visible GPUs for throughput; set to 1 to debug deterministically.",
        "dae.trainer.enable_model_summary": "Keep model summary for sanity checking layer sizes.",
        "dae.distributed.enabled": "Disabled by default; enable when running multi-node DDP explicitly.",
        "dae.distributed.backend": "NCCL is standard for multi-GPU NVIDIA distributed training.",
        "segmenter.cluster_selection_method": "EOM tends to produce a stable set of clusters; 'leaf' often yields more clusters.",
        "segmenter.min_cluster_size": "Primary HDBSCAN knob: smallest grouping considered a cluster (HDBSCAN docs).",
        "segmenter.min_cluster_pct": "Pct version of min_cluster_size for readability across table sizes.",
        "segmenter.min_cluster_threshold": "Pipeline-level threshold to steer toward ~5-20 segments.",
        "segmenter.min_samples": "Density conservativeness; set lower than min_cluster_size to reduce outliers.",
        "segmenter.min_samples_pct": "Pct version of min_samples for readability.",
        "segmenter.n_neighbors": "UMAP neighborhood size; larger values emphasize global structure (UMAP docs).",
        "segmenter.min_dist": "UMAP min_dist=0 makes clumpier embeddings, often better for clustering (UMAP docs).",
        "segmenter.n_components": "Cluster in 5-10D UMAP space to preserve structure while avoiding high-dim clustering.",
        "segmenter.metric": "Euclidean is appropriate for DAE latent spaces after scaling.",
        "segmenter.prediction_data": "Required for fast approximate_predict labeling of new points (HDBSCAN docs).",
        "segmenter.soft_clustering_batch_size": "Chunk membership computations to avoid OOM on very large N.",
        "segmenter.noise_threshold": "Low threshold keeps most points assigned while reserving a noise tail.",
        "segmenter.nnd_n_clusters": "Keep schema default unless you have an overlap strategy.",
        "segmenter.nnd_overlap_factor": "Keep schema default unless you have an overlap strategy.",
        "segmenter.knn_n_clusters": "Keep schema default unless you have an overlap strategy.",
        "segmenter.knn_overlap_factor": "Keep schema default unless you have an overlap strategy.",
        "explainability.num_features": "Number of top features to compute SHAP/global importances over (tradeoff: time vs detail).",
        "explainability.top_n": "How many top features to surface in reports.",
        "wandb.project": "W&B project name.",
    }

    if not config_debug:
        config = _strip_nulls_and_defaults(config, SEGMENTER_CONFIG_DEFAULTS)
        rationale = _filter_rationale(rationale, config)

    if return_rationale:
        return config, rationale
    return config


# ---------------------------------------------------------------------------
# YAML rendering and output
# ---------------------------------------------------------------------------
def render_config_yaml_with_comments(
    config: Dict[str, Any],
    *,
    header_comment_lines: Sequence[str] | None = None,
    rationale: Mapping[str, str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> str:
    """Render YAML with a top-of-file comment block."""
    header_comment_lines = list(header_comment_lines or [])
    rationale = dict(rationale or {})

    lines: List[str] = []
    for ln in header_comment_lines:
        ln = str(ln).rstrip("\n")
        if not ln.startswith("#"):
            ln = "# " + ln
        lines.append(ln)

    if rationale:
        lines.append("#")
        lines.append(
            "# -----------------------------------------------------------------------------"
        )
        lines.append("# HEURISTIC NOTES (why these defaults)")
        lines.append(
            "# -----------------------------------------------------------------------------"
        )
        for k in sorted(rationale.keys()):
            msg = str(rationale[k]).replace("\n", " ").strip()
            lines.append(f"# {k}: {msg}")
        lines.append(
            "# -----------------------------------------------------------------------------"
        )
        lines.append("#")

    yaml_text = yaml.safe_dump(
        config,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
    yaml_text = _annotate_yaml_with_defaults(
        yaml_text,
        config=config,
        defaults=defaults,
    )
    lines.append(yaml_text.rstrip("\n"))
    lines.append("")  # final newline
    return "\n".join(lines)


def print_config_yaml(config: Dict[str, Any]) -> None:
    """Print config as YAML to stdout."""
    text = yaml.safe_dump(
        config,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
    logger.info("[config] Generated YAML:\n%s", text)


def save_config_yaml(
    config: Dict[str, Any],
    path: Union[str, Path],
    *,
    header_comment_lines: Sequence[str] | None = None,
    rationale: Mapping[str, str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> None:
    """Save config YAML to file."""
    path = Path(path)
    if header_comment_lines or rationale or defaults:
        text = render_config_yaml_with_comments(
            config,
            header_comment_lines=header_comment_lines,
            rationale=rationale,
            defaults=defaults,
        )
    else:
        text = yaml.safe_dump(
            config, sort_keys=False, default_flow_style=False, allow_unicode=False
        )
    path.write_text(text)
    logger.info("[config] Saved config to %s", path)


__all__ = [
    "SEGMENTER_CONFIG_DEFAULTS",
    "infer_column_roles_from_data_dict",
    "suggest_autoencoder_dims",
    "suggest_feature_embed_dim",
    "suggest_corruption_probs",
    "suggest_backbone_type",
    "suggest_batch_size",
    "suggest_scheduler",
    "suggest_learning_rate",
    "suggest_compute_stats_plan",
    "suggest_target_segments",
    "suggest_segmenter_hparams",
    "suggest_explainability_hparams",
    "build_pretty_config_from_data_dict",
    "render_config_yaml_with_comments",
    "print_config_yaml",
    "save_config_yaml",
]
