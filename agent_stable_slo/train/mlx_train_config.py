"""Validated config for MLX-based GRPO LoRA training on Apple Silicon.

Mirrors GRPOTrainConfig from agent_stable_slo.utils.config with
MLX-specific fields (lora_layers, adapter_path, etc.) instead of
CUDA/BitsAndBytes settings.
"""

from __future__ import annotations

import os
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


MLX_CONFIG_VERSION = "0.1.0"


class MLXTrainConfig(BaseModel):
    """Validated config for MLX GRPO-style LoRA training."""

    model_config = ConfigDict(extra="forbid")

    # Model
    base_model: str = Field(
        description="MLX model path (HuggingFace ID or local mlx-community path)."
    )
    adapter_path: str = Field(
        default="",
        description="LoRA adapter output path. Auto-generated if empty.",
    )

    # GRPO
    group_size: int = Field(default=4, gt=0, le=16, description="GRPO group size.")
    beta: float = Field(default=0.1, ge=0.0, description="KL penalty coefficient.")

    # Optimiser
    learning_rate: float = Field(default=1e-5, gt=0)
    num_steps: int = Field(default=1000, gt=0, le=20000)

    # LoRA
    lora_rank: int = Field(default=8, gt=0)
    lora_layers: int = Field(
        default=16,
        gt=0,
        description="Number of transformer layers to apply LoRA to.",
    )

    # Memory
    grad_checkpoint: bool = Field(
        default=True,
        description="Enable gradient checkpointing to reduce memory.",
    )

    # Generation
    max_tokens: int = Field(default=256, gt=8, le=1024)
    batch_size: int = Field(default=1, gt=0)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)

    # Tasks
    tasks: List[str] = Field(description="Task JSONL file paths.")
    schema_paths: List[str] = Field(
        default_factory=list,
        description="Schema JSON paths (auto-discovered from tasks if empty).",
    )

    # Reproducibility
    seed: int = Field(default=42)

    # Reward weights (match P2 defaults)
    lam_latency: float = Field(default=0.1, ge=0.0)
    mu_cost: float = Field(default=0.01, ge=0.0)
    gamma_stability: float = Field(default=0.0, ge=0.0)

    # Logging / checkpointing
    log_path: str = Field(
        default="",
        description="JSONL log output path. Auto-generated if empty.",
    )
    checkpoint_every: int = Field(
        default=100,
        ge=0,
        description="Save LoRA adapter every N steps (0 = end only).",
    )
    eval_interval: int = Field(default=50, gt=0)

    # Version
    config_version: str = Field(default=MLX_CONFIG_VERSION, frozen=True)

    # ------------------------------------------------------------------ #
    # Validators                                                          #
    # ------------------------------------------------------------------ #

    @field_validator("tasks")
    @classmethod
    def _tasks_exist(cls, v: List[str]) -> List[str]:
        for path in v:
            if not os.path.exists(path):
                raise ValueError(f"task file not found: {path}")
        return v

    @field_validator("adapter_path")
    @classmethod
    def _ensure_adapter_path(cls, v: str) -> str:
        return v or ""

    @field_validator("log_path")
    @classmethod
    def _ensure_log_path(cls, v: str) -> str:
        return v or ""

    @field_validator("checkpoint_every")
    @classmethod
    def _checkpoint_reasonable(cls, v: int) -> int:
        if v == 1:
            raise ValueError(
                "checkpoint_every=1 is not allowed; use >=5 to limit IO churn."
            )
        return v
