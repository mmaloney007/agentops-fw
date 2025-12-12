from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


CONFIG_VERSION = "0.2.0"
CRITICAL_DEFAULTS = {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "tasks": "tasks/robust_eval_gold.jsonl",
    "lr": 1e-5,
    "steps": 500,
    "max_prompt_len": 1024,
    "max_new_tokens": 128,
}


class GRPOTrainConfig(BaseModel):
    """Validated config for GRPO-style LoRA training."""

    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct")
    tasks: str = Field(default="tasks/robust_eval_gold.jsonl")
    out: str = Field(default="")
    steps: int = Field(default=500, gt=0, le=20000)
    max_prompt_len: int = Field(default=1024, gt=32)
    max_new_tokens: int = Field(default=128, gt=8, le=1024)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    lr: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    gradient_accumulation: int = Field(default=1, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    lora_rank: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    lora_targets: str = Field(default="")
    load_in_4bit: Optional[bool] = Field(default=None)
    torch_dtype: str = Field(default="float16")
    eval_interval: int = Field(default=50, gt=0)
    deterministic: bool = Field(default=False)
    force_json_fallback: bool = Field(default=False)
    lam_latency: float = Field(default=0.0)
    mu_cost: float = Field(default=0.0)
    gamma_stability: float = Field(default=0.0)
    seed: int = Field(default=17)
    repro: bool = Field(default=False, description="Enforce deterministic ops where available.")
    cache_dataset: bool = Field(default=False)
    cache_dir: str = Field(default="out/cache")
    checkpoint_every: int = Field(default=0, ge=0)
    resume_from: Optional[str] = Field(default=None)
    no_silent_defaults: bool = Field(default=False, description="Require explicit overrides for critical hyperparams.")
    expected_dataset_hash: Optional[str] = Field(default=None, description="If set, validate dataset hash on load.")
    allow_dataset_drift: bool = Field(default=False, description="Allow dataset hash mismatches without aborting.")
    config_preset: Optional[str] = Field(default=None, description="Optional preset name used to build this config.")
    val_tasks: Optional[str] = Field(default=None, description="Optional validation tasks file.")
    val_interval: int = Field(default=0, ge=0, description="Run validation every N steps (0 = disable).")
    val_samples: int = Field(default=1, ge=1, description="Samples per validation task; best reward kept.")
    max_prompt_chars: int = Field(default=0, ge=0, description="If >0, truncate or reject prompts longer than this many characters.")
    truncate_prompts: bool = Field(default=False, description="If true, truncate prompts exceeding max_prompt_chars; otherwise raise.")
    blocklist: Optional[str] = Field(default=None, description="Comma-separated substrings to reject from prompts/outputs.")
    reject_blocklisted: bool = Field(default=False, description="If true, zero reward and mark invalid when blocklisted substrings appear.")
    ddp_backend: Optional[str] = Field(default=None, description="Optional torch.distributed backend (e.g., nccl, gloo).")
    config_version: str = Field(default=CONFIG_VERSION, frozen=True)

    @field_validator("tasks")
    @classmethod
    def _tasks_exist(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"tasks file not found: {v}")
        return v

    @field_validator("out")
    @classmethod
    def _ensure_out(cls, v: str) -> str:
        return v or ""

    @field_validator("checkpoint_every")
    @classmethod
    def _checkpoint_reasonable(cls, v: int) -> int:
        if v == 1:
            # Avoid excessive IO by discouraging per-step checkpoints
            raise ValueError("checkpoint_every=1 is not allowed; use >=5 to limit IO churn.")
        return v

    @field_validator("cache_dir")
    @classmethod
    def _expand_cache_dir(cls, v: str) -> str:
        return os.path.expanduser(v)

    @model_validator(mode="after")
    def _no_silent_defaults(self):
        if self.no_silent_defaults:
            for field, default_val in CRITICAL_DEFAULTS.items():
                if getattr(self, field) == default_val:
                    raise ValueError(f"critical param '{field}' left at default; set explicitly or disable --no-silent-defaults")
        return self

    @field_validator("val_tasks")
    @classmethod
    def _val_tasks_exist(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not os.path.exists(v):
            raise ValueError(f"val_tasks file not found: {v}")
        return v

    @field_validator("max_prompt_chars")
    @classmethod
    def _prompt_limit_reasonable(cls, v: int) -> int:
        return v


@dataclass
class DatasetFingerprint:
    tasks_path: str
    sha256: str
    num_records: int
    schema_sha256: Dict[str, str]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_or_raise(cfg: Dict[str, Any]) -> GRPOTrainConfig:
    try:
        migrated = migrate_config(cfg)
        return GRPOTrainConfig(**migrated)
    except ValidationError as exc:  # pragma: no cover - exercised in CLI use
        raise SystemExit(f"[config] invalid config:\n{exc}")


def migrate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort config migration. Today we only ensure config_version is populated;
    future versions can transform fields here before validation.
    """
    cfg = dict(cfg)
    cfg.setdefault("config_version", CONFIG_VERSION)
    return cfg
