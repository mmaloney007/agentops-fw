"""Config schema for the local LLM + reasoning training stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BackendName = Literal["cuda", "mlx"]


class ReasoningStackConfig(BaseModel):
    """Validated runtime config for two-stage local training."""

    model_config = ConfigDict(extra="forbid")

    backend: BackendName = Field(default="cuda")
    run_name: str = Field(default="reasoning_stack")
    out_dir: str = Field(default="out/reasoning_stack")
    seed: int = Field(default=17)

    tokenizer_name: str = Field(default="gpt2")
    pretrain_text_files: List[str] = Field(default_factory=list)
    reasoning_task_files: List[str] = Field(default_factory=lambda: ["tasks/robust_eval_gold.jsonl"])
    eval_task_files: List[str] = Field(default_factory=lambda: ["tasks/robust_eval_gold.jsonl"])
    max_reasoning_examples: int = Field(default=5000, gt=0)
    max_eval_examples: int = Field(default=500, gt=0)

    sequence_length: int = Field(default=256, ge=64, le=2048)
    micro_batch_size: int = Field(default=8, gt=0)
    grad_accum_steps: int = Field(default=2, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    warmup_steps: int = Field(default=50, ge=0)
    pretrain_steps: int = Field(default=1500, gt=0)
    reasoning_steps: int = Field(default=800, gt=0)
    log_every: int = Field(default=25, gt=0)

    enable_rl_stage: bool = Field(default=True)
    rl_steps: int = Field(default=600, gt=0)
    rl_group_size: int = Field(default=4, ge=2, le=16)
    rl_max_new_tokens: int = Field(default=96, ge=16, le=512)
    rl_temperature: float = Field(default=0.7, ge=0.0)
    rl_top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    rl_log_every: int = Field(default=20, gt=0)
    lam_latency: float = Field(default=0.1, ge=0.0)
    mu_cost: float = Field(default=0.01, ge=0.0)
    gamma_stability: float = Field(default=0.0, ge=0.0)

    model_hidden_size: int = Field(default=512, ge=128)
    model_layers: int = Field(default=8, ge=2)
    model_heads: int = Field(default=8, ge=2)
    model_ffn_mult: int = Field(default=4, ge=2)
    model_dropout: float = Field(default=0.1, ge=0.0, le=0.5)

    mlx_base_model: str = Field(default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    mlx_adapter_rank: int = Field(default=16, gt=0)
    mlx_lora_layers: int = Field(default=16, gt=0)
    mlx_learning_rate: float = Field(default=1e-5, gt=0)
    command_timeout_sec: int = Field(default=0, ge=0)
    dry_run: bool = Field(default=False)

    @field_validator("pretrain_text_files", "reasoning_task_files", "eval_task_files")
    @classmethod
    def _paths_must_exist(cls, paths: List[str]) -> List[str]:
        for raw in paths:
            if not Path(raw).exists():
                raise ValueError(f"file not found: {raw}")
        return paths

    @model_validator(mode="after")
    def _shape_constraints(self):
        if self.model_hidden_size % self.model_heads != 0:
            raise ValueError("model_hidden_size must be divisible by model_heads")
        if self.backend == "cuda" and not self.pretrain_text_files:
            raise ValueError("CUDA path requires pretrain_text_files for stage-1 LM training")
        return self

    def stage_dir(self, stage_name: str) -> Path:
        return Path(self.out_dir) / self.run_name / stage_name


def _resolve_file_list(items: Any, base_dir: Path) -> List[str]:
    if not isinstance(items, list):
        raise ValueError("Expected a list of file paths")

    resolved: List[str] = []
    for item in items:
        path = Path(str(item))
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        resolved.append(str(path))
    return resolved


def load_reasoning_stack_config(config_path: str) -> ReasoningStackConfig:
    """Load YAML config and resolve file paths relative to the config file."""

    cfg_path = Path(config_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base_dir = cfg_path.parent
    payload: Dict[str, Any] = dict(raw)

    for key in ("pretrain_text_files", "reasoning_task_files", "eval_task_files"):
        if key in payload:
            payload[key] = _resolve_file_list(payload[key], base_dir)

    out_dir = Path(payload.get("out_dir", "out/reasoning_stack"))
    if not out_dir.is_absolute():
        payload["out_dir"] = str((base_dir / out_dir).resolve())

    return ReasoningStackConfig(**payload)
