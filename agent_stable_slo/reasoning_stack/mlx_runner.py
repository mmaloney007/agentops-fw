"""MLX command-plan runner using practical `mlx_lm` + local GRPO script flows."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def build_mlx_commands(
    *,
    config,
    stage1_data_dir: str,
    stage2_data_dir: str,
) -> List[List[str]]:
    """Build shell-safe commands for staged MLX training."""

    stage1_adapter = config.stage_dir("mlx_stage1_adapter")
    stage1_fused = config.stage_dir("mlx_stage1_fused")
    stage2_adapter = config.stage_dir("mlx_stage2_adapter")
    stage2_fused = config.stage_dir("mlx_stage2_fused")
    stage3_dir = config.stage_dir("mlx_stage3_rl")

    repo_root = Path(__file__).resolve().parents[2]
    rl_script = repo_root / "scripts" / "reasoning" / "run_mlx_grpo_stage.py"

    commands: List[List[str]] = [
        [
            "python",
            "-m",
            "mlx_lm.lora",
            "--train",
            "--model",
            config.mlx_base_model,
            "--data",
            stage1_data_dir,
            "--iters",
            str(config.pretrain_steps),
            "--batch-size",
            str(config.micro_batch_size),
            "--learning-rate",
            str(config.mlx_learning_rate),
            "--lora-layers",
            str(config.mlx_lora_layers),
            "--adapter-path",
            str(stage1_adapter),
        ],
        [
            "python",
            "-m",
            "mlx_lm.fuse",
            "--model",
            config.mlx_base_model,
            "--adapter-path",
            str(stage1_adapter),
            "--save-path",
            str(stage1_fused),
        ],
        [
            "python",
            "-m",
            "mlx_lm.lora",
            "--train",
            "--model",
            str(stage1_fused),
            "--data",
            stage2_data_dir,
            "--iters",
            str(config.reasoning_steps),
            "--batch-size",
            str(max(1, config.micro_batch_size // 2)),
            "--learning-rate",
            str(config.mlx_learning_rate * 0.5),
            "--lora-layers",
            str(config.mlx_lora_layers),
            "--adapter-path",
            str(stage2_adapter),
        ],
        [
            "python",
            "-m",
            "mlx_lm.fuse",
            "--model",
            str(stage1_fused),
            "--adapter-path",
            str(stage2_adapter),
            "--save-path",
            str(stage2_fused),
        ],
    ]

    if config.enable_rl_stage:
        rl_cmd = [
            "python",
            str(rl_script),
            "--base-model",
            str(stage2_fused),
            "--out-dir",
            str(stage3_dir),
            "--steps",
            str(config.rl_steps),
            "--group-size",
            str(config.rl_group_size),
            "--max-tokens",
            str(config.rl_max_new_tokens),
            "--batch-size",
            str(max(1, config.micro_batch_size // 2)),
            "--temperature",
            str(config.rl_temperature),
            "--top-p",
            str(config.rl_top_p),
            "--lora-rank",
            str(config.mlx_adapter_rank),
            "--lora-layers",
            str(config.mlx_lora_layers),
            "--learning-rate",
            str(config.mlx_learning_rate),
            "--seed",
            str(config.seed),
            "--lam-latency",
            str(config.lam_latency),
            "--mu-cost",
            str(config.mu_cost),
            "--gamma-stability",
            str(config.gamma_stability),
            "--checkpoint-every",
            "100",
        ]
        for task_file in config.reasoning_task_files:
            rl_cmd.extend(["--tasks", task_file])
        commands.append(rl_cmd)

    return commands


def run_mlx_commands(commands: List[List[str]], *, dry_run: bool, timeout_sec: int) -> List[Dict[str, Any]]:
    """Execute MLX commands sequentially and return structured results."""

    results: List[Dict[str, Any]] = []

    for index, command in enumerate(commands, start=1):
        rendered = shlex.join(command)
        record: Dict[str, Any] = {
            "step": index,
            "command": rendered,
            "status": "dry_run" if dry_run else "pending",
        }

        if dry_run:
            results.append(record)
            continue

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec if timeout_sec > 0 else None,
        )

        record["status"] = "ok" if completed.returncode == 0 else "failed"
        record["returncode"] = completed.returncode
        if completed.stdout:
            record["stdout_tail"] = completed.stdout[-2000:]
        if completed.stderr:
            record["stderr_tail"] = completed.stderr[-2000:]

        results.append(record)
        if completed.returncode != 0:
            break

    return results


def write_mlx_command_log(path: str, command_results: List[Dict[str, Any]]) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(command_results, indent=2), encoding="utf-8")
    return str(out_path.resolve())
