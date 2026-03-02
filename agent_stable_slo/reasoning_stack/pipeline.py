"""End-to-end orchestration for local LLM + reasoning training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .config import ReasoningStackConfig
from .data import (
    build_pretrain_documents,
    build_reasoning_sft_examples,
    write_jsonl,
    write_mlx_lora_dataset,
)
from .mlx_runner import build_mlx_commands, run_mlx_commands, write_mlx_command_log
from .torch_trainer import train_cuda_three_stage


def _prepare_stage1_mlx_examples(docs: List[str]) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for doc in docs:
        text = doc.strip()
        if len(text) < 80:
            continue
        split = max(32, min(len(text) - 16, len(text) // 2))
        prompt = text[:split]
        completion = text[split:]
        if len(completion.strip()) < 8:
            continue
        examples.append({"prompt": prompt, "completion": completion})
    return examples


def run_reasoning_stack(config: ReasoningStackConfig) -> Dict[str, Any]:
    """Run base-LM, reasoning SFT, and optional RL optimization stages."""

    root_dir = Path(config.out_dir) / config.run_name
    root_dir.mkdir(parents=True, exist_ok=True)

    reasoning_examples = build_reasoning_sft_examples(
        task_files=config.reasoning_task_files,
        max_examples=config.max_reasoning_examples,
        seed=config.seed,
    )

    if not reasoning_examples:
        raise ValueError("No reasoning examples were generated from reasoning_task_files")

    pretrain_docs = build_pretrain_documents(config.pretrain_text_files)
    artifacts_dir = config.stage_dir("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    reasoning_jsonl = artifacts_dir / "reasoning_train.jsonl"
    write_jsonl(str(reasoning_jsonl), reasoning_examples)

    summary: Dict[str, Any] = {
        "backend": config.backend,
        "run_name": config.run_name,
        "out_dir": str(root_dir.resolve()),
        "counts": {
            "pretrain_docs": len(pretrain_docs),
            "reasoning_examples": len(reasoning_examples),
            "eval_task_files": len(config.eval_task_files),
        },
        "artifacts": {
            "reasoning_jsonl": str(reasoning_jsonl.resolve()),
        },
    }

    if config.backend == "cuda":
        if not pretrain_docs:
            raise ValueError("CUDA backend needs at least one pretraining document")

        train_summary = train_cuda_three_stage(
            pretrain_docs=pretrain_docs,
            reasoning_examples=reasoning_examples,
            config=config,
        )
        summary["training"] = train_summary

    else:
        stage1_examples = _prepare_stage1_mlx_examples(pretrain_docs)
        if not stage1_examples:
            stage1_examples = [
                {"prompt": ex["prompt"], "completion": ex["completion"]}
                for ex in reasoning_examples[: max(64, min(512, len(reasoning_examples)))]
            ]

        stage1_data = write_mlx_lora_dataset(
            out_dir=str(config.stage_dir("mlx_stage1_data")),
            examples=stage1_examples,
        )
        stage2_data = write_mlx_lora_dataset(
            out_dir=str(config.stage_dir("mlx_stage2_data")),
            examples=[
                {"prompt": ex["prompt"], "completion": ex["completion"]}
                for ex in reasoning_examples
            ],
        )

        commands = build_mlx_commands(
            config=config,
            stage1_data_dir=str(config.stage_dir("mlx_stage1_data")),
            stage2_data_dir=str(config.stage_dir("mlx_stage2_data")),
        )
        command_results = run_mlx_commands(
            commands=commands,
            dry_run=config.dry_run,
            timeout_sec=config.command_timeout_sec,
        )
        command_log = write_mlx_command_log(
            path=str(config.stage_dir("artifacts") / "mlx_commands.json"),
            command_results=command_results,
        )

        summary["training"] = {
            "backend": "mlx",
            "stage1_data": stage1_data,
            "stage2_data": stage2_data,
            "rl_enabled": config.enable_rl_stage,
            "command_log": command_log,
            "command_results": command_results,
        }

    summary_path = root_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    return summary
