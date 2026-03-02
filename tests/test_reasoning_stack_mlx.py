"""MLX command planning tests for the reasoning stack."""

from __future__ import annotations

import json
from pathlib import Path

from agent_stable_slo.reasoning_stack.config import ReasoningStackConfig
from agent_stable_slo.reasoning_stack.mlx_runner import build_mlx_commands
from agent_stable_slo.reasoning_stack.pipeline import run_reasoning_stack


def _make_task_file(root: Path) -> str:
    schema_path = root / "schema.json"
    schema_path.write_text(
        json.dumps({"type": "object", "properties": {"answer": {"type": "string"}}}),
        encoding="utf-8",
    )

    task_path = root / "task.jsonl"
    task_path.write_text(
        json.dumps(
            {
                "prompt": "Answer briefly",
                "schema_path": str(schema_path),
                "gold": {"answer": "ok"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return str(task_path)


def test_build_mlx_commands_contains_expected_tools(tmp_path):
    pretrain = tmp_path / "pretrain.txt"
    pretrain.write_text("hello world " * 20, encoding="utf-8")
    task_path = _make_task_file(tmp_path)

    cfg = ReasoningStackConfig(
        backend="mlx",
        run_name="test_mlx_cmds",
        out_dir=str(tmp_path / "out"),
        pretrain_text_files=[str(pretrain)],
        reasoning_task_files=[task_path],
        eval_task_files=[task_path],
        dry_run=True,
    )

    cmds = build_mlx_commands(
        config=cfg,
        stage1_data_dir=str(tmp_path / "d1"),
        stage2_data_dir=str(tmp_path / "d2"),
    )

    joined = "\n".join(" ".join(cmd) for cmd in cmds)
    assert "mlx_lm.lora" in joined
    assert "mlx_lm.fuse" in joined
    assert "run_mlx_grpo_stage.py" in joined
    assert cfg.mlx_base_model in joined


def test_run_reasoning_stack_mlx_dry_run(tmp_path):
    pretrain = tmp_path / "pretrain.txt"
    pretrain.write_text("reasoning corpus " * 30, encoding="utf-8")
    task_path = _make_task_file(tmp_path)

    cfg = ReasoningStackConfig(
        backend="mlx",
        run_name="test_mlx_pipeline",
        out_dir=str(tmp_path / "out"),
        pretrain_text_files=[str(pretrain)],
        reasoning_task_files=[task_path],
        eval_task_files=[task_path],
        max_reasoning_examples=16,
        dry_run=True,
    )

    summary = run_reasoning_stack(cfg)
    assert summary["backend"] == "mlx"
    assert summary["training"]["command_results"][0]["status"] == "dry_run"
    assert Path(summary["summary_path"]).exists()
