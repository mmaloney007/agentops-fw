"""Config validation tests for the reasoning stack."""

from __future__ import annotations

import json

import pytest

from agent_stable_slo.reasoning_stack.config import load_reasoning_stack_config


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_reasoning_stack_config_resolves_relative_paths(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    text_path = data_dir / "pretrain.txt"
    text_path.write_text("hello world\n\nthis is corpus text", encoding="utf-8")

    schema_path = data_dir / "schema.json"
    _write_json(schema_path, {"type": "object", "properties": {"answer": {"type": "string"}}})

    task_path = data_dir / "task.jsonl"
    record = {
        "prompt": "Return JSON",
        "schema_path": str(schema_path),
        "gold": {"answer": "ok"},
    }
    task_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "backend: mlx",
                "run_name: test_run",
                "out_dir: ./out",
                "pretrain_text_files:",
                "  - ./data/pretrain.txt",
                "reasoning_task_files:",
                "  - ./data/task.jsonl",
                "eval_task_files:",
                "  - ./data/task.jsonl",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_reasoning_stack_config(str(cfg_path))
    assert cfg.backend == "mlx"
    assert cfg.pretrain_text_files[0].endswith("pretrain.txt")
    assert cfg.reasoning_task_files[0].endswith("task.jsonl")
    assert cfg.out_dir.endswith("out")


def test_cuda_backend_requires_pretrain_files(tmp_path):
    schema_path = tmp_path / "schema.json"
    _write_json(schema_path, {"type": "object", "properties": {"x": {"type": "string"}}})

    task_path = tmp_path / "task.jsonl"
    task_path.write_text(
        json.dumps(
            {
                "prompt": "x",
                "schema_path": str(schema_path),
                "gold": {"x": "y"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "backend: cuda",
                "reasoning_task_files:",
                f"  - {task_path}",
                "eval_task_files:",
                f"  - {task_path}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        load_reasoning_stack_config(str(cfg_path))
