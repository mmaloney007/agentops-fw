"""Data conversion tests for the reasoning stack."""

from __future__ import annotations

import json

from agent_stable_slo.reasoning_stack.data import (
    build_pretrain_documents,
    build_reasoning_sft_examples,
)


def test_build_reasoning_sft_examples_from_task_file(tmp_path):
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    schema_path = schema_dir / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "bullets"],
            }
        ),
        encoding="utf-8",
    )

    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text(
        json.dumps(
            {
                "prompt": "Write a short summary.",
                "schema_path": str(schema_path),
                "gold": {"title": "QBR", "bullets": ["m1", "m2"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_reasoning_sft_examples([str(task_path)], max_examples=10, seed=1)
    assert len(rows) == 1
    assert "<think>" in rows[0]["completion"]
    assert "title" in rows[0]["completion"]


def test_build_pretrain_documents_reads_text_and_jsonl(tmp_path):
    text_path = tmp_path / "pretrain.txt"
    text_path.write_text("alpha\n\n beta", encoding="utf-8")

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps({"type": "object", "properties": {"answer": {"type": "string"}}}),
        encoding="utf-8",
    )

    task_path = tmp_path / "task.jsonl"
    task_path.write_text(
        json.dumps(
            {
                "prompt": "Say hi",
                "schema_path": str(schema_path),
                "gold": {"answer": "hi"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    docs = build_pretrain_documents([str(text_path), str(task_path)])
    assert len(docs) >= 2
    assert any("alpha" in doc for doc in docs)
    assert any("Assistant" in doc for doc in docs)
