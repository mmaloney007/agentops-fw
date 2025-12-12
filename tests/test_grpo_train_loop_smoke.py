import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_stable_slo.train.grpo_train_loop import GRPOTrainConfig, train_loop
from agent_stable_slo.utils.data import cache_dataset, fingerprint_tasks
from agent_stable_slo.train import grpo_train_loop as grpo_module


def _write_tiny_tasks(tmp_path: Path) -> Path:
    schema_path = tmp_path / "schema.json"
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "bullets": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
        },
        "required": ["answer"],
    }
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    tasks_path = tmp_path / "tasks.jsonl"
    rec = {"prompt": "Return valid JSON with keys answer and bullets.", "schema_path": str(schema_path), "gold": {"answer": "ok"}}
    tasks_path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    return tasks_path


@pytest.mark.slow
def test_grpo_train_loop_runs_one_step_tiny_model(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    tasks_path = _write_tiny_tasks(tmp_path)
    out_dir = tmp_path / "out"

    cfg = GRPOTrainConfig(
        base_model="hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
        tasks=str(tasks_path),
        out=str(out_dir),
        steps=1,
        max_prompt_len=64,
        max_new_tokens=16,
        deterministic=True,
        cache_dataset=False,
        load_in_4bit=False,
        gradient_accumulation=1,
        lr=1e-4,
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_targets="query_key_value,dense",
        eval_interval=1,
        torch_dtype="float32",
    )

    train_loop(cfg)

    log_file = out_dir / "train_log.jsonl"
    adapter_dir = out_dir / "adapter"
    assert log_file.exists(), "train loop should emit per-step JSONL log"
    assert adapter_dir.exists(), "adapter artifacts should be saved"


def test_cache_dataset_rewrites_schema_paths(tmp_path):
    tasks_path = _write_tiny_tasks(tmp_path)
    cached_tasks, fp = cache_dataset(str(tasks_path), str(tmp_path / "cache"))

    with open(cached_tasks, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())

    assert Path(rec["schema_path"]).exists()
    original_fp = fingerprint_tasks(str(tasks_path))
    assert fp.sha256 != original_fp.sha256  # schema path rewrite changes hash


def test_no_silent_defaults_guard(tmp_path):
    tasks_path = _write_tiny_tasks(tmp_path)
    with pytest.raises(ValidationError):
        GRPOTrainConfig(tasks=str(tasks_path), out=str(tmp_path / "o"), no_silent_defaults=True)


def test_guard_prompt_truncates():
    short = grpo_module._guard_prompt("abc", max_chars=2, truncate=True)
    assert short == "ab"
    with pytest.raises(ValueError):
        grpo_module._guard_prompt("abc", max_chars=2, truncate=False)


def test_expected_dataset_hash_enforced(tmp_path):
    tasks_path = _write_tiny_tasks(tmp_path)
    wrong_hash = "deadbeef"
    cfg = GRPOTrainConfig(
        base_model="hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
        tasks=str(tasks_path),
        out=str(tmp_path / "out"),
        expected_dataset_hash=wrong_hash,
    )
    with pytest.raises(ValueError):
        grpo_module.validate_fingerprint(grpo_module.fingerprint_tasks(cfg.tasks), cfg.expected_dataset_hash, allow_drift=False)
