import json
from pathlib import Path

import pytest

from agent_stable_slo.train.grpo_train_loop import GRPOTrainConfig, train_loop


def _tiny_tasks(tmp_path: Path) -> Path:
    schema_path = tmp_path / "schema.json"
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps({"prompt": "reply with JSON", "schema_path": str(schema_path)}) + "\n",
        encoding="utf-8",
    )
    return tasks_path


@pytest.mark.slow
def test_resume_from_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    tasks_path = _tiny_tasks(tmp_path)
    out_dir = tmp_path / "run"

    cfg_first = GRPOTrainConfig(
        base_model="hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
        tasks=str(tasks_path),
        out=str(out_dir),
        steps=2,
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
        checkpoint_every=2,
    )
    train_loop(cfg_first)
    ckpt_dir = out_dir / "checkpoints"
    assert ckpt_dir.exists()
    assert any(ckpt_dir.iterdir())

    cfg_resume = cfg_first.model_copy(update={"steps": 4, "resume_from": str(out_dir), "checkpoint_every": 0})
    train_loop(cfg_resume)

    log_lines = list((out_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines())
    assert len(log_lines) == 4
    last = json.loads(log_lines[-1])
    assert last["step"] == 3
