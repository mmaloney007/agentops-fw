"""Dataset preparation helpers for base LM and reasoning fine-tuning."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_HINT = (
    "You are a careful reasoning assistant. Think through the task, then output valid JSON."
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records


def _find_repo_root(task_file: Path) -> Path:
    for candidate in [task_file.parent, *task_file.parents]:
        if (candidate / "tasks").exists():
            return candidate
    return task_file.parent


def _resolve_schema_path(task_file: Path, schema_path: str) -> Path:
    raw = Path(schema_path)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates = [
        (task_file.parent / raw),
        (_find_repo_root(task_file) / raw),
        (Path.cwd() / raw),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve schema path '{schema_path}' from task file {task_file}"
    )


def _schema_keys(schema: Dict[str, Any]) -> List[str]:
    props = schema.get("properties")
    if isinstance(props, dict) and props:
        return sorted(str(k) for k in props.keys())
    required = schema.get("required")
    if isinstance(required, list) and required:
        return sorted(str(k) for k in required)
    return []


def _reasoning_completion(prompt: str, gold: Dict[str, Any], schema: Dict[str, Any]) -> str:
    keys = _schema_keys(schema)
    keys_text = ", ".join(keys) if keys else "follow schema constraints"
    gold_json = json.dumps(gold, ensure_ascii=True, sort_keys=True)
    steps = [
        f"1) Determine required output keys: {keys_text}.",
        "2) Extract only task-relevant facts from the prompt.",
        "3) Build a valid JSON object that matches the schema.",
        "4) Keep wording concise and avoid extra keys.",
    ]
    think = "\n".join(steps)
    return f"<think>\n{think}\n</think>\n{gold_json}"


def build_reasoning_sft_examples(
    task_files: List[str],
    max_examples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Convert task JSONL records to supervised reasoning examples."""

    rng = random.Random(seed)
    examples: List[Dict[str, Any]] = []

    for task_path in task_files:
        task_file = Path(task_path).resolve()
        for record in load_jsonl(str(task_file)):
            prompt = str(record.get("prompt", "")).strip()
            gold = record.get("gold", {})
            schema_ref = str(record.get("schema_path", "")).strip()
            if not prompt or not isinstance(gold, dict) or not schema_ref:
                continue

            schema_path = _resolve_schema_path(task_file, schema_ref)
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            completion = _reasoning_completion(prompt=prompt, gold=gold, schema=schema)

            examples.append(
                {
                    "prompt": f"{SYSTEM_HINT}\n\nTask:\n{prompt}",
                    "completion": completion,
                    "gold": gold,
                    "schema_path": str(schema_path),
                    "source_task": str(task_file),
                }
            )

    rng.shuffle(examples)
    return examples[:max_examples]


def build_pretrain_documents(files: List[str]) -> List[str]:
    """Load plain text or task JSONL files into pretraining documents."""

    docs: List[str] = []
    for raw_path in files:
        path = Path(raw_path).resolve()
        if path.suffix.lower() == ".jsonl":
            for record in load_jsonl(str(path)):
                prompt = str(record.get("prompt", "")).strip()
                gold = record.get("gold")
                if not prompt:
                    continue
                if isinstance(gold, dict):
                    answer = json.dumps(gold, ensure_ascii=True, sort_keys=True)
                    docs.append(f"User: {prompt}\nAssistant: {answer}")
                else:
                    docs.append(f"User: {prompt}")
            continue

        text = path.read_text(encoding="utf-8")
        chunks = [part.strip() for part in text.split("\n\n") if part.strip()]
        docs.extend(chunks)

    return docs


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_mlx_lora_dataset(
    out_dir: str,
    examples: List[Dict[str, str]],
) -> Dict[str, str]:
    """Write prompt/completion splits expected by `mlx_lm.lora --train`."""

    data_dir = Path(out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(examples)
    n_train = max(1, int(n_total * 0.9))
    n_valid = max(1, int(n_total * 0.05))
    if n_train + n_valid >= n_total:
        n_valid = 1 if n_total > 2 else 0
    train = examples[:n_train]
    valid = examples[n_train : n_train + n_valid]
    test = examples[n_train + n_valid :]
    if not test:
        test = valid[:1] if valid else train[:1]

    paths = {
        "train": str((data_dir / "train.jsonl").resolve()),
        "valid": str((data_dir / "valid.jsonl").resolve()),
        "test": str((data_dir / "test.jsonl").resolve()),
    }

    write_jsonl(paths["train"], train)
    write_jsonl(paths["valid"], valid)
    write_jsonl(paths["test"], test)
    return paths
