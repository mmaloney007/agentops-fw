import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _read_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def test_clinc_label_type_match_rate_is_full():
    labels_path = _ROOT / "tasks" / "clinc150_labels.json"
    assert labels_path.exists(), "missing tasks/clinc150_labels.json"
    labels = set(json.loads(labels_path.read_text(encoding="utf-8")))

    task_path = _ROOT / "tasks" / "clinc_en.jsonl"
    assert task_path.exists(), "missing tasks/clinc_en.jsonl"

    total = 0
    match = 0
    for rec in _read_jsonl(task_path):
        intent = (rec.get("gold") or {}).get("intent")
        if intent is None:
            continue
        total += 1
        assert isinstance(intent, str)
        if intent in labels:
            match += 1

    assert total > 0
    assert match == total
