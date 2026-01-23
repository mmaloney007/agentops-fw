#!/usr/bin/env python3
"""
Prepare CLINC150 labels and normalize tasks/clinc_en.jsonl to string intents.

This script:
- loads CLINC150 label names from datasets (clinc_oos, subset=plus)
- writes tasks/clinc150_labels.json
- rewrites tasks/clinc_en.jsonl gold intent IDs to label strings
- injects the allowed label list into prompts
- updates tasks/schemas/clinc_nlu_schema.json to include intent enum
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "tasks" / "clinc_en.jsonl"
LABELS_PATH = ROOT / "tasks" / "clinc150_labels.json"
SCHEMA_PATH = ROOT / "tasks" / "schemas" / "clinc_nlu_schema.json"


def load_labels() -> List[str]:
    ds = load_dataset("clinc_oos", "plus", split="validation")
    names = list(ds.features["intent"].names)
    if "oos" not in names:
        names.append("oos")
    return names


def build_text_to_label(labels: List[str]) -> dict:
    ds = load_dataset("clinc_oos", "plus", split="validation")
    text_to_label = {}
    for rec in ds:
        text = rec.get("text", "").strip().lower()
        label_id = rec.get("intent", "")
        if isinstance(label_id, int) and 0 <= label_id < len(labels):
            label = labels[label_id]
        else:
            label = str(label_id)
        if text and label and text not in text_to_label:
            text_to_label[text] = label
    return text_to_label


def rewrite_prompts(prompt: str, labels: List[str]) -> str:
    label_blob = "Allowed intents (CLINC150): " + json.dumps(labels, ensure_ascii=True)
    marker = "Utterance:"
    if marker in prompt:
        head, tail = prompt.split(marker, 1)
        head = head.rstrip() + "\n" + label_blob + "\n"
        return head + marker + tail
    return prompt.rstrip() + "\n" + label_blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-backup", action="store_true", help="Do not create .bak files."
    )
    args = ap.parse_args()

    labels = load_labels()
    text_to_label = build_text_to_label(labels)
    LABELS_PATH.write_text(
        json.dumps(labels, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    bak_tasks = TASKS_PATH.with_suffix(".jsonl.bak")
    bak_schema = SCHEMA_PATH.with_suffix(".json.bak")
    if not args.no_backup and not bak_tasks.exists():
        TASKS_PATH.replace(TASKS_PATH.with_suffix(".jsonl.bak"))
    if not args.no_backup and not bak_schema.exists():
        SCHEMA_PATH.replace(SCHEMA_PATH.with_suffix(".json.bak"))

    rows = []
    source_path = bak_tasks if bak_tasks.exists() else TASKS_PATH
    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            gold = rec.get("gold") or {}
            prompt = rec.get("prompt", "")
            utt = ""
            if "Utterance:" in prompt:
                utt = prompt.split("Utterance:", 1)[1].strip().lower()
            if utt and utt in text_to_label:
                gold["intent"] = text_to_label[utt]
            else:
                intent = gold.get("intent")
                if isinstance(intent, int):
                    if intent < 0 or intent >= len(labels):
                        raise ValueError(f"CLINC intent id out of range: {intent}")
                    gold["intent"] = labels[intent]
                elif isinstance(intent, str):
                    gold["intent"] = intent
                else:
                    raise ValueError(f"Unexpected intent type: {type(intent)}")
            gold["is_oos"] = bool(gold.get("intent") == "oos")
            rec["gold"] = gold
            rec["prompt"] = rewrite_prompts(prompt, labels)
            rows.append(rec)

    with TASKS_PATH.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    schema_source = bak_schema if bak_schema.exists() else SCHEMA_PATH
    schema = json.loads(schema_source.read_text(encoding="utf-8"))
    schema.setdefault("properties", {})
    schema["properties"]["intent"] = {"type": "string", "enum": labels}
    if "required" not in schema:
        schema["required"] = ["intent", "is_oos"]
    SCHEMA_PATH.write_text(
        json.dumps(schema, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] wrote {LABELS_PATH}")
    print(f"[done] rewrote {TASKS_PATH}")
    print(f"[done] updated {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
