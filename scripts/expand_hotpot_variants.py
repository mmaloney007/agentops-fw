#!/usr/bin/env python3
"""
Expand HotpotQA tasks by generating deterministic prompt variants.

Keeps Context/Question blocks intact so evidence ids remain valid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


VARIANTS = [
    "You are given context passages and a question. Return JSON with keys: answer (string), reasoning_summary (brief justification), and evidence_sent_ids (indices of supporting sentences).",
    "Answer the question using only the provided context. Return JSON with keys: answer, reasoning_summary, evidence_sent_ids.",
    "Using the context below, respond with JSON only: answer, reasoning_summary, evidence_sent_ids.",
    "Return JSON only (answer, reasoning_summary, evidence_sent_ids) based on the context and question.",
    "Provide a concise JSON response with answer, reasoning_summary, evidence_sent_ids grounded in the context.",
]


def _split_prompt(prompt: str) -> tuple[str, str]:
    marker = "Context:"
    if marker not in prompt:
        raise ValueError("Prompt missing Context: marker")
    idx = prompt.index(marker)
    return prompt[:idx].strip(), prompt[idx:]


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="tasks/hotpot_dev.jsonl")
    ap.add_argument("--out", dest="out_path", default="tasks/hotpot_dev.jsonl")
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not args.no_backup and in_path == out_path:
        bak = in_path.with_suffix(".jsonl.bak")
        if not bak.exists():
            in_path.replace(bak)
            in_path = bak

    rows = load_jsonl(in_path)
    if not rows:
        raise ValueError("No rows to expand.")

    expanded: List[dict] = []
    for rec in rows:
        prefix, rest = _split_prompt(rec.get("prompt", ""))
        for i, variant in enumerate(VARIANTS):
            new = dict(rec)
            new_id = rec.get("id", "hotpot")
            new["id"] = f"{new_id}_v{i}"
            new["prompt"] = variant + "\n\n" + rest
            expanded.append(new)

    # If we still need more, cycle variants deterministically.
    if len(expanded) < args.target:
        base = list(expanded)
        i = 0
        while len(expanded) < args.target:
            src = base[i % len(base)]
            new = dict(src)
            new["id"] = f"{src.get('id','hotpot')}_x{len(expanded)}"
            expanded.append(new)
            i += 1

    expanded = expanded[: args.target]
    write_jsonl(out_path, expanded)
    print(f"[done] wrote {out_path} ({len(expanded)} rows)")


if __name__ == "__main__":
    main()
