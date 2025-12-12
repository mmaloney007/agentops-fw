#!/usr/bin/env python3
"""Build a tasks JSONL from HotpotQA (distractor) for grounded QA + reasoning summary."""
import argparse, json, random
from pathlib import Path
from datasets import load_dataset

PROMPT_TMPL = (
    "You are given context passages and a question. Return JSON with keys: answer (string),"
    " reasoning_summary (brief justification), and evidence_sent_ids (indices of supporting sentences).\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def build_context_and_evidence(example):
    ctx_blocks = example.get("context", {})
    sentences = []
    if isinstance(ctx_blocks, dict):
        for _, sent_list in ctx_blocks.items():
            if isinstance(sent_list, list):
                for s in sent_list:
                    sentences.append(s)
    elif isinstance(ctx_blocks, list):
        for title, sent_list in ctx_blocks:
            if isinstance(sent_list, list):
                for s in sent_list:
                    sentences.append(s)
    context = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    support = example.get("supporting_facts", [])
    evidence_ids = []
    if isinstance(support, dict):
        for sid in support.get("sent_id", []):
            if isinstance(sid, int):
                evidence_ids.append(sid)
    else:
        for item in support:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                sid = item[1]
                if isinstance(sid, int):
                    evidence_ids.append(sid)
    evidence_ids = sorted(set(evidence_ids))
    return context, evidence_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", help="hotpot_qa split (train/validation)")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="tasks/hotpot_dev.jsonl")
    args = ap.parse_args()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    rng = random.Random(args.seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: args.count]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path = "tasks/schemas/hotpot_explainer_schema.json"

    with out_path.open("w", encoding="utf-8") as f:
        for i, idx in enumerate(idxs):
            ex = ds[int(idx)]
            q = ex["question"].strip()
            ans = ex.get("answer", "")
            context, evidence_ids = build_context_and_evidence(ex)
            supporting = ex.get("supporting_facts", [])
            reasoning_items = []
            if isinstance(supporting, dict):
                titles = supporting.get("title", [])
                sids = supporting.get("sent_id", [])
                for t, sid in zip(titles, sids):
                    reasoning_items.append(f"{t}:{sid}")
            else:
                for item in supporting:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        t, sid = item[0], item[1]
                        reasoning_items.append(f"{t}:{sid}")
            reasoning = "; ".join(reasoning_items)
            prompt = PROMPT_TMPL.format(context=context, question=q)
            record = {
                "id": f"hotpot_{args.split}_{i}",
                "prompt": prompt,
                "schema_path": schema_path,
                "gold": {
                    "answer": ans,
                    "reasoning_summary": reasoning,
                    "evidence_sent_ids": evidence_ids,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[done] wrote {out_path} ({len(idxs)} examples)")


if __name__ == "__main__":
    main()
