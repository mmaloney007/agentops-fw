#!/usr/bin/env python3
"""Build tasks JSONL from CLINC OOS (subset plus) for intent classification and oos flag."""
import argparse, json, random
from pathlib import Path
from datasets import load_dataset

PROMPT_TMPL = (
    "Given a user utterance, return JSON with intent (string), domain (string), and is_oos (boolean).\n"
    "Do not invent intents. If the utterance is out-of-scope, set is_oos=true and intent='oos'.\n"
    "Utterance: {utt}"
)

SCHEMA_PATH = "tasks/schemas/clinc_nlu_schema.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--subset", default="plus", help="clinc_oos subset: plus/small/imbalanced")
    ap.add_argument("--split", default="validation", help="train/validation/test")
    ap.add_argument("--out", default="tasks/clinc_en.jsonl")
    args = ap.parse_args()

    ds = load_dataset("clinc_oos", args.subset, split=args.split)
    rng = random.Random(args.seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: args.count]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, idx in enumerate(idxs):
            ex = ds[int(idx)]
            utt = ex["text"].strip()
            intent = ex.get("intent", "")
            domain = ex.get("domain", "")
            is_oos = intent == "oos"
            prompt = PROMPT_TMPL.format(utt=utt)
            record = {
                "id": f"clinc_{args.split}_{i}",
                "prompt": prompt,
                "schema_path": SCHEMA_PATH,
                "gold": {
                    "intent": intent,
                    "domain": domain,
                    "is_oos": is_oos,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[done] wrote {out_path} ({len(idxs)} examples)")


if __name__ == "__main__":
    main()
