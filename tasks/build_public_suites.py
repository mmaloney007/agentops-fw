#!/usr/bin/env python3
"""
Build public evaluation suites (GSM8K, MBPP, HumanEval-lite, TruthfulQA MC,
and a small MATH slice) into JSONL tasks compatible with our structured runner.
Each task uses a simple QA schema (answer string) so EM/F1 scoring works out of
the box. Use --limit-per 0 to take full splits; defaults to 200 per dataset to
keep runs manageable. Outputs are written to tasks/public_<name>.jsonl.
"""
import argparse, json, re
from pathlib import Path
from typing import Iterable, Dict, Any
from datasets import load_dataset

ROOT = Path(__file__).parent
QA_SCHEMA = "tasks/schemas/qa_schema.json"

def write_tasks(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {path}")

def build_gsm8k(limit: int):
    ds = load_dataset("gsm8k", "main")["train"]
    rows = []
    for i, rec in enumerate(ds):
        if limit and i >= limit: break
        # answers look like "... #### 42"
        ans = rec["answer"]
        if "####" in ans:
            ans = ans.split("####")[-1].strip()
        rows.append({
            "prompt": f"Question: {rec['question']}\nAnswer succinctly.",
            "schema_path": QA_SCHEMA,
            "gold": {"answer": ans},
            "id": f"gsm8k_{i}"
        })
    return rows

def build_mbpp(limit: int):
    ds = load_dataset("mbpp")["train"]
    rows=[]
    for i, rec in enumerate(ds):
        if limit and i >= limit: break
        rows.append({
            "prompt": f"Write Python code for: {rec['text']}\nReturn only code.",
            "schema_path": QA_SCHEMA,
            "gold": {"answer": rec.get("code", "").strip()},
            "id": f"mbpp_{i}"
        })
    return rows

def build_humaneval(limit: int):
    ds = load_dataset("openai_humaneval")["test"]
    rows=[]
    for i, rec in enumerate(ds):
        if limit and i >= limit: break
        prompt = rec["prompt"]
        canon = rec.get("canonical_solution","")
        rows.append({
            "prompt": f"Complete the function as specified:\n{prompt}\nReturn only code.",
            "schema_path": QA_SCHEMA,
            "gold": {"answer": canon},
            "id": f"humaneval_{i}"
        })
    return rows

def build_truthfulqa(limit: int):
    ds = load_dataset("truthful_qa", "multiple_choice")["validation"]
    rows=[]
    for i, rec in enumerate(ds):
        if limit and i >= limit: break
        q = rec["question"]
        correct = rec["mc1_targets"]["choices"][0] if rec.get("mc1_targets") else rec.get("best_answer","")
        rows.append({
            "prompt": f"Answer truthfully: {q}",
            "schema_path": QA_SCHEMA,
            "gold": {"answer": correct},
            "id": f"truthfulqa_{i}"
        })
    return rows

def build_math(limit: int):
    try:
        ds = load_dataset("math_dataset", "all")["train"]
    except Exception:
        print("[warn] math_dataset not available; skipping math slice.")
        return []
    rows=[]
    for i, rec in enumerate(ds):
        if limit and i >= limit: break
        prob = rec["problem"]
        sol = rec["solution"]
        # Extract final answer if present in \\boxed{}
        m = re.search(r"\\boxed\\{(.+?)\\}", sol)
        final = m.group(1).strip() if m else sol.strip()
        rows.append({
            "prompt": f"Solve: {prob}\nGive the final answer.",
            "schema_path": QA_SCHEMA,
            "gold": {"answer": final},
            "id": f"math_{i}"
        })
    return rows

BUILDERS = {
    "gsm8k": build_gsm8k,
    "mbpp": build_mbpp,
    "humaneval": build_humaneval,
    "truthfulqa": build_truthfulqa,
    "math": build_math,
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit-per", type=int, default=200, help="0 means full dataset")
    ap.add_argument("--datasets", type=str, default="gsm8k,mbpp,humaneval,truthfulqa,math")
    args=ap.parse_args()
    limit = None if args.limit_per==0 else args.limit_per
    names=[x.strip() for x in args.datasets.split(",") if x.strip()]
    for name in names:
        if name not in BUILDERS:
            print(f"[skip] unknown dataset {name}")
            continue
        rows = BUILDERS[name](limit or 0)
        write_tasks(ROOT / f"public_{name}.jsonl", rows)

if __name__=="__main__":
    main()
