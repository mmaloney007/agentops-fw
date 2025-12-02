#!/usr/bin/env python3
"""
Generate a richer MLX-friendly JSONL dataset for LoRA smoke tests on Mac.
Creates train/valid splits covering:
 - Summary with citations
 - JSON extraction
 - Tool sequence planning
 - Simple QA/math
 - Safety classification
 - Code stub JSON
"""
import json
from pathlib import Path
from random import randint, choice, seed

seed(42)
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data"

sources = [
    ("latency improved", "p95 improved", "s1"),
    ("stability worse", "disagreement up", "s2"),
    ("throughput higher", "p99 worse", "s3"),
    ("risk noted", "action assigned", "s4"),
]

def summary_examples(n=80):
    rows=[]
    for i in range(n):
        a,b,sid=choice(sources)
        prompt=f"Sources:\\n[{sid}]: {a}\\n[sx]: {b}\\nWrite 2 bullets citing sources.\\nAssistant:"
        completion=f"Bullets: {a} [{sid}]; {b} [sx]"
        rows.append({"prompt": prompt, "completion": completion})
    return rows

def json_extract_examples(n=40):
    rows=[]
    for i in range(n):
        a=randint(0,9); b=randint(0,9)
        prompt=f"Return JSON exactly {{\"a\":{a},\"b\":{b}}}.\\nAssistant:"
        completion=json.dumps({"a":a,"b":b})
        rows.append({"prompt": prompt, "completion": completion})
    return rows

def toolseq_examples(n=30):
    seqs=[["search_contracts","summarize_report"],["classify_intent"],["run_sql_query","file_ticket"]]
    rows=[]
    for i in range(n):
        steps=seqs[i%len(seqs)]
        prompt=f"Plan minimal tool steps for task: {steps[-1]}. Return steps[].\\nAssistant:"
        completion=json.dumps({"steps":steps})
        rows.append({"prompt": prompt, "completion": completion})
    return rows

def qa_examples(n=30):
    rows=[]
    for i in range(n):
        x=randint(1,50); y=randint(1,50)
        prompt=f"Question: What is {x}+{y}? Answer succinctly.\\nAssistant:"
        completion=str(x+y)
        rows.append({"prompt": prompt, "completion": completion})
    return rows

def safety_examples(n=20):
    rows=[]
    for i in range(n):
        decision="flag" if i%3==0 else "allow"
        reason="PII detected" if decision=="flag" else "No PII"
        prompt=f"Policy: data_privacy. Decide flag/allow and risk score.\\nReason: {reason}\\nAssistant:"
        completion=json.dumps({"decision":decision,"risk_score":4 if decision=="flag" else 1})
        rows.append({"prompt": prompt, "completion": completion})
    return rows

def code_examples(n=20):
    rows=[]
    for i in range(n):
        prompt="Return JSON with function signature for fn_p95(latencies) and one test."
        completion=json.dumps({"function":"fn_p95","signature":"def fn_p95(latencies):","tests":["fn_p95([1,2,3])"]})
        rows.append({"prompt": prompt, "completion": completion})
    return rows

def build_split():
    rows = summary_examples() + json_extract_examples() + toolseq_examples() + qa_examples() + safety_examples() + code_examples()
    return rows

def write_split(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def main():
    train_rows = build_split()
    valid_rows = build_split()[:60]
    write_split(OUT_DIR/"mlx_train.jsonl", train_rows)
    write_split(OUT_DIR/"mlx_valid.jsonl", valid_rows)
    print(f"[done] wrote {len(train_rows)} train and {len(valid_rows)} valid examples under {OUT_DIR}")

if __name__=="__main__":
    main()
