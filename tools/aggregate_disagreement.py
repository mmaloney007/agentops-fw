#!/usr/bin/env python3
"""
Compute disagreement across multiple eval JSONL files (same task ordering).
Usage:
  python3 tools/aggregate_disagreement.py --evals out/run1/eval.jsonl out/run2/eval.jsonl --out out/aggregate/disagreement.json
Disagreement per task is 1 - (max count of most common output)/(num outputs).
Outputs a JSON with mean disagreement and per-task entries.
"""
import argparse, json
from collections import defaultdict, Counter
from pathlib import Path

def load(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    runs=[load(p) for p in args.evals]
    if len({len(r) for r in runs}) != 1:
        raise ValueError("All eval files must have the same length/order")
    n=len(runs[0])
    disagreements=[]
    per_task=[]
    for i in range(n):
        outs=[json.dumps(run[i].get("output_json",{}), sort_keys=True) for run in runs]
        cnt=Counter(outs)
        most=cnt.most_common(1)[0][1]
        dis=1 - most/len(outs)
        disagreements.append(dis)
        per_task.append({"idx": i, "disagreement": dis})
    out_obj={"mean_disagreement": sum(disagreements)/len(disagreements) if disagreements else 0.0,
             "per_task": per_task}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_obj, indent=2))
    print(f"[done] wrote {args.out}")

if __name__=="__main__": main()
