#!/usr/bin/env python3
"""
Run an MLX model + adapter over a JSONL task file and emit eval.jsonl in the same
shape as grpo_trl.py expects, so we can reuse score_eval.py and judge_faithfulness.py.

Inputs:
  --model     Path to MLX-converted model (e.g., mlx/Qwen3-4B-Thinking-2507)
  --adapter   Path to LoRA adapters (optional)
  --tasks     JSONL tasks file (prompt, schema_path, optional gold)
  --out       Output eval.jsonl
  --max-new-tokens  Max generation length
"""
import argparse, json, time
from pathlib import Path
from typing import Dict, Any, List

import mlx.core as mx
from mlx_lm import load, generate

def load_tasks(path: str) -> List[Dict[str, Any]]:
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        obj=json.loads(line)
        rows.append(obj)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of tasks (0 = all)")
    ap.add_argument("--progress", action="store_true", help="Print progress every 10 tasks")
    args=ap.parse_args()

    model, tok = load(args.model, adapter_path=args.adapter)
    tasks = load_tasks(args.tasks)
    if args.limit and args.limit > 0:
        tasks = tasks[:args.limit]

    out_path=Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,"w",encoding="utf-8") as fo:
        for i,row in enumerate(tasks):
            prompt=row["prompt"]
            schema_path=row["schema_path"]
            schema=json.load(open(schema_path,"r",encoding="utf-8"))

            t0=time.time()
            completion = generate(
                model,
                tok,
                prompt=prompt,
                max_tokens=args.max_new_tokens,
                verbose=False,
            )
            lat_ms=(time.time()-t0)*1000.0
            ttft_ms=lat_ms  # non-streaming approximation
            try:
                out_json=json.loads(completion)
            except Exception:
                out_json={}

            rec={
                "step": i,
                "latency_ms": float(round(lat_ms,3)),
                "ttft_ms": float(round(ttft_ms,3)),
                "reward": 0.0,  # not computed here
                "json_valid": 1 if out_json else 0,
                "tokens_out": -1,
                "prompt": prompt,
                "schema_path": schema_path,
                "output_json": out_json,
            }
            if "gold" in row: rec["gold"]=row["gold"]
            fo.write(json.dumps(rec)+"\n")
            if args.progress and (i+1) % 10 == 0:
                print(f"[progress] {i+1}/{len(tasks)}")
    print(f"[done] wrote {out_path}")

if __name__=="__main__":
    main()
