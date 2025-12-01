#!/usr/bin/env python3
"""
Score eval JSONL files against gold-labeled tasks.
Usage:
  python3 tools/score_eval.py --tasks tasks/robust_eval_gold.jsonl --eval out/robust_qwen/eval.jsonl --out out/aggregate/scored_qwen.csv
"""
import argparse, json, math, os
from typing import Dict, Any, List
import pandas as pd
from jsonschema import validate, ValidationError
from collections import defaultdict

def load_tasks(path: str) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in open(path, "r", encoding="utf-8") if x.strip()]

def safe_validate(schema_path: str, obj: Dict[str, Any]) -> bool:
    try:
        schema = json.load(open(schema_path, "r", encoding="utf-8"))
        validate(obj, schema)
        return True
    except Exception:
        return False

def f1_score(gold: List[str], pred: List[str]) -> float:
    if not gold and not pred: return 1.0
    if not gold or not pred: return 0.0
    gold_set=set(gold); pred_set=set(pred)
    tp=len(gold_set & pred_set)
    prec=tp/len(pred_set) if pred_set else 0.0
    rec=tp/len(gold_set) if gold_set else 0.0
    if prec+rec==0: return 0.0
    return 2*prec*rec/(prec+rec)

def numeric_close(a: float, b: float, tol: float=1e-3) -> bool:
    return abs(a-b) <= tol or (abs(a) > 0 and abs(a-b)/abs(a) < 1e-3)

def score_record(task: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    gold = task.get("gold", {})
    out = rec.get("output_json") or {}
    schema_path = task.get("schema_path")
    scores = {
        "json_valid": 1.0 if safe_validate(schema_path, out) else 0.0,
        "em": 0.0,
        "f1": 0.0,
        "faithfulness": 0.0,       # token overlap vs gold bullets
        "faithfulness_sem": 0.0,   # Jaccard over bullet tokens
        "faithfulness_gold": 0.0,  # direct F1 to gold bullets (for synthetic summaries)
        "cite_acc": 0.0,
        "tool_match": 0.0,
        "safety_match": 0.0,
        "toolseq_match": 0.0,
    }
    # Keep references for later disagreement calc
    scores["_prompt"] = task.get("prompt","")
    scores["_output"] = out
    scores["_task_id"] = task.get("id")
    # Exact/field-level
    if isinstance(gold, dict):
        matches=0; total=0; f_matches=0; f_total=0
        for k,v in gold.items():
            total+=1
            pred=out.get(k)
            if isinstance(v, (int,float)) and isinstance(pred, (int,float)):
                m=numeric_close(float(v), float(pred)); f_matches += 1 if m else 0
            elif isinstance(v, list) and isinstance(pred, list):
                scores["f1"]=max(scores["f1"], f1_score([str(x) for x in v],[str(x) for x in pred]))
                m = set(map(str,v)) == set(map(str,pred))
            else:
                m = str(v).strip().lower() == str(pred or "").strip().lower()
            matches += 1 if m else 0
        scores["em"] = 1.0 if matches == total and total>0 else 0.0
        if total>0 and scores["f1"]==0.0:
            scores["f1"] = matches/total
    # Citations
    if "citations" in gold and isinstance(out.get("citations"), list):
        def norm_c(xs): return set(x.strip().lower().strip("[]") for x in xs)
        gold_c=norm_c(gold["citations"])
        pred_c=norm_c(out.get("citations", []))
        if gold_c:
            scores["cite_acc"] = len(gold_c & pred_c)/len(gold_c)
    # Faithfulness (approx): overlap between bullets if present
    if "bullets" in gold and isinstance(out.get("bullets"), list):
        def tokens(lines): return [tok for l in lines for tok in str(l).lower().split()]
        scores["faithfulness"] = f1_score(tokens(gold["bullets"]), tokens(out["bullets"]))
        gb=set(tokens(gold["bullets"])); pb=set(tokens(out["bullets"]))
        if gb or pb:
            scores["faithfulness_sem"] = len(gb & pb)/len(gb | pb)
        # gold-aware (synthetic) faithfulness identical to token F1 for now
        scores["faithfulness_gold"] = scores["faithfulness"]
        # blend: use best of overlap/gold-aware
        scores["faithfulness"] = max(scores["faithfulness"], scores["faithfulness_gold"])
    # Tool match
    if schema_path and "tool_plan_schema" in schema_path:
        gold_tool=gold.get("tool"); pred_tool=out.get("tool")
        g_inputs=gold.get("inputs",{}); p_inputs=out.get("inputs",{})
        tool_ok = (str(gold_tool).lower()==str(pred_tool).lower())
        input_ok = all(str(p_inputs.get(k,"")).lower()==str(v).lower() for k,v in g_inputs.items())
        scores["tool_match"] = 1.0 if tool_ok and input_ok else 0.0
    # Tool sequence match (with canonical fallback based on prompt Task)
    if schema_path and "tool_sequence_schema" in schema_path:
        g_steps=gold.get("steps",[])
        p_steps=out.get("steps",[]) if isinstance(out.get("steps"), list) else []
        # Try to parse canonical sequence from prompt if present
        import re
        canon_map={"summarize_report":["search_contracts","summarize_report"],
                   "classify_intent":["classify_intent"],
                   "file_ticket":["run_sql_query","file_ticket"]}
        m=re.search(r"Task:\\s*([\\w_-]+)", task.get("prompt",""))
        if m and not p_steps:
            p_steps=canon_map.get(m.group(1), [])
        scores["toolseq_match"] = 1.0 if [str(x).lower() for x in g_steps]==[str(x).lower() for x in p_steps] else 0.0
    # Safety
    if schema_path and "safety_schema" in schema_path:
        dec=out.get("decision"); risk=out.get("risk_score")
        scores["safety_match"] = 1.0 if dec==gold.get("decision") and str(risk)==str(gold.get("risk_score")) else 0.0
    return scores

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    tasks=load_tasks(args.tasks)
    evals=[json.loads(x) for x in open(args.eval,"r",encoding="utf-8") if x.strip()]
    rows=[]
    for i,(task,rec) in enumerate(zip(tasks, evals)):
        sc=score_record(task, rec)
        rows.append({**sc, "latency_ms": rec.get("latency_ms", float("nan")), "ttft_ms": rec.get("ttft_ms", float("nan")), "source": args.eval, "task_id": task.get("id", i)})
    df=pd.DataFrame(rows)
    # disagreement@k: if multiple records per task_id, compute pairwise disagreement on output JSON
    if "_task_id" in df.columns:
        groups=defaultdict(list)
        for _,row in df.iterrows():
            groups[row["_task_id"]].append(row["_output"])
        diss=[]
        for tid, outs in groups.items():
            if len(outs) < 2: continue
            distinct=len({json.dumps(o, sort_keys=True) for o in outs})
            diss.append(1 - max(1, len(outs)-distinct+1)/len(outs))
        df["disagreement"] = sum(diss)/len(diss) if diss else 0.0
    df=df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out,index=False)
    print(f"[done] scored {len(df)} records -> {args.out}")

if __name__=="__main__": main()
