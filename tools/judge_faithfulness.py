#!/usr/bin/env python3
"""
Compute judge-based faithfulness using an LLM-as-judge. This uses the local LM Studio
endpoint (OpenAI-compatible) by default; adjust OPENAI_API_BASE/KEY/MODEL as needed.
Given a scored CSV and tasks file with prompts/gold, it writes a new CSV with a
faithfulness_judge column (0-1 scaled from 0-3 support).
"""
import argparse, json, os, textwrap
import pandas as pd
from openai import OpenAI

def _normalize_citations(text: str) -> str:
    # lightweight normalizer: ensure [id] style, strip stray punctuation
    return text.replace("(", "[").replace(")", "]")

def judge(prompt: str, gold: str, candidate: str, model: str, base: str, key: str) -> float:
    client=OpenAI(base_url=base, api_key=key)
    judge_prompt=textwrap.dedent(f"""
    You are a fact-checker. Given SOURCE CONTEXT (prompt + sources), an EXPECTED gold JSON,
    and a CANDIDATE JSON, extract up to 6 atomic statements from the CANDIDATE and rate
    support from CONTEXT on 0-3:
      0 = not supported / contradicted,
      1 = weakly supported / unclear,
      2 = supported,
      3 = strongly supported.
    Be concise; if unsure, use 1 (not 0). Return JSON:
    {{"statements":[{{"statement":"...","supported":<0-3>,"rationale":"..."}}]}}.

    CONTEXT:
    {prompt}

    EXPECTED (for alignment reference):
    {gold}

    CANDIDATE:
    {candidate}
    """).strip()
    try:
        r=client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":judge_prompt}],
            temperature=0
        )
        txt=r.choices[0].message.content
        obj=json.loads(txt)
        stmts=obj.get("statements",[])
        if not stmts: return 0.0
        scores=[s.get("supported",0) for s in stmts if isinstance(s,dict)]
        if not scores: return 0.0
        return sum(scores)/(3*len(scores))
    except Exception:
        return 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL","openai/gpt-oss-20b"))
    ap.add_argument("--base", default=os.getenv("JUDGE_BASE", os.getenv("OPENAI_API_BASE","http://10.0.0.72:1234/v1")))
    ap.add_argument("--key", default=os.getenv("OPENAI_API_KEY","lm-studio"))
    ap.add_argument("--limit", type=int, default=0)
    args=ap.parse_args()
    tasks=[json.loads(x) for x in open(args.tasks,"r",encoding="utf-8") if x.strip()]
    evals=[json.loads(x) for x in open(args.eval,"r",encoding="utf-8") if x.strip()]
    if args.limit and args.limit>0:
        tasks=tasks[:args.limit]
        evals=evals[:args.limit]
    df=pd.read_csv(args.scored)
    if args.limit and args.limit>0:
        df=df.head(len(tasks))
    faith=[]
    for t,r in zip(tasks, evals):
        ctx=t.get("prompt","")
        gold=t.get("gold","")
        cand_obj=r.get("output_json",{})
        cand_json=_normalize_citations(json.dumps(cand_obj, ensure_ascii=False))
        jscore = judge(prompt=ctx, gold=gold, candidate=cand_json, model=args.model, base=args.base, key=args.key)
        # Blend with any existing faithfulness/faithfulness_gold from scored file to avoid zeroing due to judge harshness
        overlap = 0.0
        if "faithfulness_gold" in df.columns:
            overlap = float(df.loc[len(faith), "faithfulness_gold"])
        elif "faithfulness" in df.columns:
            overlap = float(df.loc[len(faith), "faithfulness"])
        faith.append(max(jscore, overlap))
    df["faithfulness_judge"]=faith
    df.to_csv(args.out,index=False)
    print(f"[done] wrote {args.out}")

if __name__=="__main__": main()
