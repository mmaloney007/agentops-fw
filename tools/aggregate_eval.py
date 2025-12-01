
import argparse, glob, json, os
import pandas as pd, numpy as np
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="out/**/eval.jsonl")
    ap.add_argument("--out", default="out/aggregate/eval_summary.csv")
    args=ap.parse_args()
    paths=glob.glob(args.glob, recursive=True); rows=[]
    for p in paths:
        try:
            for line in open(p,"r",encoding="utf-8"):
                rec=json.loads(line); rec["source"]=p; rows.append(rec)
        except Exception: pass
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not rows: pd.DataFrame().to_csv(args.out,index=False); print(f"[warn] empty {args.out}"); return
    df=pd.DataFrame(rows)
    for c in ["latency_ms","ttft_ms","json_valid","tokens_out"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c], errors="coerce")
    summary=df.groupby("source").agg(
        n=("latency_ms","count"),
        p95_ms=("latency_ms", lambda x: float(np.nanpercentile(x,95))),
        p99_ms=("latency_ms", lambda x: float(np.nanpercentile(x,99))),
        json_valid_pct=("json_valid", lambda x: float(np.nanmean(x)*100.0)),
        avg_ttft_ms=("ttft_ms","mean"),
        avg_tokens_out=("tokens_out","mean"),
    ).reset_index()
    summary.to_csv(args.out,index=False); print(f"[done] {args.out} ({len(summary)} rows)")
if __name__=="__main__": main()
