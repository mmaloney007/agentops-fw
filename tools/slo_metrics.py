
import argparse, glob, json, os, pandas as pd
def compute_success_at_slo(path, thresholds):
    rows=[json.loads(x) for x in open(path,"r",encoding="utf-8")]
    out=[]; n=len(rows) or 1
    for thr in thresholds:
        ok=sum(1 for r in rows if r.get("json_valid",0) and r.get("latency_ms",10**9)<=thr)
        out.append((path, thr, 100.0*ok/n))
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="out/**/eval.jsonl")
    ap.add_argument("--out", default="out/aggregate/slo_curves.csv")
    ap.add_argument("--thresholds", default="200,300,500,800")
    args=ap.parse_args()
    thr=[int(x) for x in args.thresholds.split(",")]
    rows=[]
    for p in glob.glob(args.glob, recursive=True):
        try: rows.extend(compute_success_at_slo(p, thr))
        except Exception: pass
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows, columns=["source","thr_ms","success_pct"]).to_csv(args.out,index=False)
    print(f"[done] {args.out} ({len(rows)} rows)")
if __name__=="__main__": main()
