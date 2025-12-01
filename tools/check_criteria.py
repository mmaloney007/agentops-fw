#!/usr/bin/env python3
"""
Check scored evaluation files against thresholds in criteria.yaml.
Usage:
  python3 tools/check_criteria.py --scored out/aggregate/scored_qwen_gold.csv --criteria criteria.yaml --out out/aggregate/check_qwen_gold.json
"""
import argparse, json, yaml, pandas as pd, os

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--criteria", default="criteria.yaml")
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    with open(args.criteria,"r",encoding="utf-8") as f:
        crit=yaml.safe_load(f)
    df=pd.read_csv(args.scored)
    summary={}
    # thresholds
    thresh = crit.get("structure",{})
    faith = crit.get("faithfulness",{})
    slo = crit.get("slo",{})
    summary["json_valid_pass_pct"] = (df["json_valid"] >= thresh.get("json_valid",1)).mean()
    summary["f1_pass_pct"] = (df["f1"] >= thresh.get("f1",1)).mean()
    summary["em_pass_pct"] = (df["em"] >= thresh.get("em",1)).mean()
    summary["faithfulness_pass_pct"] = (df["faithfulness"] >= faith.get("faithfulness",1)).mean()
    summary["cite_acc_pass_pct"] = (df["cite_acc"] >= faith.get("cite_acc",1)).mean()
    summary["p95_ms"] = float(df["latency_ms"].quantile(0.95))
    summary["p95_within_slo"] = summary["p95_ms"] <= slo.get("p95_ms", float("inf"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    print(f"[done] criteria check -> {args.out}")

if __name__=="__main__": main()
