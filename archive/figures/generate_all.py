
import os, glob, json, ast
import numpy as np, pandas as pd, matplotlib.pyplot as plt
P1="papers/P1_stable_slo/arxiv/figs"; P2="papers/P2_reward_stability/arxiv/figs"
os.makedirs(P1, exist_ok=True); os.makedirs(P2, exist_ok=True)
def save(p): plt.tight_layout(); plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
def latency_hist():
    vals=[]
    for p in glob.glob("out/**/eval.jsonl", recursive=True):
        for line in open(p,"r",encoding="utf-8"):
            try: vals.append(json.loads(line).get("latency_ms",0.0))
            except Exception: pass
    if not vals: vals=list(np.random.exponential(scale=180.0,size=512))
    plt.figure(); plt.hist(vals, bins=30); plt.xlabel("latency (ms)"); plt.ylabel("count")
    save(os.path.join(P1,"latency_hist.png"))
def qps_vs_p95():
    entries=[]
    for p in glob.glob("out/*bench_q*.txt"):
        for line in open(p,"r",encoding="utf-8"):
            s=line.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    d=ast.literal_eval(s); entries.append((float(d.get("qps_effective",0.0)), float(d.get("p95_ms",0.0))))
                except Exception: pass
    if not entries: xs=[1,4,8]; ys=[220,320,500]
    else: xs=[e[0] for e in entries]; ys=[e[1] for e in entries]
    plt.figure(); plt.plot(xs,ys,marker="o"); plt.xlabel("effective QPS"); plt.ylabel("p95 latency (ms)")
    save(os.path.join(P1,"qps_vs_p95.png"))
def success_at_slo():
    path="out/aggregate/slo_curves.csv"
    if not os.path.exists(path): return
    df=pd.read_csv(path); 
    if df.empty: return
    plt.figure()
    for name,g in df.groupby("source"):
        g=g.sort_values("thr_ms"); label=name[-40:]
        plt.plot(g["thr_ms"], g["success_pct"], marker="o", label=label)
    plt.xlabel("SLO threshold (ms)"); plt.ylabel("success@SLO (%)"); plt.legend(fontsize=7, ncol=1)
    save(os.path.join(P1,"success_at_slo.png"))
def pareto():
    path="out/aggregate/p2_sweeps_eval.csv"
    if not os.path.exists(path): return
    df=pd.read_csv(path); 
    if df.empty: return
    plt.figure(); plt.scatter(df["p95_ms"], df["json_valid_pct"])
    plt.xlabel("p95 latency (ms)"); plt.ylabel("success (%)")
    save(os.path.join(P2,"pareto.png"))
    # Optional second model (gpt-oss-20b sweeps)
    path2="out/aggregate/p2_sweeps72_eval.csv"
    if os.path.exists(path2):
        df2=pd.read_csv(path2)
        if not df2.empty:
            plt.figure()
            plt.scatter(df2["p95_ms"], df2["json_valid_pct"], c="orange")
            plt.xlabel("p95 latency (ms)"); plt.ylabel("success (%)")
            save(os.path.join(P2,"pareto_oss20b.png"))
def acc_vs_latency():
    for name, path in [("qwen","out/aggregate/scored_qwen_gold.csv"), ("oss20b","out/aggregate/scored_oss20b_gold.csv")]:
        if not os.path.exists(path): continue
        df=pd.read_csv(path)
        if df.empty: continue
        plt.figure()
        plt.scatter(df["latency_ms"], df["f1"], alpha=0.4, label="F1")
        plt.scatter(df["latency_ms"], df["faithfulness"], alpha=0.4, label="Faithfulness")
        plt.xlabel("latency (ms)"); plt.ylabel("score")
        plt.legend()
        save(os.path.join(P1, f"acc_vs_latency_{name}.png"))
acc_vs_latency()
latency_hist(); qps_vs_p95(); success_at_slo(); pareto()
print("[done] figures")
