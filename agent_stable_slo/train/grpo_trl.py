
import os, json, argparse, statistics
from agent_stable_slo.rollout.engine import provider_generate
from agent_stable_slo.rewards.composite import composite_reward
from agent_stable_slo.logging import wandb_utils as WL
def load_tasks(path: str):
    return [json.loads(x) for x in open(path,"r",encoding="utf-8") if x.strip()]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tasks", default="tasks/fc_tasks.jsonl")
    ap.add_argument("--out", default="out/P1_run")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--start", type=int, default=0, help="Starting step offset (for chunked runs).")
    ap.add_argument("--max-new-tokens", type=int, default=196)
    ap.add_argument("--samples", type=int, default=1, help="Number of samples per task; take best reward.")
    args=ap.parse_args()
    ds=load_tasks(args.tasks)
    lam=float(os.getenv("LAMBDA_LATENCY","0.0"))
    mu=float(os.getenv("MU_COST","0.0"))
    gamma=float(os.getenv("GAMMA_STABILITY","0.0"))
    provider=os.getenv("AOFW_PROVIDER","lmstudio")
    os.makedirs(args.out, exist_ok=True)
    eval_path=os.path.join(args.out,"eval.jsonl")
    with WL.maybe_run(name=os.path.basename(args.out),
                      config={"provider":provider,"lambda":lam,"mu":mu,"gamma":gamma,
                              "max_new_tokens":args.max_new_tokens,"tasks_file":args.tasks,
                              "start":args.start,"steps":args.steps}) as run:
        latencies, ttfts=[], []
        mode="a" if args.start>0 else "w"
        with open(eval_path,mode,encoding="utf-8") as fo:
            for i in range(args.start, args.start + args.steps):
                row=ds[i % len(ds)]
                schema=json.load(open(row["schema_path"],"r",encoding="utf-8"))
                best=None
                for _ in range(max(1, args.samples)):
                    out_json, lat_ms, ttft_ms, tokens=provider_generate(row["prompt"], schema)
                    r=composite_reward(out_json, schema, ok_success=1, latency_ms=lat_ms, tokens=max(0,tokens),
                                       lam_latency=lam, mu_cost=mu, disagreement_rate=0.0, gamma_stability=gamma)
                    cand={"step":i,"latency_ms":float(round(lat_ms,3)),"ttft_ms":float(round(ttft_ms,3)),
                          "reward":float(r),"json_valid":1 if r>0 else 0,"tokens_out":int(tokens) if isinstance(tokens,int) else -1,
                          "prompt": row["prompt"], "schema_path": row["schema_path"], "output_json": out_json}
                    if best is None or cand["reward"] > best["reward"]:
                        best=cand
                rec=best
                if "gold" in row: rec["gold"] = row["gold"]
                fo.write(json.dumps(rec)+"\n")
                latencies.append(rec["latency_ms"]); ttfts.append(rec["ttft_ms"])
                WL.log(run, {k:rec[k] for k in ["latency_ms","ttft_ms","reward","json_valid"]}, step=i)
        if latencies:
            s=sorted(latencies); p95=s[int(0.95*(len(s)-1))]; p99=s[int(0.99*(len(s)-1))]
            summary_step=args.start + len(latencies)  # keep W&B step monotonic when chunking
            WL.log(run, {"summary_p95_ms":p95,"summary_p99_ms":p99,"summary_avg_ttft_ms":statistics.mean(ttfts)}, step=summary_step)
        WL.log_artifact(run, eval_path, name=os.path.basename(args.out)+"-eval", type_="metrics")
    print(f"[done] wrote {eval_path}")
if __name__=="__main__": main()
