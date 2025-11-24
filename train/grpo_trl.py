
import os, json, argparse, statistics
from datasets import Dataset
from agent_stable_slo.rollout.engine import provider_generate
from agent_stable_slo.rewards.composite import composite_reward
from agent_stable_slo.logging import wandb_utils as WL
def load_tasks(path: str) -> Dataset:
    rows=[json.loads(x) for x in open(path,"r",encoding="utf-8")]
    return Dataset.from_list(rows)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tasks", default="tasks/fc_tasks.jsonl")
    ap.add_argument("--out", default="out/P1_run")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--max-new-tokens", type=int, default=196)
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
                              "max_new_tokens":args.max_new_tokens,"tasks_file":args.tasks}) as run:
        latencies, ttfts=[], []
        with open(eval_path,"w",encoding="utf-8") as fo:
            for i in range(min(args.steps, len(ds))):
                row=ds[i % len(ds)]
                schema=json.load(open(row["schema_path"],"r",encoding="utf-8"))
                out_json, lat_ms, ttft_ms, tokens=provider_generate(row["prompt"], schema)
                r=composite_reward(out_json, schema, ok_success=1, latency_ms=lat_ms, tokens=max(0,tokens),
                                   lam_latency=lam, mu_cost=mu, disagreement_rate=0.0, gamma_stability=gamma)
                rec={"step":i,"latency_ms":float(round(lat_ms,3)),"ttft_ms":float(round(ttft_ms,3)),
                     "reward":float(r),"json_valid":1 if r>0 else 0,"tokens_out":int(tokens) if isinstance(tokens,int) else -1}
                fo.write(json.dumps(rec)+"\n")
                latencies.append(rec["latency_ms"]); ttfts.append(rec["ttft_ms"])
                WL.log(run, {k:rec[k] for k in ["latency_ms","ttft_ms","reward","json_valid"]}, step=i)
        if latencies:
            s=sorted(latencies); p95=s[int(0.95*(len(s)-1))]; p99=s[int(0.99*(len(s)-1))]
            WL.log(run, {"summary_p95_ms":p95,"summary_p99_ms":p99,"summary_avg_ttft_ms":statistics.mean(ttfts)}, step=len(latencies))
        WL.log_artifact(run, eval_path, name=os.path.basename(args.out)+"-eval", type_="metrics")
    print(f"[done] wrote {eval_path}")
if __name__=="__main__": main()
