#!/usr/bin/env python3
"""
Stub for short GRPO-style finetune: samples tasks, computes rewards, and (placeholder)
would update the model. Currently logs rewards to W&B and writes eval.jsonl so we can
compare before/after; real optimizer integration can be added later.
"""
import os, json, argparse, statistics, time
from agent_stable_slo.train.grpo_trl import load_tasks
from agent_stable_slo.rollout.engine import provider_generate
from agent_stable_slo.rewards.composite import composite_reward
from agent_stable_slo.logging import wandb_utils as WL
from agent_stable_slo.utils.hardware import detect_hardware, recommended_defaults

def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tasks", default="tasks/robust_eval_gold.jsonl")
    ap.add_argument("--out", default=f"out/finetune_stub_{_timestamp()}")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--samples", type=int, default=1)
    args=ap.parse_args()
    hw=detect_hardware()
    hw_cfg=hw.as_dict(); hw_cfg["recommended"]=recommended_defaults(hw)
    print(f"[hardware] {hw.summary()}")
    ds=load_tasks(args.tasks)
    os.makedirs(args.out, exist_ok=True)
    eval_path=os.path.join(args.out,"eval.jsonl")
    lam=float(os.getenv("LAMBDA_LATENCY","0.0"))
    mu=float(os.getenv("MU_COST","0.0"))
    gamma=float(os.getenv("GAMMA_STABILITY","0.0"))
    provider=os.getenv("AOFW_PROVIDER","lmstudio")
    with WL.maybe_run(name=os.path.basename(args.out),
                      config={"provider":provider,"steps":args.steps,"samples":args.samples,"tasks":args.tasks,"hardware":hw_cfg}) as run:
        latencies=[]
        with open(eval_path,"w",encoding="utf-8") as fo:
            for i in range(args.steps):
                row=ds[i % len(ds)]
                schema=json.load(open(row["schema_path"],"r",encoding="utf-8"))
                best=None
                for _ in range(max(1,args.samples)):
                    out_json, lat_ms, ttft_ms, tokens=provider_generate(row["prompt"], schema)
                    r=composite_reward(out_json, schema, ok_success=1, latency_ms=lat_ms, tokens=max(0,tokens),
                                       lam_latency=lam, mu_cost=mu, disagreement_rate=0.0, gamma_stability=gamma)
                    cand={"step":i,"latency_ms":float(round(lat_ms,3)),"ttft_ms":float(round(ttft_ms,3)),
                          "reward":float(r),"json_valid":1 if r>0 else 0,"tokens_out":int(tokens) if isinstance(tokens,int) else -1,
                          "prompt": row["prompt"], "schema_path": row["schema_path"], "output_json": out_json}
                    if "gold" in row: cand["gold"]=row["gold"]
                    if best is None or cand["reward"] > best["reward"]:
                        best=cand
                fo.write(json.dumps(best)+"\n")
                latencies.append(best["latency_ms"])
                WL.log(run, {k:best[k] for k in ["latency_ms","ttft_ms","reward","json_valid"]}, step=i)
        if latencies:
            s=sorted(latencies); p95=s[int(0.95*(len(s)-1))]
            WL.log(run, {"summary_p95_ms":p95,"summary_avg_ttft_ms":statistics.mean(latencies)}, step=len(latencies))
        WL.log_artifact(run, eval_path, name=os.path.basename(args.out)+"-eval", type_="metrics")
    print(f"[done] wrote {eval_path}")

if __name__=="__main__": main()
