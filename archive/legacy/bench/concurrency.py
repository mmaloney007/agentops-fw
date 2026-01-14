
import argparse, json, time, threading, queue
from agent_stable_slo.rollout.engine import provider_generate
from agent_stable_slo.logging import wandb_utils as WL
def worker(task_q, results, schema):
    while True:
        try: prompt=task_q.get_nowait()
        except Exception: return
        try:
            j, lat_ms, ttft_ms, _=provider_generate(prompt, schema)
            ok=1 if isinstance(j,dict) and len(j)>0 else 0
        except Exception:
            lat_ms, ttft_ms, ok=1e6, 1e6, 0
        results.append((lat_ms, ttft_ms, ok)); task_q.task_done()
def run_bench(qps, duration, schema_path):
    from json import load
    with WL.maybe_run(name=f"bench-qps{qps}-dur{duration}", config={"qps":qps,"duration":duration,"schema":schema_path}) as run:
        schema=load(open(schema_path,"r",encoding="utf-8"))
        total=int(qps*duration); prompts=[f"Case {i}: return JSON conforming to the schema" for i in range(total)]
        task_q=queue.Queue(); [task_q.put(p) for p in prompts]
        results=[]; concurrency=min(qps,64)
        threads=[threading.Thread(target=worker, args=(task_q,results,schema), daemon=True) for _ in range(concurrency)]
        [t.start() for t in threads]; t0=time.time(); task_q.join(); elapsed=time.time()-t0
        lat=[r[0] for r in results]; ttft=[r[1] for r in results]; ok=[r[2] for r in results]
        if lat:
            s=sorted(lat); p95=s[int(0.95*(len(s)-1))]; p99=s[int(0.99*(len(s)-1))]
        else: p95=p99=0.0
        succ=sum(ok)/max(1,len(ok))*100.0
        out={"n":len(lat),"elapsed_s":round(elapsed,2),"qps_effective": round(len(lat)/elapsed,2) if elapsed>0 else 0.0,"p95_ms":round(p95,1),"p99_ms":round(p99,1),"success_pct":round(succ,1),"avg_ttft_ms": round(sum(ttft)/max(1,len(ttft)),1) if ttft else 0.0}
        WL.log(run, out); print(out)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--qps", type=int, default=4)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--slo-ms", type=int, default=300)
    ap.add_argument("--schema", default="tasks/schemas/fc_schema.json")
    args=ap.parse_args()
    run_bench(args.qps, args.duration, args.schema)
if __name__=="__main__": main()
