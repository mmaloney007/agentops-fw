
import os, json, argparse, hashlib, statistics
from agent_stable_slo.rollout.engine import provider_generate
from agent_stable_slo.logging import wandb_utils as WL
def hash_answer(j: dict) -> str:
    return hashlib.sha256(json.dumps(j, sort_keys=True).encode("utf-8")).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/fc_tasks.jsonl")
    ap.add_argument("--schema", default="tasks/schemas/fc_schema.json")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", default="out/stability/fc_stability.jsonl")
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tasks=[json.loads(x) for x in open(args.tasks,"r",encoding="utf-8")]
    schema=json.load(open(args.schema,"r",encoding="utf-8"))
    with WL.maybe_run(name="stability-"+os.path.basename(args.out), config={"tasks":args.tasks,"schema":args.schema,"runs":args.runs}) as run, open(args.out,"w",encoding="utf-8") as fo:
        for t in tasks:
            prompt=t["prompt"]; hashes=[]; latencies=[]
            for _ in range(args.runs):
                out_json, lat_ms, _, _=provider_generate(prompt, schema)
                hashes.append(hash_answer(out_json)); latencies.append(lat_ms)
            mode=max(set(hashes), key=hashes.count)
            disagreement=1.0 - hashes.count(mode)/len(hashes)
            rec={"prompt":prompt,"runs":args.runs,"disagreement_rate":float(disagreement),"mean_latency_ms":float(statistics.mean(latencies))}
            fo.write(json.dumps(rec)+"\n"); WL.log(run, rec)
    print(f"[done] wrote {args.out}")
if __name__=="__main__": main()
