import argparse
import hashlib
import json
import os
import statistics
from typing import Dict, List

from agent_stable_slo.rollout.engine import provider_generate
from agent_stable_slo.utils.repro import set_seed
from agent_stable_slo.logging import wandb_utils as WL


def _canonical_json(obj: Dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _equivalence_key(obj: Dict, equivalence: str) -> str:
    if equivalence == "canonical_json":
        return _canonical_json(obj)
    return _hash_text(json.dumps(obj, ensure_ascii=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/fc_tasks.jsonl")
    ap.add_argument("--schema", default="tasks/schemas/fc_schema.json")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--equivalence", default="canonical_json")
    ap.add_argument("--out", default="out/stability/fc_stability.jsonl")
    args = ap.parse_args()

    set_seed(args.seed, deterministic=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.tasks, "r", encoding="utf-8") as f:
        tasks = [json.loads(x) for x in f if x.strip()]
    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)

    run_name = "stability-" + os.path.basename(args.out)
    with (
        WL.maybe_run(
            name=run_name,
            config={
                "tasks": args.tasks,
                "schema": args.schema,
                "runs": args.runs,
                "seed": args.seed,
                "equivalence": args.equivalence,
            },
            require_online=False,
        ) as run,
        open(args.out, "w", encoding="utf-8") as fo,
    ):
        for t in tasks:
            prompt = t["prompt"]
            keys: List[str] = []
            latencies: List[float] = []
            for _ in range(args.runs):
                out_json, lat_ms, _, _ = provider_generate(prompt, schema)
                keys.append(_equivalence_key(out_json, args.equivalence))
                latencies.append(lat_ms)
            mode = max(set(keys), key=keys.count)
            disagreement = 1.0 - keys.count(mode) / len(keys)
            total_agreement = keys.count(mode) / len(keys)
            rec = {
                "prompt": prompt,
                "runs": args.runs,
                "seed": args.seed,
                "equivalence": args.equivalence,
                "disagreement_rate": float(disagreement),
                "total_agreement_rate": float(total_agreement),
                "mean_latency_ms": float(statistics.mean(latencies)),
            }
            fo.write(json.dumps(rec) + "\n")
            WL.log(run, rec)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
