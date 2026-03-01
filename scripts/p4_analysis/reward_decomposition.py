#!/usr/bin/env python3
"""P4 Analysis: Reward Decomposition.

Decomposes composite reward into schema/accuracy/latency/cost components
per model and training step using known reward weights.

Input:  results/p2_all_runs.csv
Output: results/p4_analysis/reward_decomposition.json
        results/curves/pgfplots/p4_reward_decomp.dat
"""
import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
CSV_PATH = ROOT / "results" / "p2_all_runs.csv"
OUT_JSON = ROOT / "results" / "p4_analysis" / "reward_decomposition.json"
OUT_DAT = ROOT / "results" / "curves" / "pgfplots" / "p4_reward_decomp.dat"

# Known weights from GRPO configs
LAM_LATENCY = 0.1
MU_COST = 0.05
GAMMA_STABILITY = 0.1


def load_runs():
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def decompose():
    runs = load_runs()
    results = defaultdict(lambda: defaultdict(list))

    for run in runs:
        model = run["model"]
        task = run["task"]
        seed = int(run["seed"])
        total_steps = int(run["total_steps"])

        # Skip short diagnostic runs
        if total_steps < 100:
            continue

        avg_reward = float(run["avg_reward"])
        json_valid_pct = float(run["json_valid_pct"]) / 100.0
        avg_latency_ms = float(run["avg_latency_ms"])
        avg_tokens = float(run["avg_tokens_out"])
        reward_std = float(run["reward_std"])

        # Decompose composite reward into estimated components
        r_schema = json_valid_pct  # ~1.0 * fraction valid
        r_latency = -LAM_LATENCY * avg_latency_ms / 1000.0  # normalized to seconds
        r_cost = -MU_COST * avg_tokens / 100.0  # normalized
        r_residual = avg_reward - r_schema - r_latency - r_cost

        results[model][task].append({
            "seed": seed,
            "total_steps": total_steps,
            "avg_reward": round(avg_reward, 4),
            "r_schema": round(r_schema, 4),
            "r_residual": round(max(0, r_residual), 4),
            "r_latency": round(r_latency, 4),
            "r_cost": round(r_cost, 4),
            "reward_std": round(reward_std, 4),
            "json_valid_pct": round(json_valid_pct * 100, 1),
        })

    # Aggregate per model (mean across tasks and seeds)
    model_summary = {}
    for model, tasks in sorted(results.items()):
        all_runs = [r for t_runs in tasks.values() for r in t_runs]
        n = len(all_runs)
        if n == 0:
            continue
        model_summary[model] = {
            "n_runs": n,
            "tasks": dict(tasks),
            "mean_reward": round(sum(r["avg_reward"] for r in all_runs) / n, 4),
            "mean_r_schema": round(sum(r["r_schema"] for r in all_runs) / n, 4),
            "mean_r_residual": round(sum(r["r_residual"] for r in all_runs) / n, 4),
            "mean_r_latency": round(sum(r["r_latency"] for r in all_runs) / n, 4),
            "mean_r_cost": round(sum(r["r_cost"] for r in all_runs) / n, 4),
        }

    output = {"models": model_summary, "weights": {"lam_latency": LAM_LATENCY, "mu_cost": MU_COST, "gamma_stability": GAMMA_STABILITY}}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {OUT_JSON}")

    # Write pgfplots data
    OUT_DAT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DAT, "w") as f:
        f.write("model r_schema r_residual r_latency r_cost mean_reward\n")
        for model, data in sorted(model_summary.items()):
            f.write(f"{model} {data['mean_r_schema']:.4f} {data['mean_r_residual']:.4f} "
                    f"{data['mean_r_latency']:.4f} {data['mean_r_cost']:.4f} {data['mean_reward']:.4f}\n")
    print(f"Wrote {OUT_DAT}")

    return output


if __name__ == "__main__":
    decompose()
