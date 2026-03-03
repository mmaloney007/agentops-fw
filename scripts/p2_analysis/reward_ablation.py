#!/usr/bin/env python3
"""
Reward ablation analysis for P2 Section 6.7.

Recomputes composite rewards from logged training data under different
ablation conditions (zeroing out individual reward components) to show
the absolute and delta impact of each penalty term.

The actual training used all-zero penalty weights (lam=mu=gamma=kappa=0),
so the logged reward is simply 2*json_valid.  This script computes what
the reward WOULD have been if the paper's proposed weights had been active.

Composite reward formula (from agent_stable_slo/rewards/composite.py):

    R = schema_valid + ok_success
        + latency_penalty(latency_ms, lam)
        + cost_penalty(tokens, mu)
        + stability_penalty(disagreement_rate, gamma)
        + kappa * (faithfulness - 0.5)

Where:
    schema_valid = json_valid  (since ok_success = json_valid in training)
    ok_success   = json_valid
    latency_penalty = -lam * (latency_ms / 1000)
    cost_penalty    = -mu  * (tokens / 1000)
    stability_penalty = -gamma * disagreement_rate
    faithfulness_term = kappa * (faithfulness - 0.5)

So base = 2 * json_valid, plus four additive penalty/bonus terms.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAINING_DIR = Path(__file__).resolve().parents[2] / "out" / "p2_training_20260124"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "p2_analysis"

# Paper's proposed weights (the "full composite" scenario)
LAM_FULL = 0.1
MU_FULL = 0.05
GAMMA_FULL = 0.1
KAPPA_FULL = 1.0

# Ablation conditions: name -> (lam, mu, gamma, kappa)
ABLATION_CONDITIONS = {
    "baseline":   (LAM_FULL, MU_FULL, GAMMA_FULL, KAPPA_FULL),
    "no_latency": (0.0,      MU_FULL, GAMMA_FULL, KAPPA_FULL),
    "no_cost":    (LAM_FULL, 0.0,     GAMMA_FULL, KAPPA_FULL),
    "no_stab":    (LAM_FULL, MU_FULL, 0.0,        KAPPA_FULL),
    "schema_only":(0.0,      0.0,     0.0,         0.0),
}

# Analysis windows
LAST_N_STEPS = 200   # for mean reward (converged behavior)
LAST_N_VALID = 50    # for json_valid rate


def recompute_reward(rec: dict, lam: float, mu: float, gamma: float,
                     kappa: float) -> float:
    """Recompute composite reward from a log record under given weights."""
    jv = rec["json_valid"]
    base = 2.0 * jv  # schema_valid + ok_success, both = json_valid

    lat_pen = -lam * (rec["latency_ms"] / 1000.0)
    cost_pen = -mu * (rec["tokens_out"] / 1000.0)
    stab_pen = -gamma * rec.get("disagreement_rate", 0.0)
    faith_term = kappa * (rec.get("faithfulness", 1.0) - 0.5)

    return base + lat_pen + cost_pen + stab_pen + faith_term


def load_train_log(path: Path) -> list[dict]:
    """Load a train_log.jsonl file, returning list of record dicts.

    Skips blank lines and lines that fail JSON parsing (truncated writes).
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip corrupted/truncated lines
    return records


def analyze_run(records: list[dict]) -> dict:
    """
    For one run (model/task/seed), compute ablation rewards over the
    last LAST_N_STEPS steps and json_valid rate over the last LAST_N_VALID.
    """
    total = len(records)
    if total == 0:
        return None

    # Use last N steps for converged reward analysis
    tail_reward = records[-min(LAST_N_STEPS, total):]
    tail_valid = records[-min(LAST_N_VALID, total):]

    result = {}

    # Compute mean reward under each ablation condition
    for cond_name, (lam, mu, gamma, kappa) in ABLATION_CONDITIONS.items():
        rewards = [recompute_reward(r, lam, mu, gamma, kappa) for r in tail_reward]
        result[f"{cond_name}_reward"] = mean(rewards)
        result[f"{cond_name}_reward_std"] = stdev(rewards) if len(rewards) > 1 else 0.0

    # Also store the logged (actual training) reward for reference
    logged_rewards = [r["reward"] for r in tail_reward]
    result["logged_reward"] = mean(logged_rewards)

    # json_valid rate (last 50 steps)
    valid_flags = [r["json_valid"] for r in tail_valid]
    result["last50_validity"] = mean(valid_flags)

    # Mean component values (last 200 steps) for interpretability
    result["mean_latency_ms"] = mean(r["latency_ms"] for r in tail_reward)
    result["mean_tokens_out"] = mean(r["tokens_out"] for r in tail_reward)
    result["mean_disagreement"] = mean(
        r.get("disagreement_rate", 0.0) for r in tail_reward
    )
    result["mean_faithfulness"] = mean(
        r.get("faithfulness", 1.0) for r in tail_reward
    )
    result["total_steps"] = total

    return result


def discover_runs(base_dir: Path) -> list[dict]:
    """
    Walk the training output directory and discover all model/task/seed
    combinations with a top-level train_log.jsonl.
    """
    runs = []
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        # Skip non-model directories
        if model_name in ("latex_tables",):
            continue
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            for seed_dir in sorted(task_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                log_path = seed_dir / "train_log.jsonl"
                if log_path.exists():
                    seed = int(seed_dir.name.split("_")[1])
                    runs.append({
                        "model": model_name,
                        "task": task_name,
                        "seed": seed,
                        "log_path": str(log_path),
                    })
    return runs


def main():
    if not TRAINING_DIR.exists():
        print(f"ERROR: Training directory not found: {TRAINING_DIR}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Discover and analyze all runs
    print(f"Scanning {TRAINING_DIR} for training logs...")
    runs = discover_runs(TRAINING_DIR)
    print(f"Found {len(runs)} runs across models/tasks/seeds")

    # Phase 2: Analyze each run
    per_run_results = []
    for i, run_info in enumerate(runs):
        records = load_train_log(Path(run_info["log_path"]))
        analysis = analyze_run(records)
        if analysis is None:
            continue
        entry = {**run_info, **analysis}
        del entry["log_path"]  # Don't persist absolute paths
        per_run_results.append(entry)

        if (i + 1) % 50 == 0:
            print(f"  Analyzed {i + 1}/{len(runs)} runs...")

    print(f"  Analyzed {len(per_run_results)} runs total")

    # Phase 3: Aggregate by model-task (mean across seeds)
    grouped = defaultdict(list)
    for entry in per_run_results:
        key = (entry["model"], entry["task"])
        grouped[key].append(entry)

    aggregated = []
    for (model, task), entries in sorted(grouped.items()):
        agg = {
            "model": model,
            "task": task,
            "n_seeds": len(entries),
        }
        # Average all numeric fields across seeds
        numeric_keys = [k for k in entries[0] if isinstance(entries[0][k], (int, float))
                        and k not in ("seed",)]
        for k in numeric_keys:
            vals = [e[k] for e in entries]
            agg[k] = round(mean(vals), 4)

        # Compute deltas from baseline
        bl = agg["baseline_reward"]
        for cond_name in ABLATION_CONDITIONS:
            if cond_name != "baseline":
                delta = agg[f"{cond_name}_reward"] - bl
                agg[f"delta_{cond_name}"] = round(delta, 4)

        aggregated.append(agg)

    # Phase 4: Also aggregate by model only (across all tasks)
    model_grouped = defaultdict(list)
    for entry in per_run_results:
        model_grouped[entry["model"]].append(entry)

    model_summary = []
    for model, entries in sorted(model_grouped.items()):
        ms = {
            "model": model,
            "n_runs": len(entries),
        }
        numeric_keys = [k for k in entries[0] if isinstance(entries[0][k], (int, float))
                        and k not in ("seed",)]
        for k in numeric_keys:
            vals = [e[k] for e in entries]
            ms[k] = round(mean(vals), 4)

        bl = ms["baseline_reward"]
        for cond_name in ABLATION_CONDITIONS:
            if cond_name != "baseline":
                delta = ms[f"{cond_name}_reward"] - bl
                ms[f"delta_{cond_name}"] = round(delta, 4)

        model_summary.append(ms)

    # Phase 5: Write results
    output = {
        "metadata": {
            "description": "Reward ablation analysis for P2 Section 6.7",
            "training_dir": str(TRAINING_DIR),
            "ablation_conditions": {
                k: {"lam": v[0], "mu": v[1], "gamma": v[2], "kappa": v[3]}
                for k, v in ABLATION_CONDITIONS.items()
            },
            "last_n_steps_for_reward": LAST_N_STEPS,
            "last_n_steps_for_validity": LAST_N_VALID,
            "note": (
                "Training was run with all penalty weights=0 (schema-only). "
                "This analysis recomputes what rewards WOULD have been under "
                "the paper's proposed weights."
            ),
            "formula": (
                "R = 2*json_valid "
                "- lam*(latency_ms/1000) "
                "- mu*(tokens_out/1000) "
                "- gamma*disagreement_rate "
                "+ kappa*(faithfulness - 0.5)"
            ),
        },
        "per_model_task": aggregated,
        "per_model": model_summary,
    }

    out_path = OUTPUT_DIR / "reward_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Phase 6: Print summary table
    print("\n" + "=" * 120)
    print("REWARD ABLATION SUMMARY (per model, averaged across tasks and seeds, last 200 steps)")
    print("=" * 120)
    header = (
        f"{'Model':<20s} "
        f"{'Baseline':>9s} "
        f"{'NoLat':>9s} {'dLat':>7s} "
        f"{'NoCost':>9s} {'dCost':>7s} "
        f"{'NoStab':>9s} {'dStab':>7s} "
        f"{'SchOnly':>9s} {'dSch':>7s} "
        f"{'Valid%':>7s} "
        f"{'LatMs':>8s} "
        f"{'Tok':>6s}"
    )
    print(header)
    print("-" * 120)

    for ms in model_summary:
        row = (
            f"{ms['model']:<20s} "
            f"{ms['baseline_reward']:>9.4f} "
            f"{ms['no_latency_reward']:>9.4f} {ms['delta_no_latency']:>+7.4f} "
            f"{ms['no_cost_reward']:>9.4f} {ms['delta_no_cost']:>+7.4f} "
            f"{ms['no_stab_reward']:>9.4f} {ms['delta_no_stab']:>+7.4f} "
            f"{ms['schema_only_reward']:>9.4f} {ms['delta_schema_only']:>+7.4f} "
            f"{ms['last50_validity'] * 100:>6.1f}% "
            f"{ms['mean_latency_ms']:>8.0f} "
            f"{ms['mean_tokens_out']:>6.0f}"
        )
        print(row)

    print("-" * 120)

    # Grand mean
    all_models_bl = mean(ms["baseline_reward"] for ms in model_summary)
    all_models_nl = mean(ms["no_latency_reward"] for ms in model_summary)
    all_models_nc = mean(ms["no_cost_reward"] for ms in model_summary)
    all_models_ns = mean(ms["no_stab_reward"] for ms in model_summary)
    all_models_so = mean(ms["schema_only_reward"] for ms in model_summary)
    all_models_v = mean(ms["last50_validity"] for ms in model_summary)
    all_models_lat = mean(ms["mean_latency_ms"] for ms in model_summary)
    all_models_tok = mean(ms["mean_tokens_out"] for ms in model_summary)

    print(
        f"{'GRAND MEAN':<20s} "
        f"{all_models_bl:>9.4f} "
        f"{all_models_nl:>9.4f} {all_models_nl - all_models_bl:>+7.4f} "
        f"{all_models_nc:>9.4f} {all_models_nc - all_models_bl:>+7.4f} "
        f"{all_models_ns:>9.4f} {all_models_ns - all_models_bl:>+7.4f} "
        f"{all_models_so:>9.4f} {all_models_so - all_models_bl:>+7.4f} "
        f"{all_models_v * 100:>6.1f}% "
        f"{all_models_lat:>8.0f} "
        f"{all_models_tok:>6.0f}"
    )
    print("=" * 120)

    # Phase 7: Per-task breakdown (compact)
    print("\n" + "=" * 100)
    print("PER-TASK BREAKDOWN (averaged across models and seeds)")
    print("=" * 100)

    task_grouped = defaultdict(list)
    for entry in per_run_results:
        task_grouped[entry["task"]].append(entry)

    header2 = (
        f"{'Task':<8s} "
        f"{'Baseline':>9s} "
        f"{'NoLat':>9s} {'dLat':>7s} "
        f"{'NoCost':>9s} {'dCost':>7s} "
        f"{'NoStab':>9s} {'dStab':>7s} "
        f"{'SchOnly':>9s} {'dSch':>7s} "
        f"{'Valid%':>7s}"
    )
    print(header2)
    print("-" * 100)

    for task in sorted(task_grouped.keys()):
        entries = task_grouped[task]
        bl = mean(e["baseline_reward"] for e in entries)
        nl = mean(e["no_latency_reward"] for e in entries)
        nc = mean(e["no_cost_reward"] for e in entries)
        ns = mean(e["no_stab_reward"] for e in entries)
        so = mean(e["schema_only_reward"] for e in entries)
        v = mean(e["last50_validity"] for e in entries)
        print(
            f"{task:<8s} "
            f"{bl:>9.4f} "
            f"{nl:>9.4f} {nl - bl:>+7.4f} "
            f"{nc:>9.4f} {nc - bl:>+7.4f} "
            f"{ns:>9.4f} {ns - bl:>+7.4f} "
            f"{so:>9.4f} {so - bl:>+7.4f} "
            f"{v * 100:>6.1f}%"
        )

    print("=" * 100)
    print(f"\nDone. Full results: {out_path}")


if __name__ == "__main__":
    main()
