#!/usr/bin/env python3
"""
Generate comprehensive training curve datasets from P2 GRPO training logs.

Reads train_log.jsonl files from out/p2_training_20260124/{model}/{task}/seed_{N}/
and produces:
  1. reward_curves.json       -- reward over steps per model/task, averaged across seeds
  2. validity_curves.json     -- rolling-window JSON validity rate per model/task
  3. capacity_threshold.json  -- model size vs final last-50 validity with error bars
  4. task_difficulty.json     -- steps to reach 50% validity threshold, per task/model
  5. forgetting_analysis.json -- Mixed-task validity vs avg single-task validity

Also generates pgfplots-ready .dat files for direct LaTeX inclusion.
"""

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("/home/maloney/Documents/GitHub/agentops-fw")
DATA_DIR = BASE_DIR / "out" / "p2_training_20260124"
OUT_DIR = BASE_DIR / "results" / "curves"
PGF_DIR = OUT_DIR / "pgfplots"

# Target models and their parameter counts (billions)
MODEL_SIZES = {
    "llama-3.2-1b": 1.0,
    "llama-3.2-3b": 3.0,
    "qwen2.5-3b": 3.0,
    "phi-3-mini": 3.8,
    "qwen3-4b": 4.0,
    "yi-1.5-6b": 6.0,
    "mistral-7b-v0.3": 7.0,
    "ministral-8b": 8.0,
    "llama-3.1-8b": 8.0,
    "gemma-2-9b": 9.0,
    "gemma-3-12b": 12.0,
}

# Canonical task list (exclude T6 which only llama-3.2-1b has)
CANONICAL_TASKS = ["T1", "T2", "T3", "T4", "T5", "Mixed"]

# Rolling window size for validity curves
ROLLING_WINDOW = 50

# Threshold for "learned" validity
VALIDITY_THRESHOLD = 0.50

# Minimum steps required to consider a log valid
MIN_STEPS = 20

SEEDS = [42, 123, 456]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_train_log(path: Path) -> list[dict]:
    """Load a train_log.jsonl file, returning a list of step dicts."""
    entries = []
    if not path.exists():
        return entries
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_all_data() -> dict:
    """
    Returns nested dict: data[model][task][seed] = list of step dicts
    Only includes target models with sufficient data.
    """
    data = {}
    for model in MODEL_SIZES:
        model_dir = DATA_DIR / model
        if not model_dir.is_dir():
            print(f"  [SKIP] {model}: directory not found")
            continue
        data[model] = {}
        for task in CANONICAL_TASKS:
            task_dir = model_dir / task
            if not task_dir.is_dir():
                continue
            data[model][task] = {}
            for seed in SEEDS:
                seed_dir = task_dir / f"seed_{seed}"
                log_path = seed_dir / "train_log.jsonl"
                entries = load_train_log(log_path)
                if len(entries) >= MIN_STEPS:
                    data[model][task][seed] = entries
            if not data[model][task]:
                del data[model][task]
        if not data[model]:
            del data[model]
    return data


# ---------------------------------------------------------------------------
# Utility: align seeds to a common step grid
# ---------------------------------------------------------------------------

def extract_series(entries: list[dict], field: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract (steps, values) arrays for a given field from a single seed's log."""
    steps = np.array([e["step"] for e in entries], dtype=float)
    values = np.array([e.get(field, 0) for e in entries], dtype=float)
    return steps, values


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean with edge-aware padding (shorter window at start)."""
    result = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start : i + 1])
    return result


def aggregate_seeds(
    seed_data: dict[int, list[dict]],
    field: str,
    apply_rolling: bool = False,
    window: int = ROLLING_WINDOW,
) -> dict:
    """
    Aggregate a field across seeds.
    Returns dict with keys: steps, mean, std, n_seeds, per_seed.
    All arrays aligned to the shortest common length.
    """
    all_steps = []
    all_values = []

    for seed, entries in sorted(seed_data.items()):
        steps, vals = extract_series(entries, field)
        if apply_rolling:
            vals = rolling_mean(vals, window)
        all_steps.append(steps)
        all_values.append(vals)

    if not all_values:
        return {"steps": [], "mean": [], "std": [], "n_seeds": 0, "per_seed": {}}

    # Align to shortest length
    min_len = min(len(v) for v in all_values)
    aligned = np.stack([v[:min_len] for v in all_values], axis=0)
    steps_common = all_steps[0][:min_len].tolist()

    mean_vals = np.mean(aligned, axis=0)
    std_vals = np.std(aligned, axis=0)

    per_seed = {}
    for seed, vals in zip(sorted(seed_data.keys()), all_values):
        per_seed[str(seed)] = vals[:min_len].tolist()

    return {
        "steps": steps_common,
        "mean": mean_vals.tolist(),
        "std": std_vals.tolist(),
        "n_seeds": len(seed_data),
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# 1. Reward curves
# ---------------------------------------------------------------------------

def generate_reward_curves(data: dict) -> dict:
    """Reward over steps for each model, averaged across seeds, per task."""
    result = {}
    for model in sorted(data.keys()):
        result[model] = {"size_b": MODEL_SIZES[model], "tasks": {}}
        for task in sorted(data[model].keys()):
            agg = aggregate_seeds(data[model][task], "reward")
            result[model]["tasks"][task] = agg
    return result


# ---------------------------------------------------------------------------
# 2. Validity curves (rolling window)
# ---------------------------------------------------------------------------

def generate_validity_curves(data: dict) -> dict:
    """JSON validity rate (rolling window) over steps per model/task."""
    result = {}
    for model in sorted(data.keys()):
        result[model] = {"size_b": MODEL_SIZES[model], "tasks": {}}
        for task in sorted(data[model].keys()):
            agg = aggregate_seeds(
                data[model][task], "json_valid", apply_rolling=True, window=ROLLING_WINDOW
            )
            result[model]["tasks"][task] = agg
    return result


# ---------------------------------------------------------------------------
# 3. Capacity threshold: model size vs final last-50 validity
# ---------------------------------------------------------------------------

def generate_capacity_threshold(data: dict) -> dict:
    """
    For each model, compute last-50-step JSON validity averaged across all
    single tasks (T1-T5) and all seeds.  Returns model size (x) vs validity (y)
    with error bars from seeds.
    """
    result = {"models": []}
    single_tasks = [t for t in CANONICAL_TASKS if t != "Mixed"]

    for model in sorted(data.keys(), key=lambda m: MODEL_SIZES[m]):
        seed_finals = []  # one value per (task, seed)
        task_details = {}

        for task in single_tasks:
            if task not in data[model]:
                continue
            task_seed_vals = []
            for seed, entries in data[model][task].items():
                last_n = entries[-50:] if len(entries) >= 50 else entries
                validity = np.mean([e.get("json_valid", 0) for e in last_n])
                seed_finals.append(validity)
                task_seed_vals.append(validity)
            if task_seed_vals:
                task_details[task] = {
                    "mean": float(np.mean(task_seed_vals)),
                    "std": float(np.std(task_seed_vals)),
                    "n": len(task_seed_vals),
                }

        if seed_finals:
            result["models"].append({
                "model": model,
                "size_b": MODEL_SIZES[model],
                "mean_validity": float(np.mean(seed_finals)),
                "std_validity": float(np.std(seed_finals)),
                "n_observations": len(seed_finals),
                "task_breakdown": task_details,
            })

    return result


# ---------------------------------------------------------------------------
# 4. Task difficulty: steps to reach 50% validity threshold
# ---------------------------------------------------------------------------

def steps_to_threshold(
    entries: list[dict], field: str, threshold: float, window: int
) -> int | None:
    """
    Return the first step at which the rolling mean of `field` reaches
    `threshold`, or None if never reached.
    """
    vals = np.array([e.get(field, 0) for e in entries], dtype=float)
    rm = rolling_mean(vals, window)
    indices = np.where(rm >= threshold)[0]
    if len(indices) == 0:
        return None
    return int(entries[int(indices[0])]["step"])


def generate_task_difficulty(data: dict) -> dict:
    """Per-task learning speed: steps to reach VALIDITY_THRESHOLD."""
    result = {"threshold": VALIDITY_THRESHOLD, "window": ROLLING_WINDOW, "tasks": {}}

    for task in CANONICAL_TASKS:
        task_result = {}
        for model in sorted(data.keys(), key=lambda m: MODEL_SIZES[m]):
            if task not in data[model]:
                continue
            seed_steps = []
            for seed, entries in data[model][task].items():
                s = steps_to_threshold(entries, "json_valid", VALIDITY_THRESHOLD, ROLLING_WINDOW)
                seed_steps.append(s)

            reached = [s for s in seed_steps if s is not None]
            task_result[model] = {
                "size_b": MODEL_SIZES[model],
                "seeds_reached": len(reached),
                "seeds_total": len(seed_steps),
                "mean_steps": float(np.mean(reached)) if reached else None,
                "std_steps": float(np.std(reached)) if reached else None,
                "min_steps": int(min(reached)) if reached else None,
                "max_steps": int(max(reached)) if reached else None,
                "per_seed": {
                    str(seed): s
                    for seed, s in zip(sorted(data[model][task].keys()), seed_steps)
                },
            }
        result["tasks"][task] = task_result

    return result


# ---------------------------------------------------------------------------
# 5. Forgetting analysis: Mixed vs average single-task validity
# ---------------------------------------------------------------------------

def generate_forgetting_analysis(data: dict) -> dict:
    """
    Compare Mixed task last-50 validity vs average of single-task last-50 validity.
    Positive interference_delta = Mixed does better; negative = forgetting.
    """
    single_tasks = [t for t in CANONICAL_TASKS if t != "Mixed"]
    result = {"models": []}

    for model in sorted(data.keys(), key=lambda m: MODEL_SIZES[m]):
        # Mixed validity
        mixed_vals = []
        if "Mixed" in data[model]:
            for seed, entries in data[model]["Mixed"].items():
                last_n = entries[-50:] if len(entries) >= 50 else entries
                mixed_vals.append(np.mean([e.get("json_valid", 0) for e in last_n]))

        # Average single-task validity
        single_vals = []
        per_task = {}
        for task in single_tasks:
            if task not in data[model]:
                continue
            task_seeds = []
            for seed, entries in data[model][task].items():
                last_n = entries[-50:] if len(entries) >= 50 else entries
                v = np.mean([e.get("json_valid", 0) for e in last_n])
                single_vals.append(v)
                task_seeds.append(v)
            per_task[task] = float(np.mean(task_seeds))

        if mixed_vals and single_vals:
            mixed_mean = float(np.mean(mixed_vals))
            single_mean = float(np.mean(single_vals))
            result["models"].append({
                "model": model,
                "size_b": MODEL_SIZES[model],
                "mixed_validity": mixed_mean,
                "mixed_std": float(np.std(mixed_vals)),
                "single_avg_validity": single_mean,
                "single_std": float(np.std(single_vals)),
                "interference_delta": round(mixed_mean - single_mean, 4),
                "per_task_single": per_task,
            })

    return result


# ---------------------------------------------------------------------------
# pgfplots .dat file generators
# ---------------------------------------------------------------------------

def write_pgf_reward(data: dict, reward_curves: dict):
    """Write reward_by_model.dat with columns: step  model1_mean  model1_std  ..."""
    # Find common step count across all models (use T1 as reference task)
    # We'll write one file per task to keep things clean.
    for task in CANONICAL_TASKS:
        lines = []
        models_with_task = []
        arrays = {}  # model -> (mean, std) arrays

        for model in sorted(data.keys(), key=lambda m: MODEL_SIZES[m]):
            if model in reward_curves and task in reward_curves[model]["tasks"]:
                rc = reward_curves[model]["tasks"][task]
                if rc["steps"]:
                    models_with_task.append(model)
                    arrays[model] = (rc["mean"], rc["std"])

        if not models_with_task:
            continue

        # Align to shortest
        min_len = min(len(arrays[m][0]) for m in models_with_task)
        steps = reward_curves[models_with_task[0]]["tasks"][task]["steps"][:min_len]

        # Header
        header_parts = ["step"]
        for m in models_with_task:
            safe = m.replace("-", "_").replace(".", "")
            header_parts.append(f"{safe}_mean")
            header_parts.append(f"{safe}_std")
        lines.append("  ".join(header_parts))

        # Data rows (subsample every 10 steps to keep .dat manageable)
        for i in range(0, min_len, 10):
            row = [f"{int(steps[i])}"]
            for m in models_with_task:
                row.append(f"{arrays[m][0][i]:.4f}")
                row.append(f"{arrays[m][1][i]:.4f}")
            lines.append("  ".join(row))

        out_path = PGF_DIR / f"reward_{task}.dat"
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")


def write_pgf_validity(data: dict, validity_curves: dict):
    """Write validity_by_model.dat per task."""
    for task in CANONICAL_TASKS:
        lines = []
        models_with_task = []
        arrays = {}

        for model in sorted(data.keys(), key=lambda m: MODEL_SIZES[m]):
            if model in validity_curves and task in validity_curves[model]["tasks"]:
                vc = validity_curves[model]["tasks"][task]
                if vc["steps"]:
                    models_with_task.append(model)
                    arrays[model] = (vc["mean"], vc["std"])

        if not models_with_task:
            continue

        min_len = min(len(arrays[m][0]) for m in models_with_task)
        steps = validity_curves[models_with_task[0]]["tasks"][task]["steps"][:min_len]

        header_parts = ["step"]
        for m in models_with_task:
            safe = m.replace("-", "_").replace(".", "")
            header_parts.append(f"{safe}_mean")
            header_parts.append(f"{safe}_std")
        lines.append("  ".join(header_parts))

        for i in range(0, min_len, 10):
            row = [f"{int(steps[i])}"]
            for m in models_with_task:
                row.append(f"{arrays[m][0][i]:.4f}")
                row.append(f"{arrays[m][1][i]:.4f}")
            lines.append("  ".join(row))

        out_path = PGF_DIR / f"validity_{task}.dat"
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")


def write_pgf_capacity(capacity: dict):
    """Write capacity_threshold.dat: size_b  model  mean_validity  std_validity."""
    lines = ["size_b  model  mean_validity  std_validity"]
    for entry in capacity["models"]:
        lines.append(
            f"{entry['size_b']:.1f}  {entry['model']}  "
            f"{entry['mean_validity']:.4f}  {entry['std_validity']:.4f}"
        )
    out_path = PGF_DIR / "capacity_threshold.dat"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(
    data: dict,
    reward_curves: dict,
    validity_curves: dict,
    capacity: dict,
    task_diff: dict,
    forgetting: dict,
):
    """Print a human-readable summary of key findings."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("  P2 TRAINING CURVE ANALYSIS -- KEY FINDINGS")
    print(f"{sep}\n")

    # --- Data coverage ---
    total_logs = sum(
        len(data[m][t]) for m in data for t in data[m]
    )
    print(f"Data loaded: {len(data)} models, {total_logs} seed-level logs\n")

    for model in sorted(data.keys(), key=lambda m: MODEL_SIZES[m]):
        tasks = sorted(data[model].keys())
        seeds_per_task = {t: len(data[model][t]) for t in tasks}
        steps_info = []
        for t in tasks:
            for seed, entries in data[model][t].items():
                steps_info.append(len(entries))
        avg_steps = np.mean(steps_info) if steps_info else 0
        print(
            f"  {model:20s} ({MODEL_SIZES[model]:5.1f}B): "
            f"{len(tasks)} tasks, seeds/task={seeds_per_task}, "
            f"avg {avg_steps:.0f} steps/log"
        )

    # --- Capacity threshold ---
    print(f"\n{sep}")
    print("  CAPACITY THRESHOLD (last-50 JSON validity, single tasks)")
    print(f"{sep}\n")
    print(f"  {'Model':20s} {'Size':>6s} {'Validity':>10s} {'Std':>8s}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8}")
    for entry in capacity["models"]:
        print(
            f"  {entry['model']:20s} {entry['size_b']:5.1f}B "
            f"{entry['mean_validity']:10.1%} {entry['std_validity']:8.1%}"
        )

    # Find threshold crossing
    sorted_models = sorted(capacity["models"], key=lambda x: x["size_b"])
    below_threshold = [m for m in sorted_models if m["mean_validity"] < 0.80]
    above_threshold = [m for m in sorted_models if m["mean_validity"] >= 0.80]
    if below_threshold and above_threshold:
        boundary_low = below_threshold[-1]
        boundary_high = above_threshold[0]
        print(
            f"\n  >> Capacity threshold zone: {boundary_low['size_b']:.1f}B "
            f"({boundary_low['mean_validity']:.1%}) -> "
            f"{boundary_high['size_b']:.1f}B ({boundary_high['mean_validity']:.1%})"
        )

    # --- Task difficulty ---
    print(f"\n{sep}")
    print(f"  TASK DIFFICULTY (steps to {VALIDITY_THRESHOLD:.0%} validity, window={ROLLING_WINDOW})")
    print(f"{sep}\n")
    for task in CANONICAL_TASKS:
        if task not in task_diff["tasks"]:
            continue
        td = task_diff["tasks"][task]
        reached_models = [
            m for m, v in td.items() if v.get("mean_steps") is not None
        ]
        unreached_models = [
            m for m, v in td.items() if v.get("mean_steps") is None
        ]
        if reached_models:
            avg_steps_to_thresh = np.mean([td[m]["mean_steps"] for m in reached_models])
        else:
            avg_steps_to_thresh = float("inf")

        print(f"  {task}:")
        print(f"    Models reaching threshold: {len(reached_models)}/{len(td)}")
        if reached_models:
            print(f"    Avg steps to threshold:    {avg_steps_to_thresh:.0f}")
            fastest = min(reached_models, key=lambda m: td[m]["mean_steps"])
            slowest = max(reached_models, key=lambda m: td[m]["mean_steps"])
            print(f"    Fastest: {fastest} ({td[fastest]['mean_steps']:.0f} steps)")
            print(f"    Slowest: {slowest} ({td[slowest]['mean_steps']:.0f} steps)")
        if unreached_models:
            print(f"    Never reached: {', '.join(unreached_models)}")
        print()

    # --- Forgetting analysis ---
    print(f"{sep}")
    print("  FORGETTING ANALYSIS (Mixed vs avg single-task validity)")
    print(f"{sep}\n")
    print(
        f"  {'Model':20s} {'Size':>6s} {'Mixed':>8s} {'Single':>8s} {'Delta':>8s} {'Verdict':>12s}"
    )
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    for entry in forgetting["models"]:
        delta = entry["interference_delta"]
        verdict = "FORGETTING" if delta < -0.05 else ("BOOST" if delta > 0.05 else "NEUTRAL")
        print(
            f"  {entry['model']:20s} {entry['size_b']:5.1f}B "
            f"{entry['mixed_validity']:8.1%} {entry['single_avg_validity']:8.1%} "
            f"{delta:+8.1%} {verdict:>12s}"
        )

    # Overall forgetting trend
    deltas = [e["interference_delta"] for e in forgetting["models"]]
    if deltas:
        avg_delta = np.mean(deltas)
        print(f"\n  >> Average interference delta: {avg_delta:+.1%}")
        if avg_delta < -0.03:
            print("  >> FINDING: Multi-task training shows forgetting across the board")
        elif avg_delta > 0.03:
            print("  >> FINDING: Multi-task training shows positive transfer")
        else:
            print("  >> FINDING: Multi-task interference is model-dependent")

    # --- Reward trajectory highlights ---
    print(f"\n{sep}")
    print("  REWARD TRAJECTORY HIGHLIGHTS (final mean reward, T1)")
    print(f"{sep}\n")
    for model in sorted(data.keys(), key=lambda m: MODEL_SIZES[m]):
        if "T1" in reward_curves.get(model, {}).get("tasks", {}):
            rc = reward_curves[model]["tasks"]["T1"]
            if rc["mean"]:
                final_reward = rc["mean"][-1]
                initial_reward = rc["mean"][0]
                improvement = final_reward - initial_reward
                print(
                    f"  {model:20s}: {initial_reward:.2f} -> {final_reward:.2f} "
                    f"(delta={improvement:+.2f})"
                )

    print(f"\n{sep}")
    print("  ANALYSIS COMPLETE")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading training logs...")
    data = load_all_data()

    if not data:
        print("ERROR: No valid training data found. Check DATA_DIR path.")
        sys.exit(1)

    # Create output directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PGF_DIR.mkdir(parents=True, exist_ok=True)

    # --- Generate all curve datasets ---
    print("Generating reward curves...")
    reward_curves = generate_reward_curves(data)
    with open(OUT_DIR / "reward_curves.json", "w") as f:
        json.dump(reward_curves, f, indent=2)
    print(f"  -> {OUT_DIR / 'reward_curves.json'}")

    print("Generating validity curves...")
    validity_curves = generate_validity_curves(data)
    with open(OUT_DIR / "validity_curves.json", "w") as f:
        json.dump(validity_curves, f, indent=2)
    print(f"  -> {OUT_DIR / 'validity_curves.json'}")

    print("Generating capacity threshold data...")
    capacity = generate_capacity_threshold(data)
    with open(OUT_DIR / "capacity_threshold.json", "w") as f:
        json.dump(capacity, f, indent=2)
    print(f"  -> {OUT_DIR / 'capacity_threshold.json'}")

    print("Generating task difficulty data...")
    task_diff = generate_task_difficulty(data)
    with open(OUT_DIR / "task_difficulty.json", "w") as f:
        json.dump(task_diff, f, indent=2)
    print(f"  -> {OUT_DIR / 'task_difficulty.json'}")

    print("Generating forgetting analysis...")
    forgetting = generate_forgetting_analysis(data)
    with open(OUT_DIR / "forgetting_analysis.json", "w") as f:
        json.dump(forgetting, f, indent=2)
    print(f"  -> {OUT_DIR / 'forgetting_analysis.json'}")

    # --- Generate pgfplots .dat files ---
    print("Generating pgfplots .dat files...")
    write_pgf_reward(data, reward_curves)
    write_pgf_validity(data, validity_curves)
    write_pgf_capacity(capacity)
    print(f"  -> {PGF_DIR}/")

    # List generated .dat files
    dat_files = sorted(PGF_DIR.glob("*.dat"))
    for f in dat_files:
        print(f"     {f.name}")

    # --- Print summary ---
    print_summary(data, reward_curves, validity_curves, capacity, task_diff, forgetting)


if __name__ == "__main__":
    main()
