#!/usr/bin/env python3
"""
Analyze GRPO training experiments across compute paths and models.

Reads grpo_log.jsonl files from results/experiments/*/ and produces:
- LaTeX tables (timing, power, reward) to stdout
- pgfplots .dat files to --dat-dir
- Markdown summary to stderr

Usage:
    python scripts/analyze_experiments.py \
        --results-dir results/experiments/ \
        --dat-dir results/experiments/pgfplots/
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Model name normalisation
# ---------------------------------------------------------------------------

_MODEL_ALIASES: dict[str, str] = {
    "qwen2.5-0.5b": "Qwen2.5-0.5B",
    "qwen2.5-0.5b-instruct": "Qwen2.5-0.5B",
    "stories110m": "Stories-110M",
    "smollm2-360m": "SmolLM2-360M",
    "smollm2-360m-instruct": "SmolLM2-360M",
    "SmolLM2-360M-Instruct": "SmolLM2-360M",
}

_BACKEND_ORDER = ["public", "private", "private-full", "mlx"]
_SEED_DIR_RE = re.compile(r"^seed[_-](\d+)$")


def normalize_model(raw: str) -> str:
    """Map raw model name to display name."""
    return _MODEL_ALIASES.get(raw, _MODEL_ALIASES.get(raw.lower(), raw))


def backend_sort_key(backend: str) -> int:
    try:
        return _BACKEND_ORDER.index(backend)
    except ValueError:
        return len(_BACKEND_ORDER)


def sanitize_name(name: str) -> str:
    return name.replace("/", "__")


def experiment_group_name(name: str) -> str:
    parts = name.split("/")
    if parts and _SEED_DIR_RE.match(parts[-1]):
        return "/".join(parts[:-1])
    return name


def extract_seed(name: str, records: list[dict[str, Any]]) -> int | None:
    if records and "seed" in records[0]:
        try:
            return int(records[0]["seed"])
        except (TypeError, ValueError):
            pass
    parts = name.split("/")
    if parts:
        match = _SEED_DIR_RE.match(parts[-1])
        if match:
            return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_experiment(log_path: str) -> list[dict[str, Any]]:
    """Load a single grpo_log.jsonl, returning list of step records."""
    records: list[dict[str, Any]] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def discover_experiments(results_dir: str) -> dict[str, list[dict]]:
    """
    Walk results_dir recursively looking for grpo_log.jsonl files.

    Returns dict mapping experiment_name -> list of step records.
    For nested seed runs, the experiment name includes the seed directory
    (for example: qwen_public/seed_42).
    """
    experiments: dict[str, list[dict]] = {}
    results_path = Path(results_dir)
    if not results_path.is_dir():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    for log_file in sorted(results_path.rglob("grpo_log.jsonl")):
        if "pgfplots" in log_file.parts:
            continue
        rel = log_file.relative_to(results_path)
        name = "/".join(rel.parts[:-1])
        if not name:
            name = log_file.parent.name
        records = load_experiment(str(log_file))
        if records:
            experiments[name] = records

    if not experiments:
        print(f"ERROR: no grpo_log.jsonl files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    return experiments


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(variance)


def safe_get(d: dict, *keys, default: float = 0.0) -> float:
    """Nested dict access with default."""
    current = d
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def peak_window_mean(vals: list[float], window: int) -> float:
    if not vals:
        return 0.0
    if len(vals) <= window:
        return mean(vals)
    best = -float("inf")
    for idx in range(0, len(vals) - window + 1):
        best = max(best, mean(vals[idx:idx + window]))
    return best


def as_metric_tuple(value: Any) -> tuple[float, float]:
    if isinstance(value, tuple):
        return float(value[0]), float(value[1])
    return float(value), 0.0


# ---------------------------------------------------------------------------
# Per-experiment summary
# ---------------------------------------------------------------------------


def summarise_experiment(name: str, records: list[dict]) -> dict[str, Any]:
    """Compute summary statistics for one experiment."""
    if not records:
        return {}

    # Extract model and backend from the first record
    model_raw = records[0].get("model", name)
    backend = records[0].get("backend", "unknown")
    seed = extract_seed(name, records)

    # Skip step 0 as warmup for timing/power
    timing_records = [r for r in records if r.get("step", 0) > 0]
    # If only step 0, use it anyway
    if not timing_records:
        timing_records = records

    # ---- Timing ----
    total_ms = [safe_get(r, "timing", "total_ms") for r in timing_records]
    rollout_ms = [safe_get(r, "timing", "rollout_ms") for r in timing_records]
    gradient_ms = [safe_get(r, "timing", "gradient_ms") for r in timing_records]
    ane_ms = [safe_get(r, "timing", "ane_ms") for r in timing_records]
    bwd_ane_ms = [safe_get(r, "timing", "bwd_ane_ms") for r in timing_records]

    # ---- Power ----
    # power can be a dict or absent; power_w is a scalar fallback
    cpu_w_vals = []
    gpu_w_vals = []
    ane_w_vals = []
    total_w_vals = []
    for r in timing_records:
        power = r.get("power", {})
        if isinstance(power, dict) and power:
            cpu_w_vals.append(safe_get(power, "cpu_w"))
            gpu_w_vals.append(safe_get(power, "gpu_w"))
            ane_w_vals.append(safe_get(power, "ane_w"))
            total_w_vals.append(safe_get(power, "total_w"))
        else:
            # Fallback to top-level power_w
            pw = safe_get(r, "power_w")
            total_w_vals.append(pw)

    # ---- Rewards ----
    all_rewards = [safe_get(r, "mean_reward") for r in records]
    all_valid = [safe_get(r, "json_valid_pct") for r in records]
    last_50 = records[-50:] if len(records) >= 50 else records
    final_rewards = [safe_get(r, "mean_reward") for r in last_50]
    final_valid = [safe_get(r, "json_valid_pct") for r in last_50]

    return {
        "name": name,
        "model_raw": model_raw,
        "model": normalize_model(model_raw),
        "backend": backend,
        "group_name": experiment_group_name(name),
        "seed": seed,
        "n_steps": len(records),
        "timing": {
            "total_ms": (mean(total_ms), std(total_ms)),
            "rollout_ms": (mean(rollout_ms), std(rollout_ms)),
            "gradient_ms": (mean(gradient_ms), std(gradient_ms)),
            "ane_ms": (mean(ane_ms), std(ane_ms)),
            "bwd_ane_ms": (mean(bwd_ane_ms), std(bwd_ane_ms)),
        },
        "power": {
            "cpu_w": mean(cpu_w_vals),
            "gpu_w": mean(gpu_w_vals),
            "ane_w": mean(ane_w_vals),
            "total_w": mean(total_w_vals),
        },
        "reward": {
            "mean": mean(all_rewards),
            "peak_20": peak_window_mean(all_rewards, 20),
            "final_50": mean(final_rewards),
            "mean_valid": mean(all_valid),
            "final_50_valid": mean(final_valid),
        },
    }


def aggregate_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary["model"], summary["backend"], summary["group_name"])].append(summary)

    aggregated: list[dict[str, Any]] = []
    timing_fields = ("total_ms", "rollout_ms", "gradient_ms", "ane_ms", "bwd_ane_ms")
    power_fields = ("cpu_w", "gpu_w", "ane_w", "total_w")
    reward_fields = ("mean", "peak_20", "final_50", "mean_valid", "final_50_valid")

    for (_, _, group_name), items in grouped.items():
        items.sort(key=lambda s: (s["seed"] is None, s["seed"]))
        first = items[0]
        seed_list = [item["seed"] for item in items if item["seed"] is not None]

        timing = {}
        for field in timing_fields:
            vals = [item["timing"][field][0] for item in items]
            timing[field] = (mean(vals), std(vals))

        power = {}
        for field in power_fields:
            vals = [item["power"][field] for item in items]
            power[field] = (mean(vals), std(vals))

        reward = {}
        for field in reward_fields:
            vals = [item["reward"][field] for item in items]
            reward[field] = (mean(vals), std(vals))

        aggregated.append(
            {
                "name": group_name,
                "group_name": group_name,
                "model_raw": first["model_raw"],
                "model": first["model"],
                "backend": first["backend"],
                "seed": first["seed"],
                "seeds": seed_list,
                "n_seeds": max(len(items), 1),
                "n_steps": int(round(mean([item["n_steps"] for item in items]))),
                "timing": timing,
                "power": power,
                "reward": reward,
            }
        )

    return aggregated


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def fmt_ms(val: float, sd: float) -> str:
    """Format millisecond value with optional std."""
    if val == 0.0 and sd == 0.0:
        return "---"
    if sd > 0:
        return f"{val:,.0f}$\\pm${sd:,.0f}"
    return f"{val:,.0f}"


def fmt_float(val: float, sd: float, precision: int = 3) -> str:
    if sd > 0:
        return f"{val:.{precision}f}$\\pm${sd:.{precision}f}"
    return f"{val:.{precision}f}"


def fmt_w(val: float, sd: float = 0.0) -> str:
    if val == 0.0 and sd == 0.0:
        return "---"
    return fmt_float(val, sd, precision=1)


def fmt_pct(val: float, sd: float = 0.0) -> str:
    return fmt_float(val, sd, precision=1) + r"\%"


def latex_timing_table(summaries: list[dict]) -> str:
    """Table 1: Per-step timing breakdown."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Per-step timing breakdown across compute paths (ms)}",
        r"\label{tab:timing}",
        r"\centering\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Path & Total & Rollout & Gradient & ANE fwd & ANE bwd \\",
        r"\midrule",
    ]

    prev_model = None
    for s in summaries:
        model = s["model"]
        if prev_model and model != prev_model:
            lines.append(r"\midrule")
        prev_model = model

        t = s["timing"]
        row = (
            f"{model} & {s['backend']} & "
            f"{fmt_ms(*t['total_ms'])} & "
            f"{fmt_ms(*t['rollout_ms'])} & "
            f"{fmt_ms(*t['gradient_ms'])} & "
            f"{fmt_ms(*t['ane_ms'])} & "
            f"{fmt_ms(*t['bwd_ane_ms'])} \\\\"
        )
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def latex_power_table(summaries: list[dict]) -> str:
    """Table 2: Power consumption breakdown."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Power consumption across compute paths (W)}",
        r"\label{tab:power}",
        r"\centering\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Path & CPU & GPU & ANE & Total \\",
        r"\midrule",
    ]

    prev_model = None
    for s in summaries:
        model = s["model"]
        if prev_model and model != prev_model:
            lines.append(r"\midrule")
        prev_model = model

        p = s["power"]
        cpu_w = as_metric_tuple(p["cpu_w"])
        gpu_w = as_metric_tuple(p["gpu_w"])
        ane_w = as_metric_tuple(p["ane_w"])
        total_w = as_metric_tuple(p["total_w"])
        row = (
            f"{model} & {s['backend']} & "
            f"{fmt_w(*cpu_w)} & "
            f"{fmt_w(*gpu_w)} & "
            f"{fmt_w(*ane_w)} & "
            f"{fmt_w(*total_w)} \\\\"
        )
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def latex_reward_table(summaries: list[dict]) -> str:
    """Table 3: Reward and validity metrics."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Reward and JSON validity across compute paths}",
        r"\label{tab:reward}",
        r"\centering\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Path & Mean Reward & Peak-20 Reward & Mean Valid\% & Final-50 Valid\% \\",
        r"\midrule",
    ]

    prev_model = None
    for s in summaries:
        model = s["model"]
        if prev_model and model != prev_model:
            lines.append(r"\midrule")
        prev_model = model

        rw = s["reward"]
        mean_reward = as_metric_tuple(rw["mean"])
        peak_20 = as_metric_tuple(rw["peak_20"])
        mean_valid = as_metric_tuple(rw["mean_valid"])
        final_valid = as_metric_tuple(rw["final_50_valid"])
        row = (
            f"{model} & {s['backend']} & "
            f"{fmt_float(*mean_reward, precision=3)} & "
            f"{fmt_float(*peak_20, precision=3)} & "
            f"{fmt_pct(*mean_valid)} & "
            f"{fmt_pct(*final_valid)} \\\\"
        )
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# pgfplots .dat generation
# ---------------------------------------------------------------------------


def write_dat_files(experiments: dict[str, list[dict]], dat_dir: str):
    """Write per-experiment .dat files for pgfplots."""
    os.makedirs(dat_dir, exist_ok=True)

    for name, records in sorted(experiments.items()):
        # Reward curve
        file_name = sanitize_name(name)
        dat_path = os.path.join(dat_dir, f"{file_name}_reward.dat")
        with open(dat_path, "w") as f:
            f.write("step mean_reward json_valid_pct\n")
            for r in records:
                step = r.get("step", 0)
                mr = safe_get(r, "mean_reward")
                jv = safe_get(r, "json_valid_pct")
                f.write(f"{step} {mr:.4f} {jv:.1f}\n")

        # Timing curve
        dat_path = os.path.join(dat_dir, f"{file_name}_timing.dat")
        with open(dat_path, "w") as f:
            f.write("step total_ms rollout_ms gradient_ms ane_ms bwd_ane_ms\n")
            for r in records:
                step = r.get("step", 0)
                t = r.get("timing", {})
                f.write(
                    f"{step} "
                    f"{safe_get(t, 'total_ms'):.1f} "
                    f"{safe_get(t, 'rollout_ms'):.1f} "
                    f"{safe_get(t, 'gradient_ms'):.1f} "
                    f"{safe_get(t, 'ane_ms'):.1f} "
                    f"{safe_get(t, 'bwd_ane_ms'):.1f}\n"
                )

    print(f"pgfplots .dat files written to {dat_dir}/", file=sys.stderr)


# ---------------------------------------------------------------------------
# Markdown summary (to stderr)
# ---------------------------------------------------------------------------


def markdown_summary(summaries: list[dict]) -> str:
    """Markdown table for quick review."""
    lines = [
        "## Experiment Summary",
        "",
        "| Model | Path | Seeds | Steps | Total ms | Mean Reward | Peak-20 | Final-50 Valid |",
        "|-------|------|------:|------:|--------:|------------:|--------:|---------------:|",
    ]
    for s in summaries:
        t_mean, _ = s["timing"]["total_ms"]
        rw = s["reward"]
        n_seeds = s.get("n_seeds", 1)
        mean_reward, _ = as_metric_tuple(rw["mean"])
        peak_20, _ = as_metric_tuple(rw["peak_20"])
        final_valid, _ = as_metric_tuple(rw["final_50_valid"])
        lines.append(
            f"| {s['model']} | {s['backend']} | {n_seeds} | {s['n_steps']} | "
            f"{t_mean:,.0f} | {mean_reward:.3f} | {peak_20:.3f} | "
            f"{final_valid:.1f}% |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GRPO training experiments across compute paths and models."
    )
    parser.add_argument(
        "--results-dir",
        default="results/experiments/",
        help="Directory containing experiment subdirectories with grpo_log.jsonl files",
    )
    parser.add_argument(
        "--dat-dir",
        default=None,
        help="Output directory for pgfplots .dat files (default: <results-dir>/pgfplots/)",
    )
    args = parser.parse_args()

    dat_dir = args.dat_dir or os.path.join(args.results_dir, "pgfplots")

    # Load all experiments
    experiments = discover_experiments(args.results_dir)
    print(f"Found {len(experiments)} experiments:", file=sys.stderr)
    for name in sorted(experiments.keys()):
        n = len(experiments[name])
        model = experiments[name][0].get("model", "?")
        backend = experiments[name][0].get("backend", "?")
        print(f"  {name}: {n} steps ({model} / {backend})", file=sys.stderr)
    print(file=sys.stderr)

    # Compute summaries
    summaries = []
    for name, records in experiments.items():
        s = summarise_experiment(name, records)
        if s:
            summaries.append(s)

    summaries = aggregate_summaries(summaries)
    summaries.sort(key=lambda s: (s["model"], backend_sort_key(s["backend"])))

    # Write .dat files
    write_dat_files(experiments, dat_dir)

    # LaTeX tables to stdout
    print("% " + "=" * 70)
    print("% Table 1: Timing")
    print("% " + "=" * 70)
    print(latex_timing_table(summaries))
    print()
    print("% " + "=" * 70)
    print("% Table 2: Power")
    print("% " + "=" * 70)
    print(latex_power_table(summaries))
    print()
    print("% " + "=" * 70)
    print("% Table 3: Reward")
    print("% " + "=" * 70)
    print(latex_reward_table(summaries))

    # Markdown summary to stderr
    md = markdown_summary(summaries)
    print(file=sys.stderr)
    print(md, file=sys.stderr)


if __name__ == "__main__":
    main()
