#!/usr/bin/env python3
"""
Generate comparison tables and pgfplots data from spectrum experiment results.

Reads summary.json and per-cell JSONL logs, produces:
- LaTeX table for paper
- pgfplots .dat files for timing/reward charts
- Markdown summary
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_results(out_dir: str) -> dict:
    """Load summary.json and all JSONL logs."""
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    # Enrich with step data from JSONL logs
    for key, result in summary.items():
        cell_dir = os.path.join(out_dir, key)
        log_path = os.path.join(cell_dir, "training.jsonl")
        if os.path.exists(log_path):
            steps = []
            with open(log_path) as f:
                for line in f:
                    if line.strip():
                        steps.append(json.loads(line))
            result["steps"] = steps

    return summary


def generate_latex_table(results: dict, out_path: str):
    """Generate LaTeX table for paper."""
    models = set()
    backends = set()
    for key in results:
        model, backend = key.rsplit("_", 1)
        models.add(model)
        backends.add(backend)

    models = sorted(models)
    backends = sorted(backends)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Public vs.\ Private API GRPO Performance}")
    lines.append(r"\label{tab:spectrum}")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & API & Rollout & Gradient & Total & Reward & Valid \\")
    lines.append(r" & Path & (ms) & (ms) & (ms/step) & (mean) & (\%) \\")
    lines.append(r"\midrule")

    for model in models:
        for backend in backends:
            key = f"{model}_{backend}"
            r = results.get(key, {})
            steps = r.get("steps", [])

            if steps:
                # Average across all steps
                avg_rollout = sum(s["timing"]["rollout_ms"] for s in steps) / len(steps)
                avg_gradient = sum(s["timing"]["gradient_ms"] for s in steps) / len(steps)
                avg_total = sum(s["timing"]["total_ms"] for s in steps) / len(steps)
                avg_reward = sum(s["mean_reward"] for s in steps) / len(steps)
                avg_valid = sum(s["json_valid_pct"] for s in steps) / len(steps)

                model_display = model.replace("_", r"\_")
                lines.append(
                    f"{model_display} & {backend} & "
                    f"{avg_rollout:.0f} & {avg_gradient:.0f} & {avg_total:.0f} & "
                    f"{avg_reward:.3f} & {avg_valid:.0f}\\% \\\\"
                )
            else:
                lines.append(f"{model} & {backend} & --- & --- & --- & --- & --- \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to {out_path}")


def generate_pgfplots_data(results: dict, out_dir: str):
    """Generate .dat files for pgfplots charts."""
    os.makedirs(out_dir, exist_ok=True)

    for key, r in results.items():
        steps = r.get("steps", [])
        if not steps:
            continue

        # Timing breakdown per step
        dat_path = os.path.join(out_dir, f"{key}_timing.dat")
        with open(dat_path, "w") as f:
            f.write("step rollout_ms gradient_ms total_ms\n")
            for s in steps:
                t = s["timing"]
                f.write(f"{s['step']} {t['rollout_ms']:.1f} {t['gradient_ms']:.1f} {t['total_ms']:.1f}\n")

        # Reward per step
        dat_path = os.path.join(out_dir, f"{key}_reward.dat")
        with open(dat_path, "w") as f:
            f.write("step mean_reward json_valid_pct\n")
            for s in steps:
                f.write(f"{s['step']} {s['mean_reward']:.4f} {s['json_valid_pct']:.1f}\n")

    print(f"pgfplots data saved to {out_dir}/")


def generate_markdown_summary(results: dict, out_path: str):
    """Generate markdown summary for PROGRESS.md."""
    lines = []
    lines.append("## Spectrum Experiment Results\n")
    lines.append("| Model | Backend | Steps | Mean Reward | JSON Valid | Total ms/step |")
    lines.append("|-------|---------|-------|-------------|-----------|---------------|")

    for key, r in sorted(results.items()):
        model, backend = key.rsplit("_", 1)
        steps = r.get("steps", [])
        status = r.get("status", "unknown")

        if steps:
            avg_reward = sum(s["mean_reward"] for s in steps) / len(steps)
            avg_valid = sum(s["json_valid_pct"] for s in steps) / len(steps)
            avg_total = sum(s["timing"]["total_ms"] for s in steps) / len(steps)
            lines.append(
                f"| {model} | {backend} | {len(steps)} | "
                f"{avg_reward:.3f} | {avg_valid:.0f}% | {avg_total:.0f} |"
            )
        else:
            lines.append(f"| {model} | {backend} | 0 | {status} | --- | --- |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown summary saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare public vs private ANE GRPO results")
    parser.add_argument("--results-dir", default="out/spectrum",
                        help="Directory with summary.json and cell logs")
    parser.add_argument("--latex-out", default=None,
                        help="Output path for LaTeX table (default: results_dir/table.tex)")
    parser.add_argument("--pgfplots-dir", default=None,
                        help="Output dir for pgfplots .dat files")
    parser.add_argument("--markdown-out", default=None,
                        help="Output path for markdown summary")
    args = parser.parse_args()

    results = load_results(args.results_dir)

    latex_out = args.latex_out or os.path.join(args.results_dir, "table.tex")
    generate_latex_table(results, latex_out)

    pgf_dir = args.pgfplots_dir or os.path.join(args.results_dir, "pgfplots")
    generate_pgfplots_data(results, pgf_dir)

    md_out = args.markdown_out or os.path.join(args.results_dir, "summary.md")
    generate_markdown_summary(results, md_out)


if __name__ == "__main__":
    main()
