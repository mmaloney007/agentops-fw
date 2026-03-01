#!/usr/bin/env python3
"""P4 Analysis: Generate PGFPlots Data Files.

Generates all .dat files needed for P4 paper figures from analysis outputs.

Input:  results/p4_analysis/*.json
Output: results/curves/pgfplots/p4_*.dat
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
ANALYSIS_DIR = ROOT / "results" / "p4_analysis"
PGFPLOTS_DIR = ROOT / "results" / "curves" / "pgfplots"
CURVES_DIR = ROOT / "results" / "curves"


def write_dat(path: Path, header: str, rows: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")
    print(f"Wrote {path} ({len(rows)} rows)")


def gen_reward_component_bars():
    """Generate stacked bar chart data for reward decomposition."""
    with open(ANALYSIS_DIR / "reward_decomposition.json") as f:
        data = json.load(f)
    rows = []
    for model, info in sorted(data["models"].items()):
        rows.append(f"{model} {info['mean_r_schema']:.4f} {info['mean_r_residual']:.4f} "
                    f"{abs(info['mean_r_latency']):.4f} {abs(info['mean_r_cost']):.4f}")
    write_dat(
        PGFPLOTS_DIR / "p4_reward_components.dat",
        "model schema accuracy latency cost",
        rows,
    )


def gen_forgetting_heatmap():
    """Generate heatmap data for forgetting matrix."""
    with open(ANALYSIS_DIR / "forgetting_matrix.json") as f:
        data = json.load(f)
    tasks = ["T1", "T2", "T3", "T4", "T5"]
    rows = []
    for m in sorted(data["matrix"], key=lambda x: x["size_b"]):
        vals = " ".join(f"{m['per_task_single'].get(t, 0):.4f}" for t in tasks)
        rows.append(f"{m['model']} {m['size_b']:.1f} {m['mixed_validity']:.4f} {vals}")
    write_dat(
        PGFPLOTS_DIR / "p4_forgetting_heatmap.dat",
        "model size_b mixed T1 T2 T3 T4 T5",
        rows,
    )


def gen_curve_scatter():
    """Generate scatter data for curve taxonomy (peak vs final)."""
    with open(ANALYSIS_DIR / "curve_taxonomy.json") as f:
        data = json.load(f)
    rows = []
    for r in data["runs"]:
        cat_num = {"sustained": 2, "transient": 1, "flat": 0}[r["category"]]
        rows.append(f"{r['model']} {r['task']} {r['peak_validity']:.4f} "
                    f"{r['final_validity']:.4f} {cat_num} {r['size_b']:.1f}")
    write_dat(
        PGFPLOTS_DIR / "p4_curve_scatter.dat",
        "model task peak_validity final_validity category size_b",
        rows,
    )


def gen_size_taxonomy_bars():
    """Generate bar chart for taxonomy distribution by model size."""
    with open(ANALYSIS_DIR / "curve_taxonomy.json") as f:
        data = json.load(f)
    rows = []
    for size_range, counts in sorted(data["summary"]["by_size_range"].items()):
        s = counts.get("sustained", 0)
        t = counts.get("transient", 0)
        f = counts.get("flat", 0)
        rows.append(f"\"{size_range}\" {s} {t} {f}")
    write_dat(
        PGFPLOTS_DIR / "p4_size_taxonomy.dat",
        "size_range sustained transient flat",
        rows,
    )


def gen_sample_curves():
    """Generate sample validity curves for representative models."""
    with open(CURVES_DIR / "validity_curves.json") as f:
        curves = json.load(f)
    # Pick representative models
    reps = ["llama-3.2-1b", "qwen3-4b", "gemma-2-9b"]
    task = "Mixed"

    for model in reps:
        if model not in curves:
            continue
        model_data = curves[model]
        if task not in model_data.get("tasks", {}):
            continue
        task_data = model_data["tasks"][task]
        steps = task_data.get("steps", [])
        values = task_data.get("mean") or task_data.get("validity_50_sample") or task_data.get("mean_validity") or []

        if not steps or not values:
            continue

        rows = []
        for s, v in zip(steps, values):
            rows.append(f"{int(s)} {v:.4f}")

        safe_name = model.replace(".", "_").replace("-", "_")
        write_dat(
            PGFPLOTS_DIR / f"p4_curve_{safe_name}_{task.lower()}.dat",
            "step validity",
            rows,
        )


def main():
    print("Generating P4 figure data files...\n")

    try:
        gen_reward_component_bars()
    except Exception as e:
        print(f"  Warning: reward components: {e}")

    try:
        gen_forgetting_heatmap()
    except Exception as e:
        print(f"  Warning: forgetting heatmap: {e}")

    try:
        gen_curve_scatter()
    except Exception as e:
        print(f"  Warning: curve scatter: {e}")

    try:
        gen_size_taxonomy_bars()
    except Exception as e:
        print(f"  Warning: size taxonomy: {e}")

    try:
        gen_sample_curves()
    except Exception as e:
        print(f"  Warning: sample curves: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
