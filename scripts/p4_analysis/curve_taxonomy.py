#!/usr/bin/env python3
"""P4 Analysis: Curve Taxonomy.

Classifies training curves as sustained/transient/flat based on
validity trajectory features.

Input:  results/curves/validity_curves.json
Output: results/p4_analysis/curve_taxonomy.json
        results/curves/pgfplots/p4_curves.dat
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
CURVES_PATH = ROOT / "results" / "curves" / "validity_curves.json"
OUT_JSON = ROOT / "results" / "p4_analysis" / "curve_taxonomy.json"
OUT_DAT = ROOT / "results" / "curves" / "pgfplots" / "p4_curves.dat"

# Classification thresholds
SUSTAINED_FINAL_MIN = 0.60  # final validity >= 60% = sustained
TRANSIENT_PEAK_MIN = 0.30   # peak >= 30% but final < 60% = transient
# Otherwise flat (never learns)


def classify_curve(peak: float, final: float) -> str:
    if final >= SUSTAINED_FINAL_MIN:
        return "sustained"
    elif peak >= TRANSIENT_PEAK_MIN:
        return "transient"
    else:
        return "flat"


def extract_features(steps, values):
    """Extract curve features from step-value pairs."""
    if not values or len(values) < 10:
        return None

    n = len(values)
    peak_val = max(values)
    peak_idx = values.index(peak_val)
    peak_step = steps[peak_idx] if peak_idx < len(steps) else 0
    final_val = sum(values[-50:]) / min(50, len(values[-50:]))

    # Decay rate: from peak to end
    if peak_idx < n - 1 and peak_val > 0:
        decay = (peak_val - final_val) / peak_val
    else:
        decay = 0.0

    # Time to 50% of peak (from start)
    half_peak = peak_val * 0.5
    time_to_50 = None
    for i, v in enumerate(values):
        if v >= half_peak:
            time_to_50 = steps[i] if i < len(steps) else i
            break

    # Early signal: mean of first 50 steps
    early_mean = sum(values[:50]) / min(50, len(values[:50]))

    return {
        "peak_validity": round(peak_val, 4),
        "peak_step": int(peak_step),
        "final_validity": round(final_val, 4),
        "decay_rate": round(decay, 4),
        "time_to_50pct": int(time_to_50) if time_to_50 is not None else None,
        "early_mean_50": round(early_mean, 4),
    }


def analyze():
    with open(CURVES_PATH, "r") as f:
        data = json.load(f)

    runs = []
    taxonomy_counts = defaultdict(int)
    by_size_range = defaultdict(lambda: defaultdict(int))

    for model_name, model_data in data.items():
        size_b = model_data.get("size_b", 0)

        # Size range classification
        if size_b <= 3:
            size_range = "small (<=3B)"
        elif size_b <= 7:
            size_range = "medium (4-7B)"
        else:
            size_range = "large (8B+)"

        for task_name, task_data in model_data.get("tasks", {}).items():
            steps = task_data.get("steps", [])
            # Try different validity field names
            values = task_data.get("mean") or task_data.get("validity_50_sample") or task_data.get("mean_validity") or task_data.get("validity_full")

            if not values:
                continue

            features = extract_features(steps, values)
            if features is None:
                continue

            category = classify_curve(features["peak_validity"], features["final_validity"])
            taxonomy_counts[category] += 1
            by_size_range[size_range][category] += 1

            runs.append({
                "model": model_name,
                "size_b": size_b,
                "size_range": size_range,
                "task": task_name,
                "category": category,
                **features,
            })

    # Summary
    summary = {
        "total_runs": len(runs),
        "taxonomy_counts": dict(taxonomy_counts),
        "by_size_range": {k: dict(v) for k, v in by_size_range.items()},
    }

    output = {"runs": runs, "summary": summary, "thresholds": {"sustained_final_min": SUSTAINED_FINAL_MIN, "transient_peak_min": TRANSIENT_PEAK_MIN}}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {OUT_JSON} ({len(runs)} runs classified)")

    # pgfplots data
    OUT_DAT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DAT, "w") as f:
        f.write("model task size_b category peak_validity final_validity decay_rate peak_step\n")
        for r in sorted(runs, key=lambda x: (x["size_b"], x["task"])):
            f.write(f"{r['model']} {r['task']} {r['size_b']:.1f} {r['category']} "
                    f"{r['peak_validity']:.4f} {r['final_validity']:.4f} "
                    f"{r['decay_rate']:.4f} {r['peak_step']}\n")
    print(f"Wrote {OUT_DAT}")

    return output


if __name__ == "__main__":
    analyze()
