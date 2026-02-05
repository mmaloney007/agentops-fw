#!/usr/bin/env python3
"""Threshold sensitivity analysis for P4 curve taxonomy.

Sweeps the sustained/transient/flat classification thresholds to test
robustness of the taxonomy. Also sweeps forgetting delta boundaries.

Input:
  - results/p4_analysis/curve_taxonomy.json
  - results/p4_analysis/forgetting_matrix.json

Output:
  - results/p4_analysis/threshold_sensitivity.json
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TAXONOMY_PATH = BASE_DIR / "results" / "p4_analysis" / "curve_taxonomy.json"
FORGETTING_PATH = BASE_DIR / "results" / "p4_analysis" / "forgetting_matrix.json"
OUT_PATH = BASE_DIR / "results" / "p4_analysis" / "threshold_sensitivity.json"

# Default thresholds
DEFAULT_SUSTAINED = 0.60  # final validity >= 60%
DEFAULT_TRANSIENT = 0.30  # peak >= 30% but final < sustained
DEFAULT_FORGETTING_ROBUST = -0.05
DEFAULT_FORGETTING_CATASTROPHIC = -0.30


def classify_curve(peak: float, final: float, sustained_thresh: float, transient_thresh: float) -> str:
    """Classify a curve given thresholds."""
    if final >= sustained_thresh:
        return "sustained"
    elif peak >= transient_thresh:
        return "transient"
    else:
        return "flat"


def classify_forgetting(delta: float, robust_thresh: float, catastrophic_thresh: float) -> str:
    """Classify forgetting profile given thresholds."""
    if delta > robust_thresh:
        return "robust"
    elif delta > catastrophic_thresh:
        return "selective"
    else:
        return "catastrophic"


def main():
    print("=" * 60)
    print("Threshold Sensitivity Analysis for P4 Curve Taxonomy")
    print("=" * 60)

    # Load curve taxonomy data
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

    # Filter to single-task runs only
    single_task = [r for r in taxonomy["runs"] if r["task"] != "Mixed"]
    print(f"Single-task runs: {len(single_task)}")

    # Show default classification
    default_cats = {}
    for r in single_task:
        cat = classify_curve(r["peak_validity"], r["final_validity"], DEFAULT_SUSTAINED, DEFAULT_TRANSIENT)
        default_cats[f"{r['model']}_{r['task']}"] = cat

    from collections import Counter
    print(f"Default (sustained={DEFAULT_SUSTAINED}, transient={DEFAULT_TRANSIENT}): {Counter(default_cats.values())}")

    # --- Sweep sustained threshold ---
    sustained_sweep = []
    for st in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        cats = Counter()
        for r in single_task:
            cat = classify_curve(r["peak_validity"], r["final_validity"], st, DEFAULT_TRANSIENT)
            cats[cat] += 1
        sustained_sweep.append({
            "sustained_threshold": st,
            "transient_threshold": DEFAULT_TRANSIENT,
            "distribution": dict(cats),
            "total": sum(cats.values()),
        })

    # --- Sweep transient threshold ---
    transient_sweep = []
    for tt in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        cats = Counter()
        for r in single_task:
            cat = classify_curve(r["peak_validity"], r["final_validity"], DEFAULT_SUSTAINED, tt)
            cats[cat] += 1
        transient_sweep.append({
            "sustained_threshold": DEFAULT_SUSTAINED,
            "transient_threshold": tt,
            "distribution": dict(cats),
            "total": sum(cats.values()),
        })

    # --- Joint sweep: sustained x transient ---
    joint_sweep = []
    for st in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for tt in [0.20, 0.25, 0.30, 0.35, 0.40]:
            cats = Counter()
            changes = 0
            for r in single_task:
                new_cat = classify_curve(r["peak_validity"], r["final_validity"], st, tt)
                old_cat = default_cats[f"{r['model']}_{r['task']}"]
                cats[new_cat] += 1
                if new_cat != old_cat:
                    changes += 1
            joint_sweep.append({
                "sustained_threshold": st,
                "transient_threshold": tt,
                "distribution": dict(cats),
                "changes_from_default": changes,
                "pct_stable": round(100 * (len(single_task) - changes) / len(single_task), 1),
            })

    # --- Robustness at ±10% ---
    print("\n--- Robustness at ±10% thresholds ---")
    for st_delta in [-0.10, -0.05, 0.0, 0.05, 0.10]:
        st = DEFAULT_SUSTAINED + st_delta
        changes = 0
        for r in single_task:
            new_cat = classify_curve(r["peak_validity"], r["final_validity"], st, DEFAULT_TRANSIENT)
            old_cat = default_cats[f"{r['model']}_{r['task']}"]
            if new_cat != old_cat:
                changes += 1
        pct = round(100 * (len(single_task) - changes) / len(single_task), 1)
        print(f"  sustained={st:.2f}: {changes} changes ({pct}% stable)")

    # --- Forgetting delta sensitivity ---
    forgetting_sweep = []
    if FORGETTING_PATH.exists():
        with open(FORGETTING_PATH) as f:
            forgetting = json.load(f)

        models_with_delta = []
        for entry in forgetting.get("models", forgetting.get("matrix", [])):
            if isinstance(entry, dict) and "delta" in entry:
                models_with_delta.append(entry)

        if models_with_delta:
            default_profiles = {}
            for m in models_with_delta:
                key = m.get("model", m.get("name", "unknown"))
                default_profiles[key] = classify_forgetting(
                    m["delta"], DEFAULT_FORGETTING_ROBUST, DEFAULT_FORGETTING_CATASTROPHIC
                )

            for rob_delta in [-0.05, -0.03, 0.0, 0.03, 0.05]:
                for cat_delta in [-0.05, -0.03, 0.0, 0.03, 0.05]:
                    rob = DEFAULT_FORGETTING_ROBUST + rob_delta
                    cat = DEFAULT_FORGETTING_CATASTROPHIC + cat_delta
                    profiles = Counter()
                    changes = 0
                    for m in models_with_delta:
                        key = m.get("model", m.get("name", "unknown"))
                        new_prof = classify_forgetting(m["delta"], rob, cat)
                        profiles[new_prof] += 1
                        if new_prof != default_profiles[key]:
                            changes += 1
                    forgetting_sweep.append({
                        "robust_threshold": round(rob, 3),
                        "catastrophic_threshold": round(cat, 3),
                        "distribution": dict(profiles),
                        "changes_from_default": changes,
                    })

    # --- Summary statistics ---
    # For ±10% sustained threshold: how many curves change?
    changes_at_pm10 = 0
    for r in single_task:
        cats_at_offsets = set()
        for st_delta in [-0.10, 0.0, 0.10]:
            st = DEFAULT_SUSTAINED + st_delta
            cat = classify_curve(r["peak_validity"], r["final_validity"], st, DEFAULT_TRANSIENT)
            cats_at_offsets.add(cat)
        if len(cats_at_offsets) > 1:
            changes_at_pm10 += 1
    pct_robust = round(100 * (len(single_task) - changes_at_pm10) / len(single_task), 1)
    print(f"\nCurves stable across ±10% sustained threshold: {len(single_task) - changes_at_pm10}/{len(single_task)} ({pct_robust}%)")

    # Build output
    output = {
        "default_thresholds": {
            "sustained": DEFAULT_SUSTAINED,
            "transient": DEFAULT_TRANSIENT,
            "forgetting_robust": DEFAULT_FORGETTING_ROBUST,
            "forgetting_catastrophic": DEFAULT_FORGETTING_CATASTROPHIC,
        },
        "n_single_task_curves": len(single_task),
        "default_distribution": dict(Counter(default_cats.values())),
        "sustained_sweep": sustained_sweep,
        "transient_sweep": transient_sweep,
        "joint_sweep": joint_sweep,
        "forgetting_sweep": forgetting_sweep if forgetting_sweep else "no_data",
        "robustness_summary": {
            "curves_stable_at_pm10_sustained": len(single_task) - changes_at_pm10,
            "pct_stable_at_pm10_sustained": pct_robust,
            "total_curves": len(single_task),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
