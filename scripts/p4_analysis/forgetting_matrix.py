#!/usr/bin/env python3
"""P4 Analysis: Forgetting Matrix.

Extends forgetting analysis to full model x task matrix and classifies
forgetting profiles as robust/selective/catastrophic.

Input:  results/curves/forgetting_analysis.json
Output: results/p4_analysis/forgetting_matrix.json
        results/curves/pgfplots/p4_forgetting.dat
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
FORGET_PATH = ROOT / "results" / "curves" / "forgetting_analysis.json"
OUT_JSON = ROOT / "results" / "p4_analysis" / "forgetting_matrix.json"
OUT_DAT = ROOT / "results" / "curves" / "pgfplots" / "p4_forgetting.dat"

# Thresholds for forgetting classification
ROBUST_THRESHOLD = -0.05  # delta > -0.05: robust
SELECTIVE_THRESHOLD = -0.30  # -0.30 < delta <= -0.05: selective forgetting
# delta <= -0.30: catastrophic


def classify_forgetting(delta: float) -> str:
    if delta > ROBUST_THRESHOLD:
        return "robust"
    elif delta > SELECTIVE_THRESHOLD:
        return "selective"
    else:
        return "catastrophic"


def analyze():
    with open(FORGET_PATH, "r") as f:
        data = json.load(f)

    models = data["models"]
    tasks = ["T1", "T2", "T3", "T4", "T5"]

    matrix = []
    for m in models:
        model_name = m["model"]
        per_task = m.get("per_task_single", {})

        # Build task-level forgetting: single-task validity vs mixed
        task_deltas = {}
        for task in tasks:
            single = per_task.get(task, 0.0)
            m["mixed_validity"]
            # Task-specific interference: how much does mixed hurt this task?
            task_deltas[task] = round(single, 4)

        profile = classify_forgetting(m["interference_delta"])

        matrix.append({
            "model": model_name,
            "size_b": m["size_b"],
            "mixed_validity": round(m["mixed_validity"], 4),
            "single_avg_validity": round(m["single_avg_validity"], 4),
            "interference_delta": round(m["interference_delta"], 4),
            "profile": profile,
            "per_task_single": {t: round(v, 4) for t, v in per_task.items()},
            "task_deltas": task_deltas,
        })

    # Summary statistics
    profiles = [m["profile"] for m in matrix]
    summary = {
        "n_models": len(matrix),
        "robust": sum(1 for p in profiles if p == "robust"),
        "selective": sum(1 for p in profiles if p == "selective"),
        "catastrophic": sum(1 for p in profiles if p == "catastrophic"),
        "worst_forgetter": max(matrix, key=lambda m: abs(m["interference_delta"]))["model"],
        "best_retainer": min(matrix, key=lambda m: abs(m["interference_delta"]))["model"],
    }

    output = {"matrix": matrix, "summary": summary, "thresholds": {"robust": ROBUST_THRESHOLD, "selective": SELECTIVE_THRESHOLD}}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {OUT_JSON}")

    # pgfplots data
    OUT_DAT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DAT, "w") as f:
        f.write("model size_b mixed_validity single_avg interference_delta profile\n")
        for m in sorted(matrix, key=lambda x: x["size_b"]):
            f.write(f"{m['model']} {m['size_b']:.1f} {m['mixed_validity']:.4f} "
                    f"{m['single_avg_validity']:.4f} {m['interference_delta']:.4f} {m['profile']}\n")
    print(f"Wrote {OUT_DAT}")

    return output


if __name__ == "__main__":
    analyze()
