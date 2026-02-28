#!/usr/bin/env python3
"""Generate leaderboard.json and leaderboard.csv from P1 evaluation results.

Reads:
  - results/p3_analysis/real_slo_tiers.json  (aggregate S@SLO per model, Spearman)
  - out/p1_comprehensive_20260118/all_results.json  (model info: size, vendor)

Writes:
  - results/leaderboard/leaderboard.json
  - results/leaderboard/leaderboard.csv
"""

import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SLO_TIERS_PATH = ROOT / "results" / "p3_analysis" / "real_slo_tiers.json"
ALL_RESULTS_PATH = ROOT / "out" / "p1_comprehensive_20260118" / "all_results.json"
OUT_DIR = ROOT / "results" / "leaderboard"


def load_model_info(all_results: dict) -> dict:
    """Extract {model_key: {size, vendor}} from all_results.json."""
    info = {}
    for key, val in all_results["models"].items():
        info[key] = {
            "size": val["info"]["size"],
            "vendor": val["info"]["vendor"],
        }
    return info


# Canonical display names for models (key in real_slo_tiers.json -> display)
DISPLAY_NAMES = {
    "llama-3.2-1b": "Llama-3.2-1B",
    "llama-3.2-3b": "Llama-3.2-3B",
    "qwen2.5-3b": "Qwen2.5-3B",
    "phi-3-mini": "Phi-3-mini",
    "qwen3-4b": "Qwen3-4B",
    "yi-1.5-6b": "Yi-1.5-6B",
    "mistral-7b": "Mistral-7B-v0.3",
    "falcon-mamba-7b": "Falcon-Mamba-7B",
    "ministral-8b": "Ministral-8B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "gemma-2-9b": "Gemma-2-9B",
    "gemma-3-12b": "Gemma-3-12B",
    "gpt-oss-20b": "GPT-OSS-20B",
}

# Map keys between slo_tiers (e.g. "mistral-7b") and all_results (e.g. "mistral-7b-v0.3")
SLO_TO_ALL_RESULTS_KEY = {
    "llama-3.2-1b": "llama-3.2-1b",
    "llama-3.2-3b": "llama-3.2-3b",
    "qwen2.5-3b": "qwen2.5-3b",
    "phi-3-mini": "phi-3-mini",
    "qwen3-4b": "qwen3-4b",
    "yi-1.5-6b": "yi-1.5-6b",
    "mistral-7b": "mistral-7b-v0.3",
    "falcon-mamba-7b": "falcon-mamba-7b",
    "ministral-8b": "ministral-8b",
    "llama-3.1-8b": "llama-3.1-8b",
    "gemma-2-9b": "gemma-2-9b",
    "gemma-3-12b": "gemma-3-12b",
    "gpt-oss-20b": "gpt-oss-20b",
}


def compute_ranks(models: list[dict], field: str, descending: bool = True) -> None:
    """Assign rank_{field} to each model dict based on the given field.

    Higher values get lower (better) rank when descending=True.
    """
    rank_key = field.replace("s_at_slo", "").replace("accuracy", "accuracy")
    # Build sorted list of (value, index) pairs
    indexed = [(m[field], i) for i, m in enumerate(models)]
    indexed.sort(key=lambda x: x[0], reverse=descending)
    for rank, (_, idx) in enumerate(indexed, 1):
        models[idx][f"{field}_rank"] = rank


def main():
    with open(SLO_TIERS_PATH) as f:
        slo_data = json.load(f)

    with open(ALL_RESULTS_PATH) as f:
        all_results = json.load(f)

    model_info = load_model_info(all_results)

    # Build model entries from slo_data
    models = []
    for slo_key, agg in slo_data["models"].items():
        ar_key = SLO_TO_ALL_RESULTS_KEY.get(slo_key, slo_key)
        info = model_info.get(ar_key, {"size": "?", "vendor": "?"})
        aggregate = agg["aggregate"]

        models.append({
            "name": DISPLAY_NAMES.get(slo_key, slo_key),
            "size": info["size"],
            "vendor": info["vendor"],
            "accuracy": round(aggregate["accuracy"], 4),
            "interactive_s_at_slo": round(aggregate["s_at_slo_interactive"], 4),
            "standard_s_at_slo": round(aggregate["s_at_slo_standard"], 4),
            "batch_s_at_slo": round(aggregate["s_at_slo_batch"], 4),
            "p95_latency_ms": round(aggregate["p95_latency_ms"], 1),
        })

    # Compute ranks for each metric (higher is better)
    compute_ranks(models, "accuracy")
    compute_ranks(models, "interactive_s_at_slo")
    compute_ranks(models, "standard_s_at_slo")
    compute_ranks(models, "batch_s_at_slo")

    # Sort by accuracy rank for final output (best first)
    models.sort(key=lambda m: m["accuracy_rank"])

    # Restructure into final field order
    final_models = []
    for m in models:
        final_models.append({
            "name": m["name"],
            "size": m["size"],
            "vendor": m["vendor"],
            "accuracy": m["accuracy"],
            "accuracy_rank": m["accuracy_rank"],
            "interactive_s_at_slo": m["interactive_s_at_slo"],
            "interactive_rank": m["interactive_s_at_slo_rank"],
            "standard_s_at_slo": m["standard_s_at_slo"],
            "standard_rank": m["standard_s_at_slo_rank"],
            "batch_s_at_slo": m["batch_s_at_slo"],
            "batch_rank": m["batch_s_at_slo_rank"],
            "p95_latency_ms": m["p95_latency_ms"],
        })

    # Build full JSON structure
    leaderboard = {
        "generated": str(date.today()),
        "description": "AgentSLO-Bench public leaderboard: 13 models across 3 SLO tiers",
        "tiers": {
            "interactive_2s": {
                "name": "Interactive",
                "slo_ms": slo_data["tiers"]["interactive"],
                "description": "Real-time user-facing responses",
            },
            "standard_5s": {
                "name": "Standard",
                "slo_ms": slo_data["tiers"]["standard"],
                "description": "Typical API/agent call latency budget",
            },
            "batch_30s": {
                "name": "Batch",
                "slo_ms": slo_data["tiers"]["batch"],
                "description": "Offline / background processing",
            },
        },
        "models": final_models,
        "spearman": slo_data["spearman"],
    }

    # Write JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "leaderboard.json"
    with open(json_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"Wrote {json_path}")

    # Write CSV
    csv_path = OUT_DIR / "leaderboard.csv"
    fieldnames = [
        "model", "size", "vendor",
        "accuracy", "accuracy_rank",
        "interactive_s_at_slo", "interactive_rank",
        "standard_s_at_slo", "standard_rank",
        "batch_s_at_slo", "batch_rank",
        "p95_latency_ms",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in final_models:
            writer.writerow({
                "model": m["name"],
                "size": m["size"],
                "vendor": m["vendor"],
                "accuracy": m["accuracy"],
                "accuracy_rank": m["accuracy_rank"],
                "interactive_s_at_slo": m["interactive_s_at_slo"],
                "interactive_rank": m["interactive_rank"],
                "standard_s_at_slo": m["standard_s_at_slo"],
                "standard_rank": m["standard_rank"],
                "batch_s_at_slo": m["batch_s_at_slo"],
                "batch_rank": m["batch_rank"],
                "p95_latency_ms": m["p95_latency_ms"],
            })
    print(f"Wrote {csv_path}")
    print(f"\n{len(final_models)} models in leaderboard")


if __name__ == "__main__":
    main()
