#!/usr/bin/env python3
"""
Mixed-task failure analysis for P2 training logs.

Categorizes every training step as one of:
  - valid:           json_valid == 1
  - format_collapse: output_text contains no '{' at all (model stopped producing JSON)
  - partial_json:    output_text contains '{' but json_valid == 0 (malformed JSON)
  - wrong_schema:    (subset of partial_json) output parses as JSON but doesn't
                     match the expected schema for that step

Outputs:
  results/p2_analysis/mixed_failure_analysis.json   -- full per-model breakdowns
  stdout                                            -- summary table
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = BASE_DIR / "out" / "p2_training_20260124"
SCHEMA_DIR = BASE_DIR / "out" / "cache" / "t1t5_balanced_b4a5a32e"
RESULTS_DIR = BASE_DIR / "results" / "p2_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIRST_N = 100   # "early" window
LAST_N = 100    # "late" window

# ---------------------------------------------------------------------------
# Schema loading  (for wrong_schema detection)
# ---------------------------------------------------------------------------

def load_schemas():
    """Load JSON schemas keyed by their full path."""
    schemas = {}
    for p in sorted(SCHEMA_DIR.glob("*.json")):
        with open(p) as f:
            schemas[str(p)] = json.load(f)
    return schemas

SCHEMAS = load_schemas()

# Map schema filename to the set of required top-level keys.
# This is a lightweight check: if the parsed JSON object's top-level keys
# don't overlap with the schema's required keys, it's "wrong_schema".
SCHEMA_REQUIRED_KEYS: dict[str, set[str]] = {}
for path, schema in SCHEMAS.items():
    req = set(schema.get("required", []))
    props = set(schema.get("properties", {}).keys())
    SCHEMA_REQUIRED_KEYS[path] = req if req else props


def _extract_first_json_object(text: str):
    """Try to extract the first JSON object from text (handles markdown fences)."""
    # Strip markdown code fences
    stripped = re.sub(r"```(?:json)?\s*", "", text)
    stripped = re.sub(r"```", "", stripped)

    # Find first '{' ... last matching '}'
    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def matches_schema(parsed_obj: dict, schema_path: str) -> bool:
    """Lightweight check: does the parsed object have the right top-level keys?"""
    if not isinstance(parsed_obj, dict):
        return False
    required = SCHEMA_REQUIRED_KEYS.get(schema_path, set())
    if not required:
        return True  # can't check, assume ok
    obj_keys = set(parsed_obj.keys())
    # Must have at least half the required keys to count as "right schema"
    overlap = obj_keys & required
    return len(overlap) >= len(required) / 2


# ---------------------------------------------------------------------------
# Categorize a single step
# ---------------------------------------------------------------------------

def categorize_step(rec: dict) -> str:
    """Return one of: valid, format_collapse, wrong_schema, partial_json."""
    if rec.get("json_valid") == 1:
        return "valid"

    output = rec.get("output_text", "")
    if "{" not in output:
        return "format_collapse"

    # There is a '{' but json_valid == 0.  Try to parse and check schema.
    parsed = _extract_first_json_object(output)
    if parsed is not None:
        # JSON parsed successfully but the training harness marked it invalid.
        # This means it parsed but didn't match the schema.
        return "wrong_schema"

    return "partial_json"


# ---------------------------------------------------------------------------
# Task name helper
# ---------------------------------------------------------------------------

def task_short_name(schema_path: str) -> str:
    """t1_incident_schema.json -> T1_incident"""
    base = os.path.basename(schema_path)
    base = base.replace("_schema.json", "")
    parts = base.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0].upper()}_{parts[1]}"
    return base


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_log(log_path: Path) -> dict:
    """Analyze a single train_log.jsonl, return stats dict."""
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed lines

    total = len(records)
    if total == 0:
        return {"total_steps": 0}

    categories = defaultdict(int)
    per_task = defaultdict(lambda: defaultdict(int))
    early_valid = 0
    early_total = 0
    late_valid = 0
    late_total = 0

    for rec in records:
        step = rec["step"]
        cat = categorize_step(rec)
        categories[cat] += 1

        task = task_short_name(rec.get("schema_path", "unknown"))
        per_task[task][cat] += 1
        per_task[task]["total"] += 1

        if step < FIRST_N:
            early_total += 1
            if cat == "valid":
                early_valid += 1
        if step >= total - LAST_N:
            late_total += 1
            if cat == "valid":
                late_valid += 1

    result = {
        "total_steps": total,
        "overall_validity_pct": round(100.0 * categories["valid"] / total, 2),
        "categories": {
            "valid": categories["valid"],
            "format_collapse": categories["format_collapse"],
            "wrong_schema": categories["wrong_schema"],
            "partial_json": categories["partial_json"],
        },
        "category_pcts": {
            "valid": round(100.0 * categories["valid"] / total, 2),
            "format_collapse": round(100.0 * categories["format_collapse"] / total, 2),
            "wrong_schema": round(100.0 * categories["wrong_schema"] / total, 2),
            "partial_json": round(100.0 * categories["partial_json"] / total, 2),
        },
        "early_validity_pct": round(100.0 * early_valid / early_total, 2) if early_total else None,
        "late_validity_pct": round(100.0 * late_valid / late_total, 2) if late_total else None,
        "per_task": {},
    }

    for task, counts in sorted(per_task.items()):
        t_total = counts["total"]
        result["per_task"][task] = {
            "total": t_total,
            "valid_pct": round(100.0 * counts["valid"] / t_total, 2),
            "format_collapse_pct": round(100.0 * counts["format_collapse"] / t_total, 2),
            "wrong_schema_pct": round(100.0 * counts["wrong_schema"] / t_total, 2),
            "partial_json_pct": round(100.0 * counts["partial_json"] / t_total, 2),
        }

    return result


def main():
    all_results = {}

    # Discover models
    model_dirs = sorted([
        d for d in TRAINING_DIR.iterdir()
        if d.is_dir() and (d / "Mixed").is_dir()
    ])

    for model_dir in model_dirs:
        model_name = model_dir.name
        mixed_dir = model_dir / "Mixed"
        seed_dirs = sorted([
            s for s in mixed_dir.iterdir()
            if s.is_dir() and s.name.startswith("seed_")
        ])

        seed_results = {}
        for seed_dir in seed_dirs:
            log_path = seed_dir / "train_log.jsonl"
            if not log_path.exists():
                continue
            seed_name = seed_dir.name  # e.g., seed_42
            seed_results[seed_name] = analyze_log(log_path)

        if not seed_results:
            continue

        # Filter out empty logs (total_steps == 0)
        valid_seeds = {k: v for k, v in seed_results.items() if v["total_steps"] > 0}
        if not valid_seeds:
            print(f"  SKIPPING {model_name}: all seed logs are empty", file=sys.stderr)
            continue

        # Aggregate across seeds
        n_seeds = len(valid_seeds)
        agg_valid = sum(s["overall_validity_pct"] for s in valid_seeds.values()) / n_seeds
        agg_early = [s["early_validity_pct"] for s in valid_seeds.values() if s["early_validity_pct"] is not None]
        agg_late = [s["late_validity_pct"] for s in valid_seeds.values() if s["late_validity_pct"] is not None]

        # Aggregate category percentages
        agg_cats = {}
        for cat in ["valid", "format_collapse", "wrong_schema", "partial_json"]:
            vals = [s["category_pcts"][cat] for s in valid_seeds.values()]
            agg_cats[cat] = round(sum(vals) / len(vals), 2)

        # Aggregate per-task validity
        all_tasks = set()
        for s in valid_seeds.values():
            all_tasks.update(s["per_task"].keys())

        agg_per_task = {}
        for task in sorted(all_tasks):
            task_vals = []
            task_fc = []
            task_ws = []
            task_pj = []
            for s in valid_seeds.values():
                if task in s["per_task"]:
                    task_vals.append(s["per_task"][task]["valid_pct"])
                    task_fc.append(s["per_task"][task]["format_collapse_pct"])
                    task_ws.append(s["per_task"][task]["wrong_schema_pct"])
                    task_pj.append(s["per_task"][task]["partial_json_pct"])
            agg_per_task[task] = {
                "valid_pct": round(sum(task_vals) / len(task_vals), 2) if task_vals else 0,
                "format_collapse_pct": round(sum(task_fc) / len(task_fc), 2) if task_fc else 0,
                "wrong_schema_pct": round(sum(task_ws) / len(task_ws), 2) if task_ws else 0,
                "partial_json_pct": round(sum(task_pj) / len(task_pj), 2) if task_pj else 0,
            }

        all_results[model_name] = {
            "seeds": seed_results,
            "aggregate": {
                "n_seeds": n_seeds,
                "overall_validity_pct": round(agg_valid, 2),
                "early_validity_pct": round(sum(agg_early) / len(agg_early), 2) if agg_early else None,
                "late_validity_pct": round(sum(agg_late) / len(agg_late), 2) if agg_late else None,
                "category_pcts": agg_cats,
                "per_task": agg_per_task,
            },
        }

    # -----------------------------------------------------------------------
    # Write JSON results
    # -----------------------------------------------------------------------
    out_path = RESULTS_DIR / "mixed_failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed results written to: {out_path}\n")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("=" * 120)
    print("MIXED-TASK FAILURE ANALYSIS  --  Averaged across seeds")
    print("=" * 120)

    # Table 1: Overall breakdown
    header = f"{'Model':<20} {'Valid%':>8} {'FmtColl%':>10} {'WrongSch%':>10} {'PartJSON%':>10} {'Early%':>8} {'Late%':>8} {'Delta':>8}"
    print("\n" + header)
    print("-" * len(header))

    # Sort by validity ascending to highlight failures
    sorted_models = sorted(all_results.keys(), key=lambda m: all_results[m]["aggregate"]["overall_validity_pct"])

    for model in sorted_models:
        agg = all_results[model]["aggregate"]
        cp = agg["category_pcts"]
        early = agg["early_validity_pct"]
        late = agg["late_validity_pct"]
        delta = round(late - early, 2) if early is not None and late is not None else None
        delta_str = f"{delta:+.1f}" if delta is not None else "N/A"
        early_str = f"{early:.1f}" if early is not None else "N/A"
        late_str = f"{late:.1f}" if late is not None else "N/A"

        print(f"{model:<20} {cp['valid']:>8.1f} {cp['format_collapse']:>10.1f} {cp['wrong_schema']:>10.1f} {cp['partial_json']:>10.1f} {early_str:>8} {late_str:>8} {delta_str:>8}")

    # Table 2: Per-task breakdown for worst models
    print("\n" + "=" * 120)
    print("PER-TASK VALIDITY (%) FOR MODELS WITH < 50% OVERALL VALIDITY")
    print("=" * 120)

    # Gather all task names
    all_task_names = set()
    for model in all_results:
        all_task_names.update(all_results[model]["aggregate"]["per_task"].keys())
    task_names = sorted(all_task_names)

    task_header = f"{'Model':<20}" + "".join(f" {t:>16}" for t in task_names)
    print("\n" + task_header)
    print("-" * len(task_header))

    for model in sorted_models:
        agg = all_results[model]["aggregate"]
        if agg["overall_validity_pct"] >= 50:
            continue
        row = f"{model:<20}"
        for task in task_names:
            if task in agg["per_task"]:
                val = agg["per_task"][task]["valid_pct"]
                row += f" {val:>16.1f}"
            else:
                row += f" {'N/A':>16}"
        print(row)

    # Table 3: Per-task failure mode breakdown for the WORST model
    worst_model = sorted_models[0]
    print(f"\n{'=' * 120}")
    print(f"FAILURE MODE BREAKDOWN BY TASK -- {worst_model} (worst overall)")
    print(f"{'=' * 120}")

    fm_header = f"{'Task':<20} {'Valid%':>8} {'FmtColl%':>10} {'WrongSch%':>10} {'PartJSON%':>10}"
    print("\n" + fm_header)
    print("-" * len(fm_header))

    worst_agg = all_results[worst_model]["aggregate"]["per_task"]
    for task in sorted(worst_agg.keys()):
        t = worst_agg[task]
        print(f"{task:<20} {t['valid_pct']:>8.1f} {t['format_collapse_pct']:>10.1f} {t['wrong_schema_pct']:>10.1f} {t['partial_json_pct']:>10.1f}")

    # Table 4: Learning trajectory - early vs late per model
    print(f"\n{'=' * 120}")
    print("LEARNING TRAJECTORY: EARLY (first 100 steps) vs LATE (last 100 steps) VALIDITY %")
    print(f"{'=' * 120}")

    traj_header = f"{'Model':<20} {'Early%':>8} {'Late%':>8} {'Delta':>8} {'Trajectory':>15}"
    print("\n" + traj_header)
    print("-" * len(traj_header))

    for model in sorted_models:
        agg = all_results[model]["aggregate"]
        early = agg["early_validity_pct"]
        late = agg["late_validity_pct"]
        if early is None or late is None:
            continue
        delta = round(late - early, 2)
        if delta > 5:
            traj = "IMPROVING"
        elif delta < -5:
            traj = "COLLAPSING"
        else:
            traj = "STABLE"
        print(f"{model:<20} {early:>8.1f} {late:>8.1f} {delta:>+8.1f} {traj:>15}")

    print()


if __name__ == "__main__":
    main()
