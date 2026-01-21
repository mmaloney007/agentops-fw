#!/usr/bin/env python3
"""
Direct P1 Evaluation - All 13 Models × All 5 Task Types (T1-T5)
Runs in a single process to avoid venv/import issues.

Usage:
    source .venv/bin/activate
    export OPENAI_API_BASE="http://10.11.196.166:1234/v1"
    python scripts/run_p1_direct.py
"""

import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator, ValidationError

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_stable_slo.rollout.engine import provider_generate
from agent_stable_slo.utils.data import fingerprint_tasks

# Configuration
SLO_DEADLINE_MS = 2000
OUT_DIR = Path("out/p1_comprehensive_20260118")
RESULTS_FILE = OUT_DIR / "all_results.json"

# Task files (T1-T5)
TASKS = {
    "T1": "tasks/t1_structured.jsonl",
    "T2": "tasks/t2_grounded.jsonl",
    "T3": "tasks/t3_tools.jsonl",
    "T4": "tasks/t4_bfcl.jsonl",
    "T5": "tasks/t5_swebench.jsonl",
}

# Models (LM Studio IDs verified working)
MODELS = {
    "llama-3.2-1b": {"id": "llama-3.2-1b-instruct", "size": "1B", "vendor": "Meta"},
    "llama-3.2-3b": {"id": "meta-llama_-_llama-3.2-3b-instruct", "size": "3B", "vendor": "Meta"},
    "qwen2.5-3b": {"id": "qwen2.5-3b-instruct", "size": "3B", "vendor": "Alibaba"},
    "phi-3-mini": {"id": "phi-3-mini-4k-instruct", "size": "3.8B", "vendor": "Microsoft"},
    "qwen3-4b": {"id": "qwen3-4b", "size": "4B", "vendor": "Alibaba"},
    "yi-1.5-6b": {"id": "01-ai_-_yi-1.5-6b-chat", "size": "6B", "vendor": "01.AI"},
    "mistral-7b-v0.3": {"id": "mistralai_-_mistral-7b-instruct-v0.3", "size": "7B", "vendor": "Mistral"},
    "falcon-mamba-7b": {"id": "falcon-mamba-7b-instruct", "size": "7B", "vendor": "TII"},
    "gpt-oss-20b": {"id": "openai/gpt-oss-20b", "size": "20B", "vendor": "OpenAI"},
    "ministral-8b": {"id": "ministral-8b-instruct-2410", "size": "8B", "vendor": "Mistral"},
    "llama-3.1-8b": {"id": "meta-llama-llama-3.1-8b-instruct", "size": "8B", "vendor": "Meta"},
    "gemma-2-9b": {"id": "google/gemma-2-9b", "size": "9B", "vendor": "Google"},
    "gemma-3-12b": {"id": "google/gemma-3-12b", "size": "12B", "vendor": "Google"},
}


def load_tasks(path: str) -> List[Dict[str, Any]]:
    """Load tasks from JSONL file."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate_json(obj: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate JSON against schema."""
    try:
        Draft202012Validator(schema).validate(obj)
        return True
    except ValidationError:
        return False


def score_record(task: Dict[str, Any], out_json: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, float]:
    """Score a single task output."""
    metrics: Dict[str, float] = {}
    metrics["json_valid"] = 1.0 if validate_json(out_json, schema) else 0.0

    gold = task.get("gold", {})
    ttype = task.get("task_type", "")

    if ttype == "t1":
        # T1: Incident classification
        fields = ["category", "severity", "source", "time_window"]
        matches = []
        for f in fields:
            pred = str(out_json.get(f, "")).lower().strip()
            expected = str(gold.get(f, "")).lower().strip()
            matches.append(1.0 if pred == expected else 0.0)
        metrics["t1_field_acc"] = sum(matches) / len(matches) if matches else 0.0

    elif ttype == "t2":
        # T2: Grounded reasoning
        pred_summary = str(out_json.get("short_summary", "")).lower()
        gold_summary = str(gold.get("short_summary", "")).lower()
        # Simple token overlap F1
        pred_tokens = set(pred_summary.split())
        gold_tokens = set(gold_summary.split())
        if gold_tokens:
            inter = len(pred_tokens & gold_tokens)
            prec = inter / max(1, len(pred_tokens))
            rec = inter / len(gold_tokens)
            metrics["t2_summary_f1"] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        else:
            metrics["t2_summary_f1"] = 1.0 if not pred_tokens else 0.0

    elif ttype == "t3":
        # T3: Tool calling
        tool_match = str(out_json.get("tool", "")).lower() == str(gold.get("tool", "")).lower()
        metrics["t3_tool_match"] = 1.0 if tool_match else 0.0

        gold_args = gold.get("arguments", {}) if isinstance(gold.get("arguments"), dict) else {}
        out_args = out_json.get("arguments", {}) if isinstance(out_json.get("arguments"), dict) else {}
        args_match = tool_match and all(
            str(out_args.get(k, "")).lower().strip() == str(v).lower().strip()
            for k, v in gold_args.items()
        )
        metrics["t3_args_match"] = 1.0 if args_match else 0.0
        metrics["t3_success"] = 1.0 if tool_match and args_match else 0.0

    elif ttype == "t4":
        # T4: Function calling (BFCL)
        pred_name = out_json.get("name", "")
        gold_funcs = list(gold.keys()) if isinstance(gold, dict) else []
        name_match = pred_name in gold_funcs
        metrics["t4_func_match"] = 1.0 if name_match else 0.0

    elif ttype == "t5":
        # T5: SWE-bench (patch generation)
        patch = out_json.get("patch", "")
        metrics["t5_has_patch"] = 1.0 if patch and len(str(patch).strip()) > 10 else 0.0

    return metrics


def run_eval_task(model_name: str, model_id: str, task_name: str, task_file: str) -> Dict[str, Any]:
    """Run evaluation for a single model/task combination."""
    print(f"  [{task_name}] Loading tasks from {task_file}...")
    tasks = load_tasks(task_file)

    # Set model env var
    os.environ["LMSTUDIO_MODEL"] = model_id

    # Load schema cache
    schema_cache: Dict[str, Dict[str, Any]] = {}

    results = []
    for i, rec in enumerate(tasks):
        schema_path = rec.get("schema_path")
        if not schema_path:
            print(f"    Warning: task {rec.get('id')} missing schema_path, skipping")
            continue

        # Load schema
        if schema_path not in schema_cache:
            schema_file = Path(schema_path)
            if not schema_file.exists():
                schema_file = Path(__file__).resolve().parents[1] / schema_path
            if schema_file.exists():
                schema_cache[schema_path] = json.load(open(schema_file, "r", encoding="utf-8"))
            else:
                print(f"    Warning: schema not found: {schema_path}")
                continue

        schema = schema_cache[schema_path]

        # Run inference
        try:
            out_json, lat_ms, ttft_ms, tokens = provider_generate(rec["prompt"], schema, mode="structured")
            metrics = score_record(rec, out_json, schema)
            error_msg = None
        except Exception as exc:
            out_json, lat_ms, ttft_ms, tokens = {}, 1e6, 1e6, -1
            metrics = {"json_valid": 0.0}
            error_msg = str(exc)

        results.append({
            "id": rec.get("id"),
            "task_type": rec.get("task_type"),
            "metrics": metrics,
            "latency_ms": float(lat_ms),
            "ttft_ms": float(ttft_ms),
            "tokens_out": tokens,
            "error": error_msg,
        })

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(tasks)}")

    # Compute aggregates
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms", 0) < 1e5]
    if not latencies:
        return {"error": "no valid results", "count": 0}

    sorted_lat = sorted(latencies)
    p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
    p99_idx = min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)

    # OLD: Latency-only SLO (for comparison)
    within_slo_latency_only = sum(1 for l in latencies if l <= SLO_DEADLINE_MS)

    # NEW: Joint Success@SLO = json_valid AND task_correct AND latency <= SLO
    joint_success_count = 0
    for r in results:
        metrics = r.get("metrics", {})
        lat = r.get("latency_ms", 1e6)
        ttype = r.get("task_type", "")

        # Check all three conditions
        json_valid = metrics.get("json_valid", 0) == 1.0
        within_latency = lat <= SLO_DEADLINE_MS

        # Task-specific correctness thresholds
        if ttype == "t1":
            task_correct = metrics.get("t1_field_acc", 0) >= 0.75  # 3/4 fields
        elif ttype == "t2":
            task_correct = metrics.get("t2_summary_f1", 0) >= 0.3  # meaningful overlap
        elif ttype == "t3":
            task_correct = metrics.get("t3_success", 0) == 1.0  # tool + args
        elif ttype == "t4":
            task_correct = metrics.get("t4_func_match", 0) == 1.0  # function name
        elif ttype == "t5":
            task_correct = metrics.get("t5_has_patch", 0) == 1.0  # has patch
        else:
            task_correct = True  # unknown task type, skip accuracy check

        if json_valid and task_correct and within_latency:
            joint_success_count += 1

    # Aggregate all metrics
    metric_sums: Dict[str, List[float]] = {}
    for r in results:
        for k, v in r.get("metrics", {}).items():
            if k not in metric_sums:
                metric_sums[k] = []
            metric_sums[k].append(float(v))

    summary = {
        "count": len(results),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "p95_latency_ms": round(sorted_lat[p95_idx], 1),
        "p99_latency_ms": round(sorted_lat[p99_idx], 1),
        "latency_slo_pct": round(within_slo_latency_only / len(results) * 100, 1),  # old metric
        "success_at_slo_pct": round(joint_success_count / len(results) * 100, 1),  # NEW: joint metric
    }

    for k, vals in metric_sums.items():
        if vals:
            summary[k] = round(sum(vals) / len(vals), 4)

    return summary


def load_results() -> Dict[str, Any]:
    """Load existing results."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"started": datetime.now().isoformat(), "models": {}}


def save_results(results: Dict[str, Any]):
    """Save results."""
    results["last_updated"] = datetime.now().isoformat()
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def update_progress_md(results: Dict[str, Any]):
    """Update PROGRESS.md with current results."""
    lines = [
        "# P1 Evaluation Progress",
        "",
        f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**SLO Deadline**: {SLO_DEADLINE_MS}ms",
        "",
        "## Results Summary",
        "",
        "| Model | Size | Vendor | T1 | T2 | T3 | T4 | T5 | Avg Lat (ms) | P95 (ms) | Success@SLO |",
        "|-------|------|--------|----|----|----|----|----|---------|----|-------------|",
    ]

    for model_name, model_info in MODELS.items():
        model_data = results.get("models", {}).get(model_name, {})
        tasks_done = model_data.get("tasks", {})

        def get_status(task_name):
            if task_name in tasks_done:
                data = tasks_done[task_name]
                if isinstance(data, dict) and "count" in data:
                    return "✅"
            return "⏳"

        # Aggregate metrics
        all_lats = []
        all_p95s = []
        all_slos = []
        for task_data in tasks_done.values():
            if isinstance(task_data, dict):
                if "avg_latency_ms" in task_data:
                    all_lats.append(task_data["avg_latency_ms"])
                if "p95_latency_ms" in task_data:
                    all_p95s.append(task_data["p95_latency_ms"])
                if "success_at_slo_pct" in task_data:
                    all_slos.append(task_data["success_at_slo_pct"])

        avg_lat = f"{sum(all_lats)/len(all_lats):.0f}" if all_lats else "-"
        p95 = f"{max(all_p95s):.0f}" if all_p95s else "-"
        slo = f"{sum(all_slos)/len(all_slos):.1f}%" if all_slos else "-"

        lines.append(
            f"| {model_name} | {model_info['size']} | {model_info['vendor']} | "
            f"{get_status('T1')} | {get_status('T2')} | {get_status('T3')} | "
            f"{get_status('T4')} | {get_status('T5')} | {avg_lat} | {p95} | {slo} |"
        )

    # Count progress
    total = len(MODELS) * len(TASKS)
    completed = sum(
        len([t for t in results.get("models", {}).get(m, {}).get("tasks", {}).values()
             if isinstance(t, dict) and "count" in t])
        for m in MODELS
    )
    lines.extend([
        "",
        f"**Progress**: {completed}/{total} ({completed/total*100:.1f}%)",
        "",
    ])

    Path("PROGRESS.md").write_text("\n".join(lines))
    print(f"  [SAVED] PROGRESS.md")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    print("=" * 60)
    print("P1 Comprehensive Evaluation")
    print(f"13 Models × 5 Task Types = 65 evaluations")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    for model_name, model_info in MODELS.items():
        print(f"\n>>> {model_name} ({model_info['size']}, {model_info['vendor']})")

        if model_name not in results["models"]:
            results["models"][model_name] = {"info": model_info, "tasks": {}}

        for task_name, task_file in TASKS.items():
            # Skip if already done
            existing = results["models"][model_name].get("tasks", {}).get(task_name, {})
            if isinstance(existing, dict) and "count" in existing:
                print(f"  [{task_name}] SKIP - already done ({existing['count']} records)")
                continue

            try:
                summary = run_eval_task(model_name, model_info["id"], task_name, task_file)
                results["models"][model_name]["tasks"][task_name] = summary
                print(f"  [{task_name}] DONE - {summary.get('count', 0)} records, "
                      f"{summary.get('success_at_slo_pct', 0):.1f}% SLO, "
                      f"P95={summary.get('p95_latency_ms', 0):.0f}ms")
            except Exception as e:
                print(f"  [{task_name}] ERROR: {e}")
                results["models"][model_name]["tasks"][task_name] = {"error": str(e)}

            # Save after each task
            save_results(results)
            update_progress_md(results)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Results saved to: {RESULTS_FILE}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
