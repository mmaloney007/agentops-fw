#!/usr/bin/env python3
"""
Comprehensive P1 Evaluation - All 13 Models × All 5 Task Types (T1-T5)
Tracks progress in RESULTS.json and updates PROGRESS.md

Usage:
    source .venv/bin/activate
    export OPENAI_API_BASE="http://10.11.196.166:1234/v1"
    python scripts/run_p1_comprehensive.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configuration
LMSTUDIO_BASE = os.getenv("OPENAI_API_BASE", "http://10.11.196.166:1234/v1")
OUT_DIR = Path("out/p1_comprehensive_20260118")
RESULTS_FILE = OUT_DIR / "results.json"
SLO_DEADLINE_MS = 2000

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
    "ministral-8b": {"id": "mistralai.ministral-8b-instruct-2410", "size": "8B", "vendor": "Mistral"},
    "llama-3.1-8b": {"id": "meta-llama-llama-3.1-8b-instruct", "size": "8B", "vendor": "Meta"},
    "gemma-2-9b": {"id": "google/gemma-2-9b", "size": "9B", "vendor": "Google"},
    "gemma-3-12b": {"id": "google/gemma-3-12b", "size": "12B", "vendor": "Google"},
}


def load_results():
    """Load existing results or create new structure."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {
        "started": datetime.now().isoformat(),
        "slo_deadline_ms": SLO_DEADLINE_MS,
        "models": {},
    }


def save_results(results):
    """Save results to JSON file."""
    results["last_updated"] = datetime.now().isoformat()
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def compute_metrics(predictions_file: Path) -> dict:
    """Compute metrics from predictions file."""
    if not predictions_file.exists():
        return {"error": "predictions file not found"}

    latencies = []
    json_valid = 0
    total = 0
    task_metrics = {}

    for line in predictions_file.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        total += 1
        lat = r.get("latency_ms", 0)
        latencies.append(lat)

        metrics = r.get("metrics", {})
        if metrics.get("json_valid", 0) == 1.0:
            json_valid += 1

        # Aggregate task-specific metrics
        for k, v in metrics.items():
            if k not in task_metrics:
                task_metrics[k] = []
            task_metrics[k].append(float(v))

    if not latencies:
        return {"error": "no predictions"}

    sorted_lat = sorted(latencies)
    p95_idx = int(len(sorted_lat) * 0.95)
    p99_idx = int(len(sorted_lat) * 0.99)
    within_slo = sum(1 for l in latencies if l <= SLO_DEADLINE_MS)

    result = {
        "count": total,
        "json_valid_pct": round(json_valid / total * 100, 1),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "p95_latency_ms": round(sorted_lat[p95_idx], 1),
        "p99_latency_ms": round(sorted_lat[p99_idx], 1),
        "success_at_slo_pct": round(within_slo / total * 100, 1),
    }

    # Add aggregated task metrics
    for k, vals in task_metrics.items():
        if vals:
            result[k] = round(sum(vals) / len(vals), 4)

    return result


def run_eval(model_name: str, model_id: str, task_name: str, task_file: str) -> dict:
    """Run evaluation for a single model/task combination."""
    run_name = f"p1_{model_name}_{task_name}"
    model_dir = OUT_DIR / run_name / f"lmstudio_{model_id.replace('/', '-')}"
    predictions_file = model_dir / "predictions.jsonl"

    # Skip if already done
    if predictions_file.exists():
        print(f"  [SKIP] {model_name}/{task_name} - already complete")
        return compute_metrics(predictions_file)

    print(f"  [RUN] {model_name}/{task_name}...")

    env = os.environ.copy()
    env["OPENAI_API_BASE"] = LMSTUDIO_BASE
    env["AOFW_PROVIDER"] = "lmstudio"

    # Use venv python explicitly
    venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"
    cmd = [
        str(venv_python), "scripts/eval_t_suite.py",
        "--models", f"lmstudio:{model_id}",
        "--tasks", task_file,
        "--out-dir", str(OUT_DIR),
        "--run-name", run_name,
    ]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[:200]}")
            return {"error": result.stderr[:500]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}

    return compute_metrics(predictions_file)


def update_progress_md(results: dict):
    """Update PROGRESS.md with current results."""
    progress_file = Path("PROGRESS.md")

    # Build results table
    lines = [
        "# PROGRESS.md - Checkpoint Tracker",
        "",
        f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## P1 Evaluation Results (13 Models × 5 Task Types)",
        "",
        "| Model | Size | Vendor | T1 | T2 | T3 | T4 | T5 | Avg Lat | P95 | Success@SLO |",
        "|-------|------|--------|----|----|----|----|----|---------|----|-------------|",
    ]

    for model_name, model_info in MODELS.items():
        model_results = results.get("models", {}).get(model_name, {})
        tasks_done = model_results.get("tasks", {})

        t1 = "✅" if "T1" in tasks_done else "⏳"
        t2 = "✅" if "T2" in tasks_done else "⏳"
        t3 = "✅" if "T3" in tasks_done else "⏳"
        t4 = "✅" if "T4" in tasks_done else "⏳"
        t5 = "✅" if "T5" in tasks_done else "⏳"

        # Compute aggregate metrics across all completed tasks
        all_lats = []
        all_slos = []
        for task_data in tasks_done.values():
            if isinstance(task_data, dict) and "avg_latency_ms" in task_data:
                all_lats.append(task_data.get("avg_latency_ms", 0))
                all_slos.append(task_data.get("success_at_slo_pct", 0))

        avg_lat = f"{sum(all_lats)/len(all_lats):.0f}" if all_lats else "-"
        p95 = "-"
        if tasks_done:
            p95s = [t.get("p95_latency_ms", 0) for t in tasks_done.values() if isinstance(t, dict)]
            if p95s:
                p95 = f"{max(p95s):.0f}"
        slo = f"{sum(all_slos)/len(all_slos):.1f}%" if all_slos else "-"

        lines.append(f"| {model_name} | {model_info['size']} | {model_info['vendor']} | {t1} | {t2} | {t3} | {t4} | {t5} | {avg_lat} | {p95} | {slo} |")

    # Count completion
    total_tasks = len(MODELS) * len(TASKS)
    completed = sum(
        len(results.get("models", {}).get(m, {}).get("tasks", {}))
        for m in MODELS
    )
    lines.extend([
        "",
        f"**Progress**: {completed}/{total_tasks} task combinations ({completed/total_tasks*100:.1f}%)",
        "",
        "---",
        "",
        "## Run Log",
        "",
    ])

    # Add recent runs
    for model_name in MODELS:
        model_data = results.get("models", {}).get(model_name, {})
        for task_name, task_data in model_data.get("tasks", {}).items():
            if isinstance(task_data, dict) and "count" in task_data:
                lines.append(f"- {model_name}/{task_name}: {task_data.get('count', 0)} records, {task_data.get('success_at_slo_pct', 0):.1f}% SLO")

    progress_file.write_text("\n".join(lines))
    print(f"\n[SAVED] PROGRESS.md updated")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    print("=" * 60)
    print("P1 Comprehensive Evaluation - 13 Models × 5 Task Types")
    print(f"Started: {datetime.now()}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    for model_name, model_info in MODELS.items():
        print(f"\n>>> Model: {model_name} ({model_info['size']}, {model_info['vendor']})")

        if model_name not in results["models"]:
            results["models"][model_name] = {"tasks": {}, **model_info}

        for task_name, task_file in TASKS.items():
            # Skip if already done
            if task_name in results["models"][model_name].get("tasks", {}):
                existing = results["models"][model_name]["tasks"][task_name]
                if isinstance(existing, dict) and "count" in existing:
                    print(f"  [SKIP] {task_name} - already complete ({existing.get('count')} records)")
                    continue

            metrics = run_eval(model_name, model_info["id"], task_name, task_file)
            results["models"][model_name]["tasks"][task_name] = metrics

            # Save after each task
            save_results(results)
            update_progress_md(results)

            if "error" not in metrics:
                print(f"    Done: {metrics.get('count', 0)} records, {metrics.get('success_at_slo_pct', 0):.1f}% SLO")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Results: {RESULTS_FILE}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
