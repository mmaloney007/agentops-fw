"""AgentSLO-Bench CLI.

Usage:
    agentslo-bench baseline     Print built-in 13-model P1 results
    agentslo-bench leaderboard  Generate rankings from result files
    agentslo-bench run          Evaluate an endpoint against benchmark tasks
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .slo_tiers import TIERS
from .leaderboard import load_p1_baseline, generate_leaderboard, format_markdown, format_latex
from .benchmark_runner import (
    compute_from_p1_data,
    save_results,
    TaskResult,
    BenchmarkResult,
    compute_tier_results,
)


def cmd_baseline(args: argparse.Namespace) -> None:
    """Print built-in 13-model P1 baseline results."""
    p1_data = load_p1_baseline()
    models = p1_data["models"]

    print(f"AgentSLO-Bench Baseline: {len(models)} models, 5 tasks, 3 SLO tiers\n")

    for tier in TIERS:
        print(f"=== {tier.name.upper()} TIER ({tier.deadline_ms/1000:.0f}s) ===")
        print(f"  {tier.description}")
        print(f"  Typical use: {tier.typical_use}\n")

        print(f"  {'Model':<20} {'Accuracy':>10} {'S@SLO':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")

        for model_name, model_info in models.items():
            results = compute_from_p1_data(model_info, model_name)
            total_correct = 0
            total_slo = 0
            total_count = 0
            for r in results:
                tr = r.tier_results.get(tier.name)
                if tr:
                    total_correct += tr.correct
                    total_slo += tr.success_at_slo
                    total_count += tr.total
            acc = 100.0 * total_correct / max(1, total_count)
            slo = 100.0 * total_slo / max(1, total_count)
            print(f"  {model_name:<20} {acc:>9.1f}% {slo:>9.1f}%")
        print()


def cmd_leaderboard(args: argparse.Namespace) -> None:
    """Generate leaderboard rankings."""
    leaderboard = generate_leaderboard()

    if args.format == "markdown":
        print(format_markdown(leaderboard))
    elif args.format == "latex":
        print(format_latex(leaderboard))
    else:
        print(json.dumps(leaderboard, indent=2, default=str))


def _load_task_samples(task_id: str, limit: int = 10) -> list[dict]:
    """Load sample tasks for a given task ID."""
    root = Path(__file__).parent.parent.parent
    task_files = {
        "T1": root / "tasks" / "clinc_en.jsonl",
        "T2": root / "tasks" / "hotpot_dev.jsonl",
        "T3": root / "tasks" / "fc_tasks.jsonl",
        "T4": root / "tasks" / "t4_bfcl.jsonl",
        "T5": root / "tasks" / "public_humaneval.jsonl",
    }
    task_file = task_files.get(task_id)
    if not task_file or not task_file.exists():
        return []

    samples = []
    with open(task_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
                if len(samples) >= limit:
                    break
            except json.JSONDecodeError:
                continue
    return samples


def _extract_json(content: str) -> dict:
    """Extract JSON from content, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Handle markdown code blocks
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if not part:
                continue
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue

    # Try to find JSON object in text
    import re
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def _extract_gold_function_name(gold: dict) -> str | None:
    """Extract expected function/tool name from heterogeneous gold formats."""
    if not isinstance(gold, dict) or not gold:
        return None
    if "function_name" in gold:
        return str(gold.get("function_name"))
    if "name" in gold:
        return str(gold.get("name"))
    if len(gold) == 1:
        return str(next(iter(gold.keys())))
    return None


def _compute_task_correct(
    task_id: str,
    gold: dict,
    output: dict,
    json_valid: bool,
) -> bool:
    """Task-aware correctness check for live benchmark runs.

    Conservative by design: if no comparable gold signal is available, returns False.
    """
    if not json_valid or not isinstance(output, dict):
        return False
    if not isinstance(gold, dict) or not gold:
        return False

    if task_id == "T1":
        expected_intent = gold.get("intent")
        return expected_intent is not None and str(output.get("intent")) == str(
            expected_intent
        )

    if task_id in {"T3", "T4"}:
        gold_name = _extract_gold_function_name(gold)
        if not gold_name:
            return False
        pred_name = output.get("function_name", output.get("name", output.get("tool")))
        return str(pred_name) == str(gold_name)

    if task_id == "T2" and "answer" in gold:
        return str(output.get("answer", "")).strip() == str(gold.get("answer")).strip()

    comparable = [k for k, v in gold.items() if isinstance(v, (str, int, float, bool))]
    if comparable:
        return all(str(output.get(k, "")) == str(gold.get(k, "")) for k in comparable)
    return False


def _run_live_eval(
    endpoint: str,
    model: str,
    task_id: str,
    samples: list[dict],
) -> list[TaskResult]:
    """Run live evaluation against an endpoint."""
    from openai import OpenAI

    client = OpenAI(base_url=endpoint, api_key="not-needed")
    results = []

    for i, sample in enumerate(samples):
        prompt = sample.get("prompt", sample.get("text", ""))

        # Build the request - prompt already contains full instructions
        messages = [
            {"role": "user", "content": prompt},
        ]

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
            )
            latency_ms = (time.time() - t0) * 1000

            content = response.choices[0].message.content or ""
            output = _extract_json(content)
            json_valid = bool(output)

            # Check correctness based on gold
            gold = sample.get("gold", sample.get("label", {}))
            task_correct = _compute_task_correct(task_id, gold, output, json_valid)

        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            json_valid = False
            task_correct = False
            output = {"error": str(e)}

        results.append(TaskResult(
            task_id=f"{task_id}_{i}",
            latency_ms=latency_ms,
            json_valid=json_valid,
            task_correct=task_correct,
            output=output,
        ))

        # Progress indicator
        status = "✓" if task_correct else "✗"
        print(f"  [{i+1}/{len(samples)}] {status} {latency_ms:.0f}ms", end="\r")

    print()  # Clear progress line
    return results


def cmd_run(args: argparse.Namespace) -> None:
    """Run live evaluation against an LM Studio or OpenAI-compatible endpoint."""
    print("AgentSLO-Bench Live Evaluation")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Model: {args.model}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Samples per task: {args.samples}")
    print()

    all_results = []

    for task_id in args.tasks:
        print(f"=== Task {task_id} ===")
        samples = _load_task_samples(task_id, limit=args.samples)
        if not samples:
            print(f"  No samples found for {task_id}, skipping")
            continue

        print(f"  Running {len(samples)} samples...")
        task_results = _run_live_eval(args.endpoint, args.model, task_id, samples)
        tier_results = compute_tier_results(task_results)

        all_results.append(BenchmarkResult(
            model_name=args.model,
            task_name=task_id,
            tier_results=tier_results,
            task_results=task_results,
        ))

        # Print per-tier results
        for tier in TIERS:
            tr = tier_results[tier.name]
            print(f"  {tier.name.title():12} ({tier.deadline_ms/1000:.0f}s): "
                  f"S@SLO={tr.success_at_slo_pct:.1f}% "
                  f"(Acc={tr.accuracy_pct:.1f}%, OnTime={tr.on_time_pct:.1f}%)")
        print()

    # Summary
    if all_results:
        print("=== Summary ===")
        for tier in TIERS:
            total_count = 0
            total_slo = 0
            for br in all_results:
                tr = br.tier_results.get(tier.name)
                if tr:
                    total_count += tr.total
                    total_slo += tr.success_at_slo
            slo_pct = 100.0 * total_slo / max(1, total_count)
            print(f"  {tier.name.title():12} tier: {slo_pct:.1f}% Success@SLO ({total_slo}/{total_count})")

        # Save results if output path provided
        if args.out:
            out_path = Path(args.out)
            save_results(all_results, out_path)
            print(f"\nResults saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agentslo-bench",
        description="AgentSLO-Bench: SLO-aware LLM agent benchmark",
    )
    sub = parser.add_subparsers(dest="command")

    # baseline
    sub.add_parser("baseline", help="Print built-in 13-model P1 baseline results")

    # leaderboard
    p_lead = sub.add_parser("leaderboard", help="Generate leaderboard rankings")
    p_lead.add_argument("--format", choices=["markdown", "latex", "json"], default="markdown")

    # run
    p_run = sub.add_parser("run", help="Evaluate an endpoint against benchmark tasks")
    p_run.add_argument("--endpoint", default="http://localhost:1234/v1",
                       help="OpenAI-compatible API endpoint")
    p_run.add_argument("--model", default="local-model",
                       help="Model name to use in API calls")
    p_run.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"],
                       help="Task IDs to evaluate (T1, T2, T3, T4, T5)")
    p_run.add_argument("--samples", type=int, default=10,
                       help="Number of samples per task")
    p_run.add_argument("--out", default=None,
                       help="Output path for results JSON")

    args = parser.parse_args()

    if args.command == "baseline":
        cmd_baseline(args)
    elif args.command == "leaderboard":
        cmd_leaderboard(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
