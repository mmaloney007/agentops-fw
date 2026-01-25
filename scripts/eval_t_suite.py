#!/usr/bin/env python3
"""
Evaluate the T1/T2/T3 task suites across multiple models/backends.

Usage:
  python scripts/eval_t_suite.py --models lmstudio:qwen/qwen3-4b-thinking-2507 \
    --tasks tasks/clinc_en.jsonl tasks/hotpot_dev.jsonl tasks/t3_tools.jsonl

Set AOFW_PROVIDER/LMSTUDIO_MODEL/OLLAMA_MODEL/VLLM_MODEL env vars as needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator, ValidationError

from agent_stable_slo.rollout.engine import provider_generate_raw
from agent_stable_slo.utils.data import fingerprint_tasks
from agent_stable_slo.eval.detailed_capture import (
    CaptureConfig,
    build_detailed_result,
    aggregate_stability_metrics,
)


def _norm(text: Any) -> str:
    return " ".join(str(text).lower().strip().split())


def _token_set(text: Any) -> set[str]:
    if not text:
        return set()
    return set(_norm(text).split())


def _f1_sets(pred: set[str], gold: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold:
        return 0.0
    inter = len(pred & gold)
    prec = inter / max(1, len(pred))
    rec = inter / len(gold)
    denom = prec + rec
    return (2 * prec * rec / denom) if denom else 0.0


def _load_tasks(paths: List[str], limit: int | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def _validate_json(obj: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    try:
        Draft202012Validator(schema).validate(obj)
        return True
    except ValidationError:
        return False


def _score_t1(out_json: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not gold:
        return metrics
    fields = ["category", "severity", "source", "time_window"]
    matches = []
    for f in fields:
        matches.append(
            1.0 if _norm(out_json.get(f, "")) == _norm(gold.get(f, "")) else 0.0
        )
    metrics["t1_field_acc"] = sum(matches) / len(matches)
    raw_tags = out_json.get("tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []
    pred_tags = set(_norm(t) for t in raw_tags)
    gold_tags = set(_norm(t) for t in gold.get("tags", []))
    metrics["t1_tag_f1"] = _f1_sets(pred_tags, gold_tags)
    metrics["t1_exact_match"] = (
        1.0 if metrics["t1_field_acc"] == 1.0 and metrics["t1_tag_f1"] == 1.0 else 0.0
    )
    return metrics


def _score_t2(out_json: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not gold:
        return metrics
    metrics["t2_summary_f1"] = _f1_sets(
        _token_set(out_json.get("short_summary", "")),
        _token_set(gold.get("short_summary", "")),
    )
    pred_points = []
    if isinstance(out_json.get("key_points"), list):
        pred_points = [_norm(p) for p in out_json["key_points"]]
    gold_points = [_norm(p) for p in gold.get("key_points", [])]
    metrics["t2_key_point_f1"] = _f1_sets(
        set(" ".join(pred_points).split()), set(" ".join(gold_points).split())
    )
    metrics["t2_primary_risk_match"] = (
        1.0
        if _norm(out_json.get("primary_risk", ""))
        == _norm(gold.get("primary_risk", ""))
        else 0.0
    )
    metrics["t2_action_match"] = (
        1.0
        if _norm(out_json.get("recommended_action", ""))
        == _norm(gold.get("recommended_action", ""))
        else 0.0
    )
    return metrics


def _score_t3(out_json: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not gold:
        return metrics
    tool_match = _norm(out_json.get("tool")) == _norm(gold.get("tool"))
    gold_args = (
        gold.get("arguments", {}) if isinstance(gold.get("arguments"), dict) else {}
    )
    out_args = (
        out_json.get("arguments", {})
        if isinstance(out_json.get("arguments"), dict)
        else {}
    )
    args_match = tool_match and all(
        _norm(out_args.get(k)) == _norm(v) for k, v in gold_args.items()
    )
    metrics["t3_tool_match"] = 1.0 if tool_match else 0.0
    metrics["t3_args_match"] = 1.0 if args_match else 0.0
    metrics["t3_success"] = 1.0 if tool_match and args_match else 0.0
    return metrics


def _score_t4(out_json: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    """Score T4 BFCL function calling tasks.

    BFCL gold format: {func_name: {param: [possible_values], ...}}
    Output format: {name: func_name, arguments: {param: value, ...}}
    """
    metrics: Dict[str, float] = {}
    if not gold:
        return metrics

    pred_name = out_json.get("name", "")
    pred_args = (
        out_json.get("arguments", {})
        if isinstance(out_json.get("arguments"), dict)
        else {}
    )

    # BFCL gold is {func_name: {param: [possible_values]}}
    # Check if predicted function name matches any gold function
    gold_funcs = list(gold.keys())
    name_match = pred_name in gold_funcs
    metrics["t4_func_match"] = 1.0 if name_match else 0.0

    if name_match and pred_name in gold:
        gold_args = gold[pred_name]
        if gold_args:
            # For each gold param, check if pred value is in possible values
            arg_matches = 0
            for param, possible_vals in gold_args.items():
                pred_val = pred_args.get(param)
                # possible_vals is a list of acceptable values
                if isinstance(possible_vals, list):
                    if any(_norm(pred_val) == _norm(v) for v in possible_vals):
                        arg_matches += 1
                    elif pred_val in possible_vals:
                        arg_matches += 1
                elif _norm(pred_val) == _norm(possible_vals):
                    arg_matches += 1
            metrics["t4_args_acc"] = arg_matches / len(gold_args) if gold_args else 1.0
        else:
            metrics["t4_args_acc"] = 1.0
    else:
        metrics["t4_args_acc"] = 0.0

    metrics["t4_success"] = (
        1.0 if name_match and metrics.get("t4_args_acc", 0) >= 0.99 else 0.0
    )
    return metrics


def _score_t5(out_json: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    """Score T5 SWE-bench tasks (patch generation)."""
    metrics: Dict[str, float] = {}
    if not gold:
        return metrics
    # For SWE-bench, we check if a patch was generated (basic validity)
    # Full evaluation requires running tests, which is out of scope for quick eval
    patch = out_json.get("patch", "")
    metrics["t5_has_patch"] = 1.0 if patch and len(patch.strip()) > 10 else 0.0
    # Check if patch looks like a diff
    is_diff = "diff" in patch.lower() or "@@" in patch or patch.startswith("---")
    metrics["t5_valid_diff"] = 1.0 if is_diff else 0.0
    # Token overlap with gold patch (rough similarity)
    gold_patch = gold.get("patch", "")
    if gold_patch:
        pred_tokens = _token_set(patch)
        gold_tokens = _token_set(gold_patch)
        metrics["t5_patch_f1"] = _f1_sets(pred_tokens, gold_tokens)
    else:
        metrics["t5_patch_f1"] = 0.0
    return metrics


def _extract_number(text: str) -> Optional[float]:
    """Extract a number from text, handling various formats."""
    import re
    if not text:
        return None
    # Clean up text
    text = str(text).strip()
    # Try to find a number pattern (handles negatives, decimals, commas)
    # Look for patterns like: 72, -3.14, 1,234, $50, 50%
    patterns = [
        r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?',  # with commas: 1,234.56
        r'[-+]?\d+\.?\d*',  # simple: 123.45
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.replace(',', ''))
        if matches:
            try:
                return float(matches[-1])  # Take the last number found
            except ValueError:
                continue
    return None


def _score_t6(out_json: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    """Score T6 GSM8K math tasks."""
    metrics: Dict[str, float] = {}
    if not gold:
        return metrics

    gold_answer = gold.get("answer", "")
    pred_answer = out_json.get("answer", "")

    # Extract numeric values for comparison
    gold_num = _extract_number(str(gold_answer))
    pred_num = _extract_number(str(pred_answer))

    # Exact string match (normalized)
    metrics["t6_exact_match"] = 1.0 if _norm(str(pred_answer)) == _norm(str(gold_answer)) else 0.0

    # Numeric match (handles formatting differences)
    if gold_num is not None and pred_num is not None:
        metrics["t6_numeric_match"] = 1.0 if abs(gold_num - pred_num) < 1e-6 else 0.0
    else:
        metrics["t6_numeric_match"] = 0.0

    # Overall success (either exact or numeric match)
    metrics["t6_success"] = max(metrics["t6_exact_match"], metrics["t6_numeric_match"])

    return metrics


def _score_record(
    task: Dict[str, Any], out_json: Dict[str, Any], schema: Dict[str, Any]
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["json_valid"] = 1.0 if _validate_json(out_json, schema) else 0.0
    gold = task.get("gold", {})
    ttype = task.get("task_type", "")
    if ttype == "t1":
        metrics.update(_score_t1(out_json, gold))
    elif ttype == "t2":
        metrics.update(_score_t2(out_json, gold))
    elif ttype == "t3":
        metrics.update(_score_t3(out_json, gold))
    elif ttype == "t4":
        metrics.update(_score_t4(out_json, gold))
    elif ttype == "t5":
        metrics.update(_score_t5(out_json, gold))
    elif ttype == "t6":
        metrics.update(_score_t6(out_json, gold))
    return metrics


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, List[float]] = {}
    for r in results:
        for k, v in r["metrics"].items():
            agg.setdefault(k, []).append(float(v))
    summary: Dict[str, Any] = {}
    for k, vals in agg.items():
        if not vals:
            continue
        summary[k] = round(statistics.mean(vals), 4)
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    ttfts = [r["ttft_ms"] for r in results if r.get("ttft_ms") is not None]
    summary["avg_latency_ms"] = (
        round(statistics.mean(latencies), 2) if latencies else 0.0
    )
    summary["avg_ttft_ms"] = round(statistics.mean(ttfts), 2) if ttfts else 0.0
    summary["count"] = len(results)
    return summary


def _parse_model_spec(spec: str) -> Dict[str, str]:
    if ":" not in spec:
        raise ValueError(f"model spec must be provider:model, got {spec}")
    provider, model = spec.split(":", 1)
    slug = f"{provider}_{model.replace('/', '-').replace(':', '-')}"
    return {"provider": provider, "model": model, "slug": slug}


def _set_provider_env(provider: str, model: str):
    os.environ["AOFW_PROVIDER"] = provider
    if provider == "lmstudio":
        os.environ["LMSTUDIO_MODEL"] = model
    elif provider == "ollama":
        os.environ["OLLAMA_MODEL"] = model
    elif provider == "vllm":
        os.environ["VLLM_MODEL"] = model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "tasks/clinc_en.jsonl",
            "tasks/hotpot_dev.jsonl",
            "tasks/t3_tools.jsonl",
        ],
    )
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of provider:model specs (e.g., lmstudio:qwen/qwen3-4b-thinking-2507).",
    )
    ap.add_argument(
        "--mode",
        default="structured",
        help="Decode mode: structured|text|grammar (passed to provider_generate).",
    )
    ap.add_argument(
        "--max-records", type=int, default=0, help="Optional cap on total records."
    )
    ap.add_argument(
        "--out-dir", default="out/evals", help="Directory to write per-model results."
    )
    ap.add_argument(
        "--run-name", default=None, help="Optional run name; defaults to timestamp."
    )
    ap.add_argument(
        "--capture-detailed",
        action="store_true",
        help="Capture detailed field breakdown, error taxonomy, and gold comparison.",
    )
    ap.add_argument(
        "--capture-logprobs",
        action="store_true",
        help="Request and capture logprobs from the model (if supported).",
    )
    ap.add_argument(
        "--stability-runs",
        type=int,
        default=1,
        help="Number of runs per prompt for stability analysis.",
    )
    ap.add_argument(
        "--slo-budget-ms",
        type=float,
        default=2000.0,
        help="SLO budget in milliseconds for error classification.",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers for concurrent evaluation (default: 1 = sequential).",
    )
    args = ap.parse_args()

    tasks = _load_tasks(args.tasks, args.max_records or None)
    if not tasks:
        raise SystemExit("no tasks loaded; check --tasks paths")

    run_name = args.run_name or f"t_eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_root = Path(args.out_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    schema_cache: Dict[str, Dict[str, Any]] = {}
    task_fps = {Path(p).name: fingerprint_tasks(p).as_dict() for p in args.tasks}

    # Build capture config
    capture_config = CaptureConfig(
        capture_field_breakdown=args.capture_detailed,
        capture_error_taxonomy=args.capture_detailed,
        capture_raw_output=args.capture_detailed,
        capture_gold_comparison=args.capture_detailed,
        capture_latency_decomposition=args.capture_detailed,
        capture_logprobs=args.capture_logprobs,
        stability_runs=args.stability_runs,
    )

    def _eval_single_task(
        rec: Dict[str, Any],
        schema: Dict[str, Any],
        run_idx: int,
        model_info: Dict[str, str],
        mode: str,
        capture_logprobs: bool,
        capture_detailed: bool,
        slo_budget_ms: float,
        capture_config: CaptureConfig,
    ) -> Dict[str, Any]:
        """Evaluate a single task - designed for parallel execution."""
        error_msg = None
        raw_text = ""
        logprobs_data = None
        parse_error = None
        schema_error = None

        try:
            (
                raw_text,
                out_json,
                lat_ms,
                ttft_ms,
                tokens_in,
                tokens_out,
                logprobs_data,
            ) = provider_generate_raw(
                rec["prompt"],
                schema,
                mode=mode,
                request_logprobs=capture_logprobs,
            )
            metrics = _score_record(rec, out_json, schema)
        except json.JSONDecodeError as exc:
            out_json = {}
            lat_ms, ttft_ms, tokens_in, tokens_out = 1e6, 1e6, -1, -1
            metrics = {"json_valid": 0.0}
            error_msg = str(exc)
            parse_error = str(exc)
        except Exception as exc:
            out_json = {}
            lat_ms, ttft_ms, tokens_in, tokens_out = 1e6, 1e6, -1, -1
            metrics = {"json_valid": 0.0}
            error_msg = str(exc)

        prompt_hash = hashlib.sha256(
            rec.get("prompt", "").encode("utf-8")
        ).hexdigest()

        result = {
            "id": rec.get("id"),
            "task_type": rec.get("task_type"),
            "prompt": rec.get("prompt"),
            "prompt_hash": prompt_hash,
            "output_json": out_json,
            "metrics": metrics,
            "latency_ms": float(lat_ms),
            "ttft_ms": float(ttft_ms),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "schema_path": rec.get("schema_path"),
            "error": error_msg,
            "run_idx": run_idx,
        }

        # Add gold for reference
        gold = rec.get("gold")
        if gold is not None:
            result["gold"] = gold

        # Add detailed capture if enabled
        if capture_detailed or capture_logprobs:
            detailed = build_detailed_result(
                task_id=rec.get("id", ""),
                task_type=rec.get("task_type", ""),
                prompt=rec.get("prompt", ""),
                raw_output=raw_text,
                parsed_output=out_json if out_json else None,
                gold=gold,
                schema=schema,
                parse_error=parse_error,
                schema_error=schema_error,
                latency_ms=lat_ms,
                slo_budget_ms=slo_budget_ms,
                model=model_info["model"],
                provider=model_info["provider"],
                logprobs_data=logprobs_data,
                config=capture_config,
            )
            result["detailed"] = detailed

        return result

    for spec in args.models:
        model_info = _parse_model_spec(spec)
        _set_provider_env(model_info["provider"], model_info["model"])
        model_dir = out_root / model_info["slug"]
        model_dir.mkdir(parents=True, exist_ok=True)

        # Pre-load schemas
        for rec in tasks:
            schema_path = rec.get("schema_path")
            if not schema_path:
                raise ValueError(f"task {rec.get('id')} missing schema_path")
            if schema_path not in schema_cache:
                schema_file = Path(schema_path)
                if not schema_file.exists():
                    schema_file = Path(__file__).resolve().parents[1] / schema_path
                schema_cache[schema_path] = json.load(
                    open(schema_file, "r", encoding="utf-8")
                )

        # Build work items
        work_items = []
        for rec in tasks:
            schema = schema_cache[rec.get("schema_path")]
            for run_idx in range(args.stability_runs):
                work_items.append((rec, schema, run_idx))

        results = []
        if args.parallel > 1:
            # Parallel execution
            print(f"[{model_info['slug']}] Running {len(work_items)} tasks with {args.parallel} workers...")
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {
                    executor.submit(
                        _eval_single_task,
                        rec,
                        schema,
                        run_idx,
                        model_info,
                        args.mode,
                        args.capture_logprobs,
                        args.capture_detailed,
                        args.slo_budget_ms,
                        capture_config,
                    ): (rec, run_idx)
                    for rec, schema, run_idx in work_items
                }
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if completed % 100 == 0:
                        print(f"[{model_info['slug']}] Progress: {completed}/{len(work_items)}")
        else:
            # Sequential execution
            for i, (rec, schema, run_idx) in enumerate(work_items):
                result = _eval_single_task(
                    rec,
                    schema,
                    run_idx,
                    model_info,
                    args.mode,
                    args.capture_logprobs,
                    args.capture_detailed,
                    args.slo_budget_ms,
                    capture_config,
                )
                results.append(result)
                if (i + 1) % 100 == 0:
                    print(f"[{model_info['slug']}] Progress: {i+1}/{len(work_items)}")
        preds_path = model_dir / "predictions.jsonl"
        with preds_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = _aggregate(results)
        summary.update(
            {
                "model": model_info["model"],
                "provider": model_info["provider"],
                "decode_mode": args.mode,
                "tasks": [Path(p).name for p in args.tasks],
                "task_fingerprints": task_fps,
                "stability_runs": args.stability_runs,
            }
        )

        # Add stability metrics if multiple runs
        if args.stability_runs > 1:
            stability_metrics = aggregate_stability_metrics(results, "prompt_hash")
            summary["stability"] = stability_metrics
        with (model_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[{model_info['slug']}] {summary}")


if __name__ == "__main__":
    main()
