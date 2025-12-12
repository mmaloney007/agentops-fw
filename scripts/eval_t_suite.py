#!/usr/bin/env python3
"""
Evaluate the T1/T2/T3 task suites across multiple models/backends.

Usage:
  python scripts/eval_t_suite.py --models lmstudio:qwen/qwen3-4b-thinking-2507 \
    --tasks tasks/clinc_en.jsonl tasks/hotpot_dev.jsonl tasks/t3_tools.jsonl

Set AOFW_PROVIDER/LMSTUDIO_MODEL/OLLAMA_MODEL/VLLM_MODEL env vars as needed.
"""
from __future__ import annotations

import argparse, json, os, statistics, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator, ValidationError

from agent_stable_slo.rollout.engine import provider_generate
from agent_stable_slo.utils.data import fingerprint_tasks


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
        matches.append(1.0 if _norm(out_json.get(f, "")) == _norm(gold.get(f, "")) else 0.0)
    metrics["t1_field_acc"] = sum(matches) / len(matches)
    raw_tags = out_json.get("tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []
    pred_tags = set(_norm(t) for t in raw_tags)
    gold_tags = set(_norm(t) for t in gold.get("tags", []))
    metrics["t1_tag_f1"] = _f1_sets(pred_tags, gold_tags)
    metrics["t1_exact_match"] = 1.0 if metrics["t1_field_acc"] == 1.0 and metrics["t1_tag_f1"] == 1.0 else 0.0
    return metrics


def _score_t2(out_json: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not gold:
        return metrics
    metrics["t2_summary_f1"] = _f1_sets(_token_set(out_json.get("short_summary", "")), _token_set(gold.get("short_summary", "")))
    pred_points = []
    if isinstance(out_json.get("key_points"), list):
        pred_points = [_norm(p) for p in out_json["key_points"]]
    gold_points = [_norm(p) for p in gold.get("key_points", [])]
    metrics["t2_key_point_f1"] = _f1_sets(set(" ".join(pred_points).split()), set(" ".join(gold_points).split()))
    metrics["t2_primary_risk_match"] = 1.0 if _norm(out_json.get("primary_risk", "")) == _norm(gold.get("primary_risk", "")) else 0.0
    metrics["t2_action_match"] = 1.0 if _norm(out_json.get("recommended_action", "")) == _norm(gold.get("recommended_action", "")) else 0.0
    return metrics


def _score_t3(out_json: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not gold:
        return metrics
    tool_match = _norm(out_json.get("tool")) == _norm(gold.get("tool"))
    gold_args = gold.get("arguments", {}) if isinstance(gold.get("arguments"), dict) else {}
    out_args = out_json.get("arguments", {}) if isinstance(out_json.get("arguments"), dict) else {}
    args_match = tool_match and all(_norm(out_args.get(k)) == _norm(v) for k, v in gold_args.items())
    metrics["t3_tool_match"] = 1.0 if tool_match else 0.0
    metrics["t3_args_match"] = 1.0 if args_match else 0.0
    metrics["t3_success"] = 1.0 if tool_match and args_match else 0.0
    return metrics


def _score_record(task: Dict[str, Any], out_json: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, float]:
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
    summary["avg_latency_ms"] = round(statistics.mean(latencies), 2) if latencies else 0.0
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
        default=["tasks/clinc_en.jsonl", "tasks/hotpot_dev.jsonl", "tasks/t3_tools.jsonl"],
    )
    ap.add_argument("--models", nargs="+", required=True, help="List of provider:model specs (e.g., lmstudio:qwen/qwen3-4b-thinking-2507).")
    ap.add_argument("--mode", default="structured", help="Decode mode: structured|text|grammar (passed to provider_generate).")
    ap.add_argument("--max-records", type=int, default=0, help="Optional cap on total records.")
    ap.add_argument("--out-dir", default="out/evals", help="Directory to write per-model results.")
    ap.add_argument("--run-name", default=None, help="Optional run name; defaults to timestamp.")
    args = ap.parse_args()

    tasks = _load_tasks(args.tasks, args.max_records or None)
    if not tasks:
        raise SystemExit("no tasks loaded; check --tasks paths")

    run_name = args.run_name or f"t_eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_root = Path(args.out_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    schema_cache: Dict[str, Dict[str, Any]] = {}
    task_fps = {Path(p).name: fingerprint_tasks(p).as_dict() for p in args.tasks}

    for spec in args.models:
        model_info = _parse_model_spec(spec)
        _set_provider_env(model_info["provider"], model_info["model"])
        model_dir = out_root / model_info["slug"]
        model_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for rec in tasks:
            schema_path = rec.get("schema_path")
            if not schema_path:
                raise ValueError(f"task {rec.get('id')} missing schema_path")
            if schema_path not in schema_cache:
                schema_file = Path(schema_path)
                if not schema_file.exists():
                    schema_file = Path(__file__).resolve().parents[1] / schema_path
                schema_cache[schema_path] = json.load(open(schema_file, "r", encoding="utf-8"))
            schema = schema_cache[schema_path]
            error_msg = None
            try:
                out_json, lat_ms, ttft_ms, tokens = provider_generate(rec["prompt"], schema, mode=args.mode)
                metrics = _score_record(rec, out_json, schema)
            except Exception as exc:
                out_json, lat_ms, ttft_ms, tokens = {}, 1e6, 1e6, -1
                metrics = {"json_valid": 0.0}
                error_msg = str(exc)
            results.append(
                {
                    "id": rec.get("id"),
                    "task_type": rec.get("task_type"),
                    "prompt": rec.get("prompt"),
                    "output_json": out_json,
                    "metrics": metrics,
                    "latency_ms": float(lat_ms),
                    "ttft_ms": float(ttft_ms),
                    "tokens_out": tokens,
                    "schema_path": schema_path,
                    "error": error_msg,
                }
            )
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
            }
        )
        with (model_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[{model_info['slug']}] {summary}")


if __name__ == "__main__":
    main()
