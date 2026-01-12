from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from jsonschema import Draft202012Validator

from agent_stable_slo.config.criteria_v1 import CriteriaSpec, load_criteria
from agent_stable_slo.eval.faithfulness_judge import FaithfulnessResult, score_faithfulness
from agent_stable_slo.logging import wandb_utils as WL
from agent_stable_slo.rollout.engine import DecodingMode, generate_with_mode
from agent_stable_slo.utils.data import fingerprint_tasks
from agent_stable_slo.utils.repro import env_snapshot


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _normalize_text(s: str) -> str:
    import re
    import string

    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def _exact_match(gold: str, pred: str) -> int:
    return int(_normalize_text(gold) == _normalize_text(pred))


def _f1_overlap(gold: str, pred: str) -> float:
    from collections import Counter

    g = _normalize_text(gold).split()
    p = _normalize_text(pred).split()
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    common = Counter(g) & Counter(p)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def _macro_f1(y_true: List[str], y_pred: List[str]) -> Optional[float]:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return None
    f1s: List[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom else 0.0
        f1s.append(f1)
    if not f1s:
        return None
    return float(sum(f1s) / len(f1s))


def _validate_schema(obj: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    v = Draft202012Validator(schema)
    errors = sorted(v.iter_errors(obj), key=lambda e: e.path)
    if not errors:
        return None
    err = errors[0]
    path = ".".join([str(p) for p in err.path]) if err.path else "<root>"
    return f"{path}: {err.message}"


def _canonical_json(obj: Optional[Dict[str, Any]]) -> str:
    if not obj:
        return ""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _evaluate_tiers(summary: Dict[str, Any], crit) -> Dict[str, bool]:
    """Check which tiers (Bronze/Silver/Gold) pass based on summary metrics."""
    tiers = {"bronze": False, "silver": False, "gold": False}

    # Bronze gates (structure + basic SLO)
    bronze_pass = (
        summary.get("json_valid_rate", 0.0) >= crit.structure.json_valid_min and
        summary.get("schema_valid_rate", 0.0) >= crit.structure.schema_valid_min and
        summary.get("p95_latency_ms", float('inf')) <= crit.slo.p95_ms_max
    )
    tiers["bronze"] = bronze_pass

    if not bronze_pass:
        return tiers  # Silver/Gold require Bronze

    # Silver gates (extends Bronze: adds semantic correctness + stability)
    disagreement = summary.get("stability", {}).get("disagreement_at_k", 1.0) if isinstance(summary.get("stability"), dict) else 1.0
    silver_pass = (
        summary.get("clinc_intent_macro_f1", 0.0) >= crit.accuracy.clinc_intent_macro_f1_min and
        summary.get("hotpot_answer_f1_mean", 0.0) >= crit.accuracy.hotpot_answer_f1_min and
        summary.get("hotpot_faithfulness_mean", 0.0) >= crit.faithfulness.hotpot_faithfulness_min and
        disagreement <= crit.stability.disagreement_max
    )
    tiers["silver"] = silver_pass

    if not silver_pass:
        return tiers  # Gold requires Silver

    # Gold gates (extends Silver: adds tool success + Success@SLO)
    tool_success = summary.get("tool_success_rate", 0.0)  # Will be 0.0 if T3 not present
    gold_pass = (
        tool_success >= crit.tools.tool_success_rate_min and
        summary.get("success_at_slo", 0.0) >= crit.slo.success_at_slo_min
    )
    tiers["gold"] = gold_pass

    return tiers


def _load_clinc_label_set() -> Optional[set[str]]:
    labels_path = Path("tasks/clinc150_labels.json")
    if not labels_path.exists():
        return None
    try:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        return set(str(x) for x in labels)
    except Exception:
        return None


def _clinc_label_match_rate(task_file: str, label_set: Optional[set[str]]) -> Optional[float]:
    total = 0
    match = 0
    for rec in _read_jsonl(task_file):
        gold = rec.get("gold") or {}
        intent = gold.get("intent")
        if intent is None:
            continue
        total += 1
        if isinstance(intent, str) and (label_set is None or intent in label_set):
            match += 1
    if total == 0:
        return None
    return match / total


def _parse_hotpot_prompt(prompt: str) -> Tuple[str, str]:
    if "Question:" not in prompt:
        return "", prompt
    parts = prompt.split("Question:", 1)
    context = parts[0]
    question = parts[1].strip()
    if "Context:" in context:
        context = context.split("Context:", 1)[1].strip()
    return question, context.strip()


def _tool_success(pred: Dict[str, Any], gold: Dict[str, Any]) -> int:
    if not pred or not gold:
        return 0
    tool_match = str(pred.get("tool", "")).strip().lower() == str(gold.get("tool", "")).strip().lower()
    if not tool_match:
        return 0
    gold_args = gold.get("arguments", {}) if isinstance(gold.get("arguments"), dict) else {}
    out_args = pred.get("arguments", {}) if isinstance(pred.get("arguments"), dict) else {}
    for k, v in gold_args.items():
        if str(out_args.get(k, "")).strip().lower() != str(v).strip().lower():
            return 0
    return 1


def _enforce_wandb_online() -> None:
    mode = os.getenv("WANDB_MODE", "").strip().lower()
    # Temporarily allow offline mode for faster data collection
    # TODO: Re-enable online requirement before final paper submission
    if mode == "offline":
        print("[P1] WARNING: Running in offline mode - W&B artifacts won't be uploaded immediately")
        return
    if mode and mode != "online":
        raise RuntimeError(f"WANDB_MODE must be 'online' for P1 runs. Got {mode!r}")
    os.environ["WANDB_MODE"] = "online"


def _should_use_judge(args, crit: CriteriaSpec) -> bool:
    if args.disable_judge:
        return False
    return bool(crit.judge.enabled)


def _load_schema(path: str) -> Dict[str, Any]:
    return _read_json(path)


def _maybe_set_env(provider: str, endpoint: str, model: str) -> None:
    os.environ["AOFW_PROVIDER"] = provider
    if endpoint:
        os.environ["OPENAI_API_BASE"] = endpoint
        os.environ["VLLM_API_BASE"] = endpoint
    if provider == "lmstudio":
        os.environ["LMSTUDIO_MODEL"] = model
    elif provider == "vllm":
        os.environ["VLLM_MODEL"] = model
    elif provider == "ollama":
        os.environ["OLLAMA_MODEL"] = model


def run_eval(args: argparse.Namespace) -> None:
    crit = load_criteria(args.criteria)
    mode = DecodingMode.from_str(args.mode)
    suite = crit.suites.get(args.suite)
    if not suite:
        raise SystemExit(f"suite not found: {args.suite}")
    if mode == DecodingMode.SPEC_DRIVEN_PLUS_SELFCONSISTENCY and args.self_consistency_samples < 2:
        raise SystemExit("self-consistency mode requires --self-consistency-samples >= 2")
    if args.repair_max_attempts < 0:
        raise SystemExit("--repair-max-attempts must be >= 0")

    clinc_label_set = _load_clinc_label_set()
    clinc_label_type_match_rate: Optional[float] = None
    for task in suite.tasks:
        if task.id == "t1_clinc":
            clinc_label_type_match_rate = _clinc_label_match_rate(task.task_file, clinc_label_set)
            if clinc_label_type_match_rate is not None and clinc_label_type_match_rate < 1.0:
                raise RuntimeError(
                    f"CLINC label type mismatch: match_rate={clinc_label_type_match_rate:.3f}. "
                    "Fix tasks/clinc_en.jsonl and tasks/schemas/clinc_nlu_schema.json before running."
                )

    _maybe_set_env(args.provider, args.endpoint, args.model)
    if "MAX_THOUGHT_TOKENS" not in os.environ:
        os.environ["MAX_THOUGHT_TOKENS"] = str(crit.slo.max_tokens_out)

    judge_enabled = _should_use_judge(args, crit)
    judge_base_url = args.judge_base_url or crit.judge.base_url
    judge_model = args.judge_model or crit.judge.model
    judge_temp = float(args.judge_temperature if args.judge_temperature is not None else crit.judge.temperature)
    judge_top_p = float(crit.judge.top_p)
    judge_max_tokens = int(crit.judge.max_tokens_out)

    wandb_project = args.wandb_project or crit.reporting.wandb.project
    wandb_entity = args.wandb_entity or crit.reporting.wandb.entity
    wandb_group = args.wandb_group or crit.reporting.wandb.group
    wandb_tags = args.wandb_tags or ",".join(crit.reporting.wandb.tags)

    if not wandb_project:
        raise SystemExit("W&B project is required for P1 runs; set --wandb-project or criteria.reporting.wandb.project.")

    _enforce_wandb_online()

    criteria_hash = crit.criteria_hash()
    out_dir = os.path.join(args.out_dir, crit.criteria_id, args.model.replace("/", "_"), mode.value)
    os.makedirs(out_dir, exist_ok=True)
    episodes_path = os.path.join(out_dir, "episodes.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")
    judge_traces_path = os.path.join(out_dir, "judge_traces.jsonl")
    env_path = os.path.join(out_dir, "env.json")

    env_snapshot_data = env_snapshot({"criteria_id": crit.criteria_id, "criteria_hash": criteria_hash})
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env_snapshot_data, f, indent=2)

    wb_run = WL.init_run(
        name=f"p1-{crit.criteria_id}-{args.model.replace('/', '_')}-{mode.value}",
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        tags=[t for t in wandb_tags.split(",") if t],
        config={
            "criteria_id": crit.criteria_id,
            "criteria_hash": criteria_hash,
            "suite": args.suite,
            "provider": args.provider,
            "base_url": args.endpoint,
            "model": args.model,
            "decode_mode": mode.value,
            "temperature": float(args.temperature),
            "max_retries": int(args.max_retries),
            "repair_max_attempts": int(args.repair_max_attempts),
            "self_consistency_samples": int(args.self_consistency_samples),
            "self_consistency_max_ms": int(args.self_consistency_max_ms),
            "git_rev": env_snapshot_data.get("git_rev"),
        },
        require_online=True,
    )
    import sys
    print("[P1] DEBUG: Creating episode table...", flush=True)
    sys.stderr.write("[P1] DEBUG: Creating episode table (stderr)...\n")
    sys.stderr.flush()
    wb_table = WL.create_episode_table() if wb_run else None
    print("[P1] DEBUG: Episode table created", flush=True)

    episodes: List[Dict[str, Any]] = []
    latencies: List[float] = []
    clinc_gold: List[str] = []
    clinc_pred: List[str] = []
    hotpot_em: List[int] = []
    hotpot_f1s: List[float] = []
    faith_scores: List[float] = []
    contra_rates: List[float] = []
    stability_groups: Dict[str, List[Dict[str, Any]]] = {}

    judge_trace_records: List[Dict[str, Any]] = []
    judge_trace_limit = 50

    alias = [crit.reporting.wandb.artifact_alias] if crit.reporting.wandb.artifact_alias else None
    print("[P1] DEBUG: Logging artifacts...")
    WL.log_artifact(wb_run, args.criteria, f"criteria-{crit.criteria_id}", type_="criteria", aliases=alias)

    task_fps = []
    for task in suite.tasks:
        print(f"[P1] DEBUG: Fingerprinting {task.task_file}...")
        task_fp = fingerprint_tasks(task.task_file)
        task_fps.append(task_fp.as_dict())
        WL.log_artifact(wb_run, task.task_file, f"tasks-{task.id}", type_="tasks", aliases=alias)
        WL.log_artifact(wb_run, task.output_schema, f"schema-{task.id}", type_="schema", aliases=alias)
    print("[P1] DEBUG: All artifacts logged")

    # Warmup requests to avoid cold-start latency contamination
    warmup_count = crit.reproducibility.warmup_requests
    if warmup_count > 0:
        print(f"[P1] Running {warmup_count} warmup requests...")
        warmup_schema = {"type": "object", "properties": {"warmup": {"type": "boolean"}}, "required": ["warmup"]}
        for _ in range(warmup_count):
            try:
                _ = generate_with_mode(
                    prompt="Warmup request",
                    schema=warmup_schema,
                    mode=mode,
                    temperature=0.0,
                    endpoint=args.endpoint,
                    model=args.model,
                    max_tokens=10,
                    max_retries=0,
                    repair_max_attempts=0,
                    self_consistency_samples=1,
                    self_consistency_max_ms=1000,
                    self_consistency_selection="first",
                )
            except Exception:
                pass  # Ignore warmup failures
        print("[P1] Warmup complete, starting evaluation...")

    total_examples = sum(min(args.max_examples or 999999, sum(1 for _ in _read_jsonl(t.task_file))) for t in suite.tasks)
    processed = 0
    for task in suite.tasks:
        schema = _load_schema(task.output_schema)
        for idx, rec in enumerate(_read_jsonl(task.task_file), start=1):
            if args.max_examples and idx > args.max_examples:
                break
            processed += 1
            print(f"[P1] Progress: {processed}/{total_examples} ({task.id} #{idx})", flush=True)
            prompt = rec.get("prompt", "")
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            task_instance_id = rec.get("id") or f"{task.id}:{idx}"
            repeats = args.stability_k or crit.stability.runs_per_prompt
            for rep in range(repeats):
                gen = generate_with_mode(
                    prompt=prompt,
                    schema=schema,
                    mode=mode,
                    temperature=float(args.temperature),
                    max_tokens=crit.slo.max_tokens_out,
                    max_retries=int(args.max_retries),
                    repair_max_attempts=int(args.repair_max_attempts),
                    self_consistency_samples=int(args.self_consistency_samples),
                    self_consistency_max_ms=int(args.self_consistency_max_ms),
                    self_consistency_selection=str(args.self_consistency_selection),
                )

                final = gen.final
                json_ok = int(final.parsed_json is not None and final.parse_error is None)
                schema_err = final.schema_error
                if final.parsed_json is not None and schema_err is None:
                    schema_err = _validate_schema(final.parsed_json, schema)
                schema_ok = int(json_ok == 1 and schema_err is None)

                metrics: Dict[str, Any] = {
                    "json_valid": float(json_ok),
                    "schema_valid": float(schema_ok),
                }

                if task.id == "t1_clinc":
                    gold_intent = rec.get("gold", {}).get("intent")
                    pred_intent = final.parsed_json.get("intent") if final.parsed_json else None
                    if gold_intent is not None and pred_intent is not None:
                        clinc_gold.append(str(gold_intent))
                        clinc_pred.append(str(pred_intent))
                    metrics["clinc_intent_accuracy"] = float(
                        int(gold_intent is not None and pred_intent is not None and str(gold_intent) == str(pred_intent))
                    )

                if task.id == "t2_hotpot":
                    gold_answer = rec.get("gold", {}).get("answer", "")
                    pred_answer = str(final.parsed_json.get("answer", "")) if final.parsed_json else ""
                    em = _exact_match(str(gold_answer), pred_answer)
                    f1 = _f1_overlap(str(gold_answer), pred_answer)
                    metrics["hotpot_answer_exact_match"] = float(em)
                    metrics["hotpot_answer_f1"] = float(f1)
                    hotpot_em.append(em)
                    hotpot_f1s.append(float(f1))

                    if judge_enabled and final.parsed_json:
                        question, context = _parse_hotpot_prompt(prompt)
                        faith: FaithfulnessResult = score_faithfulness(
                            base_url=judge_base_url,
                            api_key=os.getenv("OPENAI_API_KEY", ""),
                            model=judge_model,
                            question=question,
                            context=context,
                            candidate_json=final.parsed_json,
                            temperature=judge_temp,
                            top_p=judge_top_p,
                            max_tokens_out=judge_max_tokens,
                        )
                        metrics["hotpot_faithfulness"] = float(faith.faithfulness)
                        metrics["hotpot_contradiction_rate"] = float(faith.contradiction_rate)
                        faith_scores.append(float(faith.faithfulness))
                        contra_rates.append(float(faith.contradiction_rate))
                        if len(judge_trace_records) < judge_trace_limit:
                            judge_trace_records.append(
                                {
                                    "task_id": task.id,
                                    "task_instance_id": task_instance_id,
                                    "question": question,
                                    "context": context,
                                    "candidate_json": final.parsed_json,
                                    "judge_result": asdict(faith),
                                }
                            )
                    else:
                        metrics["hotpot_faithfulness"] = 0.0
                        metrics["hotpot_contradiction_rate"] = 1.0

                if task.id == "t3_tools":
                    gold = rec.get("gold", {})
                    metrics["tool_success_rate"] = float(_tool_success(final.parsed_json or {}, gold))

                # Episode-level Success@SLO (quality gates + on-time)
                quality_pass = True
                if metrics["json_valid"] < crit.structure.json_valid_min:
                    quality_pass = False
                if metrics["schema_valid"] < crit.structure.schema_valid_min:
                    quality_pass = False
                if task.id == "t1_clinc":
                    if metrics.get("clinc_intent_accuracy", 0.0) < crit.accuracy.clinc_intent_accuracy_min:
                        quality_pass = False
                if task.id == "t2_hotpot":
                    if metrics.get("hotpot_answer_f1", 0.0) < crit.accuracy.hotpot_answer_f1_min:
                        quality_pass = False
                    if metrics.get("hotpot_faithfulness", 0.0) < crit.faithfulness.hotpot_faithfulness_min:
                        quality_pass = False
                    if metrics.get("hotpot_contradiction_rate", 1.0) > crit.faithfulness.hotpot_contradiction_rate_max:
                        quality_pass = False
                if task.id == "t3_tools":
                    if metrics.get("tool_success_rate", 0.0) < crit.tools.tool_success_rate_min:
                        quality_pass = False

                on_time = float(gen.total_latency_ms) <= float(crit.slo.on_time_budget_ms)
                metrics["success_at_slo"] = float(int(quality_pass and on_time))

                episode = {
                    "task_id": task.id,
                    "task_instance_id": task_instance_id,
                    "replicate_id": rep,
                    "model": args.model,
                    "base_url": args.endpoint,
                    "decode_mode": mode.value,
                    "prompt_hash": prompt_hash,
                    "temperature": float(args.temperature),
                    "latency_ms": float(gen.total_latency_ms),
                    "provider_latency_ms": float(final.latency_ms),
                    "request_start": float(gen.request_start),
                    "request_end": float(gen.request_end),
                    "tokens_in": int(gen.total_tokens_in),
                    "tokens_out": int(gen.total_tokens_out),
                    "retry_count": int(gen.retry_count),
                    "repair_count": int(gen.repair_count),
                    "candidate_count": int(gen.candidate_count),
                    "raw_output": final.raw_text,
                    "parsed_output": final.parsed_json,
                    "parse_error": final.parse_error,
                    "schema_error": schema_err,
                    "attempts": [asdict(a) for a in gen.attempts],
                    "metrics": metrics,
                    "gold": rec.get("gold"),
                    "stability_features": {"canonical_json": _canonical_json(final.parsed_json)},
                }

                episodes.append(episode)
                latencies.append(float(gen.total_latency_ms))
                stability_groups.setdefault(task_instance_id, []).append(episode)

                WL.add_episode_row(wb_table, episode)

    # Stability summary
    disagreements: List[float] = []
    total_agreements: List[float] = []
    for _gid, eps in stability_groups.items():
        if len(eps) < (args.stability_k or crit.stability.runs_per_prompt):
            continue
        keys = [e["stability_features"]["canonical_json"] for e in eps]
        if not keys:
            continue
        mode_key = max(set(keys), key=keys.count)
        disagreements.append(1.0 - keys.count(mode_key) / len(keys))
        total_agreements.append(keys.count(mode_key) / len(keys))

    stability_summary = {
        "runs_per_prompt": int(args.stability_k or crit.stability.runs_per_prompt),
        "disagreement_at_k": float(np.mean(disagreements)) if disagreements else None,
        "total_agreement_rate_at_k": float(np.mean(total_agreements)) if total_agreements else None,
        "equivalence": crit.stability.equivalence,
    }

    summary: Dict[str, Any] = {
        "criteria_id": crit.criteria_id,
        "criteria_hash": criteria_hash,
        "suite": args.suite,
        "model": args.model,
        "base_url": args.endpoint,
        "decode_mode": mode.value,
        "num_episodes": len(episodes),
        "json_valid_rate": float(np.mean([e["metrics"]["json_valid"] for e in episodes])) if episodes else 0.0,
        "schema_valid_rate": float(np.mean([e["metrics"]["schema_valid"] for e in episodes])) if episodes else 0.0,
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else None,
        "success_at_slo": float(np.mean([e["metrics"]["success_at_slo"] for e in episodes])) if episodes else 0.0,
        "stability": stability_summary,
        "task_fingerprints": task_fps,
        "clinc_label_type_match_rate": clinc_label_type_match_rate,
        "avg_retry_count": float(np.mean([e["retry_count"] for e in episodes])) if episodes else 0.0,
        "avg_repair_count": float(np.mean([e["repair_count"] for e in episodes])) if episodes else 0.0,
        "avg_candidate_count": float(np.mean([e["candidate_count"] for e in episodes])) if episodes else 0.0,
        "retry_rate": float(np.mean([1.0 if e["retry_count"] > 0 else 0.0 for e in episodes])) if episodes else 0.0,
        "repair_rate": float(np.mean([1.0 if e["repair_count"] > 0 else 0.0 for e in episodes])) if episodes else 0.0,
        "candidate_rate": float(np.mean([1.0 if e["candidate_count"] > 1 else 0.0 for e in episodes])) if episodes else 0.0,
        "latency_boundary": {
            "client_observed_policy_call": True,
            "includes_retries": True,
            "includes_judge_latency": False,
            "concurrency": "single_client",
        },
    }

    macro_f1 = _macro_f1(clinc_gold, clinc_pred)
    summary["clinc_intent_macro_f1"] = float(macro_f1) if macro_f1 is not None else None
    summary["clinc_intent_accuracy"] = (
        float(np.mean([e["metrics"].get("clinc_intent_accuracy", 0.0) for e in episodes if e["task_id"] == "t1_clinc"]))
        if episodes
        else None
    )
    summary["hotpot_answer_exact_match_rate"] = float(np.mean(hotpot_em)) if hotpot_em else None
    summary["hotpot_answer_f1_mean"] = float(np.mean(hotpot_f1s)) if hotpot_f1s else None
    summary["hotpot_faithfulness_mean"] = float(np.mean(faith_scores)) if faith_scores else None
    summary["hotpot_contradiction_rate_mean"] = float(np.mean(contra_rates)) if contra_rates else None

    # Tier evaluation
    tier_results = _evaluate_tiers(summary, crit)
    summary["tier_bronze"] = tier_results["bronze"]
    summary["tier_silver"] = tier_results["silver"]
    summary["tier_gold"] = tier_results["gold"]

    with open(episodes_path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=True) + "\n")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    if judge_trace_records:
        with open(judge_traces_path, "w", encoding="utf-8") as f:
            for rec in judge_trace_records:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    WL.log_artifact(wb_run, episodes_path, f"episodes-{crit.criteria_id}-{mode.value}", type_="episodes", aliases=alias)
    WL.log_artifact(wb_run, summary_path, f"summary-{crit.criteria_id}-{mode.value}", type_="summary", aliases=alias)
    WL.log_artifact(wb_run, env_path, f"env-{crit.criteria_id}-{mode.value}", type_="metadata", aliases=alias)
    if judge_trace_records:
        WL.log_artifact(
            wb_run, judge_traces_path, f"judge-traces-{crit.criteria_id}-{mode.value}", type_="judge_traces", aliases=alias
        )

    WL.log_table(wb_run, wb_table, "episodes")
    WL.finish_run(wb_run)

    print(f"[P1] wrote: {episodes_path}")
    print(f"[P1] wrote: {summary_path}")
