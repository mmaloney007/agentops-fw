#!/usr/bin/env python3
"""
GRPO-style LoRA trainer with a minimal policy-gradient loop.
Features:
 - YAML/CLI config with validation (pydantic) and presets under configs/grpo.
 - Dataset fingerprinting + optional caching to make runs reproducible.
 - Structured logging (train_log.jsonl, run_log.jsonl) and manifest snapshot.
 - Checkpoint save/resume for adapter + optimizer/scaler/RNG state.
 - Optional blocklist/repro/deterministic guards.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig  # type: ignore

from agent_stable_slo.logging import wandb_utils as WL
from agent_stable_slo.logging.structured import JsonLogger
from agent_stable_slo.rewards.composite import composite_reward
from agent_stable_slo.rewards.schema_reward import schema_valid
from agent_stable_slo.utils.checkpoint import CheckpointManager
from agent_stable_slo.utils.config import (
    CONFIG_VERSION,
    GRPOTrainConfig,
    migrate_config,
    validate_or_raise,
)
from agent_stable_slo.utils.data import cache_dataset, fingerprint_tasks, validate_fingerprint
from agent_stable_slo.utils.dist import barrier, destroy_distributed, init_distributed, rank_world, seed_with_rank
from agent_stable_slo.utils.hardware import detect_hardware, recommended_defaults
from agent_stable_slo.utils.repro import atomic_write_json, env_snapshot, set_seed


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _maybe_int(val, default=None):
    try:
        return int(val)
    except Exception:
        return default


def _guard_prompt(prompt: str, max_chars: int, truncate: bool) -> str:
    if max_chars and len(prompt) > max_chars:
        if truncate:
            return prompt[:max_chars]
        raise ValueError(f"prompt length {len(prompt)} exceeds max_prompt_chars={max_chars}")
    return prompt


def _parse_json(text: str, schema: dict) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p_strip = p.strip()
            if not p_strip:
                continue
            try:
                return json.loads(p_strip)
            except Exception:
                continue
    if "answer" in schema.get("properties", {}):
        return {"answer": text.strip()}
    return {}


def _load_dataset(tasks_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = json.loads(raw)
            schema_path = rec["schema_path"]
            schema = json.load(open(schema_path, "r", encoding="utf-8"))
            rec["schema"] = schema
            rows.append(rec)
    if not rows:
        raise ValueError(f"no task records found in {tasks_path}")
    return rows


def _default_targets(model_name: str) -> List[str]:
    if "qwen" in model_name.lower():
        return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def _build_model_and_tok(cfg, hw_rec: Dict[str, Any]):
    torch_dtype = getattr(torch, cfg.torch_dtype, torch.float16)
    load_in_4bit = cfg.load_in_4bit
    if load_in_4bit is None:
        load_in_4bit = bool(hw_rec.get("recommended", {}).get("load_in_4bit", False))
    if hw_rec.get("backend") != "cuda":
        load_in_4bit = False

    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch_dtype if quant_cfg is None else None,
        device_map="auto",
        quantization_config=quant_cfg,
        trust_remote_code=True,
    )
    if quant_cfg is not None:
        model = prepare_model_for_kbit_training(model)

    targets = cfg.lora_targets.split(",") if cfg.lora_targets else _default_targets(cfg.base_model)
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=targets,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tok


def _log_to_wandb(run, step: int, reward: float, rec: Dict[str, Any]):
    metrics = {
        "reward": reward,
        "latency_ms": rec["latency_ms"],
        "ttft_ms": rec["ttft_ms"],
        "json_valid": rec["json_valid"],
        "tokens_out": rec["tokens_out"],
    }
    WL.log(run, metrics, step=step)


def _compute_logprob_loss(model, tok, prompt_ids, gen_ids, advantage, use_autocast=False, device_type: str = "cuda"):
    ctx = torch.amp.autocast(device_type=device_type) if use_autocast else nullcontext()
    with ctx:
        full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
        attn = (full_ids != tok.pad_token_id).long()
        labels = torch.full_like(full_ids, -100)
        labels[:, prompt_ids.shape[1] :] = gen_ids
        outputs = model(input_ids=full_ids, attention_mask=attn, labels=labels)
        logits = outputs.logits[:, prompt_ids.shape[1] - 1 : -1, :]
        target = gen_ids
        logprobs = F.log_softmax(logits, dim=-1)
        gathered = logprobs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        logprob_sum = gathered.sum(dim=1)  # (batch,)
        loss = -(advantage * logprob_sum).mean()
    return loss


def _violates_blocklist(texts: Sequence[str], blocked: Sequence[str]) -> bool:
    items = [t.lower() for t in texts if t]
    for b in blocked:
        b = b.strip().lower()
        if not b:
            continue
        for t in items:
            if b in t:
                return True
    return False


def _maybe_build_manifest(run_dir: Path, cfg: GRPOTrainConfig, hw: Dict[str, Any], fp_src, fp_cached, rank: int, world: int):
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        return
    manifest = {
        "config": cfg.model_dump(),
        "config_version": CONFIG_VERSION,
        "hardware": hw,
        "dataset": {"source": fp_src.as_dict(), "cached": fp_cached.as_dict() if fp_cached else None},
        "dist": {"rank": rank, "world": world},
        "env": env_snapshot(include_packages=False),
        "created_at": time.time(),
    }
    atomic_write_json(str(manifest_path), manifest)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must map to a dict: {path}")
    return data


def _resolve_config_file(config_file: Optional[str], preset: Optional[str], config_dir: str) -> Optional[str]:
    if config_file:
        return config_file
    if preset:
        path = Path(config_dir) / f"{preset}.yaml"
        alt = Path(config_dir) / f"{preset.replace('-', '_')}.yaml"
        if path.exists():
            return str(path)
        if alt.exists():
            return str(alt)
    return None


def _prepare_out_dir(cfg: GRPOTrainConfig) -> Path:
    root = cfg.resume_from or cfg.out
    out_dir = Path(root or f"out/train_grpo_lora_{_timestamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _run_validation(model, tok, val_ds, cfg, device, blocked):
    rewards: List[float] = []
    if not val_ds:
        return None
    model.eval()
    for rec in val_ds:
        prompt = _guard_prompt(rec["prompt"], cfg.max_prompt_chars, cfg.truncate_prompts)
        schema = rec["schema"]
        enc = tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_prompt_len)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=not cfg.deterministic,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                pad_token_id=tok.pad_token_id,
            )
        gen_ids = gen[:, input_ids.shape[1] :]
        txt = tok.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        out_json = _parse_json(txt, schema)
        blocked_now = _violates_blocklist([prompt, txt], blocked)
        json_valid = 0 if blocked_now and cfg.reject_blocklisted else schema_valid(out_json, schema)
        tokens_out = int(gen_ids.shape[1]) if gen_ids.numel() else 0
        reward = composite_reward(
            out_json,
            schema,
            ok_success=json_valid,
            latency_ms=0.0,
            tokens=tokens_out,
            lam_latency=float(cfg.lam_latency),
            mu_cost=float(cfg.mu_cost),
            disagreement_rate=0.0,
            gamma_stability=float(cfg.gamma_stability),
        )
        rewards.append(float(reward))
    model.train()
    if not rewards:
        return None
    return {"val_reward_mean": float(sum(rewards) / len(rewards))}


def train_loop(cfg: GRPOTrainConfig):
    rank, world = rank_world()
    if cfg.ddp_backend:
        init_distributed(cfg.ddp_backend)

    hw = detect_hardware()
    hw_cfg = hw.as_dict()
    hw_cfg["recommended"] = recommended_defaults(hw)
    print(f"[hardware] {hw.summary()}")

    seed = seed_with_rank(cfg.seed)
    set_seed(seed, deterministic=cfg.repro or cfg.deterministic)

    src_fp = fingerprint_tasks(cfg.tasks)
    validate_fingerprint(src_fp, cfg.expected_dataset_hash, allow_drift=cfg.allow_dataset_drift)
    tasks_path = cfg.tasks
    cached_fp = None
    if cfg.cache_dataset:
        tasks_path, cached_fp = cache_dataset(cfg.tasks, cfg.cache_dir)
        print(f"[dataset] cached to {tasks_path}")

    ds = _load_dataset(tasks_path)
    val_ds: List[Dict[str, Any]] = []
    if cfg.val_tasks:
        val_path = cfg.val_tasks
        if cfg.cache_dataset:
            val_path, _ = cache_dataset(cfg.val_tasks, cfg.cache_dir)
        val_ds = _load_dataset(val_path)

    out_dir = _prepare_out_dir(cfg)
    run_log = JsonLogger(str(out_dir / "run_log.jsonl"))
    train_log_path = out_dir / "train_log.jsonl"
    adapter_path = out_dir / "adapter"

    _maybe_build_manifest(out_dir, cfg, hw_cfg, src_fp, cached_fp, rank, world)

    model, tok = _build_model_and_tok(cfg, hw_cfg)
    if torch.cuda.is_available():
        device = torch.device(hw.device)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device=device.type) if device.type == "cuda" else None
    grad_accum = max(1, cfg.gradient_accumulation)
    running_baseline = 0.0

    ckpt_mgr = CheckpointManager(str(out_dir))
    start_step = 0
    append_mode = False
    if cfg.resume_from:
        log_path = Path(cfg.resume_from) / "train_log.jsonl"
        if log_path.exists():
            existing = log_path.read_text(encoding="utf-8").splitlines()
            if existing:
                try:
                    last = json.loads(existing[-1])
                    start_step = int(last.get("step", 0)) + 1
                    append_mode = True
                except Exception:
                    start_step = 0
        latest = ckpt_mgr.latest()
        if latest:
            step_loaded, running_baseline, metadata, tok = ckpt_mgr.load(
                str(latest), model, tok, opt, scaler, scheduler=None
            )
            start_step = max(start_step, step_loaded + 1)
            print(f"[resume] loaded checkpoint {latest} at step {step_loaded}")

    blocked_terms = [s for s in (cfg.blocklist or "").split(",") if s]

    lam = float(os.getenv("LAMBDA_LATENCY", cfg.lam_latency))
    mu = float(os.getenv("MU_COST", cfg.mu_cost))
    gamma = float(os.getenv("GAMMA_STABILITY", cfg.gamma_stability))

    run_cfg = {
        "provider": "hf_local",
        "base_model": cfg.base_model,
        "steps": cfg.steps,
        "tasks": tasks_path,
        "hardware": hw_cfg,
        "lr": cfg.lr,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "load_in_4bit": cfg.load_in_4bit,
        "config_version": cfg.config_version,
        "rank": rank,
        "world": world,
    }

    with WL.maybe_run(name=os.path.basename(str(out_dir)), config=run_cfg) as run, open(
        train_log_path, "a" if append_mode else "w", encoding="utf-8"
    ) as fo:
        accum_loss = 0.0
        accum_count = 0
        for step in range(start_step, cfg.steps):
            row = ds[step % len(ds)]
            prompt = _guard_prompt(row["prompt"], cfg.max_prompt_chars, cfg.truncate_prompts)
            schema = row["schema"]
            gold = row.get("gold")

            schema_hint = (
                "Return ONLY JSON that matches the provided schema exactly; no text or commentary.\n"
                "If unsure, return an empty object with required keys.\n"
            )
            full_prompt = f"{schema_hint}{prompt}"

            enc = tok(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_prompt_len,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=not cfg.deterministic,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    pad_token_id=tok.pad_token_id,
                )
            lat_ms = (time.time() - t0) * 1000.0
            ttft_ms = lat_ms

            gen_ids = gen[:, input_ids.shape[1] :]
            tokens_out = int(gen_ids.shape[1]) if gen_ids.numel() else 0
            txt = tok.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            out_json = _parse_json(txt, schema)

            blocked_now = _violates_blocklist([full_prompt, txt], blocked_terms)
            json_valid = schema_valid(out_json, schema)
            if blocked_now and cfg.reject_blocklisted:
                json_valid = 0

            reward = composite_reward(
                out_json,
                schema,
                ok_success=json_valid,
                latency_ms=lat_ms,
                tokens=tokens_out,
                lam_latency=lam,
                mu_cost=mu,
                disagreement_rate=0.0,
                gamma_stability=gamma,
            )
            if blocked_now and cfg.reject_blocklisted:
                reward = 0.0

            baseline = running_baseline
            running_baseline = 0.9 * running_baseline + 0.1 * reward
            advantage = reward - baseline
            reward_tensor = torch.tensor([advantage], device=device, dtype=model.dtype)

            loss = _compute_logprob_loss(
                model,
                tok,
                prompt_ids=input_ids,
                gen_ids=gen_ids,
                advantage=reward_tensor,
                use_autocast=device.type == "cuda",
                device_type=device.type,
            )
            loss = loss / grad_accum
            accum_loss += float(loss.detach().cpu())

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_count += 1

            if accum_count % grad_accum == 0 or step == cfg.steps - 1:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            rec = {
                "step": step,
                "prompt": prompt,
                "output_text": txt,
                "output_json": out_json,
                "reward": float(reward),
                "advantage": float(advantage),
                "latency_ms": float(round(lat_ms, 3)),
                "ttft_ms": float(round(ttft_ms, 3)),
                "json_valid": int(json_valid),
                "tokens_out": tokens_out,
                "schema_path": row["schema_path"],
                "blocked": bool(blocked_now),
            }
            if gold is not None:
                rec["gold"] = gold
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            _log_to_wandb(run, step, reward, rec)

            if cfg.checkpoint_every and cfg.checkpoint_every > 0 and (step + 1) % cfg.checkpoint_every == 0:
                ckpt_path = ckpt_mgr.save(
                    step=step,
                    model=model,
                    tokenizer=tok,
                    optimizer=opt,
                    scaler=scaler,
                    baseline=running_baseline,
                    metadata={"tasks": tasks_path, "seed": seed, "config_version": cfg.config_version},
                    scheduler=None,
                )
                run_log.info("checkpoint_saved", step=step, path=str(ckpt_path))

            if cfg.val_interval and cfg.val_interval > 0 and (step + 1) % cfg.val_interval == 0 and cfg.val_tasks:
                val_metrics = _run_validation(model, tok, val_ds, cfg, device, blocked_terms)
                if val_metrics:
                    run_log.info("val", step=step, **val_metrics)
                    if run:
                        WL.log(run, val_metrics, step=step)

            if (step + 1) % max(1, cfg.eval_interval) == 0:
                print(f"[step {step+1}] reward={reward:.3f} lat_ms={lat_ms:.1f} json_ok={json_valid} loss={accum_loss:.4f}")
                accum_loss = 0.0

    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)
    print(f"[done] wrote {train_log_path} and adapter to {adapter_path}")
    destroy_distributed()


def parse_args(argv: Optional[Sequence[str]] = None):
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config-file", type=str, default=None, help="Optional YAML config file.")
    base.add_argument(
        "--config-preset",
        type=str,
        default=None,
        help="Preset name under configs/grpo (omit .yaml).",
    )
    base.add_argument("--config-dir", type=str, default="configs/grpo", help="Directory for presets.")
    known, remaining = base.parse_known_args(argv)

    cfg_file = _resolve_config_file(known.config_file, known.config_preset, known.config_dir)
    cfg_defaults: Dict[str, Any] = {}
    if cfg_file:
        cfg_defaults = migrate_config(_load_yaml(cfg_file))
        print(f"[config] loaded {cfg_file}")

    ap = argparse.ArgumentParser(parents=[base])
    ap.add_argument("--base-model", default=cfg_defaults.get("base_model", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--tasks", default=cfg_defaults.get("tasks", "tasks/robust_eval_gold.jsonl"))
    ap.add_argument("--out", default=cfg_defaults.get("out", ""))
    ap.add_argument("--steps", type=int, default=cfg_defaults.get("steps", 500))
    ap.add_argument("--max-prompt-len", type=int, default=cfg_defaults.get("max_prompt_len", 1024))
    ap.add_argument("--max-new-tokens", type=int, default=cfg_defaults.get("max_new_tokens", 128))
    ap.add_argument("--temperature", type=float, default=cfg_defaults.get("temperature", 0.7))
    ap.add_argument("--top-p", type=float, default=cfg_defaults.get("top_p", 0.95))
    ap.add_argument("--lr", type=float, default=cfg_defaults.get("lr", 1e-5))
    ap.add_argument("--weight-decay", type=float, default=cfg_defaults.get("weight_decay", 0.01))
    ap.add_argument("--gradient-accumulation", type=int, default=cfg_defaults.get("gradient_accumulation", 1))
    ap.add_argument("--max-grad-norm", type=float, default=cfg_defaults.get("max_grad_norm", 1.0))
    ap.add_argument("--lora-rank", type=int, default=cfg_defaults.get("lora_rank", 16))
    ap.add_argument("--lora-alpha", type=int, default=cfg_defaults.get("lora_alpha", 32))
    ap.add_argument("--lora-dropout", type=float, default=cfg_defaults.get("lora_dropout", 0.05))
    ap.add_argument("--lora-targets", type=str, default=cfg_defaults.get("lora_targets", ""))
    ap.add_argument(
        "--load-in-4bit",
        type=lambda x: x.lower() in {"1", "true", "yes"},
        default=cfg_defaults.get("load_in_4bit", None),
        help="Override 4-bit loading (default picks based on hardware).",
    )
    ap.add_argument("--torch-dtype", type=str, default=cfg_defaults.get("torch_dtype", "float16"))
    ap.add_argument("--eval-interval", type=int, default=cfg_defaults.get("eval_interval", 50))
    ap.add_argument("--deterministic", action="store_true", default=cfg_defaults.get("deterministic", False))
    ap.add_argument("--force-json-fallback", action="store_true", default=cfg_defaults.get("force_json_fallback", False))
    ap.add_argument("--lam-latency", type=float, default=cfg_defaults.get("lam_latency", 0.0))
    ap.add_argument("--mu-cost", type=float, default=cfg_defaults.get("mu_cost", 0.0))
    ap.add_argument("--gamma-stability", type=float, default=cfg_defaults.get("gamma_stability", 0.0))
    ap.add_argument("--seed", type=int, default=cfg_defaults.get("seed", 17))
    ap.add_argument("--repro", action="store_true", default=cfg_defaults.get("repro", False))
    ap.add_argument("--cache-dataset", action="store_true", default=cfg_defaults.get("cache_dataset", False))
    ap.add_argument("--cache-dir", type=str, default=cfg_defaults.get("cache_dir", "out/cache"))
    ap.add_argument("--checkpoint-every", type=int, default=cfg_defaults.get("checkpoint_every", 0))
    ap.add_argument("--resume-from", type=str, default=cfg_defaults.get("resume_from", None))
    ap.add_argument("--no-silent-defaults", action="store_true", default=cfg_defaults.get("no_silent_defaults", False))
    ap.add_argument("--expected-dataset-hash", type=str, default=cfg_defaults.get("expected_dataset_hash", None))
    ap.add_argument("--allow-dataset-drift", action="store_true", default=cfg_defaults.get("allow_dataset_drift", False))
    ap.add_argument("--val-tasks", type=str, default=cfg_defaults.get("val_tasks", None))
    ap.add_argument("--val-interval", type=int, default=cfg_defaults.get("val_interval", 0))
    ap.add_argument("--val-samples", type=int, default=cfg_defaults.get("val_samples", 1))
    ap.add_argument("--max-prompt-chars", type=int, default=cfg_defaults.get("max_prompt_chars", 0))
    ap.add_argument("--truncate-prompts", action="store_true", default=cfg_defaults.get("truncate_prompts", False))
    ap.add_argument("--blocklist", type=str, default=cfg_defaults.get("blocklist", None))
    ap.add_argument("--reject-blocklisted", action="store_true", default=cfg_defaults.get("reject_blocklisted", False))
    ap.add_argument("--ddp-backend", type=str, default=cfg_defaults.get("ddp_backend", None))
    args = ap.parse_args(remaining)
    cfg_dict = vars(args)
    # Drop loader-only fields not present in the pydantic model.
    for drop in ("config_file", "config_preset", "config_dir"):
        cfg_dict.pop(drop, None)
    cfg = validate_or_raise(cfg_dict)
    return cfg


def main():
    cfg = parse_args()
    train_loop(cfg)


if __name__ == "__main__":
    main()
