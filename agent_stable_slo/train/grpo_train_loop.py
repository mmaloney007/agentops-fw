#!/usr/bin/env python3
"""
GRPO-style LoRA trainer with a minimal policy-gradient loop.
Designed to work on:
 - 4090/CUDA: bf16, optional 4-bit + LoRA
 - Mac (MPS): fp16, small batch/length (no bitsandbytes)

This is intentionally lightweight to avoid hard dependencies on TRL versions.
If trl is installed, we can later swap in GRPOTrainer; for now we run a REINFORCE-style
loop with LoRA adapters and the same composite reward used in eval.
"""
import argparse, json, os, time, math
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig  # type: ignore
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from agent_stable_slo.rewards.composite import composite_reward
from agent_stable_slo.rewards.schema_reward import schema_valid
from agent_stable_slo.logging import wandb_utils as WL
from agent_stable_slo.utils.hardware import detect_hardware, recommended_defaults


def _maybe_int(val, default=None):
    try:
        return int(val)
    except Exception:
        return default


def _parse_json(text: str, schema: dict) -> Dict[str, Any]:
    # Try direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract from fenced code blocks
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
    # Fallback: if schema has a simple "answer" field, wrap the text
    if "answer" in schema.get("properties", {}):
        return {"answer": text.strip()}
    return {}


def _load_dataset(tasks_path: str) -> Dataset:
    rows = []
    for raw in open(tasks_path, "r", encoding="utf-8"):
        if not raw.strip():
            continue
        rec = json.loads(raw)
        schema_path = rec["schema_path"]
        schema = json.load(open(schema_path, "r", encoding="utf-8"))
        rec["schema"] = schema
        rows.append(rec)
    return Dataset.from_list(rows)


def _default_targets(model_name: str) -> List[str]:
    if "qwen" in model_name.lower():
        return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def _build_model_and_tok(args, hw_rec: Dict[str, Any]):
    torch_dtype = getattr(torch, args.torch_dtype, torch.float16)
    load_in_4bit = args.load_in_4bit
    if load_in_4bit is None:
        load_in_4bit = bool(hw_rec.get("recommended", {}).get("load_in_4bit", False))
    if hw_rec.get("backend") != "cuda":
        load_in_4bit = False  # avoid bitsandbytes on MPS/CPU

    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype if quant_cfg is None else None,
        device_map="auto",
        quantization_config=quant_cfg,
        trust_remote_code=True,
    )
    if quant_cfg is not None:
        model = prepare_model_for_kbit_training(model)

    targets = args.lora_targets.split(",") if args.lora_targets else _default_targets(args.base_model)
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        lora_dropout=args.lora_dropout,
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


def _compute_logprob_loss(model, tok, prompt_ids, gen_ids, reward):
    """
    Re-run the model on prompt + generated tokens to get logprobs, then
    compute a REINFORCE-style loss: -reward * sum(logp(gen_tokens)).
    """
    # Build full sequence and mask prompt tokens from loss
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    attn = (full_ids != tok.pad_token_id).long()
    # Shift labels: predict generated tokens; ignore prompt positions
    labels = torch.full_like(full_ids, -100)
    labels[:, prompt_ids.shape[1] :] = gen_ids

    outputs = model(input_ids=full_ids, attention_mask=attn, labels=labels)
    logits = outputs.logits[:, prompt_ids.shape[1] - 1 : -1, :]
    # labels for generated tokens
    target = gen_ids
    logprobs = F.log_softmax(logits, dim=-1)
    gathered = logprobs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    # sum logprobs over generated tokens
    logprob_sum = gathered.sum(dim=1)  # (batch,)
    loss = -(reward * logprob_sum).mean()
    return loss


def train_loop(args):
    hw = detect_hardware()
    hw_cfg = hw.as_dict()
    hw_cfg["recommended"] = recommended_defaults(hw)
    print(f"[hardware] {hw.summary()}")

    ds = _load_dataset(args.tasks)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.out) / "train_log.jsonl"
    adapter_path = Path(args.out) / "adapter"

    model, tok = _build_model_and_tok(args, hw_cfg)
    if torch.cuda.is_available():
        device = torch.device(hw.device)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Optimizer on trainable (LoRA) params only
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    grad_accum = max(1, args.gradient_accumulation)
    running_baseline = 0.0

    lam = float(os.getenv("LAMBDA_LATENCY", args.lam_latency))
    mu = float(os.getenv("MU_COST", args.mu_cost))
    gamma = float(os.getenv("GAMMA_STABILITY", args.gamma_stability))

    with WL.maybe_run(
        name=os.path.basename(args.out),
        config={
            "provider": "hf_local",
            "base_model": args.base_model,
            "steps": args.steps,
            "tasks": args.tasks,
            "hardware": hw_cfg,
            "lr": args.lr,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "load_in_4bit": args.load_in_4bit,
        },
    ) as run, open(log_path, "w", encoding="utf-8") as fo:
        for step in range(args.steps):
            row = ds[step % len(ds)]
            prompt = row["prompt"]
            schema = row["schema"]
            gold = row.get("gold")

            enc = tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=args.max_prompt_len)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Sample a completion (no grad), then compute reward, then compute logprob loss with grad
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            lat_ms = (time.time() - t0) * 1000.0
            ttft_ms = lat_ms  # non-streaming approximation

            seq = gen.sequences
            gen_ids = seq[:, input_ids.shape[1] :]
            decoded = tok.batch_decode(gen_ids, skip_special_tokens=True)[0]
            out_json = _parse_json(decoded, schema)
            json_valid = schema_valid(out_json, schema)
            tokens_out = _maybe_int(gen_ids.numel(), -1)

            reward = composite_reward(
                out_json,
                schema,
                ok_success=json_valid,
                latency_ms=lat_ms,
                tokens=tokens_out if tokens_out is not None else 0,
                lam_latency=lam,
                mu_cost=mu,
                disagreement_rate=0.0,
                gamma_stability=gamma,
            )

            # Simple moving baseline to reduce variance
            baseline = running_baseline
            running_baseline = 0.9 * running_baseline + 0.1 * reward
            advantage = reward - baseline
            reward_tensor = torch.tensor([advantage], device=device, dtype=model.dtype)

            with torch.autocast(device_type=("cuda" if hw.backend == "cuda" else "cpu"), enabled=(hw.backend == "cuda")):
                loss = _compute_logprob_loss(model, tok, input_ids, gen_ids, reward_tensor) / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

            rec = {
                "step": step,
                "prompt": prompt,
                "output_text": decoded,
                "output_json": out_json,
                "reward": float(reward),
                "advantage": float(advantage),
                "latency_ms": float(round(lat_ms, 3)),
                "ttft_ms": float(round(ttft_ms, 3)),
                "json_valid": int(json_valid),
                "tokens_out": int(tokens_out) if tokens_out is not None else -1,
                "schema_path": row["schema_path"],
            }
            if gold is not None:
                rec["gold"] = gold
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            _log_to_wandb(run, step, reward, rec)

            if (step + 1) % max(1, args.eval_interval) == 0:
                print(f"[step {step+1}] reward={reward:.3f} lat_ms={lat_ms:.1f} json_ok={json_valid}")

    # Save LoRA adapter
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)
    print(f"[done] wrote {log_path} and adapter to {adapter_path}")


def parse_args():
    hw = detect_hardware()
    rec = recommended_defaults(hw)
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tasks", default="tasks/robust_eval_gold.jsonl")
    ap.add_argument("--out", default="out/train_grpo_lora")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--max-prompt-len", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--gradient-accumulation", type=int, default=1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", type=str, default="")
    ap.add_argument("--load-in-4bit", type=lambda x: x.lower() in {"1","true","yes"}, default=None,
                    help="Override 4-bit loading (default picks based on hardware).")
    ap.add_argument("--torch-dtype", type=str, default=rec.get("torch_dtype", "float16"))
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--lam-latency", type=float, default=0.0)
    ap.add_argument("--mu-cost", type=float, default=0.0)
    ap.add_argument("--gamma-stability", type=float, default=0.0)
    return ap.parse_args()


def main():
    args = parse_args()
    train_loop(args)


if __name__ == "__main__":
    main()
