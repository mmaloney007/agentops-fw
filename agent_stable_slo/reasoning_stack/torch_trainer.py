"""CUDA training helpers for a three-stage LLM + reasoning stack."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .torch_model import TinyLMConfig, build_tiny_causal_lm


def _require_torch_transformers():
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError(
            "CUDA path requires torch and transformers. Install train extras first."
        ) from exc
    return torch, F, AutoTokenizer


def _require_rewards():
    from agent_stable_slo.rewards.composite import composite_reward
    from agent_stable_slo.rewards.schema_reward import schema_valid

    return composite_reward, schema_valid


def _build_token_stream(tokenizer: Any, docs: Iterable[str], min_tokens: int) -> List[int]:
    token_ids: List[int] = []
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must provide eos_token_id")

    for doc in docs:
        text = doc.strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            token_ids.extend(ids)
            token_ids.append(eos_id)

    if len(token_ids) < min_tokens:
        raise ValueError(
            f"Token stream too small: got {len(token_ids)} tokens, need at least {min_tokens}"
        )
    return token_ids


def _sample_batch(
    torch_mod,
    token_stream,
    batch_size: int,
    seq_len: int,
    device,
) -> Tuple[object, object]:
    max_start = len(token_stream) - seq_len - 1
    starts = torch_mod.randint(low=0, high=max_start, size=(batch_size,))
    xs = []
    ys = []
    for start in starts.tolist():
        xs.append(token_stream[start : start + seq_len])
        ys.append(token_stream[start + 1 : start + seq_len + 1])

    x = torch_mod.tensor(xs, dtype=torch_mod.long, device=device)
    y = torch_mod.tensor(ys, dtype=torch_mod.long, device=device)
    return x, y


def _lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return max(0.05, step / max(1, warmup_steps))

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _train_language_model(
    *,
    stage_name: str,
    model,
    token_stream: List[int],
    train_steps: int,
    seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    log_every: int,
    run_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    torch, _, _ = _require_torch_transformers()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend selected, but torch.cuda.is_available() is False")

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    t0 = time.time()
    total_tokens = 0
    last_loss = None

    for step in range(1, train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            x, y = _sample_batch(
                torch_mod=torch,
                token_stream=token_stream,
                batch_size=micro_batch_size,
                seq_len=seq_len,
                device=device,
            )
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Model returned None loss with target_ids provided")
            (loss / grad_accum_steps).backward()
            accum_loss += float(loss.detach().item())
            total_tokens += int(x.numel())

        lr_mult = _lr_multiplier(step, warmup_steps=warmup_steps, total_steps=train_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * lr_mult

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        last_loss = accum_loss / grad_accum_steps
        if step == 1 or step % log_every == 0 or step == train_steps:
            elapsed = time.time() - t0
            record = {
                "stage": stage_name,
                "step": step,
                "train_steps": train_steps,
                "loss": round(last_loss, 6),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "tokens_seen": total_tokens,
                "tokens_per_second": round(total_tokens / max(1e-6, elapsed), 2),
                "elapsed_sec": round(elapsed, 2),
            }
            _write_jsonl(log_path, record)

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "stage": stage_name,
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "train_steps": train_steps,
            "last_loss": last_loss,
        },
        checkpoint_path,
    )

    return {
        "stage": stage_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "log": str(log_path.resolve()),
        "train_steps": train_steps,
        "last_loss": last_loss,
        "tokens_seen": total_tokens,
    }


def _parse_json(text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if not p:
                continue
            try:
                return json.loads(p)
            except Exception:
                continue

    if "answer" in schema.get("properties", {}):
        return {"answer": text.strip()}
    return {}


def _canonical_json(obj: Dict[str, Any]) -> str:
    if not isinstance(obj, dict) or not obj:
        return ""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sample_next_token(torch_mod, logits, temperature: float, top_p: float):
    if temperature <= 0.0:
        return torch_mod.argmax(logits, dim=-1, keepdim=True)

    scaled = logits / max(temperature, 1e-5)
    sorted_logits, sorted_indices = torch_mod.sort(scaled, descending=True)
    sorted_probs = torch_mod.softmax(sorted_logits, dim=-1)
    cumulative = torch_mod.cumsum(sorted_probs, dim=-1)

    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

    probs = torch_mod.softmax(sorted_logits, dim=-1)
    sample_idx = torch_mod.multinomial(probs, num_samples=1)
    return sorted_indices.gather(-1, sample_idx)


def _generate_completion(
    *,
    model,
    tokenizer,
    torch_mod,
    prompt_ids,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device,
):
    eos_id = tokenizer.eos_token_id
    cur = prompt_ids.clone()
    generated = []

    t0 = time.time()
    model.eval()
    with torch_mod.no_grad():
        for _ in range(max_new_tokens):
            if cur.shape[1] >= model.config.max_seq_len:
                cur = cur[:, -model.config.max_seq_len + 1 :]

            logits, _ = model(cur)
            next_logits = logits[:, -1, :]
            next_token = _sample_next_token(
                torch_mod,
                logits=next_logits,
                temperature=temperature,
                top_p=top_p,
            )
            cur = torch_mod.cat([cur, next_token], dim=1)
            generated.append(next_token)

            if eos_id is not None and int(next_token.item()) == int(eos_id):
                break

    model.train()
    lat_ms = (time.time() - t0) * 1000.0

    if generated:
        gen_ids = torch_mod.cat(generated, dim=1)
    else:
        fallback = eos_id if eos_id is not None else 0
        gen_ids = torch_mod.tensor([[fallback]], dtype=torch_mod.long, device=device)

    text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True).strip()
    tokens_out = int(gen_ids.shape[1])
    return gen_ids, text, lat_ms, tokens_out


def _logprob_sum_for_completion(torch_mod, F_mod, model, prompt_ids, gen_ids):
    full_ids = torch_mod.cat([prompt_ids, gen_ids], dim=1)
    logits, _ = model(full_ids[:, :-1])
    targets = full_ids[:, 1:]

    start = prompt_ids.shape[1] - 1
    logits_gen = logits[:, start:, :]
    targets_gen = targets[:, start:]

    log_probs = F_mod.log_softmax(logits_gen, dim=-1)
    gathered = log_probs.gather(2, targets_gen.unsqueeze(-1)).squeeze(-1)
    return gathered.sum()


def _prepare_rl_records(reasoning_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    schema_cache: Dict[str, Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []

    for ex in reasoning_examples:
        schema_path = str(ex.get("schema_path", ""))
        if not schema_path:
            continue
        if schema_path not in schema_cache:
            schema_cache[schema_path] = json.loads(Path(schema_path).read_text(encoding="utf-8"))

        records.append(
            {
                "prompt": str(ex.get("prompt", "")).strip(),
                "schema": schema_cache[schema_path],
                "gold": ex.get("gold", {}) if isinstance(ex.get("gold"), dict) else {},
                "schema_path": schema_path,
            }
        )
    return [r for r in records if r["prompt"]]


def _train_rl_stage(
    *,
    model,
    tokenizer,
    reasoning_examples: List[Dict[str, Any]],
    config,
    run_dir: Path,
) -> Dict[str, Any]:
    torch, F_mod, _ = _require_torch_transformers()
    composite_reward, schema_valid = _require_rewards()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend selected, but torch.cuda.is_available() is False")

    records = _prepare_rl_records(reasoning_examples)
    if not records:
        raise ValueError("No RL records available from reasoning examples")

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    device = torch.device("cuda:0")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate * 0.25, weight_decay=config.weight_decay)

    random.seed(config.seed + 3)
    torch.manual_seed(config.seed + 3)
    torch.cuda.manual_seed_all(config.seed + 3)

    t0 = time.time()
    rewards_window: List[float] = []
    valid_window: List[float] = []
    last_loss = None

    max_prompt_tokens = max(16, config.sequence_length - config.rl_max_new_tokens - 1)

    for step in range(1, config.rl_steps + 1):
        row = records[random.randint(0, len(records) - 1)]
        prompt_text = row["prompt"]
        schema = row["schema"]
        gold = row["gold"]

        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_tokens) > max_prompt_tokens:
            prompt_tokens = prompt_tokens[:max_prompt_tokens]
        if not prompt_tokens:
            eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            prompt_tokens = [eos]

        prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        completions = []
        canonical_outputs = []
        for _ in range(config.rl_group_size):
            gen_ids, text, lat_ms, tokens_out = _generate_completion(
                model=model,
                tokenizer=tokenizer,
                torch_mod=torch,
                prompt_ids=prompt_ids,
                max_new_tokens=config.rl_max_new_tokens,
                temperature=config.rl_temperature,
                top_p=config.rl_top_p,
                device=device,
            )
            out_json = _parse_json(text, schema)
            canonical = _canonical_json(out_json)
            canonical_outputs.append(canonical)

            valid = int(schema_valid(out_json, schema))
            exact = int(canonical and canonical == _canonical_json(gold))
            completions.append(
                {
                    "gen_ids": gen_ids,
                    "text": text,
                    "json": out_json,
                    "json_valid": valid,
                    "ok_success": exact,
                    "latency_ms": lat_ms,
                    "tokens_out": tokens_out,
                }
            )

        mode_count = max(canonical_outputs.count(x) for x in canonical_outputs) if canonical_outputs else 1
        disagreement_rate = 1.0 - (mode_count / max(1, len(canonical_outputs)))

        rewards = []
        for out in completions:
            r = composite_reward(
                out["json"],
                schema,
                ok_success=out["ok_success"],
                latency_ms=out["latency_ms"],
                tokens=out["tokens_out"],
                lam_latency=config.lam_latency,
                mu_cost=config.mu_cost,
                disagreement_rate=disagreement_rate,
                gamma_stability=config.gamma_stability,
            )
            rewards.append(float(r))

        reward_mean = sum(rewards) / max(1, len(rewards))
        advantages = [r - reward_mean for r in rewards]

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        for out, advantage in zip(completions, advantages):
            logprob_sum = _logprob_sum_for_completion(
                torch_mod=torch,
                F_mod=F_mod,
                model=model,
                prompt_ids=prompt_ids,
                gen_ids=out["gen_ids"],
            )
            total_loss = total_loss + (-float(advantage) * logprob_sum)

        loss = total_loss / max(1, len(completions))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        last_loss = float(loss.detach().item())
        rewards_window.append(reward_mean)
        valid_window.append(sum(out["json_valid"] for out in completions) / max(1, len(completions)))

        best_idx = max(range(len(completions)), key=lambda i: rewards[i])
        best = completions[best_idx]

        if step == 1 or step % config.rl_log_every == 0 or step == config.rl_steps:
            elapsed = time.time() - t0
            rec = {
                "stage": "stage3_rl",
                "step": step,
                "rl_steps": config.rl_steps,
                "loss": round(last_loss, 6),
                "reward_mean": round(reward_mean, 6),
                "best_reward": round(rewards[best_idx], 6),
                "best_json_valid": int(best["json_valid"]),
                "best_ok_success": int(best["ok_success"]),
                "disagreement_rate": round(disagreement_rate, 6),
                "latency_ms": round(best["latency_ms"], 3),
                "tokens_out": int(best["tokens_out"]),
                "tokens_per_second": round((step * config.rl_group_size) / max(1e-6, elapsed), 2),
                "schema_path": row["schema_path"],
            }
            _write_jsonl(log_path, rec)

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "stage": "stage3_rl",
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "train_steps": config.rl_steps,
            "last_loss": last_loss,
            "reward_mean": sum(rewards_window) / max(1, len(rewards_window)),
            "valid_mean": sum(valid_window) / max(1, len(valid_window)),
        },
        checkpoint_path,
    )

    return {
        "stage": "stage3_rl",
        "checkpoint": str(checkpoint_path.resolve()),
        "log": str(log_path.resolve()),
        "train_steps": config.rl_steps,
        "last_loss": last_loss,
        "mean_reward": sum(rewards_window) / max(1, len(rewards_window)),
        "mean_json_valid": sum(valid_window) / max(1, len(valid_window)),
    }


def train_cuda_three_stage(
    *,
    pretrain_docs: List[str],
    reasoning_examples: List[Dict[str, Any]],
    config,
) -> Dict[str, Any]:
    """Run base-LM pretraining -> reasoning SFT -> RL optimization on CUDA."""

    _, _, auto_tokenizer = _require_torch_transformers()

    tokenizer = auto_tokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = TinyLMConfig(
        vocab_size=len(tokenizer),
        max_seq_len=config.sequence_length,
        hidden_size=config.model_hidden_size,
        num_layers=config.model_layers,
        num_heads=config.model_heads,
        ffn_mult=config.model_ffn_mult,
        dropout=config.model_dropout,
    )
    model = build_tiny_causal_lm(model_config)

    pretrain_tokens = _build_token_stream(
        tokenizer=tokenizer,
        docs=pretrain_docs,
        min_tokens=config.sequence_length + 2,
    )
    stage1 = _train_language_model(
        stage_name="stage1_base_lm",
        model=model,
        token_stream=pretrain_tokens,
        train_steps=config.pretrain_steps,
        seq_len=config.sequence_length,
        micro_batch_size=config.micro_batch_size,
        grad_accum_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        log_every=config.log_every,
        run_dir=config.stage_dir("stage1_base_lm"),
        seed=config.seed,
    )

    reasoning_docs = [
        f"<|user|>\n{ex['prompt']}\n<|assistant|>\n{ex['completion']}" for ex in reasoning_examples
    ]
    reasoning_tokens = _build_token_stream(
        tokenizer=tokenizer,
        docs=reasoning_docs,
        min_tokens=config.sequence_length + 2,
    )
    stage2 = _train_language_model(
        stage_name="stage2_reasoning",
        model=model,
        token_stream=reasoning_tokens,
        train_steps=config.reasoning_steps,
        seq_len=config.sequence_length,
        micro_batch_size=max(1, config.micro_batch_size // 2),
        grad_accum_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate * 0.5,
        weight_decay=config.weight_decay,
        warmup_steps=max(10, config.warmup_steps // 2),
        log_every=config.log_every,
        run_dir=config.stage_dir("stage2_reasoning"),
        seed=config.seed + 1,
    )

    stage3 = None
    if config.enable_rl_stage:
        stage3 = _train_rl_stage(
            model=model,
            tokenizer=tokenizer,
            reasoning_examples=reasoning_examples,
            config=config,
            run_dir=config.stage_dir("stage3_rl"),
        )

    tokenizer_dir = config.stage_dir("artifacts")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    return {
        "backend": "cuda",
        "tokenizer_dir": str(tokenizer_dir.resolve()),
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
    }
