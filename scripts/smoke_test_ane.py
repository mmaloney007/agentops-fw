#!/usr/bin/env python3
"""
ANE smoke test -- loads ANEMLL-converted CoreML models and runs a short
inference to verify Apple Neural Engine execution works end-to-end.

Supports both monolithic and chunked ANEMLL model layouts, with automatic
detection via meta.yaml.

Usage:
  python scripts/smoke_test_ane.py --model-dir models/ane/qwen2.5-0.5b
  python scripts/smoke_test_ane.py --model-dir models/ane/llama-3.2-1b
  python scripts/smoke_test_ane.py --all          # test all models in models/ane/
  python scripts/smoke_test_ane.py --all --cpu     # CPU-only mode (skip ANE)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Meta.yaml parsing
# ---------------------------------------------------------------------------


def load_meta(model_dir: Path) -> Dict[str, Any]:
    """Parse meta.yaml from ANEMLL model directory."""
    meta_path = model_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.yaml not found in {model_dir}")

    import yaml

    with open(meta_path) as f:
        raw = yaml.safe_load(f)

    info = raw.get("model_info", {})
    params = info.get("parameters", {})

    return {
        "name": info.get("name", model_dir.name),
        "architecture": info.get("architecture", "unknown"),
        "model_type": info.get("model_type", "chunked"),
        "context_length": params.get("context_length", 512),
        "batch_size": params.get("batch_size", 64),
        "num_chunks": params.get("num_chunks", 0),
        "model_prefix": params.get("model_prefix", ""),
        "monolithic_model": params.get("monolithic_model", ""),
        "embeddings": params.get("embeddings", ""),
        "lm_head": params.get("lm_head", ""),
        "ffn": params.get("ffn", ""),
        "split_lm_head": params.get("split_lm_head", 8),
        "argmax_in_model": params.get("argmax_in_model", False),
        "vocab_size": params.get("vocab_size", 0),
        "lm_head_chunk_sizes": params.get("lm_head_chunk_sizes", []),
        "prefill_dynamic_slice": params.get("prefill_dynamic_slice", False),
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_coreml_model(path: Path, function_name: str = None, compute_unit=None):
    """Load a CoreML model, handling both .mlmodelc and .mlpackage.

    If the requested path doesn't exist, tries the other extension as fallback.
    """
    import coremltools as ct

    if compute_unit is None:
        compute_unit = ct.ComputeUnit.CPU_AND_NE

    # If path doesn't exist, try the other extension
    if not path.exists():
        if path.suffix == ".mlmodelc":
            alt = path.with_suffix(".mlpackage")
            if alt.exists():
                path = alt
        elif path.suffix == ".mlpackage":
            alt = path.with_suffix(".mlmodelc")
            if alt.exists():
                path = alt
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

    if path.suffix == ".mlmodelc":
        if function_name:
            try:
                return ct.models.CompiledMLModel(
                    str(path), compute_unit, function_name=function_name
                )
            except RuntimeError:
                # Multi-function compiled models may not support function_name
                # Fall back to .mlpackage if available
                mlpackage = path.with_suffix(".mlpackage")
                if mlpackage.exists():
                    return ct.models.MLModel(
                        str(mlpackage), function_name=function_name
                    )
                raise
        return ct.models.CompiledMLModel(str(path), compute_unit)
    else:
        if function_name:
            return ct.models.MLModel(str(path), function_name=function_name)
        return ct.models.MLModel(str(path))


def load_models(
    model_dir: Path, meta: Dict[str, Any], cpu_only: bool = False
) -> Dict[str, Any]:
    """Load all model components based on meta.yaml."""
    import coremltools as ct

    cu = ct.ComputeUnit.CPU_ONLY if cpu_only else ct.ComputeUnit.CPU_AND_NE

    is_monolithic = bool(meta.get("monolithic_model"))

    if is_monolithic:
        model_file = meta["monolithic_model"]
        model_path = model_dir / model_file
        if not model_path.exists():
            # Try without extension match
            candidates = list(model_dir.glob(f"{model_file}*"))
            if candidates:
                model_path = candidates[0]

        print(f"  Loading monolithic model: {model_path.name}")
        infer_model = load_coreml_model(model_path, "infer", cu)
        prefill_model = load_coreml_model(model_path, "prefill", cu)
        state = infer_model.make_state()

        return {
            "layout": "monolithic",
            "infer": infer_model,
            "prefill": prefill_model,
            "state": state,
        }
    else:
        # Chunked layout
        embed_file = meta["embeddings"]
        lmhead_file = meta["lm_head"]
        ffn_file = meta["ffn"]

        embed_path = model_dir / embed_file
        lmhead_path = model_dir / lmhead_file
        ffn_path = model_dir / ffn_file

        print(f"  Loading embed: {embed_path.name}")
        embed_model = load_coreml_model(embed_path, compute_unit=cu)

        print(f"  Loading lm_head: {lmhead_path.name}")
        lmhead_model = load_coreml_model(lmhead_path, compute_unit=cu)

        print(f"  Loading FFN: {ffn_path.name}")
        ffn_infer = load_coreml_model(ffn_path, "infer", cu)
        ffn_prefill = load_coreml_model(ffn_path, "prefill", cu)

        # Create state from infer model
        state = ffn_infer.make_state()

        return {
            "layout": "chunked",
            "embed": embed_model,
            "lmhead": lmhead_model,
            "ffn_infer": ffn_infer,
            "ffn_prefill": ffn_prefill,
            "state": state,
        }


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def load_tokenizer(model_dir: Path, meta: Dict[str, Any]):
    """Load tokenizer from model directory or HuggingFace."""
    from transformers import AutoTokenizer

    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"

    if tokenizer_path.exists():
        print(f"  Loading tokenizer from: {model_dir}")
        return AutoTokenizer.from_pretrained(str(model_dir))

    # Fallback: use architecture-specific default
    arch = meta.get("architecture", "")
    defaults = {
        "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
        "llama": "meta-llama/Llama-3.2-1B-Instruct",
        "gemma": "google/gemma-3-1b-it",
    }
    hf_name = defaults.get(arch, "")
    if hf_name:
        print(f"  Loading tokenizer from HuggingFace: {hf_name}")
        return AutoTokenizer.from_pretrained(hf_name)

    raise ValueError(f"Cannot find tokenizer for {model_dir}")


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------


def make_causal_mask(context_length: int) -> np.ndarray:
    """Create full causal attention mask. Shape: (1, 1, ctx, ctx)."""
    mask = np.zeros((1, 1, context_length, context_length), dtype=np.float16)
    for i in range(context_length):
        mask[0, 0, i, i + 1 :] = np.float16(-65504.0)
    return mask


# ---------------------------------------------------------------------------
# Argmax-in-model output decoding
# ---------------------------------------------------------------------------


def decode_argmax_output(
    output: Dict[str, Any], meta: Dict[str, Any]
) -> int:
    """Decode argmax_idx/argmax_val output to a global token index."""
    argmax_idx = output["argmax_idx"].flatten()
    argmax_val = output["argmax_val"].flatten()

    best_chunk = int(np.argmax(argmax_val))
    local_idx = int(argmax_idx[best_chunk])

    chunk_sizes = meta.get("lm_head_chunk_sizes", [])
    if chunk_sizes:
        offsets = [sum(chunk_sizes[:i]) for i in range(len(chunk_sizes))]
        return local_idx + offsets[best_chunk]

    # Fallback: uniform chunks
    vocab_size = meta.get("vocab_size", 0)
    num_chunks = len(argmax_idx)
    if num_chunks > 0 and vocab_size > 0:
        chunk_size = vocab_size // num_chunks
        return local_idx + best_chunk * chunk_size

    return local_idx


def decode_logits_output(
    output: Dict[str, Any], meta: Dict[str, Any], temperature: float = 0.0
) -> int:
    """Decode logits output to a token index."""
    split_lm_head = meta.get("split_lm_head", 8)

    # Check for split logits (logits1, logits2, ...)
    if "logits1" in output:
        parts = []
        for i in range(1, split_lm_head + 1):
            key = f"logits{i}"
            if key in output:
                parts.append(output[key])
        logits = np.concatenate(parts, axis=-1)
    elif "logits" in output:
        logits = output["logits"]
    else:
        # Find any key with 'logit' in the name
        logits = None
        for key in output:
            if "logit" in key.lower():
                logits = output[key]
                break
        if logits is None:
            raise ValueError(f"No logits found in output. Keys: {list(output.keys())}")

    # Flatten to last dim
    if logits.ndim == 3:
        last_logits = logits[0, -1, :]
    elif logits.ndim == 2:
        last_logits = logits[0, :]
    else:
        last_logits = logits.flatten()

    last_logits = last_logits.astype(np.float32)

    if temperature <= 0.0:
        return int(np.argmax(last_logits))

    scaled = last_logits / temperature
    scaled -= scaled.max()
    exp_vals = np.exp(scaled)
    probs = exp_vals / exp_vals.sum()
    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Update mask for batched prefill KV writes
# ---------------------------------------------------------------------------


def make_update_mask(
    mask_len: int, batch_pos: int, batch_size: int
) -> np.ndarray:
    """Create update mask for batched KV writes. Shape: (1, 1, mask_len, batch_size)."""
    mask = np.zeros((1, 1, mask_len, batch_size), dtype=np.float16)
    for i in range(batch_size):
        pos = batch_pos + i
        if pos < mask_len:
            mask[0, 0, pos, i] = 1.0
    return mask


def predict_with_optional_update_mask(model, inputs, state, update_mask):
    """Call model.predict, optionally adding update_mask if the model supports it."""
    if update_mask is None:
        return model.predict(inputs, state)

    supports = getattr(model, "_supports_update_mask", None)
    if supports is False:
        return model.predict(inputs, state)

    inputs_with_mask = dict(inputs)
    inputs_with_mask["update_mask"] = update_mask

    if supports is True:
        return model.predict(inputs_with_mask, state)

    try:
        output = model.predict(inputs_with_mask, state)
        model._supports_update_mask = True
        return output
    except RuntimeError as e:
        if "update_mask" in str(e):
            model._supports_update_mask = False
            return model.predict(inputs, state)
        raise


# ---------------------------------------------------------------------------
# Prefill (process prompt tokens -- fills KV cache, no output needed)
# ---------------------------------------------------------------------------


def prefill_tokens(
    models: Dict[str, Any],
    meta: Dict[str, Any],
    input_ids: List[int],
    causal_mask: np.ndarray,
) -> None:
    """Prefill the KV cache with prompt tokens.

    ANEMLL convention:
    1. Process full batch_size chunks via the 'prefill' function
    2. Process remaining tokens (< batch_size) one-at-a-time via 'infer'

    This only updates the KV cache state -- does NOT return a token.
    """
    ctx_len = meta["context_length"]
    batch_size = meta["batch_size"]
    layout = models["layout"]
    state = models["state"]

    num_tokens = len(input_ids)
    input_ids_np = np.array([input_ids], dtype=np.int32)  # [1, num_tokens]

    # --- Step 1: Full batches via prefill function ---
    batch_pos = 0
    while batch_pos + batch_size <= num_tokens:
        batch_end = batch_pos + batch_size
        batch_ids = input_ids_np[:, batch_pos:batch_end]
        position_ids = np.arange(batch_pos, batch_end, dtype=np.int32)
        batch_mask = causal_mask[:, :, batch_pos:batch_end, :].astype(np.float16)
        current_pos = np.array([batch_pos], dtype=np.int32)
        update_mask = make_update_mask(ctx_len, batch_pos, batch_size)

        if layout == "monolithic":
            prefill_model = models["prefill"]
            inputs = {
                "input_ids": batch_ids,
                "position_ids": position_ids,
                "causal_mask": batch_mask,
                "current_pos": current_pos,
            }
            predict_with_optional_update_mask(prefill_model, inputs, state, update_mask)
        else:
            embed_model = models["embed"]
            ffn_prefill = models["ffn_prefill"]
            hidden = embed_model.predict({"input_ids": batch_ids})["hidden_states"]
            inputs = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": position_ids,
                "causal_mask": batch_mask,
                "current_pos": current_pos,
            }
            predict_with_optional_update_mask(ffn_prefill, inputs, state, update_mask)

        batch_pos = batch_end

    # --- Step 2: Remaining tokens one-at-a-time via infer ---
    while batch_pos < num_tokens:
        token = input_ids_np[:, batch_pos : batch_pos + 1]
        position_ids = np.array([batch_pos], dtype=np.int32)
        single_mask = causal_mask[:, :, batch_pos : batch_pos + 1, :].astype(
            np.float16
        )
        current_pos = np.array([batch_pos], dtype=np.int32)

        if layout == "monolithic":
            infer_model = models["infer"]
            inputs = {
                "input_ids": token,
                "position_ids": position_ids,
                "causal_mask": single_mask,
                "current_pos": current_pos,
            }
            infer_model.predict(inputs, state)
        else:
            embed_model = models["embed"]
            ffn_infer = models["ffn_infer"]
            hidden = embed_model.predict({"input_ids": token})["hidden_states"]
            inputs = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": position_ids,
                "causal_mask": single_mask,
                "current_pos": current_pos,
            }
            ffn_infer.predict(inputs, state)

        batch_pos += 1


# ---------------------------------------------------------------------------
# Decode (generate one token at a time)
# ---------------------------------------------------------------------------


def decode_one_token(
    models: Dict[str, Any],
    meta: Dict[str, Any],
    token_id: int,
    pos: int,
    causal_mask: np.ndarray,
    temperature: float = 0.0,
) -> int:
    """Generate one token at position pos."""
    ctx_len = meta["context_length"]
    argmax_in_model = meta.get("argmax_in_model", False)
    layout = models["layout"]
    state = models["state"]

    token_array = np.array([[token_id]], dtype=np.int32)
    position_ids = np.array([pos], dtype=np.int32)
    single_mask = causal_mask[:, :, pos : pos + 1, :].astype(np.float16)
    current_pos = np.array([pos], dtype=np.int32)

    if layout == "monolithic":
        infer_model = models["infer"]
        inputs = {
            "input_ids": token_array,
            "position_ids": position_ids,
            "causal_mask": single_mask,
            "current_pos": current_pos,
        }
        output = infer_model.predict(inputs, state)
    else:
        embed_model = models["embed"]
        ffn_infer = models["ffn_infer"]
        lmhead_model = models["lmhead"]

        hidden = embed_model.predict({"input_ids": token_array})["hidden_states"]
        inputs = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": position_ids,
            "causal_mask": single_mask,
            "current_pos": current_pos,
        }
        output = ffn_infer.predict(inputs, state)
        hidden_out = output.get("output_hidden_states", list(output.values())[0])
        output = lmhead_model.predict(
            {"hidden_states": hidden_out.astype(np.float16)}
        )

    if argmax_in_model and "argmax_idx" in output:
        return decode_argmax_output(output, meta)
    else:
        return decode_logits_output(output, meta, temperature)


# ---------------------------------------------------------------------------
# Full generation
# ---------------------------------------------------------------------------


def generate(
    models: Dict[str, Any],
    meta: Dict[str, Any],
    tokenizer,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Run full generation: tokenize → prefill → decode loop.

    ANEMLL convention (1-indexed positions internally):
    1. Prefill all prompt tokens (fills KV cache, no output)
    2. Generate first token at pos = num_tokens (using infer + lmhead)
    3. Decode subsequent tokens one at a time
    """
    ctx_len = meta["context_length"]

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            formatted = prompt
    else:
        formatted = prompt

    input_ids = tokenizer.encode(formatted)
    tokens_in = len(input_ids)

    if tokens_in >= ctx_len:
        input_ids = input_ids[: ctx_len - max_tokens]
        tokens_in = len(input_ids)

    print(f"  Prompt tokens: {tokens_in}")

    # Create causal mask
    causal_mask = make_causal_mask(ctx_len)

    # Check for EOS
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_ids = set(eos_id)
    elif eos_id is not None:
        eos_ids = {eos_id}
    else:
        eos_ids = {2}

    # Step 1: Prefill (fills KV cache)
    t0 = time.time()
    prefill_tokens(models, meta, input_ids, causal_mask)

    # Step 2: Generate first token using the last prompt token
    # ANEMLL uses 1-indexed positions: pos=1 means position_ids=[0]
    # After prefill, we generate at pos = num_tokens (0-indexed)
    # which means feeding the last prompt token through generate_next_token
    last_token = input_ids[-1]
    next_token = decode_one_token(
        models, meta, last_token, tokens_in, causal_mask, temperature
    )
    ttft = time.time() - t0

    generated_ids = [next_token]
    pos = tokens_in + 1

    # Step 3: Decode loop
    t_decode_start = time.time()
    for step in range(max_tokens - 1):
        if next_token in eos_ids:
            break
        if pos >= ctx_len - 1:
            print(f"  Reached context limit ({ctx_len})")
            break

        next_token = decode_one_token(
            models, meta, next_token, pos, causal_mask, temperature
        )
        generated_ids.append(next_token)
        pos += 1

    t_end = time.time()

    total_time = t_end - t0
    decode_time = t_end - t_decode_start
    tokens_out = len(generated_ids)

    # Remove trailing EOS
    while generated_ids and generated_ids[-1] in eos_ids:
        generated_ids.pop()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    decode_tokens = max(tokens_out - 1, 1)
    tok_per_sec = decode_tokens / decode_time if decode_time > 0 else 0

    return {
        "text": text,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "ttft_ms": round(ttft * 1000, 1),
        "decode_ms": round(decode_time * 1000, 1),
        "total_ms": round(total_time * 1000, 1),
        "tok_per_sec": round(tok_per_sec, 1),
    }


# ---------------------------------------------------------------------------
# Smoke test runner
# ---------------------------------------------------------------------------


def run_smoke_test(
    model_dir: Path, cpu_only: bool = False, max_tokens: int = 32
) -> Dict[str, Any]:
    """Run a complete smoke test on one model."""
    print(f"\n{'='*72}")
    print(f"  ANE Smoke Test: {model_dir.name}")
    print(f"  Mode: {'CPU-only' if cpu_only else 'CPU + Neural Engine'}")
    print(f"{'='*72}")

    # Load meta
    meta = load_meta(model_dir)
    print(f"  Model: {meta['name']}")
    print(f"  Architecture: {meta['architecture']}")
    print(f"  Context: {meta['context_length']}")
    print(f"  Argmax in model: {meta['argmax_in_model']}")
    print(f"  Layout: {'monolithic' if meta.get('monolithic_model') else 'chunked'}")

    # Load tokenizer
    tokenizer = load_tokenizer(model_dir, meta)

    # Load models
    t_load = time.time()
    models = load_models(model_dir, meta, cpu_only=cpu_only)
    load_time = time.time() - t_load
    print(f"  Model load time: {load_time:.1f}s")

    # Run inference
    prompt = "What is 2 + 2? Answer briefly."
    print(f"\n  Prompt: {prompt}")

    result = generate(
        models, meta, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    print(f"\n  Output: {result['text'][:200]}")
    print(f"  Tokens in/out: {result['tokens_in']} / {result['tokens_out']}")
    print(f"  TTFT: {result['ttft_ms']:.0f} ms")
    print(f"  Decode: {result['decode_ms']:.0f} ms ({result['tok_per_sec']:.1f} tok/s)")
    print(f"  Total: {result['total_ms']:.0f} ms")

    # Determine pass/fail
    passed = (
        result["tokens_out"] > 0
        and result["ttft_ms"] < 30000  # 30s max for TTFT
        and len(result["text"].strip()) > 0
    )

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}")

    return {
        "model_dir": str(model_dir),
        "model_name": meta["name"],
        "architecture": meta["architecture"],
        "cpu_only": cpu_only,
        "load_time_s": round(load_time, 1),
        "status": status,
        **result,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="ANE smoke test for ANEMLL models")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-dir", type=Path, help="Path to ANEMLL model directory")
    group.add_argument(
        "--all", action="store_true", help="Test all models in models/ane/"
    )
    ap.add_argument(
        "--cpu", action="store_true", help="Run on CPU only (no ANE)"
    )
    ap.add_argument(
        "--max-tokens", type=int, default=32, help="Max tokens to generate (default: 32)"
    )
    ap.add_argument(
        "--models-root",
        type=Path,
        default=Path("models/ane"),
        help="Root directory for ANE models (default: models/ane)",
    )
    args = ap.parse_args()

    results = []

    if args.all:
        model_dirs = sorted(
            d for d in args.models_root.iterdir()
            if d.is_dir() and (d / "meta.yaml").exists()
        )
        if not model_dirs:
            print(f"No models found in {args.models_root}")
            return
        print(f"Found {len(model_dirs)} model(s) to test")
    else:
        model_dirs = [args.model_dir]

    for model_dir in model_dirs:
        try:
            result = run_smoke_test(
                model_dir, cpu_only=args.cpu, max_tokens=args.max_tokens
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "model_dir": str(model_dir),
                "model_name": model_dir.name,
                "status": "ERROR",
                "error": str(e),
            })

    # Summary
    print(f"\n{'='*72}")
    print(f"  Smoke Test Summary")
    print(f"{'='*72}")
    print(f"  {'Model':<35} {'Status':<8} {'TTFT':>8} {'tok/s':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        name = r.get("model_name", "?")[:35]
        status = r.get("status", "?")
        ttft = f"{r['ttft_ms']:.0f}ms" if "ttft_ms" in r else "-"
        tps = f"{r['tok_per_sec']:.1f}" if "tok_per_sec" in r else "-"
        print(f"  {name:<35} {status:<8} {ttft:>8} {tps:>8}")

    # Save results
    out_path = Path("results/ane_smoke_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Exit with error if any failed
    failures = [r for r in results if r["status"] not in ("PASS",)]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
