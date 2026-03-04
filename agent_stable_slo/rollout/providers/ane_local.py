"""
ANE (Apple Neural Engine) local inference provider.

Uses coremltools to load Anemll-converted CoreML models and runs inference
on the Apple Neural Engine via CoreML's predict API with stateful KV cache.

Model layout conventions (Anemll):
  - Chunked: *_embeddings + *_FFN_PF_* chunks + *_lm_head, each separate
  - Monolithic: single .mlmodelc with infer/prefill functions

Configuration via environment variables:
  - ANE_META_DIR: path to Anemll-converted model directory (contains meta.yaml)
  - ANE_TOKENIZER or ANE_HF_MODEL: HuggingFace tokenizer name/path
  - ANE_MAX_TOKENS: max generation tokens (default 256)
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Cache loaded models and tokenizers to avoid reloading
_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Tokenizer management
# ---------------------------------------------------------------------------


def _get_tokenizer(model_name: str):
    """Load or retrieve cached HuggingFace tokenizer."""
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]

    from transformers import AutoTokenizer

    print(f"[ane_local] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


# ---------------------------------------------------------------------------
# Meta.yaml parsing
# ---------------------------------------------------------------------------


def _load_meta_yaml(meta_path: Path) -> Dict[str, Any]:
    """Parse meta.yaml from Anemll model directory.

    Returns dict with all ANEMLL parameters needed for inference.
    """
    defaults = {
        "context_length": 2048,
        "batch_size": 64,
        "num_chunks": 0,
        "argmax_in_model": False,
        "split_lm_head": 8,
        "vocab_size": 0,
        "lm_head_chunk_sizes": [],
        "monolithic_model": "",
        "embeddings": "",
        "lm_head": "",
        "ffn": "",
    }
    if not meta_path.exists():
        return defaults

    try:
        import yaml

        with open(meta_path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            return defaults

        model_info = raw.get("model_info", {})
        params = model_info.get("parameters", {})

        return {
            "context_length": params.get("context_length", defaults["context_length"]),
            "batch_size": params.get("batch_size", defaults["batch_size"]),
            "num_chunks": params.get("num_chunks", defaults["num_chunks"]),
            "argmax_in_model": params.get("argmax_in_model", defaults["argmax_in_model"]),
            "split_lm_head": params.get("split_lm_head", defaults["split_lm_head"]),
            "vocab_size": params.get("vocab_size", defaults["vocab_size"]),
            "lm_head_chunk_sizes": params.get("lm_head_chunk_sizes", defaults["lm_head_chunk_sizes"]),
            "monolithic_model": params.get("monolithic_model", ""),
            "embeddings": params.get("embeddings", ""),
            "lm_head": params.get("lm_head", ""),
            "ffn": params.get("ffn", ""),
            "model_type": model_info.get("model_type", ""),
        }
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# CoreML model loading
# ---------------------------------------------------------------------------


def _load_coreml(path: Path, function_name: str = None, compute_unit=None):
    """Load a CoreML model, handling both .mlmodelc and .mlpackage.

    If the requested path doesn't exist, tries the other extension.
    """
    import coremltools as ct

    if compute_unit is None:
        compute_unit = ct.ComputeUnit.CPU_AND_NE

    # Try alternate extension if path doesn't exist
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
                mlpackage = path.with_suffix(".mlpackage")
                if mlpackage.exists():
                    return ct.models.MLModel(str(mlpackage), function_name=function_name)
                raise
        return ct.models.CompiledMLModel(str(path), compute_unit)
    else:
        if function_name:
            return ct.models.MLModel(str(path), function_name=function_name)
        return ct.models.MLModel(str(path))


def _get_ane_model(meta_dir: str) -> Dict[str, Any]:
    """Load or retrieve cached CoreML model components from Anemll directory.

    Reads meta.yaml for model metadata and file names, then loads the
    appropriate model components (chunked or monolithic).

    Returns dict with:
      - 'metadata': parsed meta.yaml info
      - 'layout': 'chunked' or 'monolithic'
      - For monolithic: 'infer', 'prefill', 'state'
      - For chunked: 'embed', 'ffn_infer', 'ffn_prefill', 'lmhead', 'state'
    """
    if meta_dir in _MODEL_CACHE:
        return _MODEL_CACHE[meta_dir]

    import coremltools as ct

    meta_path = Path(meta_dir) / "meta.yaml"
    metadata = _load_meta_yaml(meta_path)

    cu = ct.ComputeUnit.CPU_AND_NE
    model_dir = Path(meta_dir)

    is_monolithic = bool(metadata.get("monolithic_model"))

    if is_monolithic:
        model_file = metadata["monolithic_model"]
        model_path = model_dir / model_file

        print(f"[ane_local] Loading monolithic model: {model_path.name}")
        infer_model = _load_coreml(model_path, "infer", cu)
        prefill_model = _load_coreml(model_path, "prefill", cu)
        state = infer_model.make_state()

        components = {
            "metadata": metadata,
            "layout": "monolithic",
            "infer": infer_model,
            "prefill": prefill_model,
            "state": state,
        }
    else:
        embed_file = metadata.get("embeddings", "")
        lmhead_file = metadata.get("lm_head", "")
        ffn_file = metadata.get("ffn", "")

        if not embed_file or not lmhead_file or not ffn_file:
            raise ValueError(
                f"Chunked model missing file paths in meta.yaml: "
                f"embeddings={embed_file!r}, lm_head={lmhead_file!r}, ffn={ffn_file!r}"
            )

        embed_path = model_dir / embed_file
        lmhead_path = model_dir / lmhead_file
        ffn_path = model_dir / ffn_file

        print(f"[ane_local] Loading chunked model from: {meta_dir}")
        embed_model = _load_coreml(embed_path, compute_unit=cu)
        lmhead_model = _load_coreml(lmhead_path, compute_unit=cu)
        ffn_infer = _load_coreml(ffn_path, "infer", cu)
        ffn_prefill = _load_coreml(ffn_path, "prefill", cu)
        state = ffn_infer.make_state()

        components = {
            "metadata": metadata,
            "layout": "chunked",
            "embed": embed_model,
            "lmhead": lmhead_model,
            "ffn_infer": ffn_infer,
            "ffn_prefill": ffn_prefill,
            "state": state,
        }

    _MODEL_CACHE[meta_dir] = components
    return components


# ---------------------------------------------------------------------------
# Prompt / JSON helpers (same contract as MLX provider)
# ---------------------------------------------------------------------------


def _build_prompt_with_schema(prompt: str, schema: dict) -> str:
    """Add JSON schema instruction to prompt."""
    example = {}
    for prop, spec in schema.get("properties", {}).items():
        typ = spec.get("type", "string")
        if typ == "string":
            example[prop] = "<value>"
        elif typ == "array":
            example[prop] = []
        elif typ == "object":
            example[prop] = {}
        elif typ in ("number", "integer"):
            example[prop] = 0

    return f"""{prompt}

Respond with ONLY a JSON object like this example:
{json.dumps(example)}

Your JSON response (no explanation, just the JSON):"""


def _extract_json(raw_text: str) -> dict:
    """Extract JSON from raw model output."""
    try:
        obj = json.loads(raw_text.strip())
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    if "```" in raw_text:
        parts = raw_text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            try:
                obj = json.loads(p)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

    matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw_text)
    for m in matches:
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return {}


def _backfill_required(parsed: dict, schema: dict) -> dict:
    """Ensure all required fields exist with type-appropriate defaults."""
    if not isinstance(parsed, dict):
        return parsed
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for req in required:
        if req not in parsed:
            typ = props.get(req, {}).get("type")
            if typ == "string":
                parsed[req] = ""
            elif typ == "array":
                parsed[req] = []
            elif typ == "object":
                parsed[req] = {}
            elif typ == "boolean":
                parsed[req] = False
            elif typ in ("number", "integer"):
                parsed[req] = 0
    return parsed


# ---------------------------------------------------------------------------
# Causal mask + update mask
# ---------------------------------------------------------------------------


def _make_causal_mask(context_length: int) -> np.ndarray:
    """Create full causal attention mask. Shape: (1, 1, ctx, ctx)."""
    mask = np.zeros((1, 1, context_length, context_length), dtype=np.float16)
    for i in range(context_length):
        mask[0, 0, i, i + 1:] = np.float16(-65504.0)
    return mask


def _make_update_mask(mask_len: int, batch_pos: int, batch_size: int) -> np.ndarray:
    """Create update mask for batched KV writes. Shape: (1, 1, mask_len, batch_size)."""
    mask = np.zeros((1, 1, mask_len, batch_size), dtype=np.float16)
    for i in range(batch_size):
        pos = batch_pos + i
        if pos < mask_len:
            mask[0, 0, pos, i] = 1.0
    return mask


def _predict_with_update_mask(model, inputs, state, update_mask):
    """Call model.predict, optionally adding update_mask if supported."""
    if update_mask is None:
        return model.predict(inputs, state)

    supports = getattr(model, "_supports_update_mask", None)
    if supports is False:
        return model.predict(inputs, state)

    inputs_with = dict(inputs)
    inputs_with["update_mask"] = update_mask

    if supports is True:
        return model.predict(inputs_with, state)

    try:
        output = model.predict(inputs_with, state)
        model._supports_update_mask = True
        return output
    except RuntimeError as e:
        if "update_mask" in str(e):
            model._supports_update_mask = False
            return model.predict(inputs, state)
        raise


# ---------------------------------------------------------------------------
# Argmax-in-model output decoding
# ---------------------------------------------------------------------------


def _decode_argmax_output(output: Dict[str, Any], metadata: Dict[str, Any]) -> int:
    """Decode argmax_idx/argmax_val output to a global token index."""
    argmax_idx = output["argmax_idx"].flatten()
    argmax_val = output["argmax_val"].flatten()

    best_chunk = int(np.argmax(argmax_val))
    local_idx = int(argmax_idx[best_chunk])

    chunk_sizes = metadata.get("lm_head_chunk_sizes", [])
    if chunk_sizes:
        offsets = [sum(chunk_sizes[:i]) for i in range(len(chunk_sizes))]
        return local_idx + offsets[best_chunk]

    vocab_size = metadata.get("vocab_size", 0)
    num_chunks = len(argmax_idx)
    if num_chunks > 0 and vocab_size > 0:
        chunk_size = vocab_size // num_chunks
        return local_idx + best_chunk * chunk_size

    return local_idx


def _decode_logits_output(
    output: Dict[str, Any], metadata: Dict[str, Any], temperature: float = 0.0
) -> int:
    """Decode logits output (possibly split) to a token index."""
    split = metadata.get("split_lm_head", 8)

    if "logits1" in output:
        parts = []
        for i in range(1, split + 1):
            key = f"logits{i}"
            if key in output:
                parts.append(output[key])
        logits = np.concatenate(parts, axis=-1)
    elif "logits" in output:
        logits = output["logits"]
    else:
        logits = None
        for key in output:
            if "logit" in key.lower():
                logits = output[key]
                break
        if logits is None:
            raise ValueError(f"No logits in output. Keys: {list(output.keys())}")

    if logits.ndim == 3:
        last = logits[0, -1, :]
    elif logits.ndim == 2:
        last = logits[0, :]
    else:
        last = logits.flatten()

    last = last.astype(np.float32)

    if temperature <= 0.0:
        return int(np.argmax(last))

    scaled = last / temperature
    scaled -= scaled.max()
    exp_vals = np.exp(scaled)
    probs = exp_vals / exp_vals.sum()
    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Token generation (ANE CoreML) -- ANEMLL convention
# ---------------------------------------------------------------------------


def _prefill(
    components: Dict[str, Any],
    input_ids: List[int],
    causal_mask: np.ndarray,
) -> None:
    """Fill KV cache with prompt tokens.

    ANEMLL convention:
    1. Full batch_size chunks via prefill function
    2. Remaining tokens one-at-a-time via infer function
    """
    metadata = components["metadata"]
    ctx_len = metadata["context_length"]
    batch_size = metadata["batch_size"]
    layout = components["layout"]
    state = components["state"]

    num_tokens = len(input_ids)
    ids_np = np.array([input_ids], dtype=np.int32)

    batch_pos = 0

    # Full batches via prefill
    while batch_pos + batch_size <= num_tokens:
        batch_end = batch_pos + batch_size
        batch_ids = ids_np[:, batch_pos:batch_end]
        pos_ids = np.arange(batch_pos, batch_end, dtype=np.int32)
        batch_mask = causal_mask[:, :, batch_pos:batch_end, :].astype(np.float16)
        cur_pos = np.array([batch_pos], dtype=np.int32)
        update_mask = _make_update_mask(ctx_len, batch_pos, batch_size)

        if layout == "monolithic":
            inputs = {
                "input_ids": batch_ids,
                "position_ids": pos_ids,
                "causal_mask": batch_mask,
                "current_pos": cur_pos,
            }
            _predict_with_update_mask(components["prefill"], inputs, state, update_mask)
        else:
            hidden = components["embed"].predict({"input_ids": batch_ids})["hidden_states"]
            inputs = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_ids,
                "causal_mask": batch_mask,
                "current_pos": cur_pos,
            }
            _predict_with_update_mask(components["ffn_prefill"], inputs, state, update_mask)

        batch_pos = batch_end

    # Remaining tokens via infer
    while batch_pos < num_tokens:
        token = ids_np[:, batch_pos:batch_pos + 1]
        pos_ids = np.array([batch_pos], dtype=np.int32)
        single_mask = causal_mask[:, :, batch_pos:batch_pos + 1, :].astype(np.float16)
        cur_pos = np.array([batch_pos], dtype=np.int32)

        if layout == "monolithic":
            inputs = {
                "input_ids": token,
                "position_ids": pos_ids,
                "causal_mask": single_mask,
                "current_pos": cur_pos,
            }
            components["infer"].predict(inputs, state)
        else:
            hidden = components["embed"].predict({"input_ids": token})["hidden_states"]
            inputs = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_ids,
                "causal_mask": single_mask,
                "current_pos": cur_pos,
            }
            components["ffn_infer"].predict(inputs, state)

        batch_pos += 1


def _generate_token(
    components: Dict[str, Any],
    token_id: int,
    pos: int,
    causal_mask: np.ndarray,
    temperature: float = 0.0,
) -> int:
    """Generate one token at position pos using infer function + lmhead."""
    metadata = components["metadata"]
    layout = components["layout"]
    state = components["state"]
    argmax_in_model = metadata.get("argmax_in_model", False)

    token = np.array([[token_id]], dtype=np.int32)
    pos_ids = np.array([pos], dtype=np.int32)
    single_mask = causal_mask[:, :, pos:pos + 1, :].astype(np.float16)
    cur_pos = np.array([pos], dtype=np.int32)

    if layout == "monolithic":
        inputs = {
            "input_ids": token,
            "position_ids": pos_ids,
            "causal_mask": single_mask,
            "current_pos": cur_pos,
        }
        output = components["infer"].predict(inputs, state)
    else:
        hidden = components["embed"].predict({"input_ids": token})["hidden_states"]
        inputs = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": pos_ids,
            "causal_mask": single_mask,
            "current_pos": cur_pos,
        }
        output = components["ffn_infer"].predict(inputs, state)
        hidden_out = output.get("output_hidden_states", list(output.values())[0])
        output = components["lmhead"].predict(
            {"hidden_states": hidden_out.astype(np.float16)}
        )

    if argmax_in_model and "argmax_idx" in output:
        return _decode_argmax_output(output, metadata)
    return _decode_logits_output(output, metadata, temperature)


def _ane_generate(
    components: Dict[str, Any],
    tokenizer,
    input_ids: list,
    max_tokens: int,
    temperature: float,
    metadata: Dict[str, Any],
) -> Tuple[str, float, float, int]:
    """Run generation on ANE via CoreML using ANEMLL convention.

    1. Prefill all prompt tokens (fills KV cache)
    2. Generate first token at pos = num_tokens
    3. Decode subsequent tokens one at a time

    Returns (generated_text, latency_ms, ttft_ms, tokens_out).
    """
    ctx_len = metadata["context_length"]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_ids = set(eos_id)
    elif eos_id is not None:
        eos_ids = {eos_id}
    else:
        eos_ids = {2}

    # Create causal mask
    causal_mask = _make_causal_mask(ctx_len)

    num_tokens = len(input_ids)

    # Prefill
    t0 = time.time()
    _prefill(components, input_ids, causal_mask)

    # Generate first token
    last_token = input_ids[-1]
    next_token = _generate_token(
        components, last_token, num_tokens, causal_mask, temperature
    )
    ttft_ms = (time.time() - t0) * 1000.0

    generated_ids = [next_token]
    pos = num_tokens + 1

    # Decode loop
    for _ in range(max_tokens - 1):
        if next_token in eos_ids:
            break
        if pos >= ctx_len - 1:
            break

        next_token = _generate_token(
            components, next_token, pos, causal_mask, temperature
        )
        generated_ids.append(next_token)
        pos += 1

    latency_ms = (time.time() - t0) * 1000.0

    # Remove trailing EOS
    while generated_ids and generated_ids[-1] in eos_ids:
        generated_ids.pop()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, latency_ms, ttft_ms, len(generated_ids)


# ---------------------------------------------------------------------------
# Public API (same contract as mlx_local.generate_raw)
# ---------------------------------------------------------------------------


def generate_raw(
    prompt: str,
    schema: dict,
    mode: str = "structured",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Tuple[str, dict, float, float, int, int]:
    """Generate response using local CoreML model on Apple Neural Engine.

    Returns (raw_text, parsed_json, latency_ms, ttft_ms, tokens_in, tokens_out).
    """
    meta_dir = os.getenv("ANE_META_DIR", "")
    if not meta_dir:
        raise ValueError("ANE_META_DIR environment variable must be set")
    tokenizer_name = os.getenv("ANE_TOKENIZER") or os.getenv("ANE_HF_MODEL", "")
    if not tokenizer_name:
        raise ValueError("ANE_TOKENIZER or ANE_HF_MODEL environment variable must be set")
    max_new = max_tokens or int(os.getenv("ANE_MAX_TOKENS", "256"))

    tokenizer = _get_tokenizer(tokenizer_name)
    components = _get_ane_model(meta_dir)
    metadata = components.get("metadata", {})

    # Build prompt with schema
    full_prompt = _build_prompt_with_schema(prompt, schema)

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": full_prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            formatted = full_prompt
    else:
        formatted = full_prompt

    # Tokenize
    input_ids = tokenizer.encode(formatted)
    tokens_in = len(input_ids)

    # Truncate if needed
    ctx_len = metadata.get("context_length", 2048)
    if tokens_in >= ctx_len:
        input_ids = input_ids[:ctx_len - max_new]
        tokens_in = len(input_ids)

    # Generate
    raw_text, latency_ms, ttft_ms, tokens_out = _ane_generate(
        components, tokenizer, input_ids, max_new, temperature, metadata
    )

    # Parse JSON from response
    parsed = _extract_json(raw_text)
    parsed = _backfill_required(parsed, schema)

    return raw_text, parsed, latency_ms, ttft_ms, tokens_in, tokens_out
