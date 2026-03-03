"""
ANE (Apple Neural Engine) local inference provider.

Uses coremltools to load Anemll-converted CoreML models and runs inference
on the Apple Neural Engine via CoreML's predict API with stateful KV cache.

Model layout conventions (Anemll):
  - Chunked: embed.mlmodelc + N FFN chunks + lmhead.mlmodelc, each separate
  - Monolithic: single .mlmodelc with multiple functions

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
from typing import Any, Dict, Optional, Tuple

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
# CoreML model loading
# ---------------------------------------------------------------------------


def _get_ane_model(meta_dir: str) -> Dict[str, Any]:
    """Load or retrieve cached CoreML model components from Anemll directory.

    Reads meta.yaml for model metadata, then discovers and loads either
    chunked (.mlmodelc per component) or monolithic model layout.

    Returns dict with:
      - 'metadata': parsed meta.yaml info (context_length, num_chunks, etc.)
      - 'embed': embedding model (chunked) or None
      - 'chunks': list of FFN chunk models (chunked) or []
      - 'lmhead': lm_head model (chunked) or None
      - 'model': monolithic model (monolithic) or None
      - 'layout': 'chunked' or 'monolithic'
    """
    if meta_dir in _MODEL_CACHE:
        return _MODEL_CACHE[meta_dir]

    import coremltools as ct

    meta_path = Path(meta_dir) / "meta.yaml"
    metadata = _load_meta_yaml(meta_path)

    compute_unit = ct.ComputeUnit.CPU_AND_NE
    model_dir = Path(meta_dir)

    # Detect layout: chunked vs monolithic
    embed_path = model_dir / "embed.mlmodelc"
    lmhead_path = model_dir / "lmhead.mlmodelc"

    if embed_path.exists() and lmhead_path.exists():
        # Chunked layout
        print(f"[ane_local] Loading chunked model from: {meta_dir}")
        embed = ct.models.CompiledMLModel(str(embed_path), compute_unit)
        lmhead = ct.models.CompiledMLModel(str(lmhead_path), compute_unit)

        # Discover FFN chunks (ffn_chunk_0.mlmodelc, ffn_chunk_1.mlmodelc, ...)
        chunk_paths = sorted(model_dir.glob("ffn_chunk_*.mlmodelc"))
        chunks = [
            ct.models.CompiledMLModel(str(cp), compute_unit) for cp in chunk_paths
        ]

        components = {
            "metadata": metadata,
            "embed": embed,
            "chunks": chunks,
            "lmhead": lmhead,
            "model": None,
            "layout": "chunked",
        }
    else:
        # Monolithic layout: single model file
        print(f"[ane_local] Loading monolithic model from: {meta_dir}")
        model_paths = list(model_dir.glob("*.mlmodelc"))
        if not model_paths:
            raise FileNotFoundError(
                f"No .mlmodelc files found in {meta_dir}"
            )
        model = ct.models.CompiledMLModel(str(model_paths[0]), compute_unit)
        components = {
            "metadata": metadata,
            "embed": None,
            "chunks": [],
            "lmhead": None,
            "model": model,
            "layout": "monolithic",
        }

    _MODEL_CACHE[meta_dir] = components
    return components


def _load_meta_yaml(meta_path: Path) -> Dict[str, Any]:
    """Parse meta.yaml from Anemll model directory.

    Falls back to sensible defaults if file is missing.
    """
    defaults = {
        "context_length": 2048,
        "num_chunks": 0,
    }
    if not meta_path.exists():
        return defaults

    try:
        import yaml

        with open(meta_path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            return defaults

        # Extract from model_info.parameters.* if present
        model_info = raw.get("model_info", {})
        params = model_info.get("parameters", {})
        return {
            "context_length": params.get("context_length", defaults["context_length"]),
            "num_chunks": params.get("num_chunks", defaults["num_chunks"]),
            **{k: v for k, v in raw.items() if k != "model_info"},
        }
    except Exception:
        return defaults


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
    # Try direct parse
    try:
        obj = json.loads(raw_text.strip())
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try code blocks
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

    # Try to find JSON objects
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
# Causal mask
# ---------------------------------------------------------------------------


def _make_causal_mask(length: int, start: int = 0) -> np.ndarray:
    """Create causal attention mask as float16 numpy array.

    Shape: (1, 1, length, start + length)
    Lower triangle + diagonal = 0.0, upper triangle = -65504.0 (fp16 min).
    """
    total = start + length
    mask = np.zeros((1, 1, length, total), dtype=np.float16)
    for i in range(length):
        mask[0, 0, i, start + i + 1 :] = np.float16(-65504.0)
    return mask


# ---------------------------------------------------------------------------
# Token generation (ANE CoreML)
# ---------------------------------------------------------------------------


def _ane_generate(
    model_components: Dict[str, Any],
    tokenizer,
    input_ids: list,
    max_tokens: int,
    temperature: float,
    metadata: Dict[str, Any],
) -> Tuple[str, float, float, int]:
    """Run autoregressive generation on ANE via CoreML.

    Returns (generated_text, latency_ms, ttft_ms, tokens_out).
    Dispatches to chunked or monolithic generation based on layout.
    """
    layout = model_components.get("layout", "monolithic")

    if layout == "chunked":
        return _generate_chunked(
            model_components, tokenizer, input_ids, max_tokens, temperature, metadata
        )
    else:
        return _generate_monolithic(
            model_components, tokenizer, input_ids, max_tokens, temperature, metadata
        )


def _generate_chunked(
    components: Dict[str, Any],
    tokenizer,
    input_ids: list,
    max_tokens: int,
    temperature: float,
    metadata: Dict[str, Any],
) -> Tuple[str, float, float, int]:
    """Token-by-token generation using chunked CoreML models (embed + FFN chunks + lmhead).

    Each step:
    1. Embed current token(s)
    2. Pass through each FFN chunk with KV cache state
    3. Project via lmhead to get logits
    4. Sample next token
    """
    embed = components["embed"]
    chunks = components["chunks"]
    lmhead = components["lmhead"]

    eos_token_id = getattr(tokenizer, "eos_token_id", 2)
    generated_ids = []

    t0 = time.time()
    ttft = None

    # Create states for each chunk
    states = [chunk.make_state() for chunk in chunks]

    seq_len = len(input_ids)

    for step in range(max_tokens):
        if step == 0:
            # Prefill: process all input tokens at once
            token_array = np.array([input_ids], dtype=np.int32)
            causal_mask = _make_causal_mask(seq_len, start=0)
            pos_ids = np.arange(seq_len, dtype=np.int32).reshape(1, -1)
        else:
            # Decode: one token at a time
            token_array = np.array([[generated_ids[-1]]], dtype=np.int32)
            pos = seq_len + step - 1
            causal_mask = _make_causal_mask(1, start=pos)
            pos_ids = np.array([[pos]], dtype=np.int32)

        # Embedding
        embed_out = embed.predict({"input_ids": token_array})
        hidden = embed_out.get("hidden_states", list(embed_out.values())[0])

        # FFN chunks
        for chunk, state in zip(chunks, states):
            chunk_input = {
                "hidden_states": hidden,
                "attention_mask": causal_mask,
                "position_ids": pos_ids,
            }
            chunk_out = chunk.predict(chunk_input, state)
            hidden = chunk_out.get("hidden_states", list(chunk_out.values())[0])

        # LM head
        logits_out = lmhead.predict({"hidden_states": hidden})
        logits = logits_out.get("logits", list(logits_out.values())[0])

        # Sample
        next_token = _sample_token(logits, temperature)

        if ttft is None:
            ttft = (time.time() - t0) * 1000.0

        if next_token == eos_token_id:
            break

        generated_ids.append(next_token)

    latency_ms = (time.time() - t0) * 1000.0
    ttft_ms = ttft if ttft is not None else latency_ms

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, latency_ms, ttft_ms, len(generated_ids)


def _generate_monolithic(
    components: Dict[str, Any],
    tokenizer,
    input_ids: list,
    max_tokens: int,
    temperature: float,
    metadata: Dict[str, Any],
) -> Tuple[str, float, float, int]:
    """Token-by-token generation using a monolithic CoreML model.

    Single model handles embed + attention + lm_head in one predict call.
    """
    model = components["model"]
    eos_token_id = getattr(tokenizer, "eos_token_id", 2)
    generated_ids = []

    t0 = time.time()
    ttft = None

    # Create KV cache state
    state = model.make_state()

    seq_len = len(input_ids)

    for step in range(max_tokens):
        if step == 0:
            # Prefill
            token_array = np.array([input_ids], dtype=np.int32)
            causal_mask = _make_causal_mask(seq_len, start=0)
            pos_ids = np.arange(seq_len, dtype=np.int32).reshape(1, -1)
        else:
            # Decode
            token_array = np.array([[generated_ids[-1]]], dtype=np.int32)
            pos = seq_len + step - 1
            causal_mask = _make_causal_mask(1, start=pos)
            pos_ids = np.array([[pos]], dtype=np.int32)

        inputs = {
            "input_ids": token_array,
            "attention_mask": causal_mask,
            "position_ids": pos_ids,
        }
        output = model.predict(inputs, state)
        logits = output.get("logits", list(output.values())[0])

        # Sample
        next_token = _sample_token(logits, temperature)

        if ttft is None:
            ttft = (time.time() - t0) * 1000.0

        if next_token == eos_token_id:
            break

        generated_ids.append(next_token)

    latency_ms = (time.time() - t0) * 1000.0
    ttft_ms = ttft if ttft is not None else latency_ms

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, latency_ms, ttft_ms, len(generated_ids)


def _sample_token(logits: np.ndarray, temperature: float) -> int:
    """Sample next token from logits.

    Greedy (argmax) if temperature == 0, otherwise softmax + multinomial.
    Takes the last position's logits.
    """
    # logits shape is typically (1, seq_len, vocab) or (1, vocab)
    if logits.ndim == 3:
        last_logits = logits[0, -1, :]
    elif logits.ndim == 2:
        last_logits = logits[0, :]
    else:
        last_logits = logits.flatten()

    last_logits = last_logits.astype(np.float32)

    if temperature <= 0.0:
        return int(np.argmax(last_logits))

    # Temperature-scaled softmax sampling
    scaled = last_logits / temperature
    scaled -= scaled.max()  # numerical stability
    exp_vals = np.exp(scaled)
    probs = exp_vals / exp_vals.sum()
    return int(np.random.choice(len(probs), p=probs))


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
    model_components = _get_ane_model(meta_dir)
    metadata = model_components.get("metadata", {})

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

    # Generate
    raw_text, latency_ms, ttft_ms, tokens_out = _ane_generate(
        model_components, tokenizer, input_ids, max_new, temperature, metadata
    )

    # Parse JSON from response
    parsed = _extract_json(raw_text)

    # Backfill required fields
    parsed = _backfill_required(parsed, schema)

    return raw_text, parsed, latency_ms, ttft_ms, tokens_in, tokens_out
