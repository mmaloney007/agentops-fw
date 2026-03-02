"""
MLX local inference provider for Apple Silicon.
Uses mlx-lm to load quantized models directly on Metal GPU.
"""

import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

# Cache loaded models to avoid reloading
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _patch_qwen35_sanitizer() -> None:
    """Patch qwen3_5_moe sanitizer to drop vision_tower weights."""
    try:
        from mlx_lm.models.qwen3_5_moe import Model as Qwen35MoE

        _orig = Qwen35MoE.sanitize

        def _patched(self, weights):
            weights = {
                k: v
                for k, v in weights.items()
                if "vision_tower" not in k and "visual" not in k
            }
            return _orig(self, weights)

        if not getattr(Qwen35MoE.sanitize, "_patched", False):
            Qwen35MoE.sanitize = _patched
            Qwen35MoE.sanitize._patched = True  # type: ignore[attr-defined]
    except ImportError:
        pass


def _get_model_and_tokenizer(model_path: str):
    """Load or retrieve cached model and tokenizer via mlx-lm."""
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    _patch_qwen35_sanitizer()

    from mlx_lm import load

    print(f"[mlx_local] Loading model: {model_path}")
    model, tokenizer = load(model_path)
    _MODEL_CACHE[model_path] = (model, tokenizer)
    return model, tokenizer


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


def get_memory_stats() -> Dict[str, float]:
    """Get Metal GPU memory stats (Apple Silicon only)."""
    try:
        import mlx.core as mx

        _get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
        _get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        _get_cache = getattr(mx, "get_cache_memory", None) or mx.metal.get_cache_memory
        stats = {
            "active_memory_gb": _get_active() / (1024**3),
            "peak_memory_gb": _get_peak() / (1024**3),
            "cache_memory_gb": _get_cache() / (1024**3),
        }
        return stats
    except Exception:
        return {}


def generate_raw(
    prompt: str,
    schema: dict,
    mode: str = "structured",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Tuple[str, dict, float, float, int, int]:
    """Generate response using local MLX model on Apple Silicon."""
    from mlx_lm import generate as mlx_generate

    model_path = os.getenv("MLX_MODEL", "mlx-community/Llama-3.2-1B-Instruct-4bit")
    max_new = max_tokens or int(os.getenv("MLX_MAX_TOKENS", "2048"))

    model, tokenizer = _get_model_and_tokenizer(model_path)

    # Build prompt with schema
    full_prompt = _build_prompt_with_schema(prompt, schema)

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": full_prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = full_prompt

    # Count input tokens
    input_ids = tokenizer.encode(formatted)
    tokens_in = len(input_ids)

    # Generate
    t0 = time.time()

    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(
        temp=temperature if temperature > 0 else 0.0,
        top_p=0.95 if temperature > 0 else 0.0,
    )
    raw_text = mlx_generate(
        model, tokenizer, prompt=formatted,
        max_tokens=max_new, verbose=False, sampler=sampler,
    )

    lat_ms = (time.time() - t0) * 1000.0
    ttft_ms = lat_ms  # No streaming, so TTFT = total latency

    # Count output tokens
    output_ids = tokenizer.encode(raw_text)
    tokens_out = len(output_ids)

    # Parse JSON from response
    parsed = _extract_json(raw_text)

    # Ensure required fields exist
    if isinstance(parsed, dict):
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

    return raw_text, parsed, lat_ms, ttft_ms, tokens_in, tokens_out
