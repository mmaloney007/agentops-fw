import os
import json
import time
from typing import Optional, Tuple
from ollama import chat


def _max_tokens(default: Optional[int] = None):
    try:
        val = int(os.getenv("MAX_THOUGHT_TOKENS", str(default or 512)))
        return val if val > 0 else None
    except Exception:
        return default or 512


def generate_raw(
    prompt: str,
    schema: dict,
    mode: str = "structured",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Tuple[str, dict, float, float, int, int]:
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    t0 = time.time()
    fmt = schema if mode == "structured" else None
    if mode == "grammar":
        # Ollama supports "format":"json" with prompt hints; use schema as format fallback
        fmt = schema
    max_new = max_tokens if max_tokens is not None else _max_tokens()
    resp = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=fmt,
        options={"temperature": temperature, "num_predict": max_new},
    )
    lat_ms = (time.time() - t0) * 1000.0
    content = resp.get("message", {}).get("content")
    if not content:
        return "", {}, lat_ms, lat_ms, -1, -1
    try:
        j = json.loads(content)
    except json.JSONDecodeError:
        j = {}
    return content, j, lat_ms, lat_ms, -1, -1


def generate_json(
    prompt: str,
    schema: dict,
    mode: str = "structured",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Tuple[dict, float, float, int]:
    _raw, parsed, lat_ms, ttft_ms, _tokens_in, tokens_out = generate_raw(
        prompt, schema, mode=mode, temperature=temperature, max_tokens=max_tokens
    )
    return parsed, lat_ms, ttft_ms, tokens_out
