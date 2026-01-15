"""
Direct HuggingFace local inference provider.
Uses transformers to load models directly without needing a server.
"""

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Cache loaded models to avoid reloading
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _get_model_and_tokenizer(model_path: str):
    """Load or retrieve cached model and tokenizer."""
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    print(f"[hf_local] Loading model: {model_path}")

    # Determine if we should use 4-bit quantization based on model size
    use_4bit = os.getenv("HF_LOCAL_4BIT", "auto").lower()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if use_4bit == "true" or (use_4bit == "auto" and "12b" in model_path.lower()):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    _MODEL_CACHE[model_path] = (model, tokenizer)
    return model, tokenizer


def _build_prompt_with_schema(prompt: str, schema: dict) -> str:
    """Add JSON schema instruction to prompt."""
    # Build example from schema
    example = {}
    for prop, spec in schema.get("properties", {}).items():
        typ = spec.get("type", "string")
        if typ == "string":
            example[prop] = "<value>"
        elif typ == "array":
            example[prop] = []
        elif typ == "object":
            example[prop] = {}
        elif typ == "number" or typ == "integer":
            example[prop] = 0

    return f"""{prompt}

Respond with ONLY a JSON object like this example:
{json.dumps(example)}

Your JSON response (no explanation, just the JSON):"""


def generate_raw(
    prompt: str,
    schema: dict,
    mode: str = "structured",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Tuple[str, dict, float, float, int, int]:
    """Generate response using local HuggingFace model."""
    model_path = os.getenv("HF_LOCAL_MODEL", "./models/qwen3-4b-instruct")
    max_new = max_tokens or int(os.getenv("HF_LOCAL_MAX_TOKENS", "256"))

    model, tokenizer = _get_model_and_tokenizer(model_path)

    # Build prompt with schema
    full_prompt = _build_prompt_with_schema(prompt, schema)

    # Tokenize
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": full_prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
    else:
        input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)

    tokens_in = input_ids.shape[1]

    # Generate
    t0 = time.time()

    gen_kwargs = {
        "max_new_tokens": max_new,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)

    lat_ms = (time.time() - t0) * 1000.0
    ttft_ms = lat_ms  # No streaming, so TTFT = total latency

    # Decode
    new_tokens = outputs[0][tokens_in:]
    tokens_out = len(new_tokens)
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Parse JSON from response
    parsed = {}
    try:
        # Try direct parse
        parsed = json.loads(raw_text.strip())
    except json.JSONDecodeError:
        # Try to extract JSON from text
        import re
        # Look for JSON in code blocks
        if "```" in raw_text:
            parts = raw_text.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                try:
                    parsed = json.loads(p)
                    break
                except:
                    continue

        # Look for JSON objects
        if not parsed:
            matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text)
            for m in matches:
                try:
                    parsed = json.loads(m)
                    break
                except:
                    continue

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


def generate_json(
    prompt: str,
    schema: dict,
    mode: str = "structured",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Tuple[dict, float, float, int]:
    """Generate JSON response using local model."""
    _raw, parsed, lat_ms, ttft_ms, _tokens_in, tokens_out = generate_raw(
        prompt, schema, mode=mode, temperature=temperature, max_tokens=max_tokens
    )
    return parsed, lat_ms, ttft_ms, tokens_out
