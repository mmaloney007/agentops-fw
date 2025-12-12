
from __future__ import annotations

import os
from typing import Dict, Tuple
def provider_generate(prompt: str, schema: Dict, mode: str | None = None) -> Tuple[Dict, float, float, int]:
    backend=os.getenv("AOFW_PROVIDER","lmstudio").lower()
    decode_mode = mode or os.getenv("DECODE_MODE","structured")  # structured | text | grammar
    if backend=="lmstudio":
        from .providers.lmstudio_openai import generate_json
        return generate_json(prompt, schema, mode=decode_mode)
    if backend=="ollama":
        from .providers.ollama_structured import generate_json
        return generate_json(prompt, schema, mode=decode_mode)
    if backend=="vllm":
        from .providers.vllm_openai import generate_json
        return generate_json(prompt, schema, mode=decode_mode)
    out={k: ("example" if v.get("type")=="string" else ["a","b"]) for k,v in schema.get("properties",{}).items()}
    return out, 5.0, 5.0, -1
