
import os
from typing import Dict, Tuple
def provider_generate(prompt: str, schema: Dict) -> Tuple[Dict, float, float, int]:
    backend=os.getenv("AOFW_PROVIDER","lmstudio").lower()
    mode=os.getenv("DECODE_MODE","structured")  # structured | text | grammar
    if backend=="lmstudio":
        from .providers.lmstudio_openai import generate_json
        return generate_json(prompt, schema, mode=mode)
    if backend=="ollama":
        from .providers.ollama_structured import generate_json
        return generate_json(prompt, schema, mode=mode)
    if backend=="vllm":
        from .providers.vllm_openai import generate_json
        return generate_json(prompt, schema, mode=mode)
    out={k: ("example" if v.get("type")=="string" else ["a","b"]) for k,v in schema.get("properties",{}).items()}
    return out, 5.0, 5.0, -1
