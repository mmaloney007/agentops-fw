
import os
from typing import Dict, Tuple
def provider_generate(prompt: str, schema: Dict) -> Tuple[Dict, float, float, int]:
    backend=os.getenv("AOFW_PROVIDER","lmstudio").lower()
    if backend=="lmstudio":
        from .providers.lmstudio_openai import generate_json
        return generate_json(prompt, schema)
    if backend=="ollama":
        from .providers.ollama_structured import generate_json
        return generate_json(prompt, schema)
    if backend=="hf":
        from .providers.hf_local import HFLocal
        mdl=os.getenv("HF_LOCAL_MODEL","Qwen/Qwen2.5-7B-Instruct")
        return HFLocal(mdl).generate_json(prompt, schema)
    out={k: ("example" if v.get("type")=="string" else (["a","b"] if v.get("type")=="array" else 1))
         for k,v in schema.get("properties",{}).items()}
    return out, 5.0, 5.0, -1
