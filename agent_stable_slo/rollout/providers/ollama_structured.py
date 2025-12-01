
import os, json, time
from typing import Tuple
from ollama import chat
def _max_tokens():
    try: return int(os.getenv("MAX_THOUGHT_TOKENS","0")) or None
    except Exception: return None
def generate_json(prompt: str, schema: dict, mode: str = "structured") -> Tuple[dict, float, float, int]:
    model=os.getenv("OLLAMA_MODEL","llama3.1:8b")
    t0=time.time()
    fmt = schema if mode=="structured" else None
    if mode=="grammar":
        # Ollama supports "format":"json" with prompt hints; use schema as format fallback
        fmt = schema
    resp=chat(model=model, messages=[{"role":"user","content": prompt}], format=fmt,
              options={"temperature":0, "num_predict":_max_tokens()})
    lat_ms=(time.time()-t0)*1000.0
    content=resp.get("message",{}).get("content")
    if not content: return {}, lat_ms, lat_ms, -1
    try: j=json.loads(content)
    except json.JSONDecodeError: j={}
    return j, lat_ms, lat_ms, -1
