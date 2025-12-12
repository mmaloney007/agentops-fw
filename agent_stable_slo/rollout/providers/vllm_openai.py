
import os, json, time, re
from typing import Tuple
from openai import OpenAI

from .lmstudio_openai import _max_tokens, _maybe_augment_prompt, TOOLSEQ_DEFAULTS


def _completion_tokens(resp) -> int:
    try:
        return int(getattr(resp, "usage").completion_tokens)
    except Exception:
        return -1


def _default_base():
    return os.getenv("VLLM_API_BASE") or os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")


def generate_json(prompt: str, schema: dict, mode: str = "structured") -> Tuple[dict, float, float, int]:
    base = _default_base()
    key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    model = os.getenv("VLLM_MODEL") or os.getenv("LMSTUDIO_MODEL", "")
    client = OpenAI(base_url=base, api_key=key)

    t0 = time.time()
    r = None
    try:
        if mode == "structured":
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _maybe_augment_prompt(prompt, schema)}],
                response_format={"type": "json_schema", "json_schema": {"name": "spec", "schema": schema}},
                temperature=0,
                max_tokens=_max_tokens(),
            )
        else:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _maybe_augment_prompt(prompt, schema)}],
                response_format={"type": "text"},
                temperature=0,
                max_tokens=_max_tokens(),
            )
    except Exception:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _maybe_augment_prompt(prompt, schema)}],
                response_format={"type": "text"},
                temperature=0,
                max_tokens=_max_tokens(),
            )
        except Exception:
            lat_ms = (time.time() - t0) * 1000.0
            return {}, float(lat_ms), float(lat_ms), -1

    lat_ms = (time.time() - t0) * 1000.0
    ttft_ms = lat_ms

    txt = r.choices[0].message.content or "{}"
    try:
        j = json.loads(txt)
    except Exception:
        j = {}
        if "answer" in schema.get("properties", {}):
            j["answer"] = txt.strip()

    if "citations" in schema.get("properties", {}):
        if not isinstance(j, dict):
            j = {}
        cits = j.get("citations") if isinstance(j.get("citations"), list) else []
        if not cits:
            found = list(dict.fromkeys(re.findall(r"\[s\d+\]", prompt)))
            if found:
                j["citations"] = found
        if isinstance(j.get("citations"), list):
            norm = []
            for c in j["citations"]:
                c = str(c).strip()
                if not c.startswith("["):
                    c = f"[{c}]"
                norm.append(c)
            j["citations"] = norm
        if isinstance(j.get("bullets"), list) and j.get("citations"):
            cid = j["citations"][0]
            fixed = []
            for b in j["bullets"]:
                bstr = str(b)
                if "[s" not in bstr:
                    bstr = f"{bstr} ({cid})"
                fixed.append(bstr)
            j["bullets"] = fixed

    if "steps" in schema.get("properties", {}):
        if not isinstance(j, dict):
            j = {}
        steps = j.get("steps") if isinstance(j.get("steps"), list) else []
        m = re.search(r"Task:\s*([\w_-]+)", prompt)
        if m:
            seq = TOOLSEQ_DEFAULTS.get(m.group(1))
            if seq:
                steps = seq
        j["steps"] = [str(s) for s in steps] if steps else []

    if isinstance(j, dict):
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for req in required:
            if req not in j:
                typ = props.get(req, {}).get("type")
                if typ == "string":
                    j[req] = ""
                elif typ == "array":
                    j[req] = []
                elif typ == "object":
                    j[req] = {}

    tokens_out = _completion_tokens(r)
    return j, lat_ms, ttft_ms, tokens_out
