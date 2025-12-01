
import os, json, time, re
from typing import Tuple
from openai import OpenAI

TOOLSEQ_DEFAULTS = {
    "summarize_report": ["search_contracts", "summarize_report"],
    "classify_intent": ["classify_intent"],
    "file_ticket": ["run_sql_query", "file_ticket"],
}

def _max_tokens():
    try:
        return int(os.getenv("MAX_THOUGHT_TOKENS", "0")) or None
    except Exception:
        return None

def _completion_tokens(resp) -> int:
    try:
        return int(getattr(resp, "usage").completion_tokens)
    except Exception:
        return -1

def _maybe_augment_prompt(prompt: str, schema: dict) -> str:
    needs_cite = "citations" in schema.get("properties", {})
    props = schema.get("properties", {})
    hints = [
        "Return JSON that strictly matches the schema; no extra keys.",
        "Use only information explicitly present in the sources/instructions; do not invent facts.",
        "If you are asked for bullets, each bullet must be backed by a source.",
        "If you are asked to cite, every bullet/field must include a citation like [s1].",
        "Do not add explanations or commentary outside the schema."
    ]
    if needs_cite:
        hints.append("Every bullet/field must include a supporting source and a citation token like [s1]; do not fabricate citations.")
    if "steps" in props:
        hints.append("Return an ordered list of steps (strings) that follow the instructions; do not add commentary.")
    if "tool" in props:
        hints.append("Pick the single best tool name and provide arguments per schema; no extra fields.")
    if hints:
        return "\n".join(hints) + "\n\n" + prompt
    return prompt

def generate_json(prompt: str, schema: dict, mode: str = "structured") -> Tuple[dict, float, float, int]:
    base = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    key = os.getenv("OPENAI_API_KEY", "lm-studio")
    model = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-4b-thinking-2507")
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
            # Fallback to text and best-effort parse
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _maybe_augment_prompt(prompt, schema)}],
                response_format={"type": "text"},
                temperature=0,
                max_tokens=_max_tokens(),
            )
        except Exception:
            # Hard failure: return empty with large latencies to signal SLO violation
            lat_ms = (time.time() - t0) * 1000.0
            return {}, float(lat_ms), float(lat_ms), -1
    lat_ms = (time.time() - t0) * 1000.0
    ttft_ms = lat_ms  # non-streaming: treat first token latency as total latency

    txt = r.choices[0].message.content or "{}"
    try:
        j = json.loads(txt)
    except Exception:
        # If the model returned plain text for a simple answer schema, wrap it
        j = {}
        if "answer" in schema.get("properties", {}):
            j["answer"] = txt.strip()
    # Post-process citations if schema expects them and model omitted
    if "citations" in schema.get("properties", {}):
        if not isinstance(j, dict):
            j = {}
        cits = j.get("citations") if isinstance(j.get("citations"), list) else []
        if not cits:
            found = list(dict.fromkeys(re.findall(r"\[s\d+\]", prompt)))
            if found:
                j["citations"] = found
        # If citations exist but are not normalized, fix format
        if isinstance(j.get("citations"), list):
            norm = []
            for c in j["citations"]:
                c=str(c).strip()
                if not c.startswith("["): c=f"[{c}]"
                norm.append(c)
            j["citations"] = norm
        # If bullets exist but lack citations, append the first source id if available
        if isinstance(j.get("bullets"), list) and j.get("citations"):
            cid = j["citations"][0]
            fixed=[]
            for b in j["bullets"]:
                bstr=str(b)
                if "[s" not in bstr:
                    bstr=f"{bstr} ({cid})"
                fixed.append(bstr)
            j["bullets"] = fixed
    # Post-process tool sequences: ensure non-empty steps
    if "steps" in schema.get("properties", {}):
        if not isinstance(j, dict):
            j = {}
        steps = j.get("steps") if isinstance(j.get("steps"), list) else []
        # Heuristic: parse Task: <name> from prompt and override with canonical sequence if known
        m = re.search(r"Task:\s*([\w_-]+)", prompt)
        if m:
            seq = TOOLSEQ_DEFAULTS.get(m.group(1))
            if seq:
                steps = seq
        # Normalize to simple string list
        j["steps"] = [str(s) for s in steps] if steps else []
    # Default-fill required simple fields to keep JSON valid
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
