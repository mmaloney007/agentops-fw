
import os, json, time
from typing import Tuple
from openai import OpenAI
def _max_tokens():
    try: return int(os.getenv("MAX_THOUGHT_TOKENS","0")) or None
    except Exception: return None
def generate_json(prompt: str, schema: dict) -> Tuple[dict, float, float, int]:
    base=os.getenv("OPENAI_API_BASE","http://localhost:1234/v1")
    key=os.getenv("OPENAI_API_KEY","lm-studio")
    model=os.getenv("LMSTUDIO_MODEL","llama-3.1-8b-instruct")
    client=OpenAI(base_url=base, api_key=key)
    want_stream=os.getenv("AOFW_STREAM","0")=="1"
    max_new=_max_tokens()
    if want_stream:
        t0=time.time(); ttft=None; chunks=[]
        with client.chat.completions.stream(
            model=model,
            messages=[{"role":"user","content": prompt}],
            response_format={"type":"json_schema","json_schema":{"name":"spec","schema":schema}},
            temperature=0, max_tokens=max_new
        ) as stream:
            for ev in stream:
                if getattr(ev,"type",None)=="response.delta":
                    if ttft is None: ttft=(time.time()-t0)*1000.0
                    delta=ev.delta or ""
                    if isinstance(delta,str): chunks.append(delta)
            final=stream.get_final_response()
            text="".join(chunks) if chunks else final.choices[0].message.content
            lat_ms=(time.time()-t0)*1000.0
            try: j=json.loads(text)
            except Exception: j={}
            return j, lat_ms, (ttft or lat_ms), -1
    t0=time.time()
    r=client.chat.completions.create(model=model,
        messages=[{"role":"user","content": prompt}],
        response_format={"type":"json_schema","json_schema":{"name":"spec","schema":schema}},
        temperature=0, max_tokens=max_new)
    lat_ms=(time.time()-t0)*1000.0
    txt=r.choices[0].message.content
    try: j=json.loads(txt)
    except Exception: j={}
    return j, lat_ms, lat_ms, -1
