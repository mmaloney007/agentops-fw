
import os, json, time
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
class HFLocal:
    def __init__(self, model_id: str):
        self.tok=AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model=AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    def generate_json(self, prompt: str, schema: dict) -> Tuple[dict,float,float,int]:
        max_new=int(os.getenv("MAX_THOUGHT_TOKENS","0")) or 256
        t0=time.time(); x=self.tok(prompt, return_tensors="pt").to(self.model.device)
        in_len=x["input_ids"].shape[-1]
        y=self.model.generate(**x, max_new_tokens=max_new, temperature=0.0)
        out_len=y.shape[-1]; text=self.tok.decode(y[0], skip_special_tokens=True)
        lat_ms=(time.time()-t0)*1000.0
        s=text[text.find("{"): text.rfind("}")+1]
        try: j=json.loads(s)
        except json.JSONDecodeError: j={}
        tokens_out=max(0, out_len-in_len)
        return j, lat_ms, lat_ms, tokens_out
