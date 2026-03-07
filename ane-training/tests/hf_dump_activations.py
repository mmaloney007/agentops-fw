#!/usr/bin/env python3
"""Dump intermediate activations from HuggingFace Qwen2.5-0.5B for comparison with our C implementation."""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello"

print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

# Encode prompt
ids = tok.encode(PROMPT, add_special_tokens=False)
print(f"Token IDs: {ids}")
input_ids = torch.tensor([ids])

# Hook to capture activations
activations = {}

def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations[name] = output[0].detach()
        else:
            activations[name] = output.detach()
    return hook_fn

# Register hooks
model.model.embed_tokens.register_forward_hook(make_hook("embed"))
model.model.layers[0].input_layernorm.register_forward_hook(make_hook("layer0_attn_norm"))
model.model.layers[0].self_attn.q_proj.register_forward_hook(make_hook("layer0_q"))
model.model.layers[0].self_attn.k_proj.register_forward_hook(make_hook("layer0_k"))
model.model.layers[0].self_attn.v_proj.register_forward_hook(make_hook("layer0_v"))
model.model.layers[0].self_attn.register_forward_hook(make_hook("layer0_attn_out"))
model.model.layers[0].register_forward_hook(make_hook("layer0_out"))
model.model.layers[23].register_forward_hook(make_hook("layer23_out"))
model.model.norm.register_forward_hook(make_hook("final_norm"))

# Run forward
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1]  # last position

print("\n=== WEIGHTS ===")
emb = model.model.embed_tokens.weight.data
print(f"embed_tokens.weight[0,:10] = {emb[0,:10].tolist()}")
print(f"embed_tokens.weight[{ids[0]},:10] = {emb[ids[0],:10].tolist()}")

wq0 = model.model.layers[0].self_attn.q_proj.weight.data
bq0 = model.model.layers[0].self_attn.q_proj.bias.data
print(f"layer0.q_proj.weight[0,:10] = {wq0[0,:10].tolist()}")
print(f"layer0.q_proj.bias[:10] = {bq0[:10].tolist()}")

rms_attn0 = model.model.layers[0].input_layernorm.weight.data
print(f"layer0.input_layernorm.weight[:10] = {rms_attn0[:10].tolist()}")

rms_final = model.model.norm.weight.data
print(f"model.norm.weight[:10] = {rms_final[:10].tolist()}")

print("\n=== ACTIVATIONS ===")
for name in ["embed", "layer0_attn_norm", "layer0_q", "layer0_k", "layer0_v",
             "layer0_attn_out", "layer0_out", "layer23_out", "final_norm"]:
    if name in activations:
        a = activations[name]
        if a.dim() == 3:
            vals = a[0, 0, :10].tolist()  # batch=0, pos=0, first 10 dims
        else:
            vals = a[:10].tolist()
        print(f"{name}[0,0,:10] = {vals}")

print("\n=== LOGITS (last pos) ===")
topk = torch.topk(logits, 10)
for i in range(10):
    tid = topk.indices[i].item()
    val = topk.values[i].item()
    tok_str = tok.decode([tid])
    print(f"  #{i+1}: id={tid} logit={val:.6f} token='{tok_str}'")

# Also dump specific logit values
print(f"\nlogits[0] = {logits[0].item():.6f}")
print(f"logits.max() = {logits.max().item():.6f}")
print(f"logits.min() = {logits.min().item():.6f}")
print(f"logits.mean() = {logits.mean().item():.6f}")
print(f"logits.std() = {logits.std().item():.6f}")
