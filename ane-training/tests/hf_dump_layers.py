#!/usr/bin/env python3
"""Dump per-layer hidden state norms for the chat prompt to find where divergence starts."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = '<|im_start|>system\nYou are a helpful assistant. Always respond with valid JSON only, no other text.<|im_end|>\n<|im_start|>user\nExtract name and age: John Smith is 30 years old.\nRespond with valid JSON matching this schema: {"name":"string","age":"number"}<|im_end|>\n<|im_start|>assistant\n'

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

ids = tok.encode(PROMPT, add_special_tokens=False)
input_ids = torch.tensor([ids])

# Capture hidden states at every layer
layer_outputs = {}
def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            layer_outputs[name] = output[0].detach()
        else:
            layer_outputs[name] = output.detach()
    return hook_fn

model.model.embed_tokens.register_forward_hook(make_hook("embed"))
for i in range(24):
    model.model.layers[i].register_forward_hook(make_hook(f"layer{i}"))
model.model.norm.register_forward_hook(make_hook("final_norm"))

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1]

# Print hidden state norms at last position for each layer
print("Layer-by-layer hidden state norm at LAST position:")
last = len(ids) - 1
for name in ["embed"] + [f"layer{i}" for i in range(24)] + ["final_norm"]:
    if name in layer_outputs:
        h = layer_outputs[name][0, last]  # last position
        norm = h.norm().item()
        first5 = h[:5].tolist()
        print(f"  {name:12s}: norm={norm:12.4f}  first5={[f'{v:.6f}' for v in first5]}")

# Also print at position 0 for comparison
print("\nLayer-by-layer hidden state norm at FIRST position:")
for name in ["embed"] + [f"layer{i}" for i in range(24)] + ["final_norm"]:
    if name in layer_outputs:
        h = layer_outputs[name][0, 0]
        norm = h.norm().item()
        first5 = h[:5].tolist()
        print(f"  {name:12s}: norm={norm:12.4f}  first5={[f'{v:.6f}' for v in first5]}")

# Also check: what's a 2-token test?
print("\n\n=== 2-TOKEN TEST ===")
ids2 = tok.encode("Hi", add_special_tokens=False)
print(f"2-token IDs: {ids2}")
input_ids2 = torch.tensor([ids2])
with torch.no_grad():
    outputs2 = model(input_ids2)
    logits2 = outputs2.logits[0, -1]

topk = torch.topk(logits2, 5)
for i in range(5):
    tid = topk.indices[i].item()
    val = topk.values[i].item()
    print(f"  #{i+1}: id={tid} logit={val:.6f}")
print(f"logits2.max() = {logits2.max().item():.6f}")
print(f"logits2.std() = {logits2.std().item():.6f}")
