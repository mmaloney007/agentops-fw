#!/usr/bin/env python3
"""Dump activations from HuggingFace for full chat prompt."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = '<|im_start|>system\nYou are a helpful assistant. Always respond with valid JSON only, no other text.<|im_end|>\n<|im_start|>user\nExtract name and age: John Smith is 30 years old.\nRespond with valid JSON matching this schema: {"name":"string","age":"number"}<|im_end|>\n<|im_start|>assistant\n'

print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

# Encode prompt
ids = tok.encode(PROMPT, add_special_tokens=False)
print(f"Prompt length: {len(ids)} tokens")
print(f"Token IDs: {ids}")
input_ids = torch.tensor([ids])

# Run forward
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1]  # last position

print(f"\n=== LOGITS (position {len(ids)-1}) ===")
topk = torch.topk(logits, 15)
for i in range(15):
    tid = topk.indices[i].item()
    val = topk.values[i].item()
    tok_str = tok.decode([tid]).replace('\n', '\\n')
    print(f"  #{i+1}: id={tid} logit={val:.6f} token='{tok_str}'")

print(f"\nlogits[0] = {logits[0].item():.6f}")
print(f"logits.max() = {logits.max().item():.6f}")
print(f"logits.min() = {logits.min().item():.6f}")
print(f"logits.mean() = {logits.mean().item():.6f}")
print(f"logits.std() = {logits.std().item():.6f}")

# Also print greedy generation
print("\n=== GREEDY GENERATION (20 tokens) ===")
gen_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False, temperature=None, top_p=None)
gen_text = tok.decode(gen_ids[0, len(ids):])
print(f"Generated: '{gen_text}'")
gen_list = gen_ids[0, len(ids):].tolist()
print(f"Token IDs: {gen_list}")
