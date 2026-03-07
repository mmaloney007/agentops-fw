#!/usr/bin/env python3
"""Generate CoreML .mlpackage models from safetensors weights for ANE dispatch.

Creates per-layer SDPA and FFN kernels, plus an output kernel, with weights baked in.
Uses coremltools MIL Builder to construct compute graphs that CoreML dispatches to ANE.

Kernel types per layer:
  - SDPA: RMSNorm → Q/K/V projections (conv1x1)
  - FFN:  RMSNorm → gate/up (conv1x1) → SiLU → mul → down (conv1x1) → residual add

Output kernel:
  - Final RMSNorm → classifier projection (lm_head)

Usage:
  python scripts/gen_coreml_models.py \\
    --weights ~/.cache/huggingface/.../model.safetensors \\
    --config stories110m \\
    --output-dir models/stories110m_coreml/
"""

import argparse
import os
import sys
import numpy as np

# Suppress noisy warnings from coremltools
import warnings
warnings.filterwarnings("ignore")

# Import torch BEFORE coremltools to avoid fork/segfault issues
import torch  # noqa: F401

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb

# Model configurations matching shared/model_config.h
CONFIGS = {
    "stories110m": {
        "dim": 768,
        "hidden_dim": 2048,
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 12,
        "head_dim": 64,
        "vocab_size": 32000,
        "seq_len": 256,
        "rms_norm_eps": 1e-5,
        "qkv_bias": False,
        "tie_embeddings": True,
    },
    "qwen05b": {
        "dim": 896,
        "hidden_dim": 4864,
        "n_layers": 24,
        "n_heads": 14,
        "n_kv_heads": 2,
        "head_dim": 64,
        "vocab_size": 151936,
        "seq_len": 256,
        "rms_norm_eps": 1e-6,
        "qkv_bias": True,
        "tie_embeddings": True,
    },
    "smollm2": {
        "dim": 960,
        "hidden_dim": 2560,
        "n_layers": 32,
        "n_heads": 15,
        "n_kv_heads": 5,
        "head_dim": 64,
        "vocab_size": 49152,
        "seq_len": 256,
        "rms_norm_eps": 1e-5,
        "qkv_bias": False,
        "tie_embeddings": True,
    },
}


def load_safetensors(path):
    """Load safetensors file, return dict of numpy arrays.
    Uses torch for loading to handle bfloat16 (numpy doesn't support bf16)."""
    import torch
    from safetensors.torch import load_file
    torch_tensors = load_file(path)
    tensors = {}
    for key, t in torch_tensors.items():
        tensors[key] = t.float().numpy()
    del torch_tensors
    return tensors


def build_sdpa_kernel(layer_idx, cfg, weights):
    """Build SDPA kernel: RMSNorm + Q/K/V projections.

    Input:  [1, dim, 1, seq] fp32
    Output: q [1, q_dim, 1, seq] fp32, k [1, kv_dim, 1, seq] fp32, v [1, kv_dim, 1, seq] fp32
    """
    dim = cfg["dim"]
    seq = cfg["seq_len"]
    n_heads = cfg["n_heads"]
    n_kv = cfg["n_kv_heads"]
    hd = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    q_dim = n_heads * hd
    kv_dim = n_kv * hd

    # Load weights for this layer
    prefix = f"model.layers.{layer_idx}"
    gamma = weights[f"{prefix}.input_layernorm.weight"].astype(np.float16).reshape(1, dim, 1, 1)
    wq = weights[f"{prefix}.self_attn.q_proj.weight"].astype(np.float16).reshape(q_dim, dim, 1, 1)
    wk = weights[f"{prefix}.self_attn.k_proj.weight"].astype(np.float16).reshape(kv_dim, dim, 1, 1)
    wv = weights[f"{prefix}.self_attn.v_proj.weight"].astype(np.float16).reshape(kv_dim, dim, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, seq))])
    def prog(x):
        x16 = mb.cast(x=x, dtype="fp16", name="x16")
        # RMSNorm
        sq = mb.mul(x=x16, y=x16, name="sq")
        mean_sq = mb.reduce_mean(x=sq, axes=[1], keep_dims=True, name="mean_sq")
        eps_c = mb.const(val=np.float16(eps), name="eps")
        mean_eps = mb.add(x=mean_sq, y=eps_c, name="mean_eps")
        rrms = mb.rsqrt(x=mean_eps, name="rrms")
        xnorm = mb.mul(x=x16, y=rrms, name="xnorm")
        g = mb.const(val=gamma, name="gamma")
        xn = mb.mul(x=xnorm, y=g, name="xn")
        # QKV projections
        Wq = mb.const(val=wq, name="Wq")
        Wk = mb.const(val=wk, name="Wk")
        Wv = mb.const(val=wv, name="Wv")
        q = mb.conv(x=xn, weight=Wq, name="q_conv")
        k = mb.conv(x=xn, weight=Wk, name="k_conv")
        v = mb.conv(x=xn, weight=Wv, name="v_conv")
        q32 = mb.cast(x=q, dtype="fp32", name="q")
        k32 = mb.cast(x=k, dtype="fp32", name="k")
        v32 = mb.cast(x=v, dtype="fp32", name="v")
        return q32, k32, v32

    return prog


def build_ffn_kernel(layer_idx, cfg, weights):
    """Build FFN kernel: RMSNorm + SwiGLU + residual.

    Input:  [1, dim, 1, seq] fp32
    Output: [1, dim, 1, seq] fp32 (with residual added)
    """
    dim = cfg["dim"]
    seq = cfg["seq_len"]
    hdim = cfg["hidden_dim"]
    eps = cfg["rms_norm_eps"]

    prefix = f"model.layers.{layer_idx}"
    gamma = weights[f"{prefix}.post_attention_layernorm.weight"].astype(np.float16).reshape(1, dim, 1, 1)
    w1 = weights[f"{prefix}.mlp.gate_proj.weight"].astype(np.float16).reshape(hdim, dim, 1, 1)
    w3 = weights[f"{prefix}.mlp.up_proj.weight"].astype(np.float16).reshape(hdim, dim, 1, 1)
    w2 = weights[f"{prefix}.mlp.down_proj.weight"].astype(np.float16).reshape(dim, hdim, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, seq))])
    def prog(x):
        x16 = mb.cast(x=x, dtype="fp16", name="x16")
        # RMSNorm
        sq = mb.mul(x=x16, y=x16, name="sq")
        mean_sq = mb.reduce_mean(x=sq, axes=[1], keep_dims=True, name="mean_sq")
        eps_c = mb.const(val=np.float16(eps), name="eps")
        mean_eps = mb.add(x=mean_sq, y=eps_c, name="mean_eps")
        rrms = mb.rsqrt(x=mean_eps, name="rrms")
        xnorm = mb.mul(x=x16, y=rrms, name="xnorm")
        g = mb.const(val=gamma, name="gamma")
        xn = mb.mul(x=xnorm, y=g, name="xn")
        # SwiGLU
        W1 = mb.const(val=w1, name="W1")
        W3 = mb.const(val=w3, name="W3")
        W2 = mb.const(val=w2, name="W2")
        gate = mb.conv(x=xn, weight=W1, name="gate")
        up = mb.conv(x=xn, weight=W3, name="up")
        gate_silu = mb.silu(x=gate, name="gate_silu")
        hidden = mb.mul(x=gate_silu, y=up, name="hidden")
        down = mb.conv(x=hidden, weight=W2, name="down")
        # Residual
        out = mb.add(x=x16, y=down, name="out_res")
        out32 = mb.cast(x=out, dtype="fp32", name="out")
        return out32

    return prog


def build_output_kernel(cfg, weights):
    """Build output kernel: Final RMSNorm + classifier.

    Input:  [1, dim, 1, 1] fp32 (last token only)
    Output: [1, vocab, 1, 1] fp32 (logits)
    """
    dim = cfg["dim"]
    vocab = cfg["vocab_size"]
    eps = cfg["rms_norm_eps"]

    gamma = weights["model.norm.weight"].astype(np.float16).reshape(1, dim, 1, 1)

    # Classifier weights: either lm_head or shared with embeddings
    if "lm_head.weight" in weights:
        wout = weights["lm_head.weight"].astype(np.float16).reshape(vocab, dim, 1, 1)
    else:
        wout = weights["model.embed_tokens.weight"].astype(np.float16).reshape(vocab, dim, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, 1))])
    def prog(x):
        x16 = mb.cast(x=x, dtype="fp16", name="x16")
        sq = mb.mul(x=x16, y=x16, name="sq")
        mean_sq = mb.reduce_mean(x=sq, axes=[1], keep_dims=True, name="mean_sq")
        eps_c = mb.const(val=np.float16(eps), name="eps")
        mean_eps = mb.add(x=mean_sq, y=eps_c, name="mean_eps")
        rrms = mb.rsqrt(x=mean_eps, name="rrms")
        xnorm = mb.mul(x=x16, y=rrms, name="xnorm")
        g = mb.const(val=gamma, name="gamma")
        xn = mb.mul(x=xnorm, y=g, name="xn")
        Wout = mb.const(val=wout, name="Wout")
        logits = mb.conv(x=xn, weight=Wout, name="logits_conv")
        logits32 = mb.cast(x=logits, dtype="fp32", name="logits")
        return logits32

    return prog


def build_output_kernel_fullseq(cfg, weights):
    """Build output kernel for full-sequence forward (training).

    Input:  [1, dim, 1, seq] fp32
    Output: [1, vocab, 1, seq] fp32 (logits for all positions)
    """
    dim = cfg["dim"]
    seq = cfg["seq_len"]
    vocab = cfg["vocab_size"]
    eps = cfg["rms_norm_eps"]

    gamma = weights["model.norm.weight"].astype(np.float16).reshape(1, dim, 1, 1)

    if "lm_head.weight" in weights:
        wout = weights["lm_head.weight"].astype(np.float16).reshape(vocab, dim, 1, 1)
    else:
        wout = weights["model.embed_tokens.weight"].astype(np.float16).reshape(vocab, dim, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, seq))])
    def prog(x):
        x16 = mb.cast(x=x, dtype="fp16", name="x16")
        sq = mb.mul(x=x16, y=x16, name="sq")
        mean_sq = mb.reduce_mean(x=sq, axes=[1], keep_dims=True, name="mean_sq")
        eps_c = mb.const(val=np.float16(eps), name="eps")
        mean_eps = mb.add(x=mean_sq, y=eps_c, name="mean_eps")
        rrms = mb.rsqrt(x=mean_eps, name="rrms")
        xnorm = mb.mul(x=x16, y=rrms, name="xnorm")
        g = mb.const(val=gamma, name="gamma")
        xn = mb.mul(x=xnorm, y=g, name="xn")
        Wout = mb.const(val=wout, name="Wout")
        logits = mb.conv(x=xn, weight=Wout, name="logits_conv")
        logits32 = mb.cast(x=logits, dtype="fp32", name="logits")
        return logits32

    return prog


def convert_and_save(prog, path):
    """Convert MIL program to CoreML and save as .mlpackage."""
    model = ct.convert(
        prog,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    model.save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate CoreML .mlpackage models")
    parser.add_argument("--weights", required=True, help="Path to safetensors model file")
    parser.add_argument("--config", required=True, choices=CONFIGS.keys(),
                        help="Model configuration")
    parser.add_argument("--output-dir", required=True, help="Output directory for .mlpackage files")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to generate (default: all)")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading weights from {args.weights}...")
    weights = load_safetensors(args.weights)
    print(f"  {len(weights)} tensors loaded")

    n_layers = cfg["n_layers"]
    layers = range(n_layers) if args.layers is None else [int(x) for x in args.layers.split(",")]

    total = len(layers) * 2 + 2  # SDPA + FFN per layer + output + output_fullseq
    done = 0

    for l in layers:
        # SDPA kernel
        path = os.path.join(args.output_dir, f"layer_{l:02d}_sdpa.mlpackage")
        if os.path.exists(path):
            print(f"  [{done+1}/{total}] layer_{l:02d}_sdpa — exists, skipping")
        else:
            print(f"  [{done+1}/{total}] layer_{l:02d}_sdpa — generating...")
            prog = build_sdpa_kernel(l, cfg, weights)
            convert_and_save(prog, path)
            print(f"           → {path}")
        done += 1

        # FFN kernel
        path = os.path.join(args.output_dir, f"layer_{l:02d}_ffn.mlpackage")
        if os.path.exists(path):
            print(f"  [{done+1}/{total}] layer_{l:02d}_ffn  — exists, skipping")
        else:
            print(f"  [{done+1}/{total}] layer_{l:02d}_ffn  — generating...")
            prog = build_ffn_kernel(l, cfg, weights)
            convert_and_save(prog, path)
            print(f"           → {path}")
        done += 1

    # Output kernel (single token — for generation)
    path = os.path.join(args.output_dir, "output.mlpackage")
    if os.path.exists(path):
        print(f"  [{done+1}/{total}] output — exists, skipping")
    else:
        print(f"  [{done+1}/{total}] output — generating...")
        prog = build_output_kernel(cfg, weights)
        convert_and_save(prog, path)
        print(f"           → {path}")
    done += 1

    # Output kernel (full sequence — for training forward)
    path = os.path.join(args.output_dir, "output_fullseq.mlpackage")
    if os.path.exists(path):
        print(f"  [{done+1}/{total}] output_fullseq — exists, skipping")
    else:
        print(f"  [{done+1}/{total}] output_fullseq — generating...")
        prog = build_output_kernel_fullseq(cfg, weights)
        convert_and_save(prog, path)
        print(f"           → {path}")
    done += 1

    print(f"\nDone! {done} kernels in {args.output_dir}")
    print(f"  {len(layers)} layers × 2 (SDPA + FFN) + 2 output = {done} total")


if __name__ == "__main__":
    main()
