#!/usr/bin/env python3
"""Generate CoreML .mlpackage backward (dx) kernels from safetensors weights.

Creates per-layer backward activation gradient kernels with weights baked in.
These compute d_loss/d_input for each layer's sub-modules during backpropagation.

Kernel types per layer:
  - FFN backward:  d_xnorm from d_x through SwiGLU (gate/up/down projections)
  - Wo backward:   d_attn_out from d_x through output projection
  - QKV backward:  d_xnorm from d_q/d_k/d_v through Q/K/V projections

Weight transpose logic (backward dx through linear y = Wx):
  Forward conv weight: W [out, in, 1, 1]
  Backward dx: d_x = W^T @ d_y -> conv(d_y, W^T) where W^T = [in, out, 1, 1]

Usage:
  python scripts/gen_backward_kernels.py \\
    --weights weights/stories110m/model.safetensors \\
    --config stories110m \\
    --output-dir models/stories110m_coreml/ \\
    --layers 0
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

# Model configurations matching gen_coreml_models.py
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


def build_ffn_bwd_kernel(layer_idx, cfg, weights):
    """Build FFN backward dx kernel.

    Computes gradient through SwiGLU FFN: down_proj <- silu(gate) * up <- gate/up projections.

    Inputs:
      d_x       [1, dim, 1, seq]   — gradient from downstream (after residual split)
      gate_raw  [1, hdim, 1, seq]  — saved gate_proj output (pre-SiLU) from forward
      up_val    [1, hdim, 1, seq]  — saved up_proj output from forward

    Output:
      d_xnorm   [1, dim, 1, seq]  — gradient w.r.t. FFN input (post-RMSNorm)
    """
    dim = cfg["dim"]
    seq = cfg["seq_len"]
    hdim = cfg["hidden_dim"]

    prefix = f"model.layers.{layer_idx}"

    # Load forward weights and transpose for backward dx
    # Forward: y = conv(x, W) with W [out, in, 1, 1]
    # Backward dx: d_x = conv(d_y, W^T) with W^T [in, out, 1, 1]
    w2 = weights[f"{prefix}.mlp.down_proj.weight"]   # [dim, hdim]
    w1 = weights[f"{prefix}.mlp.gate_proj.weight"]    # [hdim, dim]
    w3 = weights[f"{prefix}.mlp.up_proj.weight"]      # [hdim, dim]

    # Transpose: swap out/in dims, then reshape as conv weight
    w2_t = w2.T.astype(np.float16).reshape(hdim, dim, 1, 1)   # [hdim, dim, 1, 1]
    w1_t = w1.T.astype(np.float16).reshape(dim, hdim, 1, 1)   # [dim, hdim, 1, 1]
    w3_t = w3.T.astype(np.float16).reshape(dim, hdim, 1, 1)   # [dim, hdim, 1, 1]

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, seq)),    # d_x
        mb.TensorSpec(shape=(1, hdim, 1, seq)),   # gate_raw
        mb.TensorSpec(shape=(1, hdim, 1, seq)),   # up_val
    ])
    def prog(d_x, gate_raw, up_val):
        # Cast inputs to fp16
        d_x16 = mb.cast(x=d_x, dtype="fp16", name="d_x16")
        gate16 = mb.cast(x=gate_raw, dtype="fp16", name="gate16")
        up16 = mb.cast(x=up_val, dtype="fp16", name="up16")

        # Step 1: Backward through down_proj: d_gated = W2^T @ d_x
        W2t = mb.const(val=w2_t, name="W2t")
        d_gated = mb.conv(x=d_x16, weight=W2t, name="d_gated")

        # Step 2: Recompute gate_silu = silu(gate_raw) for backward
        gate_silu = mb.silu(x=gate16, name="gate_silu")

        # Step 3: Backward through element-wise mul (hidden = gate_silu * up)
        # d_gate_silu = d_gated * up_val
        d_gate_silu = mb.mul(x=d_gated, y=up16, name="d_gate_silu")
        # d_up = d_gated * gate_silu
        d_up = mb.mul(x=d_gated, y=gate_silu, name="d_up")

        # Step 4: SiLU backward: d_gate = d_gate_silu * silu_grad(gate_raw)
        # silu(x) = x * sigmoid(x)
        # silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sig = mb.sigmoid(x=gate16, name="sig")
        one = mb.const(val=np.float16(1.0), name="one")
        one_minus_sig = mb.sub(x=one, y=sig, name="one_minus_sig")
        x_times_oms = mb.mul(x=gate16, y=one_minus_sig, name="x_times_oms")
        one_plus_term = mb.add(x=one, y=x_times_oms, name="one_plus_term")
        silu_grad = mb.mul(x=sig, y=one_plus_term, name="silu_grad")
        d_gate = mb.mul(x=d_gate_silu, y=silu_grad, name="d_gate")

        # Step 5: Backward through gate/up projections
        # d_xnorm = W1^T @ d_gate + W3^T @ d_up
        W1t = mb.const(val=w1_t, name="W1t")
        W3t = mb.const(val=w3_t, name="W3t")
        d_xn_gate = mb.conv(x=d_gate, weight=W1t, name="d_xn_gate")
        d_xn_up = mb.conv(x=d_up, weight=W3t, name="d_xn_up")
        d_xnorm = mb.add(x=d_xn_gate, y=d_xn_up, name="d_xnorm_add")

        # Cast output to fp32
        out = mb.cast(x=d_xnorm, dtype="fp32", name="d_xnorm")
        return out

    return prog


def build_wo_bwd_kernel(layer_idx, cfg, weights):
    """Build Wo backward dx kernel.

    Backward through the attention output projection: attn_out = Wo @ concat_heads.

    Input:  d_x        [1, dim, 1, seq] — gradient from downstream
    Output: d_attn_out  [1, dim, 1, seq] — gradient w.r.t. concat_heads
    """
    dim = cfg["dim"]
    seq = cfg["seq_len"]

    prefix = f"model.layers.{layer_idx}"

    # Wo: [dim, dim] forward. Transpose for backward dx.
    wo = weights[f"{prefix}.self_attn.o_proj.weight"]  # [dim, dim]
    wo_t = wo.T.astype(np.float16).reshape(dim, dim, 1, 1)  # [dim, dim, 1, 1]

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, seq))])
    def prog(d_x):
        d_x16 = mb.cast(x=d_x, dtype="fp16", name="d_x16")
        Wot = mb.const(val=wo_t, name="Wot")
        d_attn = mb.conv(x=d_x16, weight=Wot, name="d_attn_conv")
        out = mb.cast(x=d_attn, dtype="fp32", name="d_attn_out")
        return out

    return prog


def build_qkv_bwd_kernel(layer_idx, cfg, weights):
    """Build QKV backward dx kernel.

    Backward through Q/K/V projections: q = Wq @ xnorm, k = Wk @ xnorm, v = Wv @ xnorm.

    Inputs:
      d_q  [1, q_dim, 1, seq]   — gradient w.r.t. Q
      d_k  [1, kv_dim, 1, seq]  — gradient w.r.t. K
      d_v  [1, kv_dim, 1, seq]  — gradient w.r.t. V

    Output:
      d_xnorm [1, dim, 1, seq]  — gradient w.r.t. xnorm (input to QKV projections)
    """
    dim = cfg["dim"]
    seq = cfg["seq_len"]
    n_heads = cfg["n_heads"]
    n_kv = cfg["n_kv_heads"]
    hd = cfg["head_dim"]
    q_dim = n_heads * hd
    kv_dim = n_kv * hd

    prefix = f"model.layers.{layer_idx}"

    # Load forward weights and transpose for backward dx
    wq = weights[f"{prefix}.self_attn.q_proj.weight"]  # [q_dim, dim]
    wk = weights[f"{prefix}.self_attn.k_proj.weight"]  # [kv_dim, dim]
    wv = weights[f"{prefix}.self_attn.v_proj.weight"]  # [kv_dim, dim]

    wq_t = wq.T.astype(np.float16).reshape(dim, q_dim, 1, 1)   # [dim, q_dim, 1, 1]
    wk_t = wk.T.astype(np.float16).reshape(dim, kv_dim, 1, 1)  # [dim, kv_dim, 1, 1]
    wv_t = wv.T.astype(np.float16).reshape(dim, kv_dim, 1, 1)  # [dim, kv_dim, 1, 1]

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, q_dim, 1, seq)),   # d_q
        mb.TensorSpec(shape=(1, kv_dim, 1, seq)),  # d_k
        mb.TensorSpec(shape=(1, kv_dim, 1, seq)),  # d_v
    ])
    def prog(d_q, d_k, d_v):
        # Cast inputs to fp16
        dq16 = mb.cast(x=d_q, dtype="fp16", name="dq16")
        dk16 = mb.cast(x=d_k, dtype="fp16", name="dk16")
        dv16 = mb.cast(x=d_v, dtype="fp16", name="dv16")

        # Backward: d_xnorm = Wq^T @ d_q + Wk^T @ d_k + Wv^T @ d_v
        Wqt = mb.const(val=wq_t, name="Wqt")
        Wkt = mb.const(val=wk_t, name="Wkt")
        Wvt = mb.const(val=wv_t, name="Wvt")

        d_xn_q = mb.conv(x=dq16, weight=Wqt, name="d_xn_q")
        d_xn_k = mb.conv(x=dk16, weight=Wkt, name="d_xn_k")
        d_xn_v = mb.conv(x=dv16, weight=Wvt, name="d_xn_v")

        d_xn_qk = mb.add(x=d_xn_q, y=d_xn_k, name="d_xn_qk")
        d_xnorm = mb.add(x=d_xn_qk, y=d_xn_v, name="d_xnorm_add")

        out = mb.cast(x=d_xnorm, dtype="fp32", name="d_xnorm")
        return out

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
    parser = argparse.ArgumentParser(
        description="Generate CoreML backward (dx) kernels for ANE training"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to safetensors model file")
    parser.add_argument("--config", required=True, choices=CONFIGS.keys(),
                        help="Model configuration")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for .mlpackage files")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to generate (default: all)")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading weights from {args.weights}...")
    weights = load_safetensors(args.weights)
    print(f"  {len(weights)} tensors loaded")

    n_layers = cfg["n_layers"]
    layers = (range(n_layers) if args.layers is None
              else [int(x) for x in args.layers.split(",")])

    total = len(layers) * 3  # FFN bwd + Wo bwd + QKV bwd per layer
    done = 0

    for l in layers:
        # FFN backward kernel
        path = os.path.join(args.output_dir, f"layer_{l:02d}_ffn_bwd.mlpackage")
        if os.path.exists(path):
            print(f"  [{done+1}/{total}] layer_{l:02d}_ffn_bwd — exists, skipping")
        else:
            print(f"  [{done+1}/{total}] layer_{l:02d}_ffn_bwd — generating...")
            prog = build_ffn_bwd_kernel(l, cfg, weights)
            convert_and_save(prog, path)
            print(f"           -> {path}")
        done += 1

        # Wo backward kernel
        path = os.path.join(args.output_dir, f"layer_{l:02d}_wo_bwd.mlpackage")
        if os.path.exists(path):
            print(f"  [{done+1}/{total}] layer_{l:02d}_wo_bwd  — exists, skipping")
        else:
            print(f"  [{done+1}/{total}] layer_{l:02d}_wo_bwd  — generating...")
            prog = build_wo_bwd_kernel(l, cfg, weights)
            convert_and_save(prog, path)
            print(f"           -> {path}")
        done += 1

        # QKV backward kernel
        path = os.path.join(args.output_dir, f"layer_{l:02d}_qkv_bwd.mlpackage")
        if os.path.exists(path):
            print(f"  [{done+1}/{total}] layer_{l:02d}_qkv_bwd — exists, skipping")
        else:
            print(f"  [{done+1}/{total}] layer_{l:02d}_qkv_bwd — generating...")
            prog = build_qkv_bwd_kernel(l, cfg, weights)
            convert_and_save(prog, path)
            print(f"           -> {path}")
        done += 1

    print(f"\nDone! {done} backward kernels in {args.output_dir}")
    print(f"  {len(layers)} layers x 3 (FFN bwd + Wo bwd + QKV bwd) = {done} total")


if __name__ == "__main__":
    main()
