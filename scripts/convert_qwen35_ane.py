#!/usr/bin/env python3
"""
Qwen3.5 DeltaNet → CoreML/ANE converter.

First-of-its-kind converter for Gated DeltaNet hybrid architectures on
Apple Neural Engine.  Handles both DeltaNet (recurrent linear attention)
and standard GQA attention layers in a single CoreML model.

Architecture:
  24 layers = 6 × (3 × [GatedDeltaNet → MLP] → 1 × [GatedAttention → MLP])
  - 18 DeltaNet layers (recurrent state, O(1) memory per token)
  - 6 Full Attention layers (KV cache, standard GQA)
  - Tied word embeddings (embed_tokens == lm_head)

State management:
  - KV cache: [2*6, num_kv_heads, context_length, attn_head_dim]
  - DeltaNet recurrent state: [18, Hv, Dv_per_head, Dk_per_head]
  - Conv1d sliding window: [18, conv_dim, kernel_size-1]

Usage:
  python scripts/convert_qwen35_ane.py --model Qwen/Qwen3.5-0.8B
  python scripts/convert_qwen35_ane.py --model Qwen/Qwen3.5-0.8B --context 512
  python scripts/convert_qwen35_ane.py --test-only  # skip conversion, test existing
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

MODEL_DTYPE = torch.float16
DEVICE = "cpu"  # Tracing always on CPU; CoreML handles HW dispatch
# ANE requires state tensor dimensions to be multiples of 32.
# Conv cache kernel_size-1 = 3 must be padded.
ANE_CONV_PAD = 32  # Pad conv cache trailing dim to this value


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────


class Qwen35Config:
    """Parsed Qwen3.5 text configuration from config.json."""

    def __init__(self, raw: dict):
        # The text config may be nested under 'text_config'
        tc = raw.get("text_config", raw)

        self.hidden_size: int = tc["hidden_size"]
        self.num_hidden_layers: int = tc["num_hidden_layers"]
        self.intermediate_size: int = tc["intermediate_size"]
        self.vocab_size: int = tc["vocab_size"]
        self.rms_norm_eps: float = tc.get("rms_norm_eps", 1e-6)
        self.tie_word_embeddings: bool = tc.get("tie_word_embeddings", True)

        # Full attention layers
        self.num_attention_heads: int = tc["num_attention_heads"]
        self.num_key_value_heads: int = tc["num_key_value_heads"]
        self.head_dim: int = tc.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.full_attention_interval: int = tc.get("full_attention_interval", 4)
        self.attn_output_gate: bool = tc.get("attn_output_gate", True)

        # RoPE
        self.rope_theta: float = tc.get("rope_theta", 10000000.0)
        self.partial_rotary_factor: float = tc.get("partial_rotary_factor", 0.25)
        self.rotary_dim: int = int(self.head_dim * self.partial_rotary_factor)

        # DeltaNet linear attention
        self.linear_num_key_heads: int = tc.get("linear_num_key_heads", 16)
        self.linear_num_value_heads: int = tc.get("linear_num_value_heads", 16)
        self.linear_key_head_dim: int = tc.get("linear_key_head_dim", 128)
        self.linear_value_head_dim: int = tc.get("linear_value_head_dim", 128)
        self.linear_conv_kernel_dim: int = tc.get("linear_conv_kernel_dim", 4)

        # Derived DeltaNet dimensions (these are PER-HEAD dims in the config)
        # Total projection sizes computed from weight shapes at load time
        self.dk_total: int = 0  # filled by weight loader
        self.dv_total: int = 0
        self.conv_dim: int = 0

        # Layer type schedule
        self.layer_types: List[str] = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.full_attention_interval == 0:
                self.layer_types.append("full_attention")
            else:
                self.layer_types.append("linear_attention")

        self.num_attn_layers = sum(1 for t in self.layer_types if t == "full_attention")
        self.num_delta_layers = sum(1 for t in self.layer_types if t == "linear_attention")

        # Runtime context length (set by converter)
        self.context_length: int = 512
        self.state_length: int = 512


# ──────────────────────────────────────────────────────────────────────
# ANE-Optimized Layer Implementations
# ──────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """RMSNorm matching Qwen3.5's convention: weight stores (w - 1.0).

    Qwen3.5 initializes norm weight to zeros and applies it as (1 + weight),
    so the stored parameter is the deviation from unity.  The forward is:
        output = rms_norm(x) * (1 + weight)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Stored weight represents (actual_scale - 1.0), matching Qwen3.5 convention
        self.weight = nn.Parameter(torch.zeros(dim, dtype=MODEL_DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).to(MODEL_DTYPE)


class RMSNormGated(nn.Module):
    """RMSNormGated matching Qwen3.5's Qwen3_5RMSNormGated convention.

    Unlike RMSNorm, the gated variant stores weight directly (initialized to ones),
    NOT as (weight - 1.0).  Applied as: weight * rms_norm(x) * silu(z).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=MODEL_DTYPE))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        # Norm first, then scale by weight, then gate — matching reference order
        normed = (self.weight.float() * normed.to(x.dtype).float())
        return (normed * F.silu(z.float())).to(MODEL_DTYPE)


def _rms_norm_no_weight(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Bare RMS normalization (no learnable weight) matching MLX's rms_norm(x, None, eps).

    Unlike F.layer_norm, this does NOT subtract the mean — it normalizes by
    root-mean-square only.  This is critical for DeltaNet Q/K normalization.
    """
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return x / rms


class SiLUMLP(nn.Module):
    """SiLU-gated MLP using Conv2d for ANE."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Conv2d(hidden, intermediate, 1, bias=False, dtype=MODEL_DTYPE)
        self.up_proj = nn.Conv2d(hidden, intermediate, 1, bias=False, dtype=MODEL_DTYPE)
        self.down_proj = nn.Conv2d(intermediate, hidden, 1, bias=False, dtype=MODEL_DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, hidden, 1, S] (Conv2d format)
        return self.down_proj(F.silu(self.gate_proj(x).float()).to(MODEL_DTYPE) * self.up_proj(x))


# ──────────────────────────────────────────────────────────────────────
# DeltaNet Layer
# ──────────────────────────────────────────────────────────────────────


class GatedDeltaNetLayer(nn.Module):
    """Gated DeltaNet recurrent linear attention for ANE.

    State: fixed-size matrix [Hv, Dv_per_head, Dk_per_head] per layer.
    Uses Conv2d projections for ANE compatibility.

    Per-step computation (decode, seq_len=1):
      1. Project input → qkv, a, b, z
      2. Conv1d with sliding window cache
      3. Split into q, k, v; normalize
      4. Compute decay: g = exp(-exp(A_log) * softplus(a + dt_bias))
      5. Delta rule update:
         state *= g
         mem = (state * k).sum(-1)
         delta = (v - mem) * sigmoid(b)
         state += outer(delta, k)
         y = (state * q).sum(-1)
      6. Output gate: RMSNormGated(y, z) → out_proj
    """

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.Hk = config.linear_num_key_heads
        self.Hv = config.linear_num_value_heads
        self.kernel_size = config.linear_conv_kernel_dim

        # These get set properly during weight loading when we know tensor shapes
        self.dk_total = 0
        self.dv_total = 0
        self.conv_dim = 0
        self.Dk_per_head = 0
        self.Dv_per_head = 0

        # Projections (Conv2d, sizes set during weight loading)
        # Placeholders — rebuilt in _init_projections after we know dims
        self.in_proj_qkv: Optional[nn.Conv2d] = None
        self.in_proj_a: Optional[nn.Conv2d] = None
        self.in_proj_b: Optional[nn.Conv2d] = None
        self.in_proj_z: Optional[nn.Conv2d] = None
        self.out_proj: Optional[nn.Conv2d] = None

        # Conv1d (depthwise, set during weight loading)
        self.conv1d: Optional[nn.Conv1d] = None

        # Learnable parameters
        self.A_log = nn.Parameter(torch.zeros(1, dtype=MODEL_DTYPE))  # resized during load
        self.dt_bias = nn.Parameter(torch.zeros(1, dtype=MODEL_DTYPE))

        # Norm (gated) — operates PER-HEAD on Dv_per_head dimension
        self.norm: Optional[RMSNormGated] = None

    def _init_projections(self, dk_total: int, dv_total: int):
        """Initialize projection layers once we know dimensions from weights."""
        self.dk_total = dk_total
        self.dv_total = dv_total
        self.conv_dim = 2 * dk_total + dv_total
        self.Dk_per_head = dk_total // self.Hk
        self.Dv_per_head = dv_total // self.Hv

        h = self.hidden_size
        self.in_proj_qkv = nn.Conv2d(h, self.conv_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.in_proj_a = nn.Conv2d(h, self.Hv, 1, bias=False, dtype=MODEL_DTYPE)
        self.in_proj_b = nn.Conv2d(h, self.Hv, 1, bias=False, dtype=MODEL_DTYPE)
        self.in_proj_z = nn.Conv2d(h, dv_total, 1, bias=False, dtype=MODEL_DTYPE)
        self.out_proj = nn.Conv2d(dv_total, h, 1, bias=False, dtype=MODEL_DTYPE)

        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, self.kernel_size,
            groups=self.conv_dim, bias=False, dtype=MODEL_DTYPE,
        )

        # Norm operates per-head: weight shape = [Dv_per_head]
        self.norm = RMSNormGated(self.Dv_per_head, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        delta_state: torch.Tensor,
        conv_cache: torch.Tensor,
        position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [1, hidden, 1, S] (Conv2d format)
            delta_state: [1, Hv, Dv_per_head, Dk_per_head] recurrent state
            conv_cache: [1, conv_dim, kernel-1] sliding window
            position: scalar position index

        Returns:
            output: [1, hidden, 1, S]
            new_delta_state: same shape as delta_state
            new_conv_cache: same shape as conv_cache
        """
        B, C, _, S = hidden_states.shape
        h = hidden_states.float()

        # 1. Project inputs
        qkv = self.in_proj_qkv(hidden_states).squeeze(2)  # [1, conv_dim, S]
        a = self.in_proj_a(hidden_states).squeeze(2)  # [1, Hv, S]
        b_raw = self.in_proj_b(hidden_states).squeeze(2)  # [1, Hv, S]
        z = self.in_proj_z(hidden_states).squeeze(2).permute(0, 2, 1)  # [1, S, dv_total]

        # 2. Conv1d with sliding window (cache may be ANE-padded)
        # Extract actual cache entries (first kernel-1 positions from padded cache)
        actual_cache = conv_cache[:, :, :self.kernel_size - 1]
        # Prepend cached tokens: [1, conv_dim, kernel-1 + S]
        conv_input = torch.cat([actual_cache, qkv], dim=2)
        # Save new cache (last kernel-1 elements, padded to ANE_CONV_PAD)
        new_actual = conv_input[:, :, -(self.kernel_size - 1):]
        # Pad to ANE_CONV_PAD for ANE alignment (zeros beyond kernel-1)
        pad_size = conv_cache.shape[2] - (self.kernel_size - 1)
        if pad_size > 0:
            new_conv_cache = F.pad(new_actual, (0, pad_size))
        else:
            new_conv_cache = new_actual
        # Apply depthwise conv1d + SiLU
        conv_out = F.silu(self.conv1d(conv_input).float())  # [1, conv_dim, S]

        # 3. Split into q, k, v and reshape
        q_flat = conv_out[:, :self.dk_total, :]  # [1, dk_total, S]
        k_flat = conv_out[:, self.dk_total:2*self.dk_total, :]  # [1, dk_total, S]
        v_flat = conv_out[:, 2*self.dk_total:, :]  # [1, dv_total, S]

        # Reshape to per-head: [B, H, S, D_per_head]
        q = q_flat.view(1, self.Hk, self.Dk_per_head, S).permute(0, 1, 3, 2).float()
        k = k_flat.view(1, self.Hk, self.Dk_per_head, S).permute(0, 1, 3, 2).float()
        v = v_flat.view(1, self.Hv, self.Dv_per_head, S).permute(0, 1, 3, 2).float()

        # 4. Normalize q, k with RMS norm (NOT layer_norm — no mean subtraction)
        inv_scale_k = (self.Dk_per_head ** -0.5)
        inv_scale_q = inv_scale_k ** 2
        q = inv_scale_q * _rms_norm_no_weight(q)
        k = inv_scale_k * _rms_norm_no_weight(k)

        # Expand k,v heads to match Hv if Hk != Hv
        if self.Hk != self.Hv:
            repeat = self.Hv // self.Hk
            k = k.repeat(1, repeat, 1, 1)
            q = q.repeat(1, repeat, 1, 1)  # q also uses Hk heads

        # 5. Compute decay and beta gate
        a_vals = a.permute(0, 2, 1).float()  # [1, S, Hv]
        b_vals = b_raw.permute(0, 2, 1).float()  # [1, S, Hv]

        # g = exp(-exp(A_log) * softplus(a + dt_bias))
        A = torch.exp(self.A_log.float())
        gate_input = F.softplus(a_vals + self.dt_bias.float())
        g = torch.exp(-A * gate_input)  # [1, S, Hv], decay in (0, 1)
        beta = torch.sigmoid(b_vals)  # [1, S, Hv], update gate in (0, 1)

        # 6. Sequential delta rule update over S tokens
        state = delta_state.float()  # [1, Hv, Dv_per_head, Dk_per_head]
        outputs = []

        for t in range(S):
            q_t = q[:, :, t, :]  # [1, Hv, Dk_per_head]
            k_t = k[:, :, t, :]  # [1, Hv, Dk_per_head]
            v_t = v[:, :, t, :]  # [1, Hv, Dv_per_head]
            g_t = g[:, t, :]     # [1, Hv]
            beta_t = beta[:, t, :]  # [1, Hv]

            # Decay
            state = state * g_t.unsqueeze(-1).unsqueeze(-1)

            # Read memory: dot product state with k
            # state: [1, Hv, Dv, Dk], k_t: [1, Hv, Dk]
            mem = (state * k_t.unsqueeze(-2)).sum(dim=-1)  # [1, Hv, Dv]

            # Delta update
            delta = (v_t - mem) * beta_t.unsqueeze(-1)  # [1, Hv, Dv]

            # Write: outer product of delta and k
            state = state + delta.unsqueeze(-1) * k_t.unsqueeze(-2)

            # Read output
            y_t = (state * q_t.unsqueeze(-2)).sum(dim=-1)  # [1, Hv, Dv]
            outputs.append(y_t)

        # Stack outputs: [1, Hv, Dv_per_head, S] → [1, S, Hv, Dv_per_head]
        y = torch.stack(outputs, dim=-1)  # [1, Hv, Dv_per_head, S]
        y = y.permute(0, 3, 1, 2)  # [1, S, Hv, Dv_per_head]

        # Reshape z to per-head: [1, S, dv_total] → [1, S, Hv, Dv_per_head]
        z_heads = z.view(1, S, self.Hv, self.Dv_per_head)

        # 7. Gated output (norm operates per-head on last dim = Dv_per_head)
        y_normed = self.norm(y.to(MODEL_DTYPE), z_heads.to(MODEL_DTYPE))  # [1, S, Hv, Dv_per_head]

        # Flatten heads: [1, S, Hv, Dv_per_head] → [1, S, dv_total]
        y_flat = y_normed.reshape(1, S, self.dv_total)

        # Reshape for Conv2d out_proj: [1, dv_total, 1, S]
        y_conv = y_flat.permute(0, 2, 1).unsqueeze(2)
        output = self.out_proj(y_conv)  # [1, hidden, 1, S]

        return output, state.to(MODEL_DTYPE), new_conv_cache


# ──────────────────────────────────────────────────────────────────────
# Full Attention Layer (Gated)
# ──────────────────────────────────────────────────────────────────────


class GatedAttentionLayer(nn.Module):
    """Standard GQA attention with output sigmoid gate, for ANE."""

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attn_output_gate = config.attn_output_gate
        self.rotary_dim = config.rotary_dim

        # Q projection: if output_gate, q_proj outputs 2x for sigmoid gate
        q_out = self.num_heads * self.head_dim * (2 if self.attn_output_gate else 1)
        kv_out = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Conv2d(self.hidden_size, q_out, 1, bias=False, dtype=MODEL_DTYPE)
        self.k_proj = nn.Conv2d(self.hidden_size, kv_out, 1, bias=False, dtype=MODEL_DTYPE)
        self.v_proj = nn.Conv2d(self.hidden_size, kv_out, 1, bias=False, dtype=MODEL_DTYPE)
        self.o_proj = nn.Conv2d(self.num_heads * self.head_dim, self.hidden_size, 1,
                                bias=False, dtype=MODEL_DTYPE)

        # Q/K normalization
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # RoPE (pre-cached)
        self.rope_theta = config.rope_theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        attn_layer_idx: int,
        cos_cached: torch.Tensor,
        sin_cached: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [1, hidden, 1, S]
            kv_cache: [2*num_attn_layers, num_kv_heads, ctx_len, head_dim]
            causal_mask: [1, 1, S, ctx_len]
            position_ids: [S]
            attn_layer_idx: index into the 6 attention layers (0-5)
            cos_cached, sin_cached: [max_len, rotary_dim]

        Returns:
            output: [1, hidden, 1, S]
            kv_cache: updated
        """
        B, C, _, S = hidden_states.shape

        # Project
        q_raw = self.q_proj(hidden_states).squeeze(2)  # [1, q_out, S]
        k_raw = self.k_proj(hidden_states).squeeze(2)  # [1, kv_out, S]
        v_raw = self.v_proj(hidden_states).squeeze(2)  # [1, kv_out, S]

        # Reshape Q: handle output gate (doubled Q)
        if self.attn_output_gate:
            q_and_gate = q_raw.view(1, self.num_heads, 2 * self.head_dim, S)
            q = q_and_gate[:, :, :self.head_dim, :].permute(0, 1, 3, 2).float()
            gate = q_and_gate[:, :, self.head_dim:, :].permute(0, 1, 3, 2).float()
        else:
            q = q_raw.view(1, self.num_heads, self.head_dim, S).permute(0, 1, 3, 2).float()
            gate = None

        k = k_raw.view(1, self.num_kv_heads, self.head_dim, S).permute(0, 1, 3, 2).float()
        v = v_raw.view(1, self.num_kv_heads, self.head_dim, S).permute(0, 1, 3, 2).float()

        # Q/K normalization (returns float16, cast back to float32 for attention math)
        q = self.q_norm(q).float()
        k = self.k_norm(k).float()

        # Apply partial RoPE (only first rotary_dim dimensions)
        if self.rotary_dim > 0:
            q_rot = q[..., :self.rotary_dim]
            q_pass = q[..., self.rotary_dim:]
            k_rot = k[..., :self.rotary_dim]
            k_pass = k[..., self.rotary_dim:]

            cos = cos_cached[position_ids].unsqueeze(0).unsqueeze(0)  # [1, 1, S, rotary_dim]
            sin = sin_cached[position_ids].unsqueeze(0).unsqueeze(0)

            q_rot = q_rot * cos + _rotate_half(q_rot) * sin
            k_rot = k_rot * cos + _rotate_half(k_rot) * sin

            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        # Update KV cache
        k_idx = attn_layer_idx * 2
        v_idx = k_idx + 1

        if S == 1:
            # Decode: one-hot mask for position update (trace-friendly for CoreML)
            ctx_len = kv_cache.shape[2]
            pos_mask = F.one_hot(position_ids[0:1].long(), num_classes=ctx_len).to(MODEL_DTYPE)
            pos_mask = pos_mask.unsqueeze(-1)  # [1, ctx, 1]
            k_new = k.squeeze(0).to(MODEL_DTYPE)  # [kv_heads, 1, head_dim]
            v_new = v.squeeze(0).to(MODEL_DTYPE)
            kv_cache[k_idx] = kv_cache[k_idx] * (1.0 - pos_mask) + k_new * pos_mask
            kv_cache[v_idx] = kv_cache[v_idx] * (1.0 - pos_mask) + v_new * pos_mask
        else:
            # Prefill: direct slice assignment (Python only, not traced)
            pos_start = position_ids[0].item()
            kv_cache[k_idx, :, pos_start:pos_start + S, :] = k.squeeze(0).to(MODEL_DTYPE)
            kv_cache[v_idx, :, pos_start:pos_start + S, :] = v.squeeze(0).to(MODEL_DTYPE)

        # Read full cache for attention
        k_full = kv_cache[k_idx].unsqueeze(0).float()  # [1, kv_heads, ctx, head_dim]
        v_full = kv_cache[v_idx].unsqueeze(0).float()

        # Expand KV heads for GQA using expand+reshape (ANE-friendly, no repeat_interleave)
        # [1, kv_heads, ctx, dim] → [1, kv_heads, 1, ctx, dim] → expand → reshape
        # Produces interleaved order: [kv0, kv0, kv0, kv0, kv1, kv1, kv1, kv1]
        if self.num_kv_groups > 1:
            g = self.num_kv_groups
            # [1, kv_heads, ctx, dim] → [1, kv_heads, 1, ctx, dim]
            k_full = k_full.unsqueeze(2).expand(-1, -1, g, -1, -1)
            k_full = k_full.reshape(1, self.num_heads, -1, self.head_dim)
            v_full = v_full.unsqueeze(2).expand(-1, -1, g, -1, -1)
            v_full = v_full.reshape(1, self.num_heads, -1, self.head_dim)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k_full.transpose(-2, -1)) * scale
        attn_weights = attn_weights + causal_mask.float()
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v_full)  # [1, num_heads, S, head_dim]

        # Apply output gate
        if gate is not None:
            attn_out = attn_out * torch.sigmoid(gate)

        # Reshape and project
        attn_out = attn_out.permute(0, 1, 3, 2)  # [1, num_heads, head_dim, S]
        attn_out = attn_out.reshape(1, self.num_heads * self.head_dim, S)
        attn_out = attn_out.unsqueeze(2).to(MODEL_DTYPE)  # [1, out_dim, 1, S]
        output = self.o_proj(attn_out)

        return output, kv_cache


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half for RoPE: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Full Model
# ──────────────────────────────────────────────────────────────────────


class Qwen35ForANE(nn.Module):
    """Qwen3.5 model optimized for Apple Neural Engine via CoreML.

    Combines DeltaNet and GQA attention layers with Conv2d projections,
    unified state management, and RMSNorm-via-LayerNorm optimization.
    """

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config = config
        h = config.hidden_size

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, h, dtype=MODEL_DTYPE)

        # Build layers
        self.layers = nn.ModuleList()
        self.input_layernorms = nn.ModuleList()
        self.post_attention_layernorms = nn.ModuleList()
        self.mlps = nn.ModuleList()

        self.attn_layer_indices = []  # maps global layer idx → attn cache idx
        attn_idx = 0
        delta_idx = 0
        self.delta_layer_indices = []  # maps global layer idx → delta state idx

        for i in range(config.num_hidden_layers):
            self.input_layernorms.append(RMSNorm(h, config.rms_norm_eps))
            self.post_attention_layernorms.append(RMSNorm(h, config.rms_norm_eps))
            self.mlps.append(SiLUMLP(h, config.intermediate_size))

            if config.layer_types[i] == "full_attention":
                self.layers.append(GatedAttentionLayer(config, i))
                self.attn_layer_indices.append(attn_idx)
                self.delta_layer_indices.append(-1)
                attn_idx += 1
            else:
                self.layers.append(GatedDeltaNetLayer(config, i))
                self.attn_layer_indices.append(-1)
                self.delta_layer_indices.append(delta_idx)
                delta_idx += 1

        # Final norm
        self.norm = RMSNorm(h, config.rms_norm_eps)

        # LM head (tied to embeddings)
        # We'll use a Conv2d with tied weights for tracing
        self.lm_head_weight = None  # set during weight loading

        # Pre-compute RoPE cos/sin
        self._init_rope(config)

    def _init_rope(self, config: Qwen35Config):
        """Pre-compute rotary embedding cos/sin tables."""
        dim = config.rotary_dim
        if dim == 0:
            self.register_buffer("cos_cached", torch.zeros(1, 1))
            self.register_buffer("sin_cached", torch.zeros(1, 1))
            return

        inv_freq = 1.0 / (config.rope_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32) / dim
        ))
        max_len = max(config.context_length, config.state_length) * 2
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_len, dim]
        self.register_buffer("cos_cached", emb.cos().to(MODEL_DTYPE))
        self.register_buffer("sin_cached", emb.sin().to(MODEL_DTYPE))

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        kv_cache: torch.Tensor,
        delta_states: torch.Tensor,
        conv_caches: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [1, S]
            position_ids: [S]
            causal_mask: [1, 1, S, context_length]
            kv_cache: [2*num_attn, kv_heads, ctx, head_dim]
            delta_states: [num_delta, Hv, Dv_per_head, Dk_per_head]
            conv_caches: [num_delta, conv_dim, kernel-1]

        Returns:
            logits: [1, S, vocab_size]
            kv_cache: updated
            delta_states: updated
        """
        # Embed
        hidden = self.embed_tokens(input_ids)  # [1, S, hidden]

        # Process through layers
        for i, layer in enumerate(self.layers):
            # Pre-norm
            normed = self.input_layernorms[i](hidden)

            # Reshape for Conv2d: [1, S, H] → [1, H, 1, S]
            normed_conv = normed.permute(0, 2, 1).unsqueeze(2)

            if self.config.layer_types[i] == "full_attention":
                attn_idx = self.attn_layer_indices[i]
                residual, kv_cache = layer(
                    normed_conv, kv_cache, causal_mask, position_ids,
                    attn_idx, self.cos_cached, self.sin_cached,
                )
                # Reshape back: [1, H, 1, S] → [1, S, H]
                residual = residual.squeeze(2).permute(0, 2, 1)
            else:
                d_idx = self.delta_layer_indices[i]
                d_state = delta_states[d_idx:d_idx+1]
                c_cache = conv_caches[d_idx:d_idx+1]
                residual, new_d_state, new_c_cache = layer(
                    normed_conv, d_state, c_cache, position_ids,
                )
                delta_states[d_idx] = new_d_state.squeeze(0)
                conv_caches[d_idx] = new_c_cache.squeeze(0)
                residual = residual.squeeze(2).permute(0, 2, 1)

            hidden = hidden + residual

            # Post-attention norm + MLP
            normed2 = self.post_attention_layernorms[i](hidden)
            normed2_conv = normed2.permute(0, 2, 1).unsqueeze(2)
            mlp_out = self.mlps[i](normed2_conv)
            hidden = hidden + mlp_out.squeeze(2).permute(0, 2, 1)

        # Final norm
        hidden = self.norm(hidden)

        # LM head (tied to embed_tokens)
        logits = F.linear(hidden.float(), self.embed_tokens.weight.float())

        return logits.to(MODEL_DTYPE), kv_cache, delta_states


# ──────────────────────────────────────────────────────────────────────
# Weight Loading
# ──────────────────────────────────────────────────────────────────────


def load_qwen35_weights(
    model: Qwen35ForANE,
    config: Qwen35Config,
    model_path: Path,
) -> None:
    """Load Qwen3.5 weights from HuggingFace safetensors into ANE model."""
    from safetensors.torch import load_file

    # Find safetensors files
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files in {model_path}")

    print(f"  Loading weights from {len(st_files)} file(s)...")
    all_weights: Dict[str, torch.Tensor] = {}
    for f in st_files:
        all_weights.update(load_file(str(f), device="cpu"))

    # Determine DeltaNet dimensions from weight shapes
    # Find first DeltaNet layer's in_proj_qkv weight
    for i, ltype in enumerate(config.layer_types):
        if ltype == "linear_attention":
            qkv_key = f"model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight"
            if qkv_key in all_weights:
                qkv_shape = all_weights[qkv_key].shape
                conv_dim = qkv_shape[0]  # output features

                # Also check in_proj_z to get dv_total
                z_key = f"model.language_model.layers.{i}.linear_attn.in_proj_z.weight"
                dv_total = all_weights[z_key].shape[0]
                dk_total = (conv_dim - dv_total) // 2

                config.dk_total = dk_total
                config.dv_total = dv_total
                config.conv_dim = conv_dim

                print(f"  DeltaNet dims: dk_total={dk_total}, dv_total={dv_total}, "
                      f"conv_dim={conv_dim}")
                print(f"  Per-head: Dk={dk_total // config.linear_num_key_heads}, "
                      f"Dv={dv_total // config.linear_num_value_heads}")
                break

    # Initialize DeltaNet projections now that we know dimensions
    for i, layer in enumerate(model.layers):
        if isinstance(layer, GatedDeltaNetLayer):
            layer._init_projections(config.dk_total, config.dv_total)

    # Map weights
    prefix = "model.language_model."

    # Embeddings
    model.embed_tokens.weight.data = all_weights[prefix + "embed_tokens.weight"].to(MODEL_DTYPE)

    # Final norm
    model.norm.weight.data = all_weights[prefix + "norm.weight"].to(MODEL_DTYPE)

    # Per-layer weights
    for i in range(config.num_hidden_layers):
        lp = f"{prefix}layers.{i}."

        # Layer norms
        model.input_layernorms[i].weight.data = all_weights[lp + "input_layernorm.weight"].to(MODEL_DTYPE)
        model.post_attention_layernorms[i].weight.data = all_weights[lp + "post_attention_layernorm.weight"].to(MODEL_DTYPE)

        # MLP (reshape Linear → Conv2d [out, in, 1, 1])
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            w = all_weights[lp + f"mlp.{proj_name}.weight"].to(MODEL_DTYPE)
            getattr(model.mlps[i], proj_name).weight.data = w.view(w.shape[0], w.shape[1], 1, 1)

        layer = model.layers[i]

        if config.layer_types[i] == "linear_attention":
            la_prefix = lp + "linear_attn."

            # Projections → Conv2d
            for proj in ["in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"]:
                w = all_weights[la_prefix + f"{proj}.weight"].to(MODEL_DTYPE)
                getattr(layer, proj).weight.data = w.view(w.shape[0], w.shape[1], 1, 1)

            # Conv1d weight: HF stores [out_channels, in_channels/groups, kernel_size]
            # For depthwise conv (groups=conv_dim): [6144, 1, 4] — already correct
            conv_w = all_weights[la_prefix + "conv1d.weight"].to(MODEL_DTYPE)
            if conv_w.ndim == 2:
                # [channels, kernel] → [channels, 1, kernel]
                conv_w = conv_w.unsqueeze(1)
            layer.conv1d.weight.data = conv_w

            # A_log and dt_bias
            layer.A_log = nn.Parameter(all_weights[la_prefix + "A_log"].to(MODEL_DTYPE))
            layer.dt_bias = nn.Parameter(all_weights[la_prefix + "dt_bias"].to(MODEL_DTYPE))

            # Norm weight
            layer.norm.weight.data = all_weights[la_prefix + "norm.weight"].to(MODEL_DTYPE)

        else:
            sa_prefix = lp + "self_attn."

            # Q/K/V/O projections → Conv2d
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                w = all_weights[sa_prefix + f"{proj}.weight"].to(MODEL_DTYPE)
                getattr(layer, proj).weight.data = w.view(w.shape[0], w.shape[1], 1, 1)

            # Q/K norms
            layer.q_norm.weight.data = all_weights[sa_prefix + "q_norm.weight"].to(MODEL_DTYPE)
            layer.k_norm.weight.data = all_weights[sa_prefix + "k_norm.weight"].to(MODEL_DTYPE)

    print(f"  Loaded {len(all_weights)} weight tensors successfully.")


# ──────────────────────────────────────────────────────────────────────
# CoreML Conversion
# ──────────────────────────────────────────────────────────────────────


class Qwen35Wrapper(nn.Module):
    """Wrapper for CoreML conversion with stateful buffers.

    CoreML StateType requires states to be named_buffers that are
    modified in-place during forward(). The forward() only takes
    non-state inputs; states live as self.kv_cache etc.
    """

    def __init__(self, model: Qwen35ForANE, kv_shape, delta_shape, conv_shape):
        super().__init__()
        self.model = model
        # Register states as buffers so CoreML can track them
        self.register_buffer("kv_cache", torch.zeros(kv_shape, dtype=MODEL_DTYPE))
        self.register_buffer("delta_states", torch.zeros(delta_shape, dtype=MODEL_DTYPE))
        self.register_buffer("conv_caches", torch.zeros(conv_shape, dtype=MODEL_DTYPE))

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits, _, _ = self.model(
            input_ids, position_ids, causal_mask,
            self.kv_cache, self.delta_states, self.conv_caches,
        )
        return logits


def convert_to_coreml(
    model: Qwen35ForANE,
    config: Qwen35Config,
    output_dir: Path,
    context_length: int = 512,
) -> Path:
    """Convert the Qwen3.5 model to CoreML format.

    Creates a stateful CoreML model with:
    - KV cache state for attention layers
    - DeltaNet recurrent state
    - Conv1d sliding window cache

    Returns path to the compiled .mlmodelc directory.
    """
    import coremltools as ct

    model.eval()

    Dk_per_head = config.dk_total // config.linear_num_key_heads
    Dv_per_head = config.dv_total // config.linear_num_value_heads

    # State tensor shapes
    kv_shape = (2 * config.num_attn_layers, config.num_key_value_heads,
                context_length, config.head_dim)
    delta_shape = (config.num_delta_layers, config.linear_num_value_heads,
                   Dv_per_head, Dk_per_head)
    conv_shape = (config.num_delta_layers, config.conv_dim,
                  ANE_CONV_PAD)

    print(f"  State shapes:")
    print(f"    KV cache:      {kv_shape}")
    print(f"    Delta states:  {delta_shape}")
    print(f"    Conv caches:   {conv_shape}")

    # Build wrapper with state buffers
    wrapper = Qwen35Wrapper(model, kv_shape, delta_shape, conv_shape)
    wrapper.eval()

    # Sample inputs for tracing (single token decode)
    sample_input_ids = torch.zeros(1, 1, dtype=torch.long)
    sample_position_ids = torch.zeros(1, dtype=torch.long)
    sample_causal_mask = torch.zeros(1, 1, 1, context_length, dtype=MODEL_DTYPE)

    # Trace (states are self.kv_cache etc., not forward args)
    print(f"  Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (sample_input_ids, sample_position_ids, sample_causal_mask),
        )

    # Define CoreML inputs (non-state only)
    inputs = [
        ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="position_ids", shape=(1,), dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, context_length),
                       dtype=np.float16),
    ]

    # Define CoreML states (name must match named_buffers keys)
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=kv_shape, dtype=np.float16),
            name="kv_cache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=delta_shape, dtype=np.float16),
            name="delta_states",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=conv_shape, dtype=np.float16),
            name="conv_caches",
        ),
    ]

    # Convert
    print(f"  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        states=states,
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    mlpackage_path = output_dir / "qwen35_deltanet.mlpackage"
    mlmodel.save(str(mlpackage_path))
    print(f"  Saved to: {mlpackage_path}")

    # Compile .mlpackage → .mlmodelc
    print(f"  Compiling to .mlmodelc...")
    import shutil
    compiled_path = output_dir / "qwen35_deltanet.mlmodelc"
    compiled_url = ct.utils.compile_model(str(mlpackage_path))
    if compiled_path.exists():
        shutil.rmtree(compiled_path)
    shutil.move(compiled_url, str(compiled_path))
    print(f"  Compiled to: {compiled_path}")

    # Write meta.yaml
    meta = {
        "model_info": {
            "name": "Qwen3.5-DeltaNet-ANE",
            "architecture": "hybrid_deltanet_gqa",
            "parameters": {
                "hidden_size": config.hidden_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attn_layers": config.num_attn_layers,
                "num_delta_layers": config.num_delta_layers,
                "vocab_size": config.vocab_size,
                "context_length": context_length,
                "dk_total": config.dk_total,
                "dv_total": config.dv_total,
                "conv_dim": config.conv_dim,
            },
        },
    }
    import yaml
    meta_path = output_dir / "meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False)
    print(f"  meta.yaml written to {meta_path}")

    return mlpackage_path


# ──────────────────────────────────────────────────────────────────────
# Inference Test
# ──────────────────────────────────────────────────────────────────────


def test_pytorch_forward(
    model: Qwen35ForANE,
    config: Qwen35Config,
    tokenizer,
    prompt: str = "What is 2 + 2?",
    max_tokens: int = 32,
) -> str:
    """Test the PyTorch model with greedy decoding (no CoreML)."""
    model.eval()

    Dk_per_head = config.dk_total // config.linear_num_key_heads
    Dv_per_head = config.dv_total // config.linear_num_value_heads
    ctx = config.context_length

    # Tokenize
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            formatted = prompt
    else:
        formatted = prompt

    input_ids = tokenizer.encode(formatted)
    print(f"  Input tokens: {len(input_ids)}")

    # Initialize states
    kv_cache = torch.zeros(
        2 * config.num_attn_layers, config.num_key_value_heads, ctx, config.head_dim,
        dtype=MODEL_DTYPE,
    )
    delta_states = torch.zeros(
        config.num_delta_layers, config.linear_num_value_heads, Dv_per_head, Dk_per_head,
        dtype=MODEL_DTYPE,
    )
    conv_caches = torch.zeros(
        config.num_delta_layers, config.conv_dim, config.linear_conv_kernel_dim - 1,
        dtype=MODEL_DTYPE,
    )

    generated_ids = []
    eos_id = getattr(tokenizer, "eos_token_id", None) or 248044

    t0 = time.time()

    with torch.no_grad():
        # Prefill: process all input tokens in one forward pass
        seq_len = len(input_ids)
        ids = torch.tensor([input_ids], dtype=torch.long)  # [1, seq_len]
        pos_ids = torch.arange(seq_len, dtype=torch.long)  # [seq_len]
        mask = _make_causal_mask(seq_len, 0, ctx)  # [1, 1, seq_len, ctx]

        logits, kv_cache, delta_states = model(
            ids, pos_ids, mask, kv_cache, delta_states, conv_caches,
        )

        ttft = (time.time() - t0) * 1000

        # Diagnostics: check logits health
        last_logits = logits[0, -1].float()
        top5_vals, top5_ids = torch.topk(last_logits, 5)
        print(f"  Logits stats: min={last_logits.min():.2f}, max={last_logits.max():.2f}, "
              f"mean={last_logits.mean():.4f}, std={last_logits.std():.2f}")
        print(f"  Top-5 tokens: {[(int(i), f'{v:.2f}') for i, v in zip(top5_ids, top5_vals)]}")
        decoded_top5 = [tokenizer.decode([int(i)]) for i in top5_ids]
        print(f"  Top-5 decoded: {decoded_top5}")

        # Decode
        next_token = int(logits[0, -1].argmax())
        generated_ids.append(next_token)

        for step in range(max_tokens - 1):
            if next_token == eos_id:
                break

            pos = len(input_ids) + step
            ids = torch.tensor([[next_token]], dtype=torch.long)
            pos_ids = torch.tensor([pos], dtype=torch.long)
            mask = _make_causal_mask(1, pos, ctx)

            logits, kv_cache, delta_states = model(
                ids, pos_ids, mask, kv_cache, delta_states, conv_caches,
            )
            next_token = int(logits[0, -1].argmax())
            generated_ids.append(next_token)

    elapsed = (time.time() - t0) * 1000
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"  TTFT: {ttft:.0f}ms, Total: {elapsed:.0f}ms, "
          f"Tokens: {len(generated_ids)}, "
          f"Tok/s: {len(generated_ids) / (elapsed / 1000):.1f}")
    print(f"  Output: {text[:200]}")
    print(f"  Raw token IDs: {generated_ids[:20]}")

    return text


def _make_causal_mask(
    seq_len: int, start: int, total: int
) -> torch.Tensor:
    """Create causal attention mask [1, 1, seq_len, total]."""
    mask = torch.zeros(1, 1, seq_len, total, dtype=MODEL_DTYPE)
    # Mask future positions with large negative
    for i in range(seq_len):
        mask[0, 0, i, start + i + 1:] = torch.finfo(MODEL_DTYPE).min
    return mask


def test_coreml_model(
    model_dir: Path,
    tokenizer,
    prompt: str = "What is 2 + 2?",
    max_tokens: int = 32,
    context_length: int = 512,
) -> str:
    """Test the converted CoreML model with greedy decoding."""
    import coremltools as ct

    mlpackage_path = model_dir / "qwen35_deltanet.mlpackage"
    if not mlpackage_path.exists():
        raise FileNotFoundError(f"No .mlpackage found at {mlpackage_path}")

    print(f"  Loading CoreML model from {mlpackage_path}")
    model = ct.models.MLModel(str(mlpackage_path), compute_units=ct.ComputeUnit.ALL)
    state = model.make_state()

    # Tokenize
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            formatted = prompt
    else:
        formatted = prompt

    input_ids = tokenizer.encode(formatted)
    print(f"  Input tokens: {len(input_ids)}")

    eos_id = getattr(tokenizer, "eos_token_id", None) or 248044
    generated = []

    t0 = time.time()

    # Prefill: one token at a time through CoreML
    for i, tok in enumerate(input_ids):
        mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
        mask[0, 0, 0, i + 1:] = np.float16(-65504.0)
        out = model.predict({
            "input_ids": np.array([[tok]], dtype=np.int32),
            "position_ids": np.array([i], dtype=np.int32),
            "causal_mask": mask,
        }, state)

    ttft = (time.time() - t0) * 1000

    # First generated token from last prefill logits
    logits = out["logits"]
    next_tok = int(np.argmax(logits[0, 0]))
    generated.append(next_tok)

    # Diagnostics
    top5_ids = np.argsort(logits[0, 0])[-5:][::-1]
    top5_vals = logits[0, 0, top5_ids]
    print(f"  Logits: min={logits.min():.2f}, max={logits.max():.2f}, std={logits.std():.2f}")
    print(f"  Top-5: {[(int(i), f'{v:.2f}') for i, v in zip(top5_ids, top5_vals)]}")
    decoded_top5 = [tokenizer.decode([int(i)]) for i in top5_ids]
    print(f"  Top-5 decoded: {decoded_top5}")

    # Decode
    for step in range(max_tokens - 1):
        if next_tok == eos_id:
            break
        pos = len(input_ids) + step
        mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
        mask[0, 0, 0, pos + 1:] = np.float16(-65504.0)
        out = model.predict({
            "input_ids": np.array([[next_tok]], dtype=np.int32),
            "position_ids": np.array([pos], dtype=np.int32),
            "causal_mask": mask,
        }, state)
        next_tok = int(np.argmax(out["logits"][0, 0]))
        generated.append(next_tok)

    elapsed = (time.time() - t0) * 1000
    text = tokenizer.decode(generated, skip_special_tokens=True)
    tok_per_s = len(generated) / ((elapsed - ttft) / 1000) if elapsed > ttft else 0

    print(f"  TTFT: {ttft:.0f}ms, Total: {elapsed:.0f}ms, "
          f"Tokens: {len(generated)}, Tok/s: {tok_per_s:.1f}")
    print(f"  Output: {text[:200]}")
    return text


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def find_hf_model_path(model_id: str) -> Path:
    """Find downloaded HuggingFace model in cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    slug = f"models--{model_id.replace('/', '--')}"
    model_dir = cache_dir / slug / "snapshots"

    if model_dir.exists():
        snapshots = sorted(model_dir.iterdir())
        if snapshots:
            return snapshots[-1]

    # Try downloading
    print(f"  Model not in cache, downloading {model_id}...")
    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id)
    return Path(path)


def main():
    ap = argparse.ArgumentParser(
        description="Convert Qwen3.5 (DeltaNet) to CoreML for Apple Neural Engine."
    )
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                    help="HuggingFace model ID (default: Qwen/Qwen3.5-0.8B)")
    ap.add_argument("--output", default="models/ane/qwen3.5-deltanet",
                    help="Output directory for CoreML model")
    ap.add_argument("--context", type=int, default=512,
                    help="Context length (default: 512)")
    ap.add_argument("--test-pytorch", action="store_true",
                    help="Test PyTorch model only (skip CoreML conversion)")
    ap.add_argument("--test-only", action="store_true",
                    help="Skip conversion, test existing CoreML model")
    ap.add_argument("--prompt", default="What is 2 + 2?",
                    help="Test prompt")
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="Max tokens to generate in test")

    args = ap.parse_args()

    print(f"\n{'='*72}")
    print(f"  Qwen3.5 DeltaNet → CoreML/ANE Converter")
    print(f"{'='*72}\n")

    # 1. Find model
    print(f"[1/4] Locating model: {args.model}")
    model_path = find_hf_model_path(args.model)
    print(f"  Path: {model_path}")

    # 2. Load config
    print(f"\n[2/4] Loading configuration")
    config_path = model_path / "config.json"
    with open(config_path) as f:
        raw_config = json.load(f)
    config = Qwen35Config(raw_config)
    config.context_length = args.context
    config.state_length = args.context

    print(f"  hidden_size:      {config.hidden_size}")
    print(f"  num_layers:       {config.num_hidden_layers}")
    print(f"  attn_layers:      {config.num_attn_layers}")
    print(f"  delta_layers:     {config.num_delta_layers}")
    print(f"  vocab_size:       {config.vocab_size}")
    print(f"  context:          {config.context_length}")
    print(f"  attn_heads:       {config.num_attention_heads} (Q), {config.num_key_value_heads} (KV)")
    print(f"  delta_heads:      {config.linear_num_key_heads} (K), {config.linear_num_value_heads} (V)")
    print(f"  head_dim (attn):  {config.head_dim}")
    print(f"  partial_rotary:   {config.partial_rotary_factor}")

    # 3. Build and load model
    print(f"\n[3/4] Building ANE-optimized model and loading weights")
    model = Qwen35ForANE(config)
    load_qwen35_weights(model, config, model_path)
    model.eval()

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # 4. Test or convert
    if args.test_pytorch:
        print(f"\n[4/4] Testing PyTorch forward pass")
        test_pytorch_forward(model, config, tokenizer, args.prompt, args.max_tokens)
    elif args.test_only:
        print(f"\n[4/4] Testing existing CoreML model")
        test_coreml_model(Path(args.output), tokenizer, args.prompt, args.max_tokens,
                          args.context)
    else:
        print(f"\n[4/4] Converting to CoreML")
        # First test PyTorch
        print(f"\n  --- PyTorch sanity check ---")
        test_pytorch_forward(model, config, tokenizer, args.prompt, args.max_tokens)

        print(f"\n  --- CoreML conversion ---")
        output_dir = Path(args.output)
        convert_to_coreml(model, config, output_dir, args.context)

        print(f"\n  --- CoreML inference test ---")
        test_coreml_model(output_dir, tokenizer, args.prompt, args.max_tokens,
                          args.context)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
