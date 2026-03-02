"""Small decoder-only Transformer used for CUDA local training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("PyTorch is required for CUDA path. Install torch first.") from exc
    return torch, nn, F


@dataclass(frozen=True)
class TinyLMConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_mult: int = 4
    dropout: float = 0.1


def build_tiny_causal_lm(config: TinyLMConfig):
    """Factory to avoid importing torch at module import time."""

    torch, nn, F = _require_torch()

    class _CausalSelfAttention(nn.Module):
        def __init__(self, cfg: TinyLMConfig):
            super().__init__()
            self.n_heads = cfg.num_heads
            self.head_dim = cfg.hidden_size // cfg.num_heads
            self.qkv = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=False)
            self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
            self.dropout = cfg.dropout

        def forward(self, x):
            batch_size, seq_len, channels = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)

            q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
            return self.out_proj(attn)

    class _TransformerBlock(nn.Module):
        def __init__(self, cfg: TinyLMConfig):
            super().__init__()
            inner = cfg.hidden_size * cfg.ffn_mult
            self.ln1 = nn.LayerNorm(cfg.hidden_size)
            self.attn = _CausalSelfAttention(cfg)
            self.ln2 = nn.LayerNorm(cfg.hidden_size)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.hidden_size, inner),
                nn.GELU(),
                nn.Linear(inner, cfg.hidden_size),
                nn.Dropout(cfg.dropout),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class TinyCausalLM(nn.Module):
        def __init__(self, cfg: TinyLMConfig):
            super().__init__()
            self.config = cfg
            self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
            self.drop = nn.Dropout(cfg.dropout)
            self.blocks = nn.ModuleList([_TransformerBlock(cfg) for _ in range(cfg.num_layers)])
            self.norm = nn.LayerNorm(cfg.hidden_size)
            self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

            self.head.weight = self.token_embed.weight
            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(module):
            if hasattr(module, "weight") and module.__class__.__name__.startswith("Linear"):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            if module.__class__.__name__ == "Embedding":
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(
            self,
            input_ids,
            target_ids: Optional[object] = None,
        ) -> Tuple[object, Optional[object]]:
            batch_size, seq_len = input_ids.shape
            if seq_len > self.config.max_seq_len:
                raise ValueError(
                    f"Input sequence ({seq_len}) exceeds max_seq_len ({self.config.max_seq_len})"
                )

            positions = torch.arange(seq_len, device=input_ids.device)
            x = self.token_embed(input_ids) + self.pos_embed(positions)[None, :, :]
            x = self.drop(x)
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.norm(x))

            loss = None
            if target_ids is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=-100,
                )
            return logits, loss

    return TinyCausalLM(config)
