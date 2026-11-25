"""
Causal Multi-Head Self-Attention with Rotary Position Embeddings (RoPE)
for ARC-LB-01. Supports both Dense (nn.Linear) and Native LittleBitLinear.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.littlebit.native_operations import LittleBitLinear


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotated * sin


def _make_proj(
    in_features: int,
    out_features: int,
    use_littlebit: bool,
    split_dim: int,
    residual: bool,
    use_itq: bool,
    quant_func_name: str,
    smooth_alpha: float,
) -> nn.Module:
    if use_littlebit:
        return LittleBitLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            split_dim=split_dim,
            residual=residual,
            use_itq=use_itq,
            quant_func_name=quant_func_name,
            smooth_alpha=smooth_alpha,
        )
    return nn.Linear(in_features, out_features, bias=False)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 128,
        use_littlebit: bool = False,
        split_dim: int = 24,
        residual: bool = False,
        use_itq: bool = True,
        quant_func_name: str = "SmoothSign",
        smooth_alpha: float = 100.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        kwargs = dict(
            use_littlebit=use_littlebit,
            split_dim=split_dim,
            residual=residual,
            use_itq=use_itq,
            quant_func_name=quant_func_name,
            smooth_alpha=smooth_alpha,
        )
        self.q_proj = _make_proj(d_model, d_model, **kwargs)
        self.k_proj = _make_proj(d_model, d_model, **kwargs)
        self.v_proj = _make_proj(d_model, d_model, **kwargs)
        self.o_proj = _make_proj(d_model, d_model, **kwargs)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_out)

