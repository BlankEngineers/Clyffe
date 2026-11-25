"""
Feed-Forward Network (MLP) for ARC-LB-01.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architecture.attention import _make_proj


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        use_littlebit: bool = False,
        split_dim: int = 24,
        residual: bool = False,
        use_itq: bool = True,
        quant_func_name: str = "SmoothSign",
        smooth_alpha: float = 100.0,
    ):
        super().__init__()
        kwargs = dict(
            use_littlebit=use_littlebit,
            split_dim=split_dim,
            residual=residual,
            use_itq=use_itq,
            quant_func_name=quant_func_name,
            smooth_alpha=smooth_alpha,
        )
        self.w_in = _make_proj(d_model, d_ff, **kwargs)
        self.w_out = _make_proj(d_ff, d_model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.gelu(self.w_in(x)))

