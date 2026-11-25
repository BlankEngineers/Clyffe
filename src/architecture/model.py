"""
ARC-LB-01 Tiny Decoder-Only Language Model Architecture.
"""

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architecture.attention import CausalSelfAttention
from src.architecture.embeddings import LittleBitEmbedding
from src.architecture.feedforward import FeedForward
from src.littlebit.native_operations import LittleBitLinear


@dataclass
class ARCLB01Config:
    vocab_size: int = 512
    d_model: int = 96
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 192
    max_seq_len: int = 128
    use_littlebit: bool = False
    littlebit_embeddings: bool = False
    littlebit_lm_head: bool = True
    split_dim: int = 24
    residual: bool = False
    use_itq: bool = True
    quant_func_name: str = "SmoothSign"
    smooth_alpha: float = 100.0
    tie_weights: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight.to(x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, config: ARCLB01Config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            use_littlebit=config.use_littlebit,
            split_dim=config.split_dim,
            residual=config.residual,
            use_itq=config.use_itq,
            quant_func_name=config.quant_func_name,
            smooth_alpha=config.smooth_alpha,
        )
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            use_littlebit=config.use_littlebit,
            split_dim=config.split_dim,
            residual=config.residual,
            use_itq=config.use_itq,
            quant_func_name=config.quant_func_name,
            smooth_alpha=config.smooth_alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ARCLB01Model(nn.Module):
    """Tiny decoder-only Transformer supporting Dense and Native LittleBit-2 modes."""

    def __init__(self, config: ARCLB01Config):
        super().__init__()
        self.config = config

        if config.use_littlebit and config.littlebit_embeddings:
            self.token_emb: nn.Module = LittleBitEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.d_model,
                split_dim=config.split_dim,
                use_itq=config.use_itq,
                quant_func_name=config.quant_func_name,
                smooth_alpha=config.smooth_alpha,
            )
        else:
            self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm_f = RMSNorm(config.d_model)

        if config.use_littlebit and config.littlebit_lm_head and not config.tie_weights:
            self.lm_head: nn.Module = LittleBitLinear(
                in_features=config.d_model,
                out_features=config.vocab_size,
                bias=False,
                split_dim=config.split_dim,
                residual=config.residual,
                use_itq=config.use_itq,
                quant_func_name=config.quant_func_name,
                smooth_alpha=config.smooth_alpha,
            )
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            if config.tie_weights and isinstance(self.token_emb, nn.Embedding):
                self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    def pack_for_inference(self) -> None:
        """Pack all LittleBitLinear and LittleBitEmbedding modules into int32 binary buffers."""
        for mod in self.modules():
            if isinstance(mod, (LittleBitLinear, LittleBitEmbedding)):
                mod.pack_for_inference()

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

