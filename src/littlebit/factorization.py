"""
Low-rank SVD factorization and effective bit-width rank estimation
for LittleBit / LittleBit-2.

Reference: Lee et al., "LittleBit: Ultra Low-Bit Quantization via Latent Factorization"
(NeurIPS 2025, arXiv:2506.13771).
"""

from typing import Optional, Tuple
import torch


def estimate_split_dim(
    in_features: int,
    out_features: int,
    eff_bit_target: Optional[float],
    residual: bool = False,
) -> Optional[float]:
    """Estimate floating-point latent rank s for a target effective BPW."""
    if eff_bit_target is None or in_features * out_features == 0:
        return None
    a, b = in_features, out_features
    base = a + b + 16
    if residual:
        numerator = a * b * eff_bit_target - 32 * (a + b)
        denominator = 2 * base
    else:
        numerator = a * b * eff_bit_target - 16 * (a + b)
        denominator = base
    return numerator / denominator if denominator else None


def finalize_split_dim(
    split_float: Optional[float],
    split_default: int = 16,
    min_split_dim: int = 8,
    max_split_dim: Optional[int] = None,
) -> int:
    """Round latent rank down to a multiple of 8 and clamp to [min_split_dim, max_split_dim]."""
    cand = split_float if split_float is not None else split_default
    cand_int = int(cand) if cand is not None else 0
    cand_int = (cand_int // 8) * 8
    if cand_int <= 0:
        cand_int = min_split_dim
    rank = max(cand_int, min_split_dim)
    if max_split_dim is not None:
        rank = min(rank, max_split_dim)
    return rank


def compute_eff_bits(
    in_features: int,
    out_features: int,
    split_dim: int,
    residual: bool = False,
) -> float:
    """Compute actual effective bits per weight including 16-bit scale overhead."""
    a, b, s = in_features, out_features, split_dim
    if a * b == 0:
        return float("inf")
    if residual:
        num = s * 2 * (a + b + 16) + 32 * (a + b)
    else:
        num = s * (a + b + 16) + 16 * (a + b)
    return num / (a * b)


def svd_factorize(
    W: torch.Tensor,
    split_dim: int,
    exact: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factorize W (out_features, in_features) into U (out_features, s) and V (s, in_features)
    via truncated SVD with symmetric singular value splitting:
        U = U_s @ diag(sqrt(S_s)),  V = diag(sqrt(S_s)) @ V_s^T.
    """
    if W.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {tuple(W.shape)}")
    out_features, in_features = W.shape
    max_rank = min(out_features, in_features)
    if split_dim > max_rank:
        raise ValueError(
            f"split_dim={split_dim} exceeds min(out_features, in_features)={max_rank}"
        )

    W_f = W.detach().float()
    if exact or max_rank <= 512:
        U_full, S_full, Vh_full = torch.linalg.svd(W_f, full_matrices=False)
        U_t = U_full[:, :split_dim]
        S_t = S_full[:split_dim]
        Vh_t = Vh_full[:split_dim, :]
    else:
        U_t, S_t, V_t = torch.svd_lowrank(W_f, q=split_dim)
        Vh_t = V_t.t()

    sqrt_S = torch.sqrt(torch.clamp(S_t, min=0.0))
    U = (U_t * sqrt_S.unsqueeze(0)).contiguous()
    V = (sqrt_S.unsqueeze(1) * Vh_t).contiguous()
    return U.to(W.dtype), V.to(W.dtype)

