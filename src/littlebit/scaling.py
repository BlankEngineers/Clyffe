"""
Dual-SVID Multi-Scale Magnitude Restoration for LittleBit / LittleBit-2.

Computes rank-1 SVD on |U| and |V| to extract row, column, and latent scales:
    |U| ~ u1^T @ u2,   |V| ~ v1^T @ v2
"""

from typing import Tuple
import torch


def rank_one_svd_abs(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rank-1 SVD approximation of a non-negative 2D matrix X (m, n):
        X ~ s1^T @ s2
    Returns:
        s1: shape (1, m)
        s2: shape (1, n)
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(X.shape)}")
    X_f = X.detach().float()
    U, S, Vh = torch.linalg.svd(X_f, full_matrices=False)
    sqrt_s0 = torch.sqrt(torch.clamp(S[0], min=0.0))
    # Ensure non-negative sign convention via first left singular vector
    sign_fix = 1.0 if U[:, 0].sum() >= 0 else -1.0
    s1 = (sign_fix * U[:, :1] * sqrt_s0).t().contiguous()
    s2 = (sign_fix * sqrt_s0 * Vh[:1, :]).contiguous()
    return s1.to(X.dtype), s2.to(X.dtype)


def dual_svid(
    U: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Dual-SVID scale factors (u1, u2, v1, v2) from latent factors U and V.
    """
    u1, u2 = rank_one_svd_abs(torch.abs(U))
    v1, v2 = rank_one_svd_abs(torch.abs(V))
    return u1, u2, v1, v2

