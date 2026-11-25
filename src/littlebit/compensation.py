"""
Scale fusion and residual compensation mechanisms for LittleBit / LittleBit-2.
"""

import torch


def fuse_latent_scales(v1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    """
    Fuse the inner latent scale vectors (v1 of shape (1, s) and u2 of shape (1, s))
    into a single s-dimensional scale vector for zero-overhead inference.
    """
    return (v1 * u2).contiguous()


def compute_residual_target(W: torch.Tensor, W_approx: torch.Tensor) -> torch.Tensor:
    """Compute residual error matrix W_R = W - W_approx for residual LittleBit factorization."""
    return (W.float() - W_approx.float()).to(W.dtype)

