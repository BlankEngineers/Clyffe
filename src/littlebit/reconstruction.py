"""
Explicit weight reconstruction and numerical error metrics for validation/auditing.
Not used in native LittleBit inference runtime.
"""

from typing import Dict, Optional
import torch


def reconstruct_branch(
    Uq: torch.Tensor,
    Vq: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct a single LittleBit branch:
        W_hat = (Uq * (u1^T @ u2)) @ (Vq * (v1^T @ v2))
    """
    U_scaled = Uq.float() * (u1.float().t() @ u2.float())
    V_scaled = Vq.float() * (v1.float().t() @ v2.float())
    return (U_scaled @ V_scaled).contiguous()


def reconstruct_weight(
    Uq: torch.Tensor,
    Vq: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    residual_branch: Optional[tuple] = None,
) -> torch.Tensor:
    """Reconstruct full weight matrix including optional residual branch."""
    W_hat = reconstruct_branch(Uq, Vq, u1, u2, v1, v2)
    if residual_branch is not None:
        W_hat = W_hat + reconstruct_branch(*residual_branch)
    return W_hat


def reconstruction_metrics(W: torch.Tensor, W_hat: torch.Tensor) -> Dict[str, float]:
    """Compute reconstruction quality metrics between dense W and reconstructed W_hat."""
    W_f = W.detach().float()
    W_h = W_hat.detach().float()
    diff = W_f - W_h

    frob_w = torch.linalg.norm(W_f, ord="fro").item()
    frob_err = torch.linalg.norm(diff, ord="fro").item()
    rel_err = frob_err / max(frob_w, 1e-12)
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()

    w_flat = W_f.reshape(-1)
    wh_flat = W_h.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        w_flat.unsqueeze(0), wh_flat.unsqueeze(0)
    ).item()

    return {
        "frobenius_norm_W": frob_w,
        "frobenius_error": frob_err,
        "relative_error": rel_err,
        "rmse": rmse,
        "cosine_similarity": cos_sim,
    }

