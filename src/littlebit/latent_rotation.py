"""
Internal Latent Rotation for LittleBit-2.

Exploits rotational invariance of latent factorization:
    U @ V = (U @ R) @ (R^T @ V)  for any orthogonal R in O(s).

Reference: Lee & Kim, "LittleBit-2: Maximizing the Spectral Energy Gain
in Sub-1-Bit LLMs via Latent Geometry Alignment" (ICML 2026, arXiv:2603.00042).
"""

from typing import Optional, Tuple
import torch


def random_orthogonal(
    dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a random orthogonal matrix R in O(dim) via QR decomposition."""
    mat = torch.randn((dim, dim), device=device, dtype=torch.float32, generator=generator)
    q, r = torch.linalg.qr(mat)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(0)
    return q.to(dtype)


def apply_latent_rotation(
    U: torch.Tensor,
    V: torch.Tensor,
    R: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply orthogonal rotation R to latent factors: U' = U @ R, V' = R^T @ V."""
    U_rot = (U @ R).contiguous()
    V_rot = (R.t() @ V).contiguous()
    return U_rot, V_rot


def verify_orthogonality(R: torch.Tensor, atol: float = 1e-5) -> Tuple[bool, float]:
    """Check ||R^T R - I||_F <= atol."""
    eye = torch.eye(R.shape[0], device=R.device, dtype=torch.float32)
    err = torch.linalg.norm(R.float().t() @ R.float() - eye, ord="fro").item()
    return err <= atol, err

