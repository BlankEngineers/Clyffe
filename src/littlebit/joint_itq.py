"""
Joint Iterative Quantization (Joint-ITQ) for LittleBit-2.

Solves:
    min_{R in O(s), B in {-1,+1}^{(b+a) x s}} || B - X_combined @ R ||_F^2
where X_combined = [U; V^T].

Reference: Lee & Kim, "LittleBit-2" (ICML 2026, arXiv:2603.00042).
"""

from typing import Dict, List, Optional, Tuple
import torch

from src.littlebit.binary_factors import binarize_sign
from src.littlebit.latent_rotation import random_orthogonal


def compute_joint_itq_rotation(
    U: torch.Tensor,
    V: torch.Tensor,
    n_iter: int = 50,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """
    Compute optimal orthogonal rotation R in O(s) aligning [U; V^T]
    with the binary hypercube {-1, +1}^s.

    Returns:
        R: Orthogonal rotation matrix of shape (s, s).
        diagnostics: Dict containing 'quant_error' per iteration.
    """
    with torch.no_grad():
        X_combined = torch.cat([U.detach().float(), V.detach().float().t()], dim=0)
        _, dim = X_combined.shape
        device = X_combined.device

        R = random_orthogonal(dim, device=device, dtype=torch.float32, generator=generator)
        errors: List[float] = []

        for _ in range(n_iter):
            Z = X_combined @ R
            B = binarize_sign(Z)
            err = torch.linalg.norm(B - Z, ord="fro").item()
            errors.append(err)

            M = B.t() @ X_combined
            U_p, _, Vh_p = torch.linalg.svd(M, full_matrices=False)
            R = Vh_p.t() @ U_p.t()

        # Final error after last Procrustes step
        Z_final = X_combined @ R
        B_final = binarize_sign(Z_final)
        errors.append(torch.linalg.norm(B_final - Z_final, ord="fro").item())

        return R.to(U.dtype), {"quant_error": errors}

