"""
ARC-LB-01 LittleBit-2 Mathematical Computing Engine.

Independent implementation based on:
- LittleBit (Lee et al., NeurIPS 2025, arXiv:2506.13771)
- LittleBit-2 (Lee & Kim, ICML 2026, arXiv:2603.00042)
"""

from src.littlebit.binary_factors import binarize_sign, pack_binary, unpack_binary
from src.littlebit.compensation import compute_residual_target, fuse_latent_scales
from src.littlebit.factorization import (
    compute_eff_bits,
    estimate_split_dim,
    finalize_split_dim,
    svd_factorize,
)
from src.littlebit.joint_itq import compute_joint_itq_rotation
from src.littlebit.latent_rotation import (
    apply_latent_rotation,
    random_orthogonal,
    verify_orthogonality,
)
from src.littlebit.native_operations import LittleBitLinear, native_factorized_forward
from src.littlebit.qat import SmoothSign, STEBinary, get_quant_func
from src.littlebit.reconstruction import (
    reconstruct_branch,
    reconstruct_weight,
    reconstruction_metrics,
)
from src.littlebit.representation import LittleBitMemoryReport, compute_memory_report
from src.littlebit.scaling import dual_svid, rank_one_svd_abs

__all__ = [
    "estimate_split_dim",
    "finalize_split_dim",
    "compute_eff_bits",
    "svd_factorize",
    "random_orthogonal",
    "apply_latent_rotation",
    "verify_orthogonality",
    "compute_joint_itq_rotation",
    "binarize_sign",
    "pack_binary",
    "unpack_binary",
    "rank_one_svd_abs",
    "dual_svid",
    "fuse_latent_scales",
    "compute_residual_target",
    "STEBinary",
    "SmoothSign",
    "get_quant_func",
    "reconstruct_branch",
    "reconstruct_weight",
    "reconstruction_metrics",
    "LittleBitMemoryReport",
    "compute_memory_report",
    "native_factorized_forward",
    "LittleBitLinear",
]

