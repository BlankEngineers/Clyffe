"""
Memory accounting and representation metadata for LittleBit-2 layers.
"""

from dataclasses import dataclass
from src.littlebit.factorization import compute_eff_bits


@dataclass(frozen=True)
class LittleBitMemoryReport:
    in_features: int
    out_features: int
    split_dim: int
    residual: bool
    dense_params: int
    dense_fp16_bits: int
    binary_factor_bits: int
    scale_fp16_bits: int
    total_littlebit_bits: int
    actual_bpw: float
    compression_ratio_vs_fp16: float


def compute_memory_report(
    in_features: int,
    out_features: int,
    split_dim: int,
    residual: bool = False,
) -> LittleBitMemoryReport:
    """Compute exact bit-level storage and effective BPW for a LittleBitLinear layer."""
    a, b, s = in_features, out_features, split_dim
    dense_params = a * b
    dense_fp16_bits = 16 * dense_params

    mult = 2 if residual else 1
    binary_factor_bits = mult * s * (a + b)
    scale_fp16_bits = mult * 16 * (a + b + s)
    total_bits = binary_factor_bits + scale_fp16_bits

    actual_bpw = compute_eff_bits(a, b, s, residual=residual)
    compression_ratio = dense_fp16_bits / max(total_bits, 1)

    return LittleBitMemoryReport(
        in_features=a,
        out_features=b,
        split_dim=s,
        residual=residual,
        dense_params=dense_params,
        dense_fp16_bits=dense_fp16_bits,
        binary_factor_bits=binary_factor_bits,
        scale_fp16_bits=scale_fp16_bits,
        total_littlebit_bits=total_bits,
        actual_bpw=actual_bpw,
        compression_ratio_vs_fp16=compression_ratio,
    )

