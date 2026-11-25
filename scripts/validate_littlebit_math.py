"""
Phase 3 Synthetic Matrix Validation Script for LittleBit-2.

Compares SVD-only vs LittleBit-2 (Joint-ITQ) vs Residual LittleBit-2
across random (Marchenko-Pastur) and structured low-rank matrices.
"""

import json
import torch

from src.littlebit import (
    LittleBitLinear,
    reconstruct_branch,
    reconstruct_weight,
    reconstruction_metrics,
    binarize_sign,
)


def run_synthetic_validation():
    torch.manual_seed(42)
    results = []

    configs = [
        ("random_gaussian_96x96_s24", torch.randn(96, 96) * 0.1, 24, False),
        ("random_gaussian_96x192_s32", torch.randn(96, 192) * 0.1, 32, False),
        ("random_gaussian_96x192_s24_res", torch.randn(96, 192) * 0.1, 24, True),
        (
            "structured_lowrank_96x192_s24",
            (torch.randn(96, 16) @ torch.randn(16, 192)) * 0.05 + torch.randn(96, 192) * 0.005,
            24,
            False,
        ),
    ]

    for name, W, split_dim, residual in configs:
        for use_itq in (False, True):
            layer = LittleBitLinear(
                in_features=W.shape[1],
                out_features=W.shape[0],
                split_dim=split_dim,
                residual=residual,
                use_itq=use_itq,
                itq_n_iter=50,
            )
            layer.initialize_from_dense(W)
            res_branch = None
            if residual:
                res_branch = (
                    binarize_sign(layer.U_R),
                    binarize_sign(layer.V_R),
                    layer.u1_R,
                    layer.u2_R,
                    layer.v1_R,
                    layer.v2_R,
                )
            W_hat = reconstruct_weight(
                binarize_sign(layer.U),
                binarize_sign(layer.V),
                layer.u1,
                layer.u2,
                layer.v1,
                layer.v2,
                residual_branch=res_branch,
            )
            metrics = reconstruction_metrics(W, W_hat)
            mem = layer.memory_report()
            results.append(
                {
                    "case": name,
                    "use_itq": use_itq,
                    "split_dim": split_dim,
                    "residual": residual,
                    "actual_bpw": round(mem.actual_bpw, 4),
                    "compression_vs_fp16": round(mem.compression_ratio_vs_fp16, 3),
                    "relative_error": round(metrics["relative_error"], 4),
                    "cosine_similarity": round(metrics["cosine_similarity"], 4),
                }
            )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_synthetic_validation()

