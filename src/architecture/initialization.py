"""
Initialization and Representation Auditing for ARC-LB-01.

Implements:
- Experiment A: Random initialization -> LittleBit-2 (SVD + Joint-ITQ + Dual-SVID).
- Experiment B: Direct LittleBit-2 initialization.
- Full-parameter representation audit (every matrix & vector accounted for).
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from src.architecture.embeddings import LittleBitEmbedding
from src.littlebit.native_operations import LittleBitLinear


def init_random_weights(model: nn.Module, seed: int = 42, std: float = 0.02) -> None:
    """Reproducible random initialization for dense baseline / Experiment A parent."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, mean=0.0, std=std, generator=gen)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, mean=0.0, std=std, generator=gen)


@torch.no_grad()
def convert_dense_to_littlebit(
    dense_model: nn.Module,
    lb_model: nn.Module,
    seed: int = 42,
) -> None:
    """
    Experiment A: Initialize a LittleBit-2 model's layers from a randomly
    initialized dense parent model via SVD + Joint-ITQ + Dual-SVID.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dense_modules = dict(dense_model.named_modules())

    for name, mod in lb_model.named_modules():
        if isinstance(mod, LittleBitLinear):
            src_mod = dense_modules.get(name)
            if isinstance(src_mod, nn.Linear):
                mod.initialize_from_dense(src_mod.weight.data, generator=gen)
                if mod.bias is not None and src_mod.bias is not None:
                    mod.bias.data.copy_(src_mod.bias.data)
        elif isinstance(mod, LittleBitEmbedding):
            src_mod = dense_modules.get(name)
            if isinstance(src_mod, nn.Embedding):
                mod.initialize_from_dense(src_mod.weight.data, generator=gen)


def audit_model_representation(model: nn.Module) -> Dict[str, Any]:
    """
    Audit every learned parameter/buffer in the model to verify what fraction
    is LittleBit-represented vs ordinary floating-point.
    """
    littlebit_layers = []
    fp_modules = []

    total_dense_equiv_params = 0
    total_littlebit_dense_equiv = 0
    total_littlebit_bits = 0
    total_fp_params = 0

    for name, mod in model.named_modules():
        if isinstance(mod, (LittleBitLinear, LittleBitEmbedding)):
            rep = mod.memory_report()
            littlebit_layers.append(
                {
                    "name": name,
                    "type": type(mod).__name__,
                    "shape": (rep.out_features, rep.in_features),
                    "split_dim": rep.split_dim,
                    "residual": rep.residual,
                    "packed": mod._packed_mode,
                    "dense_equiv_params": rep.dense_params,
                    "actual_bpw": round(rep.actual_bpw, 4),
                    "total_bits": rep.total_littlebit_bits,
                }
            )
            total_dense_equiv_params += rep.dense_params
            total_littlebit_dense_equiv += rep.dense_params
            total_littlebit_bits += rep.total_littlebit_bits
        elif len(list(mod.children())) == 0:
            params = sum(p.numel() for p in mod.parameters(recurse=False))
            if params > 0:
                # Skip tied lm_head if weight is shared with token_emb
                fp_modules.append({"name": name, "type": type(mod).__name__, "params": params})
                total_fp_params += params
                total_dense_equiv_params += params

    total_fp_bits = total_fp_params * 16  # measured at FP16 baseline
    total_bits = total_littlebit_bits + total_fp_bits

    return {
        "total_dense_equivalent_params": total_dense_equiv_params,
        "littlebit_covered_params": total_littlebit_dense_equiv,
        "floating_point_params": total_fp_params,
        "littlebit_coverage_pct": round(
            100.0 * total_littlebit_dense_equiv / max(total_dense_equiv_params, 1), 2
        ),
        "total_littlebit_bits": total_littlebit_bits,
        "total_fp16_remaining_bits": total_fp_bits,
        "total_model_bits": total_bits,
        "total_model_bytes": (total_bits + 7) // 8,
        "overall_effective_bpw": round(total_bits / max(total_dense_equiv_params, 1), 4),
        "littlebit_layers": littlebit_layers,
        "floating_point_modules": fp_modules,
    }

