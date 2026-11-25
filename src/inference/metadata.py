"""
Model Self-Information & Metadata Retrieval for ARC-LB-01.
"""

import os
from typing import Any, Dict

from src.architecture.initialization import audit_model_representation
from src.architecture.model import ARCLB01Model
from src.training.checkpoint import compute_sha256


def build_model_metadata(
    model: ARCLB01Model,
    checkpoint_path: str,
    version: str = "0.1.0",
) -> Dict[str, Any]:
    audit = audit_model_representation(model)
    storage_bytes = os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 0
    sha256 = compute_sha256(checkpoint_path) if os.path.exists(checkpoint_path) else "N/A"
    return {
        "name": "Tesla 1.1",
        "organization": "Blank Engineers (ARC III Ecosystem)",
        "version": version,
        "architecture": f"Decoder-only Transformer ({model.config.n_layers}L, d={model.config.d_model}, h={model.config.n_heads}, d_ff={model.config.d_ff})",
        "parameter_count_dense_equiv": audit["total_dense_equivalent_params"],
        "littlebit_coverage_pct": audit["littlebit_coverage_pct"],
        "overall_effective_bpw": audit["overall_effective_bpw"],
        "vocab_size": model.config.vocab_size,
        "context_length": model.config.max_seq_len,
        "weight_representation": (
            f"LittleBit-2 (split_dim={model.config.split_dim}, use_itq={model.config.use_itq}, residual={model.config.residual})"
            if model.config.use_littlebit
            else "Dense FP32 Baseline"
        ),
        "checkpoint_identifier": os.path.basename(checkpoint_path),
        "checkpoint_sha256": sha256,
        "actual_storage_bytes": storage_bytes,
    }

