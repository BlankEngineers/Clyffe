"""
Compression, Storage, and Bit-Per-Weight Audit for ARC-LB-01.
"""

import json
import os
from typing import Dict
import torch

from src.architecture.initialization import audit_model_representation
from src.training.checkpoint import compute_sha256, load_checkpoint


def evaluate_compression(ckpt_path: str) -> Dict[str, object]:
    model, meta = load_checkpoint(ckpt_path, device="cpu")
    audit = audit_model_representation(model)
    file_bytes = os.path.getsize(ckpt_path)
    return {
        "checkpoint": ckpt_path,
        "sha256": compute_sha256(ckpt_path),
        "checkpoint_file_bytes": file_bytes,
        "use_littlebit": model.config.use_littlebit,
        "split_dim": model.config.split_dim,
        "use_itq": model.config.use_itq,
        "residual": model.config.residual,
        **{k: v for k, v in audit.items() if k not in ("littlebit_layers", "floating_point_modules")},
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ARC-LB-01-FINAL_best.pt"
    print(json.dumps(evaluate_compression(path), indent=2))

