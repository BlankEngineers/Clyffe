"""
Checkpoint management and SHA-256 verification for ARC-LB-01.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional
import torch

from src.architecture.model import ARCLB01Config, ARCLB01Model


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(
    model: ARCLB01Model,
    save_dir: str,
    tag: str,
    metadata: Optional[Dict[str, Any]] = None,
    pack_littlebit: bool = False,
) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    if pack_littlebit and model.config.use_littlebit:
        # Clone or pack in-place on CPU copy to avoid mutating training model
        import copy
        export_model = copy.deepcopy(model).cpu()
        export_model.pack_for_inference()
        state = export_model.state_dict()
        packed_flag = True
    else:
        state = model.state_dict()
        packed_flag = False

    ckpt_path = os.path.join(save_dir, f"{tag}.pt")
    payload = {
        "config": model.config.to_dict(),
        "packed_littlebit": packed_flag,
        "state_dict": state,
        "metadata": metadata or {},
    }
    torch.save(payload, ckpt_path)
    sha = compute_sha256(ckpt_path)

    meta_path = os.path.join(save_dir, f"{tag}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": tag,
                "checkpoint_file": ckpt_path,
                "sha256": sha,
                "size_bytes": os.path.getsize(ckpt_path),
                "packed_littlebit": packed_flag,
                "config": model.config.to_dict(),
                "metadata": metadata or {},
            },
            f,
            indent=2,
        )
    return {"checkpoint": ckpt_path, "meta": meta_path, "sha256": sha}


def load_checkpoint(ckpt_path: str, device: str = "cpu") -> tuple[ARCLB01Model, dict]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ARCLB01Config(**payload["config"])
    model = ARCLB01Model(cfg)
    if payload.get("packed_littlebit", False) and cfg.use_littlebit:
        model.pack_for_inference()
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload.get("metadata", {})

