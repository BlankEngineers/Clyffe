"""
Offline Checkpoint & Tokenizer Loader for ARC-LB-01 Native Inference.
"""

from typing import Tuple
from src.architecture.model import ARCLB01Model
from src.tokenizer.tokenizer import ARCTokenizer
from src.training.checkpoint import load_checkpoint


def load_runtime_model(
    checkpoint_path: str,
    tokenizer_dir: str = "checkpoints/tokenizer",
    device: str = "cpu",
) -> Tuple[ARCLB01Model, ARCTokenizer, dict]:
    tokenizer = ARCTokenizer.load(tokenizer_dir)
    model, ckpt_meta = load_checkpoint(checkpoint_path, device=device)
    if model.config.use_littlebit:
        # Ensure packed native mode is active
        model.pack_for_inference()
    return model, tokenizer, ckpt_meta

