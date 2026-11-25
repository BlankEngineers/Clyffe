"""
Loss function for ARC-LB-01 next-token prediction.
"""

import torch
import torch.nn.functional as F


def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard causal cross-entropy loss ignoring -100 prompt/pad positions."""
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(B * T, V), labels.reshape(B * T), ignore_index=-100)

