"""
Training metrics: token accuracy and gradient norm monitoring.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0.0
    correct = ((preds == labels) & mask).sum().item()
    return float(correct) / float(mask.sum().item())


def compute_grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            total_sq += float(torch.sum(g * g).item())
    return total_sq ** 0.5

