"""
Optimizer and learning rate schedule for ARC-LB-01.
"""

import math
from typing import Tuple
import torch
import torch.nn as nn


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 3e-3,
    weight_decay: float = 0.01,
    total_steps: int = 1000,
    warmup_steps: int = 50,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

