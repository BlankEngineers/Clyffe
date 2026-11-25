"""
Quantization-Aware Training (QAT) surrogate gradient functions for LittleBit.

Implements:
- STEBinary: Straight-Through Estimator clipped to (-1, 1).
- SmoothSign: Smooth tanh-derivative surrogate with sharpness alpha.
"""

from typing import Callable
import torch


class _STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        y = x.sign()
        y[y == 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        mask = (x > -1.0) & (x < 1.0)
        return grad_output * mask.to(grad_output.dtype)


class _SmoothSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
        ctx.alpha = float(alpha)
        ctx.save_for_backward(x)
        y = x.sign()
        y[y == 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        # Clamp alpha * x inside tanh to prevent overflow in fp16/bf16
        ax = torch.clamp(alpha * x.float(), min=-20.0, max=20.0)
        grad_in = grad_output.float() * alpha * (1.0 - torch.tanh(ax) ** 2)
        return grad_in.to(grad_output.dtype), None


STEBinary = _STEBinary.apply


def SmoothSign(x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    """Apply SmoothSign binarization with configurable surrogate sharpness alpha."""
    return _SmoothSign.apply(x, alpha)


def get_quant_func(name: str = "SmoothSign", alpha: float = 100.0) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a binarization callable by name ('SmoothSign' or 'STEBinary')."""
    if name == "STEBinary":
        return STEBinary
    if name == "SmoothSign":
        return lambda x: _SmoothSign.apply(x, alpha)
    raise ValueError(f"Unknown quant_func: {name!r}")

