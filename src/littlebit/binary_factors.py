"""
Binary latent factor representation and 32-bit LSB-first packing/unpacking.

Maps {-1, +1} int8 matrices into row-major int32 words (+1 -> bit 0, -1 -> bit 1).
"""

from typing import Tuple
import torch


def binarize_sign(x: torch.Tensor) -> torch.Tensor:
    """Binarize tensor to {-1, +1}, mapping exact zeros to +1."""
    y = x.sign()
    y[y == 0] = 1
    return y


def pack_binary(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack a 2D {-1, +1} int8 tensor into row-major int32 words (LSB-first).

    Encoding: +1 -> bit 0, -1 -> bit 1.
    Columns not divisible by 32 are padded on the right with +1 (bit 0).
    """
    if tensor.dtype != torch.int8:
        raise TypeError(f"Expected torch.int8 tensor, got {tensor.dtype}")
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(tensor.shape)}")

    n_rows, n_cols = tensor.shape
    words_per_row = (n_cols + 31) // 32
    pad = words_per_row * 32 - n_cols
    device = tensor.device

    if pad > 0:
        pad_tensor = torch.ones((n_rows, pad), dtype=torch.int8, device=device)
        tensor = torch.cat([tensor, pad_tensor], dim=1)

    bits = ((1 - tensor) // 2).to(torch.int32).reshape(n_rows, words_per_row, 32)
    shifts = torch.arange(32, dtype=torch.int32, device=device)
    packed = (bits << shifts).sum(dim=2, dtype=torch.int32)
    return packed.contiguous()


def unpack_binary(
    packed: torch.Tensor,
    original_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Unpack a row-major int32 packed tensor back to {-1, +1} int8 of original_shape.
    """
    if packed.ndim != 2 or packed.dtype != torch.int32:
        raise ValueError(
            f"Expected 2D int32 packed tensor, got shape {tuple(packed.shape)}, dtype {packed.dtype}"
        )
    n_rows, n_cols = int(original_shape[0]), int(original_shape[1])
    words_per_row = (n_cols + 31) // 32
    if packed.shape != (n_rows, words_per_row):
        raise ValueError(
            f"Shape mismatch: expected {(n_rows, words_per_row)}, got {tuple(packed.shape)}"
        )

    shifts = torch.arange(32, dtype=torch.int32, device=packed.device)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).to(torch.int8)
    bits = bits.reshape(n_rows, words_per_row * 32)[:, :n_cols]
    return (1 - 2 * bits).contiguous()

