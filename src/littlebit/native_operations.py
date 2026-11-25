"""
Native LittleBit-2 Linear Layer and Factorized Forward Operations.

Executes forward computation directly in the factorized binary latent space:
    y = ((((x * v2) @ Vq^T) * (v1 * u2)) @ Uq^T) * u1
without materializing the dense weight matrix W_hat.
"""

from typing import Dict, Optional
import math
import torch
import torch.nn as nn

from src.littlebit.binary_factors import binarize_sign, pack_binary, unpack_binary
from src.littlebit.compensation import compute_residual_target, fuse_latent_scales
from src.littlebit.factorization import (
    compute_eff_bits,
    estimate_split_dim,
    finalize_split_dim,
    svd_factorize,
)
from src.littlebit.joint_itq import compute_joint_itq_rotation
from src.littlebit.latent_rotation import apply_latent_rotation
from src.littlebit.qat import get_quant_func
from src.littlebit.reconstruction import reconstruct_branch
from src.littlebit.representation import LittleBitMemoryReport, compute_memory_report
from src.littlebit.scaling import dual_svid


def native_factorized_forward(
    x: torch.Tensor,
    Vq: torch.Tensor,
    Uq: torch.Tensor,
    v2: torch.Tensor,
    v1u2: torch.Tensor,
    u1: torch.Tensor,
) -> torch.Tensor:
    """
    Native factorized matrix-vector/matrix-matrix multiplication without dense W_hat:
        x:    (..., in_features)
        Vq:   (split_dim, in_features) in {-1, +1}
        Uq:   (out_features, split_dim) in {-1, +1}
        v2:   (1, in_features)
        v1u2: (1, split_dim)
        u1:   (1, out_features)
    """
    h = (x * v2) @ Vq.t()
    h = h * v1u2
    y = (h @ Uq.t()) * u1
    return y


class LittleBitLinear(nn.Module):
    """
    Linear layer backed by LittleBit-2 binary latent factorization.

    Supports:
    - Experiment A: `initialize_from_dense(W)` using SVD + optional Joint-ITQ + Dual-SVID.
    - Experiment B: `initialize_direct()` initializing latent factors & scales directly.
    - Native packed inference: `pack_for_inference()` packs Uq, Vq into int32 words
      and removes float latent parameters U, V, u2, v1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        split_dim: Optional[int] = None,
        eff_bit: Optional[float] = 1.0,
        min_split_dim: int = 8,
        residual: bool = False,
        use_itq: bool = True,
        itq_n_iter: int = 50,
        quant_func_name: str = "SmoothSign",
        smooth_alpha: float = 100.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.use_itq = use_itq
        self.itq_n_iter = itq_n_iter
        self.quant_func_name = quant_func_name
        self.smooth_alpha = smooth_alpha
        self.quant_func = get_quant_func(quant_func_name, alpha=smooth_alpha)

        max_rank = min(in_features, out_features)
        if split_dim is None:
            est = estimate_split_dim(in_features, out_features, eff_bit, residual=residual)
            split_dim = finalize_split_dim(
                est,
                split_default=min(16, max_rank),
                min_split_dim=min(min_split_dim, max_rank),
                max_split_dim=max_rank,
            )
        else:
            split_dim = min(split_dim, max_rank)

        self.split_dim = split_dim
        self.register_buffer(
            "_eff_bit_actual",
            torch.tensor(compute_eff_bits(in_features, out_features, split_dim, residual)),
        )
        self._packed_mode = False

        # Trainable QAT latent parameters (active when not packed)
        self.U = nn.Parameter(torch.empty(out_features, split_dim))
        self.V = nn.Parameter(torch.empty(split_dim, in_features))
        self.u1 = nn.Parameter(torch.empty(1, out_features))
        self.u2 = nn.Parameter(torch.empty(1, split_dim))
        self.v1 = nn.Parameter(torch.empty(1, split_dim))
        self.v2 = nn.Parameter(torch.empty(1, in_features))

        if residual:
            self.U_R = nn.Parameter(torch.empty(out_features, split_dim))
            self.V_R = nn.Parameter(torch.empty(split_dim, in_features))
            self.u1_R = nn.Parameter(torch.empty(1, out_features))
            self.u2_R = nn.Parameter(torch.empty(1, split_dim))
            self.v1_R = nn.Parameter(torch.empty(1, split_dim))
            self.v2_R = nn.Parameter(torch.empty(1, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize directly by default so layer is always valid
        self.initialize_direct()

    def _decompose_branch(
        self,
        W: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ):
        U, V = svd_factorize(W, self.split_dim, exact=True)
        if self.use_itq:
            R, _ = compute_joint_itq_rotation(
                U, V, n_iter=self.itq_n_iter, generator=generator
            )
            U, V = apply_latent_rotation(U, V, R)
        u1, u2, v1, v2 = dual_svid(U, V)
        return U, V, u1, u2, v1, v2

    @torch.no_grad()
    def initialize_from_dense(
        self,
        W: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Experiment A: Initialize LittleBit-2 parameters from a dense matrix W."""
        if W.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Expected W shape {(self.out_features, self.in_features)}, got {tuple(W.shape)}"
            )
        U, V, u1, u2, v1, v2 = self._decompose_branch(W, generator=generator)
        self.U.copy_(U)
        self.V.copy_(V)
        self.u1.copy_(u1)
        self.u2.copy_(u2)
        self.v1.copy_(v1)
        self.v2.copy_(v2)

        if self.residual:
            W_approx = reconstruct_branch(
                binarize_sign(U), binarize_sign(V), u1, u2, v1, v2
            )
            W_res = compute_residual_target(W, W_approx)
            U_R, V_R, u1_R, u2_R, v1_R, v2_R = self._decompose_branch(
                W_res, generator=generator
            )
            self.U_R.copy_(U_R)
            self.V_R.copy_(V_R)
            self.u1_R.copy_(u1_R)
            self.u2_R.copy_(u2_R)
            self.v1_R.copy_(v1_R)
            self.v2_R.copy_(v2_R)

    @torch.no_grad()
    def initialize_direct(self, generator: Optional[torch.Generator] = None) -> None:
        """
        Experiment B: Direct LittleBit-2 latent initialization without a dense parent matrix.
        Scales are calibrated so Var(W_hat) ~ 1 / in_features (Xavier/He scale).
        """
        # Target std of reconstructed weight ~ 1 / sqrt(in_features)
        # Since W_hat = (Uq * u1^T u2) @ (Vq * v1^T v2), each element sums split_dim terms.
        # Setting U, V ~ N(0, 0.02) keeps values inside SmoothSign active region,
        # and scale vectors absorb the fan-in variance.
        scale_val = (1.0 / (self.in_features * self.split_dim)) ** 0.25
        for p in (self.U, self.V):
            p.normal_(mean=0.0, std=0.02, generator=generator)
        self.u1.fill_(math.sqrt(scale_val))
        self.u2.fill_(math.sqrt(scale_val))
        self.v1.fill_(math.sqrt(scale_val))
        self.v2.fill_(math.sqrt(scale_val))

        if self.residual:
            for p in (self.U_R, self.V_R):
                p.normal_(mean=0.0, std=0.02, generator=generator)
            res_scale = 0.5 * scale_val
            self.u1_R.fill_(math.sqrt(res_scale))
            self.u2_R.fill_(math.sqrt(res_scale))
            self.v1_R.fill_(math.sqrt(res_scale))
            self.v2_R.fill_(math.sqrt(res_scale))

    def _forward_branch_qat(
        self,
        x: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor,
        u1: torch.Tensor,
        u2: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
    ) -> torch.Tensor:
        Uq = self.quant_func(U.to(x.dtype))
        Vq = self.quant_func(V.to(x.dtype))
        v1u2 = fuse_latent_scales(v1.to(x.dtype), u2.to(x.dtype))
        return native_factorized_forward(
            x, Vq, Uq, v2.to(x.dtype), v1u2, u1.to(x.dtype)
        )

    def _forward_branch_packed(
        self,
        x: torch.Tensor,
        U_packed: torch.Tensor,
        V_packed: torch.Tensor,
        u_shape: tuple,
        v_shape: tuple,
        u1: torch.Tensor,
        v1u2: torch.Tensor,
        v2: torch.Tensor,
    ) -> torch.Tensor:
        Uq = unpack_binary(U_packed, u_shape).to(x.dtype)
        Vq = unpack_binary(V_packed, v_shape).to(x.dtype)
        return native_factorized_forward(
            x, Vq, Uq, v2.to(x.dtype), v1u2.to(x.dtype), u1.to(x.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)

        if self._packed_mode:
            y = self._forward_branch_packed(
                x_2d,
                self.U_packed,
                self.V_packed,
                (self.out_features, self.split_dim),
                (self.split_dim, self.in_features),
                self.u1_inf,
                self.v1u2_inf,
                self.v2_inf,
            )
            if self.residual:
                y = y + self._forward_branch_packed(
                    x_2d,
                    self.U_R_packed,
                    self.V_R_packed,
                    (self.out_features, self.split_dim),
                    (self.split_dim, self.in_features),
                    self.u1_R_inf,
                    self.v1u2_R_inf,
                    self.v2_R_inf,
                )
        else:
            y = self._forward_branch_qat(
                x_2d, self.U, self.V, self.u1, self.u2, self.v1, self.v2
            )
            if self.residual:
                y = y + self._forward_branch_qat(
                    x_2d,
                    self.U_R,
                    self.V_R,
                    self.u1_R,
                    self.u2_R,
                    self.v1_R,
                    self.v2_R,
                )

        if self.bias is not None:
            y = y + self.bias.to(y.dtype)

        return y.reshape(*orig_shape, self.out_features)

    @torch.no_grad()
    def pack_for_inference(self) -> Dict[str, torch.Tensor]:
        """
        Convert trained QAT parameters into packed int32 binary buffers + float16 scales.
        Deletes floating-point latent parameters U, V, u2, v1 so inference uses native packed storage.
        """
        if self._packed_mode:
            return self.export_packed_state()

        def _pack_branch(suffix: str, U: nn.Parameter, V: nn.Parameter, u1, u2, v1, v2):
            U_bin = binarize_sign(U.data).to(torch.int8)
            V_bin = binarize_sign(V.data).to(torch.int8)
            self.register_buffer(f"U{suffix}_packed", pack_binary(U_bin))
            self.register_buffer(f"V{suffix}_packed", pack_binary(V_bin))
            self.register_buffer(f"u1{suffix}_inf", u1.data.half().contiguous())
            self.register_buffer(
                f"v1u2{suffix}_inf", fuse_latent_scales(v1.data, u2.data).half()
            )
            self.register_buffer(f"v2{suffix}_inf", v2.data.half().contiguous())

        _pack_branch("", self.U, self.V, self.u1, self.u2, self.v1, self.v2)
        for name in ("U", "V", "u1", "u2", "v1", "v2"):
            delattr(self, name)

        if self.residual:
            _pack_branch(
                "_R",
                self.U_R,
                self.V_R,
                self.u1_R,
                self.u2_R,
                self.v1_R,
                self.v2_R,
            )
            for name in ("U_R", "V_R", "u1_R", "u2_R", "v1_R", "v2_R"):
                delattr(self, name)

        self._packed_mode = True
        return self.export_packed_state()

    def export_packed_state(self) -> Dict[str, torch.Tensor]:
        """Return dictionary of packed buffers for inspection/benchmarking."""
        out = {
            "U_packed": self.U_packed,
            "V_packed": self.V_packed,
            "u1_inf": self.u1_inf,
            "v1u2_inf": self.v1u2_inf,
            "v2_inf": self.v2_inf,
        }
        if self.residual:
            out.update(
                {
                    "U_R_packed": self.U_R_packed,
                    "V_R_packed": self.V_R_packed,
                    "u1_R_inf": self.u1_R_inf,
                    "v1u2_R_inf": self.v1u2_R_inf,
                    "v2_R_inf": self.v2_R_inf,
                }
            )
        return out

    def memory_report(self) -> LittleBitMemoryReport:
        return compute_memory_report(
            self.in_features, self.out_features, self.split_dim, self.residual
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"split_dim={self.split_dim}, residual={self.residual}, "
            f"use_itq={self.use_itq}, eff_bit_actual={self._eff_bit_actual.item():.4f}, "
            f"packed={self._packed_mode}"
        )

