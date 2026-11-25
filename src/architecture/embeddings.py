"""
Token Embeddings for ARC-LB-01 (Dense and LittleBit-2 Factorized).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.littlebit.binary_factors import binarize_sign, pack_binary, unpack_binary
from src.littlebit.factorization import compute_eff_bits, svd_factorize
from src.littlebit.joint_itq import compute_joint_itq_rotation
from src.littlebit.latent_rotation import apply_latent_rotation
from src.littlebit.qat import get_quant_func
from src.littlebit.representation import LittleBitMemoryReport, compute_memory_report
from src.littlebit.scaling import dual_svid
from src.littlebit.compensation import fuse_latent_scales


class LittleBitEmbedding(nn.Module):
    """
    Token embedding table represented using LittleBit-2 binary latent factors:
        E_hat = (Uq * (u1^T @ u2)) @ (Vq * (v1^T @ v2))
    where E has shape (vocab_size, embedding_dim).
    Forward lookup indexes only the requested rows of Uq and u1 without
    materializing the full (vocab_size, embedding_dim) table.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        split_dim: int = 24,
        use_itq: bool = True,
        itq_n_iter: int = 50,
        quant_func_name: str = "SmoothSign",
        smooth_alpha: float = 100.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.split_dim = min(split_dim, num_embeddings, embedding_dim)
        self.use_itq = use_itq
        self.itq_n_iter = itq_n_iter
        self.quant_func = get_quant_func(quant_func_name, alpha=smooth_alpha)
        self._packed_mode = False

        self.U = nn.Parameter(torch.empty(num_embeddings, self.split_dim))
        self.V = nn.Parameter(torch.empty(self.split_dim, embedding_dim))
        self.u1 = nn.Parameter(torch.empty(1, num_embeddings))
        self.u2 = nn.Parameter(torch.empty(1, self.split_dim))
        self.v1 = nn.Parameter(torch.empty(1, self.split_dim))
        self.v2 = nn.Parameter(torch.empty(1, embedding_dim))
        self.initialize_direct()

    @torch.no_grad()
    def initialize_direct(self, generator: Optional[torch.Generator] = None) -> None:
        self.U.normal_(mean=0.0, std=0.02, generator=generator)
        self.V.normal_(mean=0.0, std=0.02, generator=generator)
        scale_val = (0.04 / self.split_dim) ** 0.25
        self.u1.fill_(scale_val)
        self.u2.fill_(scale_val)
        self.v1.fill_(scale_val)
        self.v2.fill_(scale_val)

    @torch.no_grad()
    def initialize_from_dense(
        self,
        W: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        U, V = svd_factorize(W, self.split_dim, exact=True)
        if self.use_itq:
            R, _ = compute_joint_itq_rotation(U, V, n_iter=self.itq_n_iter, generator=generator)
            U, V = apply_latent_rotation(U, V, R)
        u1, u2, v1, v2 = dual_svid(U, V)
        self.U.copy_(U)
        self.V.copy_(V)
        self.u1.copy_(u1)
        self.u2.copy_(u2)
        self.v1.copy_(v1)
        self.v2.copy_(v2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._packed_mode:
            Uq = unpack_binary(self.U_packed, (self.num_embeddings, self.split_dim)).float()
            Vq = unpack_binary(self.V_packed, (self.split_dim, self.embedding_dim)).float()
            u1_rows = self.u1_inf.float().squeeze(0)[input_ids]
            Uq_rows = Uq[input_ids]
            v1u2 = self.v1u2_inf.float()
            v2 = self.v2_inf.float()
        else:
            Uq = self.quant_func(self.U)
            Vq = self.quant_func(self.V)
            u1_rows = self.u1.squeeze(0)[input_ids]
            Uq_rows = Uq[input_ids]
            v1u2 = fuse_latent_scales(self.v1, self.u2)
            v2 = self.v2

        # (B, T, s) * u1_rows.unsqueeze(-1) -> @ Vq -> * v2
        latent = (Uq_rows * u1_rows.unsqueeze(-1)) * v1u2
        return (latent @ Vq) * v2

    @torch.no_grad()
    def pack_for_inference(self) -> None:
        if self._packed_mode:
            return
        self.register_buffer("U_packed", pack_binary(binarize_sign(self.U.data).to(torch.int8)))
        self.register_buffer("V_packed", pack_binary(binarize_sign(self.V.data).to(torch.int8)))
        self.register_buffer("u1_inf", self.u1.data.half().contiguous())
        self.register_buffer("v1u2_inf", fuse_latent_scales(self.v1.data, self.u2.data).half())
        self.register_buffer("v2_inf", self.v2.data.half().contiguous())
        for name in ("U", "V", "u1", "u2", "v1", "v2"):
            delattr(self, name)
        self._packed_mode = True

    def memory_report(self) -> LittleBitMemoryReport:
        return compute_memory_report(self.embedding_dim, self.num_embeddings, self.split_dim, False)

