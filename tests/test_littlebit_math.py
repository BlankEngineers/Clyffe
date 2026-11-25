"""
Unit tests for ARC-LB-01 LittleBit-2 Mathematical Computing Engine.
"""

import torch
import pytest

from src.littlebit import (
    LittleBitLinear,
    SmoothSign,
    STEBinary,
    apply_latent_rotation,
    binarize_sign,
    compute_eff_bits,
    compute_joint_itq_rotation,
    compute_memory_report,
    dual_svid,
    native_factorized_forward,
    pack_binary,
    random_orthogonal,
    reconstruct_branch,
    reconstruction_metrics,
    svd_factorize,
    unpack_binary,
    verify_orthogonality,
)


def test_svd_factorization_and_rotation_invariance():
    torch.manual_seed(42)
    W = torch.randn(64, 96)
    U, V = svd_factorize(W, split_dim=24)
    assert U.shape == (64, 24)
    assert V.shape == (24, 96)

    R = random_orthogonal(24)
    ok, err = verify_orthogonality(R)
    assert ok, f"Orthogonality failed: {err}"

    U_rot, V_rot = apply_latent_rotation(U, V, R)
    assert torch.allclose(U @ V, U_rot @ V_rot, atol=1e-4)


def test_joint_itq_reduces_quantization_error():
    torch.manual_seed(7)
    W = torch.randn(96, 128)
    U, V = svd_factorize(W, split_dim=32)
    X_comb = torch.cat([U, V.t()], dim=0)
    err_before = torch.linalg.norm(binarize_sign(X_comb) - X_comb, ord="fro").item()

    R, diag = compute_joint_itq_rotation(U, V, n_iter=40)
    ok, ortho_err = verify_orthogonality(R)
    assert ok, f"ITQ rotation not orthogonal: {ortho_err}"

    err_after = diag["quant_error"][-1]
    assert err_after < err_before, f"Expected Joint-ITQ to reduce error: {err_after} >= {err_before}"


def test_binary_pack_unpack_roundtrip():
    torch.manual_seed(11)
    for shape in [(16, 32), (33, 65), (96, 24)]:
        raw = torch.randn(shape)
        b = binarize_sign(raw).to(torch.int8)
        assert set(torch.unique(b).tolist()).issubset({-1, 1})
        packed = pack_binary(b)
        assert packed.dtype == torch.int32
        unpacked = unpack_binary(packed, shape)
        assert unpacked.dtype == torch.int8
        assert torch.equal(b, unpacked)


def test_native_forward_equals_reconstructed_matmul():
    torch.manual_seed(123)
    W = torch.randn(64, 96)
    U, V = svd_factorize(W, split_dim=24)
    R, _ = compute_joint_itq_rotation(U, V, n_iter=30)
    U, V = apply_latent_rotation(U, V, R)
    u1, u2, v1, v2 = dual_svid(U, V)
    Uq = binarize_sign(U)
    Vq = binarize_sign(V)

    x = torch.randn(5, 10, 96)
    y_native = native_factorized_forward(x, Vq, Uq, v2, v1 * u2, u1)
    W_hat = reconstruct_branch(Uq, Vq, u1, u2, v1, v2)
    y_dense_ref = x @ W_hat.t()
    assert torch.allclose(y_native, y_dense_ref, atol=1e-4, rtol=1e-4)


def test_littlebit_linear_qat_and_packed_inference():
    torch.manual_seed(99)
    W = torch.randn(96, 96)
    layer = LittleBitLinear(96, 96, split_dim=24, residual=True, use_itq=True)
    layer.initialize_from_dense(W)

    x = torch.randn(4, 16, 96, requires_grad=True)
    y_qat = layer(x)
    loss = y_qat.pow(2).mean()
    loss.backward()
    assert layer.U.grad is not None
    assert layer.V.grad is not None
    assert layer.u1.grad is not None
    assert x.grad is not None

    with torch.no_grad():
        y_before_pack = layer(x).detach()
        layer.pack_for_inference()
        assert layer._packed_mode is True
        assert not hasattr(layer, "U")
        assert not hasattr(layer, "V")
        y_after_pack = layer(x).detach()
        assert torch.allclose(y_before_pack, y_after_pack, atol=1e-2, rtol=1e-2)


def test_qat_surrogate_gradients():
    x = torch.tensor([-2.0, -0.005, 0.0, 0.005, 2.0], requires_grad=True)
    y_ste = STEBinary(x)
    y_ste.sum().backward()
    assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))

    x2 = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    y_smooth = SmoothSign(x2, 100.0)
    y_smooth.sum().backward()
    assert x2.grad[1].item() == pytest.approx(100.0, rel=1e-4)
    assert x2.grad[0].item() < 1e-4


def test_memory_accounting():
    rep = compute_memory_report(96, 192, split_dim=24, residual=False)
    assert rep.actual_bpw == pytest.approx(compute_eff_bits(96, 192, 24, False))
    assert rep.compression_ratio_vs_fp16 > 1.0

