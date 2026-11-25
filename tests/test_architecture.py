"""
Unit tests for ARC-LB-01 Tokenizer and Model Architecture.
"""

import torch

from src.architecture import (
    ARCLB01Config,
    ARCLB01Model,
    audit_model_representation,
    convert_dense_to_littlebit,
    init_random_weights,
)
from src.tokenizer import ARCTokenizer


def test_tokenizer_digit_awareness_and_roundtrip(tmp_path):
    texts = [
        "Hello! How can I help you?",
        "I am ARC-LB-01, developed by ARC III.",
        "25 + 35 = 60",
        "144 ÷ 12 = 12",
        "Gravity is the attraction between objects with mass.",
    ]
    tok = ARCTokenizer.train_from_texts(texts, vocab_size=256, min_frequency=1)
    ids = tok.encode("25 + 35")
    decoded = tok.decode(ids)
    assert "2" in decoded and "5" in decoded and "+" in decoded

    qa_ids = tok.encode_qa("Hello", "Hello! How can I help you?")
    assert qa_ids[0] == tok.bos_id
    assert qa_ids[-1] == tok.eos_id

    save_dir = str(tmp_path / "tok")
    tok.save(save_dir)
    tok2 = ARCTokenizer.load(save_dir)
    assert tok2.encode("144 ÷ 12") == tok.encode("144 ÷ 12")


def test_dense_and_littlebit_model_forward_backward_and_pack():
    cfg_dense = ARCLB01Config(vocab_size=256, d_model=96, n_layers=2, n_heads=4, d_ff=192, use_littlebit=False)
    dense = ARCLB01Model(cfg_dense)
    init_random_weights(dense, seed=42)

    cfg_lb = ARCLB01Config(
        vocab_size=256,
        d_model=96,
        n_layers=2,
        n_heads=4,
        d_ff=192,
        use_littlebit=True,
        littlebit_embeddings=True,
        littlebit_lm_head=True,
        split_dim=24,
    )
    lb = ARCLB01Model(cfg_lb)
    convert_dense_to_littlebit(dense, lb, seed=42)

    audit = audit_model_representation(lb)
    assert audit["littlebit_coverage_pct"] > 99.0

    x = torch.randint(0, 256, (2, 16))
    logits = lb(x)
    assert logits.shape == (2, 16, 256)
    loss = logits.pow(2).mean()
    loss.backward()

    lb.pack_for_inference()
    with torch.no_grad():
        logits_packed = lb(x)
        assert logits_packed.shape == (2, 16, 256)

