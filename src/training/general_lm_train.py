"""
General Language Model Training & Evaluation for Clyffe v0.2 (ARC-LB-01).

Upgrades capacity to:
  d_model=128, n_layers=6, n_heads=4, d_ff=384, max_seq_len=256, vocab=4096
  (~2.23M dense-equiv params, 99.93% LittleBit-2 coverage at split_dim=32).

Stages:
  - Tokenizer training (4k vocab) + unseen sentence round-trip check
  - Stage A: 20-example LittleBit-2 overfitting & gradient-flow verification
  - Stage B/D: FP warm-start causal LM pretraining -> LittleBit-2 Joint-ITQ conversion -> QAT
  - Stage C/E/G: Multi-turn SFT QAT on Dolly-15k, ARC-Easy, GSM8K, conversions, multi-turn, OOS
  - Stage F: Held-out generalization evaluation & GENERAL_LM_REPORT.md
"""

import json
import math
import os
import random
import time
import tracemalloc
import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.eval_utils import normalize_text, token_f1
from src.architecture.embeddings import LittleBitEmbedding
from src.architecture.initialization import (
    audit_model_representation,
    convert_dense_to_littlebit,
    init_random_weights,
)
from src.architecture.model import ARCLB01Config, ARCLB01Model
from src.inference.runtime import ARCLB01Runtime
from src.littlebit.binary_factors import binarize_sign
from src.littlebit.native_operations import LittleBitLinear
from src.tokenizer.tokenizer import ARCTokenizer
from src.training.checkpoint import save_checkpoint
from src.training.dataset import collate_batch
from src.training.general_datasets import CausalTextDataset, MultiTurnSFTDataset
from src.training.loss import compute_lm_loss
from src.training.metrics import compute_grad_norm, compute_token_accuracy
from src.training.optimizer import build_optimizer_and_scheduler


DATA_DIR = "datasets/processed/general_v02"
TOK_DIR = "checkpoints/clyffe_v0.2_tokenizer"
EXP_DIR = "experiments/general_v02"
RELEASE_ID = "clyffe-v0.2-general-20261125"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_general_tokenizer(vocab_size: int = 4096):
    texts = []
    with open(os.path.join(DATA_DIR, "pretrain_train.jsonl"), "r", encoding="utf-8") as f:
        for l in f:
            texts.append(json.loads(l)["text"])
    with open(os.path.join(DATA_DIR, "sft_train.jsonl"), "r", encoding="utf-8") as f:
        for l in f:
            for u, a in json.loads(l)["turns"]:
                texts.append(u)
                texts.append(a)

    tok = ARCTokenizer.train_from_texts(texts, vocab_size=vocab_size, min_frequency=3)
    tok.save(TOK_DIR)

    # Measure fertility & unseen round-trip
    sample_words, sample_toks = 0, 0
    for t in texts[:500]:
        sample_words += len(t.split())
        sample_toks += len(tok.encode(t))
    fertility = round(sample_toks / max(1, sample_words), 4)

    unseen_checks = [
        "Explain gravity in simple English.",
        "What is the difference between RAM and ROM?",
        "Convert 5 kilometres into metres.",
        "Calculate 345 + 678.",
    ]
    roundtrip = {s: tok.decode(tok.encode(s)) for s in unseen_checks}
    return tok, {"vocab_size": tok.vocab_size, "fertility": fertility, "roundtrip": roundtrip}


def run_stage_a_littlebit_overfit(tok: ARCTokenizer, pt_ds: CausalTextDataset):
    """Verify LittleBit-2 gradients flow to all U, V, u1, u2, v1, v2 and overfit 16 blocks."""
    cfg = ARCLB01Config(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=6,
        n_heads=4,
        d_ff=384,
        max_seq_len=256,
        use_littlebit=True,
        littlebit_embeddings=True,
        littlebit_lm_head=True,
        split_dim=32,
        use_itq=True,
        quant_func_name="SmoothSign",
        smooth_alpha=15.0,
    )
    model = ARCLB01Model(cfg)
    inp, lbl = collate_batch([pt_ds[i] for i in range(min(16, len(pt_ds)))], pad_id=tok.pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    losses = []
    grad_ok = False
    for step in range(1, 41):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(inp)
        loss = compute_lm_loss(logits, lbl)
        loss.backward()
        if step == 1:
            n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
            n_total = sum(1 for _ in model.parameters())
            grad_ok = (n_with_grad == n_total)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in (1, 10, 20, 40):
            acc = compute_token_accuracy(logits, lbl)
            losses.append({"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4)})
            print(f"[Stage A LB2 Overfit] step={step:02d} loss={loss.item():.4f} acc={acc:.4f}")
    return {"grad_all_params_nonzero": grad_ok, "trajectory": losses}


def _eval_loader(model, loader):
    model.eval()
    tl, ta, nb = 0.0, 0.0, 0
    with torch.no_grad():
        for inp, lbl in loader:
            logits = model(inp)
            tl += float(compute_lm_loss(logits, lbl).item())
            ta += compute_token_accuracy(logits, lbl)
            nb += 1
    vl = tl / max(1, nb)
    return round(vl, 5), round(math.exp(min(vl, 20.0)), 3), round(ta / max(1, nb), 4)


def main():
    set_seed(42)
    os.makedirs(EXP_DIR, exist_ok=True)
    t_start = time.time()

    tok, tok_info = train_general_tokenizer(4096)
    print("Tokenizer:", json.dumps(tok_info, indent=2))

    pt_train = CausalTextDataset(os.path.join(DATA_DIR, "pretrain_train.jsonl"), tok, block_size=256)
    pt_val = CausalTextDataset(os.path.join(DATA_DIR, "pretrain_val.jsonl"), tok, block_size=256)
    sft_train = MultiTurnSFTDataset(os.path.join(DATA_DIR, "sft_train.jsonl"), tok, max_seq_len=256)
    sft_val = MultiTurnSFTDataset(os.path.join(DATA_DIR, "sft_val.jsonl"), tok, max_seq_len=256)
    print(f"Blocks: pt_train={len(pt_train)}, pt_val={len(pt_val)} | SFT: train={len(sft_train)}, val={len(sft_val)}")

    stage_a = run_stage_a_littlebit_overfit(tok, pt_train)

    pt_train_loader = DataLoader(pt_train, batch_size=16, shuffle=True, collate_fn=lambda b: collate_batch(b, tok.pad_id))
    pt_val_loader = DataLoader(pt_val, batch_size=16, shuffle=False, collate_fn=lambda b: collate_batch(b, tok.pad_id))
    sft_train_loader = DataLoader(sft_train, batch_size=32, shuffle=True, collate_fn=lambda b: collate_batch(b, tok.pad_id))
    sft_val_loader = DataLoader(sft_val, batch_size=32, shuffle=False, collate_fn=lambda b: collate_batch(b, tok.pad_id))

    # Build FP parent for warm start (4 epochs pretraining)
    dense_cfg = ARCLB01Config(
        vocab_size=tok.vocab_size, d_model=128, n_layers=6, n_heads=4, d_ff=384, max_seq_len=256, use_littlebit=False
    )
    dense_model = ARCLB01Model(dense_cfg)
    init_random_weights(dense_model, seed=42)

    warm_epochs = 4
    opt_w, sched_w = build_optimizer_and_scheduler(dense_model, lr=4e-3, total_steps=warm_epochs * len(pt_train_loader), warmup_steps=15)
    for ep in range(1, warm_epochs + 1):
        dense_model.train()
        el, nb = 0.0, 0
        for inp, lbl in pt_train_loader:
            opt_w.zero_grad(set_to_none=True)
            loss = compute_lm_loss(dense_model(inp), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_model.parameters(), 1.0)
            opt_w.step()
            sched_w.step()
            el += float(loss.item())
            nb += 1
        vl, vppl, va = _eval_loader(dense_model, pt_val_loader)
        print(f"[FP Warm-Start] ep={ep}/{warm_epochs} train_loss={el/nb:.4f} val_loss={vl:.4f} val_ppl={vppl}")

    # Convert to LittleBit-2 (SVD + Joint-ITQ T=50 + Dual-SVID, all learned matrices)
    lb_cfg = ARCLB01Config(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=6,
        n_heads=4,
        d_ff=384,
        max_seq_len=256,
        use_littlebit=True,
        littlebit_embeddings=True,
        littlebit_lm_head=True,
        split_dim=32,
        use_itq=True,
        quant_func_name="SmoothSign",
        smooth_alpha=15.0,
    )
    model = ARCLB01Model(lb_cfg)
    convert_dense_to_littlebit(dense_model, model, seed=42)

    # Stage B/D: Causal LM Pretraining QAT (6 epochs)
    pt_qat_epochs = 6
    opt_pt, sched_pt = build_optimizer_and_scheduler(model, lr=3e-3, total_steps=pt_qat_epochs * len(pt_train_loader), warmup_steps=15)
    pt_history = []
    for ep in range(1, pt_qat_epochs + 1):
        model.train()
        el, nb = 0.0, 0
        for inp, lbl in pt_train_loader:
            opt_pt.zero_grad(set_to_none=True)
            loss = compute_lm_loss(model(inp), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_pt.step()
            sched_pt.step()
            el += float(loss.item())
            nb += 1
        vl, vppl, va = _eval_loader(model, pt_val_loader)
        row = {"stage": "pretrain_qat", "epoch": ep, "train_loss": round(el / nb, 4), "val_loss": vl, "val_ppl": vppl, "val_acc": va}
        pt_history.append(row)
        print(f"[Pretrain QAT] ep={ep}/{pt_qat_epochs} train_loss={row['train_loss']} val_loss={vl} val_ppl={vppl}")

    # Stage C/E/G: Multi-Turn SFT QAT (14 epochs)
    sft_epochs = 14
    opt_sft, sched_sft = build_optimizer_and_scheduler(model, lr=3e-3, total_steps=sft_epochs * len(sft_train_loader), warmup_steps=20)
    best_sft_val = float("inf")
    sft_history = []
    for ep in range(1, sft_epochs + 1):
        model.train()
        el, nb = 0.0, 0
        for inp, lbl in sft_train_loader:
            opt_sft.zero_grad(set_to_none=True)
            loss = compute_lm_loss(model(inp), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_sft.step()
            sched_sft.step()
            el += float(loss.item())
            nb += 1
        vl, vppl, va = _eval_loader(model, sft_val_loader)
        row = {"stage": "sft_qat", "epoch": ep, "train_loss": round(el / nb, 4), "val_loss": vl, "val_ppl": vppl, "val_acc": va}
        sft_history.append(row)
        print(f"[SFT QAT] ep={ep}/{sft_epochs} train_loss={row['train_loss']} val_loss={vl} val_acc={va}")
        if vl < best_sft_val:
            best_sft_val = vl
            save_checkpoint(model, "checkpoints", f"{RELEASE_ID}_best", metadata=row, pack_littlebit=True)

    save_checkpoint(model, "checkpoints", RELEASE_ID, metadata=sft_history[-1], pack_littlebit=True)

    # Stage F: Native Offline Evaluation on Held-Out Prompts & Multi-Turn
    tracemalloc.start()
    rt = ARCLB01Runtime(f"checkpoints/{RELEASE_ID}_best.pt", tokenizer_dir=TOK_DIR, device="cpu")
    unseen_prompts = [
        "Explain gravity in simple English.",
        "What is the difference between RAM and ROM?",
        "Why does ice melt?",
        "Convert 5 kilometres into metres.",
        "Calculate 345 + 678.",
        "What is the stock price of Apple today?",
    ]
    sample_outputs = {}
    latencies = []
    for p in unseen_prompts:
        t0 = time.perf_counter()
        out = rt.respond(p, max_new_tokens=48)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        sample_outputs[p] = out

    # Multi-turn check
    rt.reset_session()
    m1 = rt.respond("My name is Hari.", use_history=True)
    m2 = rt.respond("What is my name?", use_history=True)
    m3 = rt.respond("Explain gravity.", use_history=True)

    # Held-out SFT F1
    heldout_rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, "sft_heldout_test.jsonl"))][:80]
    f1s = []
    for r in heldout_rows:
        u0, a0 = r["turns"][0]
        pred = rt.respond(u0, max_new_tokens=40, use_history=False)
        f1s.append(token_f1(pred, a0))

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        "release_id": RELEASE_ID,
        "elapsed_sec": round(time.time() - t_start, 2),
        "tokenizer": tok_info,
        "stage_a_overfit": stage_a,
        "pretrain_history": pt_history,
        "sft_history": sft_history,
        "best_sft_val_loss": best_sft_val,
        "heldout_80_avg_f1": round(sum(f1s) / max(1, len(f1s)), 4),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "peak_ram_mb": round(peak_bytes / (1024 * 1024), 2),
        "metadata": rt.metadata,
        "sample_outputs": sample_outputs,
        "multi_turn": {"t1": m1, "t2": m2, "t3": m3},
    }
    with open(os.path.join(EXP_DIR, "general_v02_results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Clyffe v0.2 — General Language Model Training Report (ARC-LB-01)",
        "",
        f"- **Checkpoint:** `checkpoints/{RELEASE_ID}_best.pt` (`{rt.metadata['actual_storage_bytes']}` bytes)",
        f"- **Architecture:** {rt.metadata['architecture']}",
        f"- **Dense-Equiv Params:** {rt.metadata['parameter_count_dense_equiv']} | **LittleBit-2 Coverage:** {rt.metadata['littlebit_coverage_pct']}% | **Effective BPW:** {rt.metadata['overall_effective_bpw']}",
        f"- **Tokenizer:** `vocab_size={tok_info['vocab_size']}`, `fertility={tok_info['fertility']}`",
        f"- **Stage A LittleBit-2 Gradients All Non-Zero:** {stage_a['grad_all_params_nonzero']} (final loss `{stage_a['trajectory'][-1]['loss']}`)",
        f"- **Best SFT Val Loss:** `{best_sft_val}` | **Held-out 80-sample Avg F1:** `{report['heldout_80_avg_f1']*100:.2f}%`",
        f"- **Avg Native Packed Latency:** `{report['avg_latency_ms']} ms` | **Peak RAM:** `{report['peak_ram_mb']} MB`",
        "",
        "## Unseen Prompt Outputs",
        "",
    ]
    for p, o in sample_outputs.items():
        md_lines.append(f"- **{p}** -> `{o}`")
    md_lines += [
        "",
        "## Multi-Turn Conversation Check",
        f"- `My name is Hari.` -> `{m1}`",
        f"- `What is my name?` -> `{m2}`",
        f"- `Explain gravity.` -> `{m3}`",
    ]
    with open("evaluation/GENERAL_LM_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()

