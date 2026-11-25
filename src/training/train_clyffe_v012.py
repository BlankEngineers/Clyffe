"""
Clyffe v0.1.2 Training & Evaluation Pipeline (ARC-LB-01).

Architecture: 4L, d_model=128, n_heads=4, d_ff=256, max_seq_len=192, vocab=2048
(~1.05M dense-equiv params, 99.90%+ LittleBit-2 coverage at split_dim=32).

Experiments:
  - Exp A: Tokenizer validation (2,048-token digit-aware BPE)
  - Exp B: 20-example LittleBit-2 overfitting test (required before full training)
  - Exp C-G: FP Warm-Start -> LittleBit-2 Joint-ITQ -> Pretrain QAT -> SFT QAT
    comparing SmoothSign vs STEBinary, plus full Phase 11 held-out evaluation.
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

from src.architecture.initialization import (
    audit_model_representation,
    convert_dense_to_littlebit,
    init_random_weights,
)
from src.architecture.model import ARCLB01Config, ARCLB01Model
from src.inference.runtime import ARCLB01Runtime
from src.tokenizer.tokenizer import ARCTokenizer
from src.training.checkpoint import save_checkpoint
from src.training.dataset import collate_batch
from src.training.general_datasets import CausalTextDataset, MultiTurnSFTDataset
from src.training.loss import compute_lm_loss
from src.training.metrics import compute_grad_norm, compute_token_accuracy
from src.training.optimizer import build_optimizer_and_scheduler


DATA_DIR = "datasets/processed/clyffe_v012"
TOK_DIR = "checkpoints/clyffe_v0.1.2_tokenizer"
EXP_DIR = "experiments/clyffe_v012"
DATE_TAG = "20261125"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def exp_a_tokenizer(vocab_size: int = 2048):
    texts = []
    with open(os.path.join(DATA_DIR, "pretrain_train.jsonl"), "r", encoding="utf-8") as f:
        for l in f:
            texts.append(json.loads(l)["text"])
    with open(os.path.join(DATA_DIR, "sft_train.jsonl"), "r", encoding="utf-8") as f:
        for l in f:
            for u, a in json.loads(l)["turns"]:
                texts.append(u)
                texts.append(a)

    tok = ARCTokenizer.train_from_texts(texts, vocab_size=vocab_size, min_frequency=2)
    tok.save(TOK_DIR)

    checks = [
        "2*2",
        "345+678",
        "I am not in a good mood.",
        "Explain gravity in simple English.",
        "What is the difference between RAM and ROM?",
    ]
    rt = {s: tok.decode(tok.encode(s)) for s in checks}
    return tok, {"vocab_size": tok.vocab_size, "roundtrip": rt}


def exp_b_littlebit_overfit(tok: ARCTokenizer, sft_ds: MultiTurnSFTDataset):
    cfg = ARCLB01Config(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=256,
        max_seq_len=192,
        use_littlebit=True,
        littlebit_embeddings=True,
        littlebit_lm_head=True,
        split_dim=32,
        use_itq=True,
        quant_func_name="SmoothSign",
        smooth_alpha=15.0,
    )
    model = ARCLB01Model(cfg)
    inp, lbl = collate_batch([sft_ds[i] for i in range(20)], pad_id=tok.pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-3)

    traj = []
    grad_ok = False
    for step in range(1, 101):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(inp)
        loss = compute_lm_loss(logits, lbl)
        loss.backward()
        if step == 1:
            n_nz = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
            n_tot = sum(1 for _ in model.parameters())
            grad_ok = (n_nz == n_tot)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in (1, 25, 50, 75, 100):
            acc = compute_token_accuracy(logits, lbl)
            traj.append({"step": step, "loss": round(loss.item(), 4), "tf_acc": round(acc, 4)})
            print(f"[Exp B LB2 Overfit] step={step:3d} loss={loss.item():.4f} tf_acc={acc:.4f}")
    return {"grad_all_params_nonzero": grad_ok, "trajectory": traj}


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


def train_variant(
    variant_name: str,
    release_id: str,
    quant_func_name: str,
    tok: ARCTokenizer,
    dense_warm: ARCLB01Model,
    pt_train_loader: DataLoader,
    pt_val_loader: DataLoader,
    sft_train_loader: DataLoader,
    sft_val_loader: DataLoader,
):
    t0 = time.time()
    lb_cfg = ARCLB01Config(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=256,
        max_seq_len=192,
        use_littlebit=True,
        littlebit_embeddings=True,
        littlebit_lm_head=True,
        split_dim=32,
        use_itq=True,
        quant_func_name=quant_func_name,
        smooth_alpha=15.0,
    )
    model = ARCLB01Model(lb_cfg)
    convert_dense_to_littlebit(dense_warm, model, seed=42)

    # Pretrain QAT (6 epochs)
    pt_epochs = 6
    opt_pt, sched_pt = build_optimizer_and_scheduler(
        model, lr=3e-3, total_steps=pt_epochs * len(pt_train_loader), warmup_steps=15
    )
    pt_hist = []
    for ep in range(1, pt_epochs + 1):
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
        pt_hist.append({"epoch": ep, "train_loss": round(el / nb, 4), "val_loss": vl, "val_ppl": vppl})
        print(f"[{variant_name} Pretrain] ep={ep}/{pt_epochs} train_loss={el/nb:.4f} val_loss={vl} ppl={vppl}")

    # SFT QAT (18 epochs)
    sft_epochs = 18
    opt_sft, sched_sft = build_optimizer_and_scheduler(
        model, lr=3e-3, total_steps=sft_epochs * len(sft_train_loader), warmup_steps=20
    )
    sft_hist = []
    best_val = float("inf")
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
        row = {"epoch": ep, "train_loss": round(el / nb, 4), "val_loss": vl, "val_ppl": vppl, "val_acc": va}
        sft_hist.append(row)
        print(f"[{variant_name} SFT] ep={ep}/{sft_epochs} train_loss={row['train_loss']} val_loss={vl} acc={va}")
        if vl < best_val:
            best_val = vl
            save_checkpoint(model, "checkpoints", f"{release_id}_best", metadata=row, pack_littlebit=True)

    save_checkpoint(model, "checkpoints", release_id, metadata=sft_hist[-1], pack_littlebit=True)

    # Evaluate on Phase 11 held-out prompts
    tracemalloc.start()
    rt = ARCLB01Runtime(f"checkpoints/{release_id}_best.pt", tokenizer_dir=TOK_DIR, device="cpu")
    heldout_rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, "heldout_eval.jsonl"))]
    eval_outputs = []
    latencies = []
    for r in heldout_rows:
        t_p = time.perf_counter()
        out = rt.respond(r["input"], max_new_tokens=48)
        latencies.append((time.perf_counter() - t_p) * 1000.0)
        eval_outputs.append({"category": r["category"], "input": r["input"], "output": out})

    # Multi-turn check
    rt.reset_session()
    mt1 = rt.respond("My name is Hari.", use_history=True)
    mt2 = rt.respond("What is my name?", use_history=True)

    _, peak_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "variant": variant_name,
        "release_id": release_id,
        "elapsed_sec": round(time.time() - t0, 2),
        "best_sft_val_loss": best_val,
        "final_train_loss": sft_hist[-1]["train_loss"],
        "pretrain_history": pt_hist,
        "sft_history": sft_hist,
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "peak_ram_mb": round(peak_b / (1024 * 1024), 2),
        "metadata": rt.metadata,
        "heldout_outputs": eval_outputs,
        "multi_turn": {"t1": mt1, "t2": mt2},
    }


def main():
    set_seed(42)
    os.makedirs(EXP_DIR, exist_ok=True)

    tok, tok_info = exp_a_tokenizer(2048)
    print("Exp A Tokenizer:", json.dumps(tok_info, indent=2))

    pt_train = CausalTextDataset(os.path.join(DATA_DIR, "pretrain_train.jsonl"), tok, block_size=192)
    pt_val = CausalTextDataset(os.path.join(DATA_DIR, "pretrain_val.jsonl"), tok, block_size=192)
    sft_train = MultiTurnSFTDataset(os.path.join(DATA_DIR, "sft_train.jsonl"), tok, max_seq_len=192)
    sft_val = MultiTurnSFTDataset(os.path.join(DATA_DIR, "sft_val.jsonl"), tok, max_seq_len=192)
    print(f"Blocks: pt_train={len(pt_train)}, pt_val={len(pt_val)} | SFT: train={len(sft_train)}, val={len(sft_val)}")

    exp_b = exp_b_littlebit_overfit(tok, sft_train)
    assert exp_b["grad_all_params_nonzero"], "LittleBit-2 gradient check failed!"

    pt_train_loader = DataLoader(pt_train, batch_size=24, shuffle=True, collate_fn=lambda b: collate_batch(b, tok.pad_id))
    pt_val_loader = DataLoader(pt_val, batch_size=24, shuffle=False, collate_fn=lambda b: collate_batch(b, tok.pad_id))
    sft_train_loader = DataLoader(sft_train, batch_size=32, shuffle=True, collate_fn=lambda b: collate_batch(b, tok.pad_id))
    sft_val_loader = DataLoader(sft_val, batch_size=32, shuffle=False, collate_fn=lambda b: collate_batch(b, tok.pad_id))

    # Shared FP Warm Start (4 epochs pretrain)
    dense_cfg = ARCLB01Config(
        vocab_size=tok.vocab_size, d_model=128, n_layers=4, n_heads=4, d_ff=256, max_seq_len=192, use_littlebit=False
    )
    dense_warm = ARCLB01Model(dense_cfg)
    init_random_weights(dense_warm, seed=42)
    opt_w, sched_w = build_optimizer_and_scheduler(dense_warm, lr=4e-3, total_steps=4 * len(pt_train_loader), warmup_steps=15)
    for ep in range(1, 5):
        dense_warm.train()
        el, nb = 0.0, 0
        for inp, lbl in pt_train_loader:
            opt_w.zero_grad(set_to_none=True)
            loss = compute_lm_loss(dense_warm(inp), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_warm.parameters(), 1.0)
            opt_w.step()
            sched_w.step()
            el += float(loss.item())
            nb += 1
        vl, vppl, _ = _eval_loader(dense_warm, pt_val_loader)
        print(f"[FP Warm-Start] ep={ep}/4 train_loss={el/nb:.4f} val_loss={vl} ppl={vppl}")

    res_smooth = train_variant(
        "C1_SmoothSign",
        f"clyffe-v0.1.2-C-smoothsign-{DATE_TAG}",
        "SmoothSign",
        tok,
        dense_warm,
        pt_train_loader,
        pt_val_loader,
        sft_train_loader,
        sft_val_loader,
    )
    res_ste = train_variant(
        "C2_STEBinary",
        f"clyffe-v0.1.2-C-ste-{DATE_TAG}",
        "STEBinary",
        tok,
        dense_warm,
        pt_train_loader,
        pt_val_loader,
        sft_train_loader,
        sft_val_loader,
    )

    summary = {
        "tokenizer": tok_info,
        "exp_b_overfit": exp_b,
        "variants": [res_smooth, res_ste],
    }
    with open(os.path.join(EXP_DIR, "v012_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Clyffe v0.1.2 — Diagnosis, Repair & General LM Report (ARC-LB-01)",
        "",
        f"- **Architecture:** 4L, `d_model=128`, `n_heads=4`, `d_ff=256`, `ctx=192`, `vocab={tok_info['vocab_size']}`",
        f"- **Exp B (20-example LittleBit-2 Overfit):** grad_all_nonzero={exp_b['grad_all_params_nonzero']}, final_loss={exp_b['trajectory'][-1]['loss']}, tf_acc={exp_b['trajectory'][-1]['tf_acc']}",
        "",
        "## SmoothSign vs STEBinary Comparison",
        "",
        "| Variant | Best SFT Val Loss | Final Train Loss | Dense Params | LB2 Coverage | Actual BPW | Disk Bytes | Avg Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in (res_smooth, res_ste):
        md = v["metadata"]
        lines.append(
            f"| {v['variant']} | {v['best_sft_val_loss']} | {v['final_train_loss']} | "
            f"{md['parameter_count_dense_equiv']} | {md['littlebit_coverage_pct']}% | "
            f"{md['overall_effective_bpw']} | {md['actual_storage_bytes']} | {v['avg_latency_ms']} ms |"
        )
    lines += ["", "## Phase 11 Held-Out Prompt Outputs (C1_SmoothSign)", ""]
    for item in res_smooth["heldout_outputs"]:
        lines.append(f"- **[{item['category']}]** `{item['input']}` -> `{item['output']}`")

    with open("evaluation/CLYFFE_V012_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

