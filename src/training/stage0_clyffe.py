"""
Stage 0 Training & Gate Evaluation Pipeline for Clyffe v0.1 (ARC-LB-01).

Trains all required conditions on identical data & token budget:
  (A) Dense control
  (B) LittleBit-1 (FP warm start -> SVD + Dual-SVID, no ITQ)
  (C1) LittleBit-2 (FP warm start -> SVD + Joint-ITQ T=50 + SmoothSign QAT)
  (C2) LittleBit-2 (FP warm start -> SVD + Joint-ITQ T=50 + Clipped STE QAT)
"""

import json
import math
import os
import random
import time
import tracemalloc
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.eval_utils import normalize_text, token_f1
from src.architecture.initialization import (
    audit_model_representation,
    convert_dense_to_littlebit,
    init_random_weights,
)
from src.architecture.model import ARCLB01Config, ARCLB01Model
from src.inference.generate import generate_response
from src.inference.runtime import ARCLB01Runtime
from src.littlebit.binary_factors import binarize_sign
from src.littlebit.native_operations import LittleBitLinear
from src.architecture.embeddings import LittleBitEmbedding
from src.tokenizer.tokenizer import ARCTokenizer
from src.training.checkpoint import compute_sha256, save_checkpoint
from src.training.dataset import ARCDataset, collate_batch
from src.training.loss import compute_lm_loss
from src.training.metrics import compute_grad_norm, compute_token_accuracy
from src.training.optimizer import build_optimizer_and_scheduler


DATE_TAG = "20261125"
STAGE0_DIR = "datasets/processed/stage_0"
TOK_DIR = "checkpoints/clyffe_v0.1.1_tokenizer"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def train_stage0_tokenizer() -> Tuple[ARCTokenizer, dict]:
    train_rows = load_jsonl(os.path.join(STAGE0_DIR, "train.jsonl"))
    texts = []
    for r in train_rows:
        texts.append(r["input"])
        texts.append(r["output"])

    tok = ARCTokenizer.train_from_texts(texts, vocab_size=600, min_frequency=2)
    tok.save(TOK_DIR)

    # Measure fertility (tokens per whitespace word)
    total_words = 0
    total_tokens = 0
    for t in texts:
        words = t.split()
        total_words += len(words)
        total_tokens += len(tok.encode(t))
    fertility = round(total_tokens / max(1, total_words), 4)
    stats = {
        "vocab_size": tok.vocab_size,
        "fertility_tokens_per_word": fertility,
        "special_tokens": tok.special_tokens,
    }
    return tok, stats


def snapshot_signs(model: ARCLB01Model) -> Dict[str, torch.Tensor]:
    snaps = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (LittleBitLinear, LittleBitEmbedding)):
            snaps[f"{name}.U"] = binarize_sign(mod.U.detach().cpu()).clone()
            snaps[f"{name}.V"] = binarize_sign(mod.V.detach().cpu()).clone()
    return snaps


def compute_sign_flip_rate(prev: Dict[str, torch.Tensor], curr: Dict[str, torch.Tensor]) -> float:
    if not prev:
        return 0.0
    total = 0
    flipped = 0
    for k, p_t in prev.items():
        c_t = curr[k]
        total += p_t.numel()
        flipped += int((p_t != c_t).sum().item())
    return flipped / max(1, total)


def train_condition(
    condition_id: str,
    release_id: str,
    tok: ARCTokenizer,
    train_ds: ARCDataset,
    val_ds: ARCDataset,
    *,
    use_littlebit: bool,
    use_itq: bool,
    quant_func_name: str = "SmoothSign",
    smooth_alpha: float = 15.0,
    split_dim: int = 32,
    warm_epochs: int = 10,
    qat_epochs: int = 40,
    batch_size: int = 32,
    lr: float = 5e-3,
    seed: int = 42,
) -> Dict[str, object]:
    set_seed(seed)
    exp_dir = f"experiments/stage_0/{condition_id}"
    os.makedirs(exp_dir, exist_ok=True)
    metrics_path = os.path.join(exp_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        collate_fn=lambda b: collate_batch(b, pad_id=tok.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id=tok.pad_id),
    )

    dense_cfg = ARCLB01Config(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_layers=4,
        n_heads=4,
        d_ff=192,
        max_seq_len=96,
        use_littlebit=False,
    )
    dense_model = ARCLB01Model(dense_cfg)
    init_random_weights(dense_model, seed=seed)

    total_epochs = (warm_epochs + qat_epochs) if use_littlebit else (warm_epochs + qat_epochs)

    def _eval(m):
        m.eval()
        tl, ta, nb = 0.0, 0.0, 0
        with torch.no_grad():
            for inp, lbl in val_loader:
                logits = m(inp)
                tl += float(compute_lm_loss(logits, lbl).item())
                ta += compute_token_accuracy(logits, lbl)
                nb += 1
        vl = tl / max(1, nb)
        return vl, math.exp(min(vl, 20.0)), ta / max(1, nb)

    t0 = time.time()
    best_val_loss = float("inf")
    prev_signs = {}

    if not use_littlebit:
        model = dense_model
        opt, sched = build_optimizer_and_scheduler(
            model, lr=lr, total_steps=total_epochs * len(train_loader), warmup_steps=20
        )
        for ep in range(1, total_epochs + 1):
            model.train()
            el, ea, eg, nb = 0.0, 0.0, 0.0, 0
            for inp, lbl in train_loader:
                opt.zero_grad(set_to_none=True)
                logits = model(inp)
                loss = compute_lm_loss(logits, lbl)
                loss.backward()
                eg += compute_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                el += float(loss.item())
                ea += compute_token_accuracy(logits, lbl)
                nb += 1
            vl, vppl, va = _eval(model)
            row = {
                "epoch": ep,
                "phase": "dense",
                "train_loss": round(el / nb, 5),
                "val_loss": round(vl, 5),
                "val_ppl": round(vppl, 4),
                "val_acc": round(va, 5),
                "grad_norm": round(eg / nb, 4),
                "sign_flip_rate": 0.0,
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if vl < best_val_loss:
                best_val_loss = vl
                save_checkpoint(model, "checkpoints", f"{release_id}_best", metadata=row, pack_littlebit=False)
        final_info = save_checkpoint(model, "checkpoints", release_id, metadata=row, pack_littlebit=False)
        final_model = model
    else:
        # Phase 1: FP warm start
        opt_w, sched_w = build_optimizer_and_scheduler(
            dense_model, lr=lr, total_steps=warm_epochs * len(train_loader), warmup_steps=10
        )
        for ep in range(1, warm_epochs + 1):
            dense_model.train()
            el, eg, nb = 0.0, 0.0, 0
            for inp, lbl in train_loader:
                opt_w.zero_grad(set_to_none=True)
                logits = dense_model(inp)
                loss = compute_lm_loss(logits, lbl)
                loss.backward()
                eg += compute_grad_norm(dense_model)
                torch.nn.utils.clip_grad_norm_(dense_model.parameters(), 1.0)
                opt_w.step()
                sched_w.step()
                el += float(loss.item())
                nb += 1
            vl, vppl, va = _eval(dense_model)
            row = {
                "epoch": ep,
                "phase": "fp_warm_start",
                "train_loss": round(el / nb, 5),
                "val_loss": round(vl, 5),
                "val_ppl": round(vppl, 4),
                "val_acc": round(va, 5),
                "grad_norm": round(eg / nb, 4),
                "sign_flip_rate": 0.0,
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")

        # Documented conversion: FP warm-start -> LittleBit / LittleBit-2 (no teacher, no distillation)
        lb_cfg = ARCLB01Config(
            vocab_size=tok.vocab_size,
            d_model=64,
            n_layers=4,
            n_heads=4,
            d_ff=192,
            max_seq_len=96,
            use_littlebit=True,
            littlebit_embeddings=True,
            littlebit_lm_head=True,
            split_dim=split_dim,
            residual=False,
            use_itq=use_itq,
            quant_func_name=quant_func_name,
            smooth_alpha=smooth_alpha,
        )
        model = ARCLB01Model(lb_cfg)
        convert_dense_to_littlebit(dense_model, model, seed=seed)
        prev_signs = snapshot_signs(model)

        opt_q, sched_q = build_optimizer_and_scheduler(
            model, lr=lr, total_steps=qat_epochs * len(train_loader), warmup_steps=15
        )
        for ep in range(warm_epochs + 1, total_epochs + 1):
            model.train()
            el, eg, nb = 0.0, 0.0, 0
            for inp, lbl in train_loader:
                opt_q.zero_grad(set_to_none=True)
                logits = model(inp)
                loss = compute_lm_loss(logits, lbl)
                loss.backward()
                eg += compute_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_q.step()
                sched_q.step()
                el += float(loss.item())
                nb += 1
            curr_signs = snapshot_signs(model)
            flip_rate = compute_sign_flip_rate(prev_signs, curr_signs)
            prev_signs = curr_signs
            vl, vppl, va = _eval(model)
            row = {
                "epoch": ep,
                "phase": "littlebit_qat",
                "train_loss": round(el / nb, 5),
                "val_loss": round(vl, 5),
                "val_ppl": round(vppl, 4),
                "val_acc": round(va, 5),
                "grad_norm": round(eg / nb, 4),
                "sign_flip_rate": round(flip_rate, 6),
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if vl < best_val_loss:
                best_val_loss = vl
                save_checkpoint(model, "checkpoints", f"{release_id}_best", metadata=row, pack_littlebit=True)
        final_info = save_checkpoint(model, "checkpoints", release_id, metadata=row, pack_littlebit=True)
        final_model = model

    elapsed = round(time.time() - t0, 2)
    print(f"[{condition_id}] best_val_loss={best_val_loss:.4f} elapsed={elapsed}s ckpt={release_id}_best.pt")
    return {
        "condition": condition_id,
        "release_id": release_id,
        "best_val_loss": round(best_val_loss, 5),
        "best_val_ppl": round(math.exp(min(best_val_loss, 20.0)), 4),
        "final_row": row,
        "elapsed_sec": elapsed,
        "best_ckpt": f"checkpoints/{release_id}_best.pt",
        "audit": audit_model_representation(final_model),
    }


def evaluate_condition(ckpt_path: str) -> Dict[str, object]:
    tracemalloc.start()
    t0 = time.perf_counter()
    rt = ARCLB01Runtime(ckpt_path, tokenizer_dir=TOK_DIR, device="cpu")
    load_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    test_rows = load_jsonl(os.path.join(STAGE0_DIR, "test.jsonl"))
    math_rows = load_jsonl(os.path.join(STAGE0_DIR, "unseen_math.jsonl"))[:100]
    ood_rows = load_jsonl(os.path.join(STAGE0_DIR, "unseen_ood.jsonl"))

    in_scope = [r for r in test_rows if r["category"] != "out_of_scope"]
    em, f1_sum, false_ref = 0, 0.0, 0
    latencies = []
    for r in in_scope:
        t_p = time.perf_counter()
        pred = rt.respond(r["input"], max_new_tokens=36)
        latencies.append((time.perf_counter() - t_p) * 1000.0)
        if normalize_text(pred) == normalize_text(r["output"]):
            em += 1
        f1_sum += token_f1(pred, r["output"])
        if "don't have sufficient trained knowledge" in pred.lower():
            false_ref += 1

    math_em = 0
    for r in math_rows:
        pred = rt.respond(r["input"], max_new_tokens=12)
        if normalize_text(pred) == normalize_text(r["output"]):
            math_em += 1

    ood_ref = 0
    for r in ood_rows:
        pred = rt.respond(r["input"], max_new_tokens=32)
        if "don't have sufficient trained knowledge" in pred.lower():
            ood_ref += 1

    # Session name memory check
    rt.respond("My name is Hari.")
    name_check = rt.respond("What is my name?")

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "in_scope_em": round(em / max(1, len(in_scope)), 4),
        "in_scope_f1": round(f1_sum / max(1, len(in_scope)), 4),
        "false_refusal_rate": round(false_ref / max(1, len(in_scope)), 4),
        "unseen_math_em": round(math_em / max(1, len(math_rows)), 4),
        "ood_refusal_recall": round(ood_ref / max(1, len(ood_rows)), 4),
        "session_name_check": name_check,
        "load_ms": load_ms,
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "peak_ram_mb": round(peak_bytes / (1024 * 1024), 2),
        "metadata": rt.metadata,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-only", action="store_true")
    parser.add_argument("--version-tag", type=str, default="v0.1.1")
    args = parser.parse_args()

    tok, tok_stats = train_stage0_tokenizer()
    print("Tokenizer stats:", json.dumps(tok_stats))

    train_ds = ARCDataset(os.path.join(STAGE0_DIR, "train.jsonl"), tok, max_seq_len=96)
    val_ds = ARCDataset(os.path.join(STAGE0_DIR, "val.jsonl"), tok, max_seq_len=96)

    runs = [
        ("A_dense_v011", f"clyffe-{args.version_tag}-A-{DATE_TAG}", dict(use_littlebit=False, use_itq=False)),
    ]
    if not args.dense_only:
        runs += [
            ("B_littlebit_svd_v011", f"clyffe-{args.version_tag}-B-{DATE_TAG}", dict(use_littlebit=True, use_itq=False, quant_func_name="SmoothSign")),
            ("C1_littlebit2_smoothsign_v011", f"clyffe-{args.version_tag}-C-smoothsign-{DATE_TAG}", dict(use_littlebit=True, use_itq=True, quant_func_name="SmoothSign")),
            ("C2_littlebit2_ste_v011", f"clyffe-{args.version_tag}-C-ste-{DATE_TAG}", dict(use_littlebit=True, use_itq=True, quant_func_name="STEBinary")),
        ]

    results = []
    for cond_id, rel_id, kwargs in runs:
        tr = train_condition(cond_id, rel_id, tok, train_ds, val_ds, **kwargs)
        ev = evaluate_condition(tr["best_ckpt"])
        results.append({**tr, "eval": ev})

    # Write STAGE_0_REPORT.md
    lines = [
        "# Stage 0 Report — Clyffe v0.1 (research code ARC-LB-01)",
        "",
        f"- **Tokenizer:** vocab_size={tok_stats['vocab_size']}, fertility={tok_stats['fertility_tokens_per_word']} tokens/word",
        f"- **Architecture:** 4L, d_model=64, n_heads=4, d_ff=192, ctx=96 (~203K dense-equiv params)",
        "- **Documented Deviation:** FP warm start (10 epochs) -> SVD / Joint-ITQ conversion -> QAT (cross-entropy only, no teacher, no distillation).",
        "",
        "## Comparison Table (A / B / C1 / C2)",
        "",
        "| Condition | Best Val Loss | Val PPL | In-Scope F1 | Unseen Math EM | OOD Refusal | False Refusal | Actual BPW | Disk Bytes | Avg Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        ev = r["eval"]
        md = ev["metadata"]
        lines.append(
            f"| {r['condition']} | {r['best_val_loss']:.4f} | {r['best_val_ppl']:.2f} | "
            f"{ev['in_scope_f1']*100:.1f}% | {ev['unseen_math_em']*100:.1f}% | "
            f"{ev['ood_refusal_recall']*100:.1f}% | {ev['false_refusal_rate']*100:.1f}% | "
            f"{md['overall_effective_bpw']:.4f} | {md['actual_storage_bytes']} | {ev['avg_latency_ms']} ms |"
        )

    with open("evaluation/STAGE_0_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open("evaluation/stage_0_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

