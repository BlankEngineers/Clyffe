"""
Main Training Script for ARC-LB-01 Experiments:
- dense: Dense Baseline Control (ARC-LB-01-INITIAL)
- littlebit_svd: Experiment A without Joint-ITQ (ARC-LB-01-LITTLEBIT)
- littlebit_joint_itq: Experiment A with Joint-ITQ + QAT (ARC-LB-01-JOINT-ITQ / QAT / FINAL)
- littlebit_direct: Experiment B direct LittleBit-2 optimization from init
"""

import argparse
import json
import os
import random
import numpy as np
import torch

from src.architecture.initialization import convert_dense_to_littlebit, init_random_weights
from src.architecture.model import ARCLB01Config, ARCLB01Model
from src.tokenizer.tokenizer import ARCTokenizer
from src.training.dataset import ARCDataset
from src.training.trainer import Trainer, TrainingConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train ARC-LB-01")
    parser.add_argument(
        "--mode",
        choices=["dense", "littlebit_svd", "littlebit_joint_itq", "littlebit_direct"],
        required=True,
    )
    parser.add_argument("--release_tag", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--split_dim", type=int, default=24)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--smooth_alpha", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    tok_dir = "checkpoints/tokenizer"
    if not os.path.exists(os.path.join(tok_dir, "vocab.json")):
        texts = []
        for path in ("datasets/train/train.jsonl", "datasets/validation/val.jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    texts.append(rec["input"])
                    texts.append(rec["output"])
        tok = ARCTokenizer.train_from_texts(texts, vocab_size=512, min_frequency=2)
        tok.save(tok_dir)
    else:
        tok = ARCTokenizer.load(tok_dir)

    train_ds = ARCDataset("datasets/train/train.jsonl", tok, max_seq_len=128)
    val_ds = ARCDataset("datasets/validation/val.jsonl", tok, max_seq_len=128)

    use_lb = args.mode != "dense"
    use_itq = args.mode in ("littlebit_joint_itq", "littlebit_direct")

    cfg = ARCLB01Config(
        vocab_size=tok.vocab_size,
        d_model=96,
        n_layers=2,
        n_heads=4,
        d_ff=192,
        max_seq_len=128,
        use_littlebit=use_lb,
        littlebit_embeddings=use_lb,
        littlebit_lm_head=use_lb,
        split_dim=args.split_dim,
        residual=args.residual,
        use_itq=use_itq,
        quant_func_name="SmoothSign",
        smooth_alpha=args.smooth_alpha,
        tie_weights=False,
    )
    model = ARCLB01Model(cfg)

    if args.mode == "dense":
        init_random_weights(model, seed=args.seed)
    elif args.mode in ("littlebit_svd", "littlebit_joint_itq"):
        dense_parent = ARCLB01Model(
            ARCLB01Config(
                vocab_size=tok.vocab_size,
                d_model=96,
                n_layers=2,
                n_heads=4,
                d_ff=192,
                max_seq_len=128,
                use_littlebit=False,
            )
        )
        init_random_weights(dense_parent, seed=args.seed)
        convert_dense_to_littlebit(dense_parent, model, seed=args.seed)
    elif args.mode == "littlebit_direct":
        # Already directly initialized in LittleBit representation
        pass

    tcfg = TrainingConfig(
        experiment_name=args.mode,
        release_tag=args.release_tag,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        save_dir="checkpoints",
        log_dir=f"experiments/{args.mode}",
        pack_best=use_lb,
    )
    trainer = Trainer(model, train_ds, val_ds, tcfg, pad_id=tok.pad_id)
    summary = trainer.train()
    print(f"Completed {args.mode} -> best_val_loss={summary['best_val_loss']}")


if __name__ == "__main__":
    main()

