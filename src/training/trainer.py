"""
Reproducible Training Engine for ARC-LB-01.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
import time
from typing import Dict, List
import torch
from torch.utils.data import DataLoader

from src.architecture.initialization import audit_model_representation
from src.architecture.model import ARCLB01Model
from src.training.checkpoint import save_checkpoint
from src.training.dataset import ARCDataset, collate_batch
from src.training.loss import compute_lm_loss
from src.training.metrics import compute_grad_norm, compute_token_accuracy
from src.training.optimizer import build_optimizer_and_scheduler


@dataclass
class TrainingConfig:
    experiment_name: str
    release_tag: str
    epochs: int = 25
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    save_dir: str = "checkpoints"
    log_dir: str = "experiments"
    pack_best: bool = True


class Trainer:
    def __init__(
        self,
        model: ARCLB01Model,
        train_ds: ARCDataset,
        val_ds: ARCDataset,
        cfg: TrainingConfig,
        pad_id: int = 0,
    ):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.pad_id = pad_id

        gen = torch.Generator().manual_seed(cfg.seed)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=gen,
            collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
        )

        total_steps = max(1, cfg.epochs * len(self.train_loader))
        warmup_steps = max(10, int(0.05 * total_steps))
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            self.model,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        for inp, lbl in self.val_loader:
            inp, lbl = inp.to(self.device), lbl.to(self.device)
            logits = self.model(inp)
            loss = compute_lm_loss(logits, lbl)
            total_loss += float(loss.item())
            total_acc += compute_token_accuracy(logits, lbl)
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        return {
            "val_loss": avg_loss,
            "val_ppl": math.exp(min(avg_loss, 20.0)),
            "val_token_acc": total_acc / max(1, n_batches),
        }

    def train(self) -> Dict[str, object]:
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        os.makedirs(self.cfg.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        history: List[Dict[str, float]] = []
        t0 = time.time()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            ep_loss = 0.0
            ep_acc = 0.0
            ep_gnorm = 0.0
            n_batches = 0

            for inp, lbl in self.train_loader:
                inp, lbl = inp.to(self.device), lbl.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(inp)
                loss = compute_lm_loss(logits, lbl)
                loss.backward()
                gnorm = compute_grad_norm(self.model)
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                ep_loss += float(loss.item())
                ep_acc += compute_token_accuracy(logits, lbl)
                ep_gnorm += gnorm
                n_batches += 1

            train_loss = ep_loss / max(1, n_batches)
            train_acc = ep_acc / max(1, n_batches)
            avg_gnorm = ep_gnorm / max(1, n_batches)
            val_metrics = self.evaluate()

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "train_token_acc": round(train_acc, 5),
                "grad_norm": round(avg_gnorm, 5),
                "val_loss": round(val_metrics["val_loss"], 5),
                "val_ppl": round(val_metrics["val_ppl"], 4),
                "val_token_acc": round(val_metrics["val_token_acc"], 5),
                "lr": round(self.scheduler.get_last_lr()[0], 6),
            }
            history.append(row)
            print(
                f"[{self.cfg.experiment_name}] Epoch {epoch:02d}/{self.cfg.epochs:02d} "
                f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
                f"val_acc={row['val_token_acc']:.4f} gnorm={row['grad_norm']:.3f}"
            )

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                save_checkpoint(
                    self.model,
                    self.cfg.save_dir,
                    f"{self.cfg.release_tag}_best",
                    metadata={"epoch": epoch, **row, "audit": audit_model_representation(self.model)},
                    pack_littlebit=self.cfg.pack_best,
                )

        # Save latest/final checkpoint
        final_info = save_checkpoint(
            self.model,
            self.cfg.save_dir,
            self.cfg.release_tag,
            metadata={"epoch": self.cfg.epochs, **history[-1], "audit": audit_model_representation(self.model)},
            pack_littlebit=self.cfg.pack_best,
        )

        peak_vram_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            if self.device.type == "cuda"
            else 0.0
        )

        summary = {
            "experiment_name": self.cfg.experiment_name,
            "release_tag": self.cfg.release_tag,
            "device": str(self.device),
            "elapsed_sec": round(time.time() - t0, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "best_val_loss": round(best_val_loss, 5),
            "final_metrics": history[-1],
            "checkpoint": final_info,
            "training_config": asdict(self.cfg),
            "history": history,
        }
        log_path = os.path.join(self.cfg.log_dir, f"{self.cfg.experiment_name}_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

