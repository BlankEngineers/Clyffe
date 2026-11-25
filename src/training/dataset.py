"""
Dataset and batching utilities for ARC-LB-01.
Masks prompt tokens with -100 so next-token cross-entropy trains on assistant outputs.
"""

import json
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset

from src.tokenizer.tokenizer import ARCTokenizer


class ARCDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: ARCTokenizer, max_seq_len: int = 128):
        self.records: List[Dict[str, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        prompt_ids = (
            [self.tokenizer.bos_id, self.tokenizer.user_id]
            + self.tokenizer.encode(rec["input"])
            + [self.tokenizer.sep_id, self.tokenizer.assistant_id]
        )
        ans_ids = self.tokenizer.encode(rec["output"]) + [self.tokenizer.eos_id]
        full = (prompt_ids + ans_ids)[: self.max_seq_len]

        input_ids = full[:-1]
        target_ids = full[1:]

        # Mask prompt tokens in target_ids (everything before first assistant output token)
        prompt_len = max(0, len(prompt_ids) - 1)
        labels = [-100 if i < prompt_len else tid for i, tid in enumerate(target_ids)]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0):
    max_len = max(x.shape[0] for x, _ in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, (inp, lbl) in enumerate(batch):
        L = inp.shape[0]
        input_ids[i, :L] = inp
        labels[i, :L] = lbl
    return input_ids, labels

