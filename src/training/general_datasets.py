"""
Datasets for General Autoregressive Language Modeling & Multi-Turn SFT.
"""

import json
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

from src.tokenizer.tokenizer import ARCTokenizer


class CausalTextDataset(Dataset):
    """Packs natural-language documents into fixed-length causal LM blocks."""

    def __init__(self, jsonl_path: str, tokenizer: ARCTokenizer, block_size: int = 256):
        token_stream: List[int] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ids = [tokenizer.bos_id] + tokenizer.encode(rec["text"]) + [tokenizer.eos_id]
                token_stream.extend(ids)

        self.blocks: List[torch.Tensor] = []
        for i in range(0, len(token_stream) - block_size - 1, block_size):
            chunk = token_stream[i : i + block_size + 1]
            self.blocks.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        blk = self.blocks[idx]
        return blk[:-1], blk[1:]


class MultiTurnSFTDataset(Dataset):
    """
    Encodes single-turn and multi-turn conversations:
        [BOS] ([USER] msg [SEP] [ASSISTANT] reply [EOS])+
    Masks user/prompt tokens with -100; supervises all assistant reply + EOS tokens.
    """

    def __init__(self, jsonl_path: str, tokenizer: ARCTokenizer, max_seq_len: int = 256):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                full_ids: List[int] = [tokenizer.bos_id]
                loss_mask: List[bool] = [False]

                for user_msg, asst_msg in rec["turns"]:
                    u_ids = [tokenizer.user_id] + tokenizer.encode(user_msg) + [tokenizer.sep_id, tokenizer.assistant_id]
                    a_ids = tokenizer.encode(asst_msg) + [tokenizer.eos_id]
                    full_ids.extend(u_ids)
                    loss_mask.extend([False] * len(u_ids))
                    full_ids.extend(a_ids)
                    loss_mask.extend([True] * len(a_ids))

                full_ids = full_ids[:max_seq_len]
                loss_mask = loss_mask[:max_seq_len]
                if len(full_ids) < 2:
                    continue
                inp = full_ids[:-1]
                tgt = full_ids[1:]
                m = loss_mask[1:]
                labels = [tid if keep else -100 for tid, keep in zip(tgt, m)]
                if any(l != -100 for l in labels):
                    self.samples.append(
                        (torch.tensor(inp, dtype=torch.long), torch.tensor(labels, dtype=torch.long))
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

