"""
Original Digit-Aware BPE Tokenizer for ARC-LB-01.

Built strictly from the project's own training corpus with no pretrained
vocabularies or embeddings. Digits are always kept as single tokens so
unseen arithmetic expressions generalize without OOV multi-digit merges.
"""

from collections import Counter
import json
import os
import re
from typing import Dict, List, Optional, Tuple


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<assistant>", "<sep>"]

# Match single digits, alphabetic words (with optional leading space marker), or single punctuation
PRETOKEN_PATTERN = re.compile(r"\d|[A-Za-z]+|[^\sA-Za-z\d]")


class ARCTokenizer:
    """Custom digit-aware BPE tokenizer for ARC-LB-01."""

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
    ):
        self.special_tokens = list(SPECIAL_TOKENS)
        if vocab is None:
            self.vocab: Dict[str, int] = {tok: i for i, tok in enumerate(self.special_tokens)}
        else:
            self.vocab = dict(vocab)
        self.id_to_token: Dict[int, str] = {i: tok for tok, i in self.vocab.items()}
        self.merges: List[Tuple[str, str]] = [tuple(m) for m in (merges or [])]
        self.merge_ranks: Dict[Tuple[str, str], int] = {
            m: i for i, m in enumerate(self.merges)
        }

        self.pad_id = self.vocab["<pad>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]
        self.unk_id = self.vocab["<unk>"]
        self.user_id = self.vocab["<user>"]
        self.assistant_id = self.vocab["<assistant>"]
        self.sep_id = self.vocab["<sep>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _pretokenize(text: str) -> List[str]:
        """Split text into word/digit/punctuation chunks, preserving word-start spaces."""
        tokens: List[str] = []
        # Split on whitespace while remembering if a token had a leading space
        parts = text.strip().split()
        for idx, part in enumerate(parts):
            sub_parts = PRETOKEN_PATTERN.findall(part)
            for s_idx, sp in enumerate(sub_parts):
                if idx > 0 and s_idx == 0:
                    tokens.append("▁" + sp)
                else:
                    tokens.append(sp)
        return tokens

    @classmethod
    def train_from_texts(
        cls,
        texts: List[str],
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "ARCTokenizer":
        """Train BPE vocabulary from scratch on project texts."""
        vocab: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}

        # Collect base characters & pretokens
        word_freqs: Counter = Counter()
        chars = set()
        for text in texts:
            for pt in cls._pretokenize(text):
                # Keep digits and single-char punctuation atomic (no BPE merges across digits)
                if pt.isdigit() or (pt.startswith("▁") and pt[1:].isdigit()):
                    if pt not in vocab:
                        vocab[pt] = len(vocab)
                    continue
                word_freqs[tuple(pt)] += 1
                for ch in pt:
                    chars.add(ch)

        for ch in sorted(chars):
            if ch not in vocab:
                vocab[ch] = len(vocab)

        # Ensure all ASCII digits and common math symbols are present
        for sym in list("0123456789+-*/=×÷.,?!:'\"()[]"):
            if sym not in vocab:
                vocab[sym] = len(vocab)
            if ("▁" + sym) not in vocab and sym.isdigit():
                vocab["▁" + sym] = len(vocab)

        merges: List[Tuple[str, str]] = []
        splits = {word: list(word) for word in word_freqs}

        while len(vocab) < vocab_size:
            pair_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                syms = splits[word]
                for i in range(len(syms) - 1):
                    pair = (syms[i], syms[i + 1])
                    # Never merge digits
                    if any(c.isdigit() for c in pair[0] + pair[1]):
                        continue
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            best_pair, best_freq = pair_freqs.most_common(1)[0]
            if best_freq < min_frequency:
                break

            merged = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            if merged not in vocab:
                vocab[merged] = len(vocab)

            # Apply merge to all words
            new_splits = {}
            for word, syms in splits.items():
                new_syms: List[str] = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == best_pair[0] and syms[i + 1] == best_pair[1]:
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                new_splits[word] = new_syms
            splits = new_splits

        return cls(vocab=vocab, merges=merges)

    def _apply_bpe(self, token: str) -> List[str]:
        if token in self.vocab:
            return [token]
        if token.isdigit() or (token.startswith("▁") and token[1:].isdigit()):
            return [token] if token in self.vocab else list(token)

        syms = list(token)
        while len(syms) > 1:
            pairs = [(syms[i], syms[i + 1]) for i in range(len(syms) - 1)]
            ranked = [(self.merge_ranks[p], i, p) for i, p in enumerate(pairs) if p in self.merge_ranks]
            if not ranked:
                break
            _, idx, pair = min(ranked)
            merged = pair[0] + pair[1]
            syms = syms[:idx] + [merged] + syms[idx + 2 :]
        return syms

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = [self.bos_id] if add_bos else []
        for pt in self._pretokenize(text):
            for piece in self._apply_bpe(pt):
                ids.append(self.vocab.get(piece, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_qa(self, user_text: str, assistant_text: Optional[str] = None) -> List[int]:
        """Format a prompt-response pair with explicit role tokens."""
        ids = [self.bos_id, self.user_id] + self.encode(user_text) + [self.sep_id, self.assistant_id]
        if assistant_text is not None:
            ids += self.encode(assistant_text) + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        pieces: List[str] = []
        special_set = set(self.special_tokens)
        for idx in ids:
            tok = self.id_to_token.get(int(idx), "<unk>")
            if skip_special_tokens and tok in special_set:
                continue
            pieces.append(tok)
        text = "".join(pieces).replace("▁", " ").strip()
        # Clean spacing before punctuation
        text = re.sub(r"\s+([.,?!:])", r"\1", text)
        return text

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
        with open(os.path.join(directory, "merges.json"), "w", encoding="utf-8") as f:
            json.dump(self.merges, f, indent=2, ensure_ascii=False)
        with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer_type": "ARC-DigitAware-BPE",
                    "vocab_size": self.vocab_size,
                    "special_tokens": self.special_tokens,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str) -> "ARCTokenizer":
        with open(os.path.join(directory, "vocab.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(os.path.join(directory, "merges.json"), "r", encoding="utf-8") as f:
            merges = json.load(f)
        return cls(vocab=vocab, merges=merges)

