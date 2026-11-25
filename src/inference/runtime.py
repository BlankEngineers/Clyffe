"""
Offline Native Inference Runtime for Clyffe (ARC-LB-01).
Supports multi-turn conversation history, session-level name memory,
verified metadata reporting, and arbitrary user prompts.
"""

import re
from typing import Dict, List, Optional, Tuple

from src.inference.generate import generate_response
from src.inference.loader import load_runtime_model
from src.inference.metadata import build_model_metadata


class ARCLB01Runtime:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_dir: str = "checkpoints/tokenizer",
        device: str = "cpu",
    ):
        self.model, self.tokenizer, self.ckpt_meta = load_runtime_model(
            checkpoint_path, tokenizer_dir=tokenizer_dir, device=device
        )
        self.metadata = build_model_metadata(self.model, checkpoint_path)
        self.session_memory: Dict[str, Optional[str]] = {"user_name": None}
        self.history: List[Tuple[str, str]] = []

    def reset_session(self) -> None:
        self.session_memory = {"user_name": None}
        self.history = []

    def respond(
        self,
        user_input: str,
        max_new_tokens: int = 64,
        use_history: bool = False,
    ) -> str:
        text = user_input.strip()
        if not text:
            return "Please enter a question or message."

        # Session-level name tracking (explicitly separate from learned weights)
        m = re.match(r"^(?:my name is|i am)\s+([A-Za-z]+)[.!?\s]*$", text, re.IGNORECASE)
        if m:
            self.session_memory["user_name"] = m.group(1)

        if re.match(r"^what is my name\??$", text, re.IGNORECASE) and not use_history:
            if self.session_memory["user_name"]:
                return f"Your name is {self.session_memory['user_name']}."
            return "You haven't told me your name in this session yet."

        # Verified metadata reporting
        if re.search(
            r"\b(model specifications|your specs|parameter count|vocabulary size|context length|checkpoint version)\b",
            text,
            re.IGNORECASE,
        ):
            md = self.metadata
            return (
                f"Name: {md['name']} | Org: {md['organization']} | Version: {md['version']} | "
                f"Params (dense equiv): {md['parameter_count_dense_equiv']} | "
                f"Representation: {md['weight_representation']} | "
                f"Vocab: {md['vocab_size']} | Context: {md['context_length']} | "
                f"Checkpoint: {md['checkpoint_identifier']} ({md['actual_storage_bytes']} bytes)"
            )

        hist = self.history[-4:] if use_history else None
        reply = generate_response(
            self.model,
            self.tokenizer,
            text,
            max_new_tokens=max_new_tokens,
            history=hist,
        )
        if use_history:
            self.history.append((text, reply))
        return reply
