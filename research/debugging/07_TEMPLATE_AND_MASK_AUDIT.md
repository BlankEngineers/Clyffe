# Check 7 — Training & Inference Template and Mask Audit

- **Training Sequence (`src/training/dataset.py`):**
  `[BOS] [USER] <encoded input> [SEP] [ASSISTANT] <encoded output> [EOS]`
  Labels shift by 1 (`full[1:]`), with `len(prompt_ids) - 1` prompt tokens masked to `-100`.
- **Inference Sequence (`src/inference/generate.py`):**
  `[BOS] [USER] <encoded input> [SEP] [ASSISTANT]` -> byte-identical prefix.
- **Confirmed Mismatch in Web Frontend (`frontend/server.py`):**
  `ServerState._tok_for_ckpt()` only checked `"V2"` and `"V3"` in checkpoint names, so `clyffe-v0.1-*` checkpoints loaded `checkpoints/tokenizer` (v0.1 Tesla, `vocab=512`) instead of `checkpoints/clyffe_v0.1_tokenizer` (`vocab=600`).

