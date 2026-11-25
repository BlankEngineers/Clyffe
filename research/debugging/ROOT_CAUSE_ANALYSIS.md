# Root Cause Analysis — Clyffe v0.1 Stage 0

## Confirmed Root Causes
1. **Frontend Tokenizer Mismatch (`frontend/server.py`):**
   - **Symptom:** Identical `"ello 6Hey f nqu th:"` gibberish across checkpoints A, B, C-smoothsign, C-ste in the web UI.
   - **Evidence:** `ServerState._tok_for_ckpt("clyffe-v0.1-A-20261125_best.pt")` returned `checkpoints/tokenizer` (`vocab=512`) instead of `checkpoints/clyffe_v0.1_tokenizer` (`vocab=600`).
   - **Fix:** Route `clyffe-v0.1` checkpoints to `checkpoints/clyffe_v0.1_tokenizer` (and store `tokenizer_dir` inside checkpoint metadata so loaders never guess).
2. **Case-Sensitive BPE + Missing Lowercase / Unpunctuated Prompts (`scripts/build_stage0_clyffe_data.py`):**
   - **Symptom:** Even with the right tokenizer, Dense A outputted `'5'` for `'hello'` and `'Myic!'` for `'what is your name'`.
   - **Evidence:** `Hello` was token `[481]`, whereas `hello` was `['h', 'ello']` (`[64, 331]`), which had zero training examples; dominant arithmetic prior (`~640` math rows) caused fallback to digit `'5'`.
   - **Fix:** Include lowercase and unpunctuated variants (`hello`, `hi`, `who are you`, `what is your name`, etc.) in `scripts/build_stage0_clyffe_data.py` (train split only, keeping val/test disjoint) and balance arithmetic count (`320` instead of `800`).

## Problems Not Reproduced (Ruled Out)
- Weight reload failure (Check 2 proved distinct tensor hashes and logits).
- Model/loss/optimizer/masking bug (Check 5 proved 100% memorization on 20 examples in 50 steps).
- Off-by-one character deletion in tokenizer (Check 6 proved exact `hello -> hello` round-trip).

