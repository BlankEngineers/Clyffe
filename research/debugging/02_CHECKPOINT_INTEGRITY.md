# Check 2 — Checkpoint Weight Hash & Reload Integrity

- **A vs Fresh Init (`seed=42`):** `max_abs_diff = 0.926384`, `mean_abs_diff = 0.099965` (weights genuinely updated).
- **First-Tensor Hash & Fixed-Input (`"hello"`) Top-5 Logit Indices on Reload:**
  - `A`: `35` tensors, `token_emb.weight` hash `38891d6c42a416cc`, top-5 logits `[11, 218, 7, 13, 15]`
  - `B`: `164` tensors, `token_emb.U_packed` hash `536e9a049313cf1b`, top-5 logits `[11, 13, 10, 420, 257]`
  - `C-smoothsign`: `164` tensors, `token_emb.U_packed` hash `2a20c23e82691fe0`, top-5 logits `[481, 257, 569, 218, 187]`
  - `C-ste`: `164` tensors, `token_emb.U_packed` hash `86f346bea67d51ca`, top-5 logits `[420, 481, 26, 180, 417]`

**Findings:** All checkpoints have distinct tensor weights and distinct logits, and `load_checkpoint` cleanly replaces parameters on reload.

