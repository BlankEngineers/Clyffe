# Stage 0 Report — Clyffe v0.1.1 (research code ARC-LB-01)

- **Tokenizer:** `checkpoints/clyffe_v0.1.1_tokenizer`, `vocab_size=600`, `fertility=1.5587` tokens/word
- **Architecture:** 4L, `d_model=64`, `n_heads=4`, `d_ff=192`, `ctx=96` (~203K dense-equiv params)
- **Documented Deviation:** FP warm start (10 epochs) -> SVD / Joint-ITQ (`T=50`) conversion -> 40-epoch cross-entropy QAT (no teacher, no distillation). All learned matrices (`q/k/v/o_proj`, `w_in/w_out`, `token_emb`, `lm_head`) are in LittleBit-2 form in B/C1/C2.

## Comparison Table (A / B / C1 / C2)

| Condition | Best Val Loss | Val PPL | In-Scope F1 | Unseen Math EM | OOD Refusal | False Refusal | Actual BPW | Disk Bytes | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_dense_v011` | `0.0829` | `1.09` | `69.2%` | `3.0%` | `55.0%` | `0.0%` | `16.0000` | `979,648` | `24.33 ms` |
| `B_littlebit_svd_v011` | `0.0802` | `1.08` | `70.2%` | `3.0%` | `62.5%` | `0.0%` | `1.1727` | `92,819` | `48.48 ms` |
| `C1_littlebit2_smoothsign_v011` | `0.0782` | `1.08` | `68.4%` | `1.0%` | `52.5%` | `0.0%` | `1.1727` | `94,817` | `51.54 ms` |
| `C2_littlebit2_ste_v011` | `0.0824` | `1.09` | `68.8%` | `2.0%` | `75.0%` | `0.0%` | `1.1727` | `93,499` | `50.06 ms` |
