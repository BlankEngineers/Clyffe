# Stage 0 Report — Clyffe v0.1 (research code ARC-LB-01)

- **Tokenizer:** vocab_size=600, fertility=1.5949 tokens/word
- **Architecture:** 4L, d_model=64, n_heads=4, d_ff=192, ctx=96 (~203K dense-equiv params)
- **Documented Deviation:** FP warm start (10 epochs) -> SVD / Joint-ITQ conversion -> QAT (cross-entropy only, no teacher, no distillation).

## Comparison Table (A / B / C1 / C2)

| Condition | Best Val Loss | Val PPL | In-Scope F1 | Unseen Math EM | OOD Refusal | False Refusal | Actual BPW | Disk Bytes | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_dense | 0.2064 | 1.23 | 54.5% | 0.0% | 67.5% | 0.0% | 16.0000 | 979566 | 9.52 ms |
| B_littlebit_svd | 0.2244 | 1.25 | 47.6% | 3.0% | 47.5% | 0.0% | 1.1727 | 92479 | 43.42 ms |
| C1_littlebit2_smoothsign | 0.2319 | 1.26 | 50.0% | 1.0% | 95.0% | 0.0% | 1.1727 | 94413 | 42.07 ms |
| C2_littlebit2_ste | 0.2361 | 1.27 | 49.2% | 1.0% | 77.5% | 0.0% | 1.1727 | 93159 | 41.75 ms |
