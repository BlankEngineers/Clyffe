# Stage 0 Report — Clyffe v0.1 (research code ARC-LB-01)

- **Tokenizer:** vocab_size=600, fertility=1.5587 tokens/word
- **Architecture:** 4L, d_model=64, n_heads=4, d_ff=192, ctx=96 (~203K dense-equiv params)
- **Documented Deviation:** FP warm start (10 epochs) -> SVD / Joint-ITQ conversion -> QAT (cross-entropy only, no teacher, no distillation).

## Comparison Table (A / B / C1 / C2)

| Condition | Best Val Loss | Val PPL | In-Scope F1 | Unseen Math EM | OOD Refusal | False Refusal | Actual BPW | Disk Bytes | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_dense_v011 | 0.0829 | 1.09 | 69.2% | 3.0% | 55.0% | 0.0% | 16.0000 | 979648 | 24.33 ms |
