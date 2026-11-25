# Check 4 — Teacher-Forced Accuracy on `datasets/processed/stage_0/train.jsonl` (First 50 Examples)

| Checkpoint | Correct / Total Answer Tokens | Token-Level Accuracy |
|---|---:|---:|
| A (Dense) | `596 / 603` | `98.84%` |
| B (LittleBit-SVD) | `577 / 603` | `95.69%` |
| C-smoothsign | `577 / 603` | `95.69%` |
| C-ste | `574 / 603` | `95.19%` |

**Findings:** All models learned their training examples at >95% teacher-forced accuracy (`98.84%` for Dense A).

