# Check 5 — 20-Example Dense Overfitting Test

- **Model:** Dense `ARCLB01Model` (`4L, d=64, h=4, d_ff=192, ctx=96`)
- **Dataset:** 20 fixed examples from `datasets/processed/stage_0/train.jsonl`
- **Optimizer:** `AdamW(lr=5e-3)`

| Step | Train Loss | Teacher-Forced Acc | Grad Norm |
|---:|---:|---:|---:|
| `1` | `6.40635` | `0.0000` | `2.3211` |
| `10` | `4.70560` | `0.1302` | `0.7762` |
| `25` | `2.66304` | `0.4512` | `0.9789` |
| `50` | `0.22174` | `1.0000` | `0.2150` |
| `100` | `0.00912` | `1.0000` | `0.0124` |
| `150` | `0.00498` | `1.0000` | `0.0068` |

**Conclusion:** **PASS.** Model, causal attention, target shifting, `-100` loss masking, and optimizer work correctly.

