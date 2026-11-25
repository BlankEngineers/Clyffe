# Check 1 — Training and Validation Loss Audit

`vocab_size = 600`, `ln(600) = 6.3969`.

| Checkpoint | Final train loss | Best validation loss | Best epoch | ln(vocab_size) |
|---|---:|---:|---:|---:|
| A | `0.00283` | `0.20644` | `28 / 50` | `6.3969` |
| B | `0.18566` | `0.22440` | `47 / 50` | `6.3969` |
| C-smoothsign | `0.17993` | `0.23193` | `48 / 50` | `6.3969` |
| C-ste | `0.19633` | `0.23611` | `46 / 50` | `6.3969` |

**Findings:** All four models converged far below `ln(vocab_size) = 6.3969`. Gradients, optimizer updates, and loss computation functioned properly during training.

