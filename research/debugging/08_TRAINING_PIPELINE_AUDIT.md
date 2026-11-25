# Check 8 — Training Pipeline & Best Checkpoint Audit

- **Epochs Configured & Executed:** 50 epochs (`35` batch steps/epoch = `1,750` optimizer steps for Dense A).
- **Dataset Consumed:** `1,114` train / `137` val (`0` overlap).
- **Best Checkpoint Logic:** Saves `<release_id>_best.pt` whenever `val_loss < best_val_loss`.
- **Data Imbalance Found:** `train.jsonl` contained `~640` arithmetic rows (`1..50` `+`/`-`) vs `~176` greeting rows and `~160` identity rows, and lacked lowercase + unpunctuated forms (`hello`, `hi`, `who are you`, `what is your name`).

