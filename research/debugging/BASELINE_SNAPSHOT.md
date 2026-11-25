# Baseline Snapshot — Clyffe v0.1 Stage 0 (Frozen)

**Snapshot Date:** 2026-11-25  
**Git Commit Before Debug Branch:** `8ddb346`  
**Tokenizer:** `checkpoints/clyffe_v0.1_tokenizer/` (`vocab_size=600`)  
**Dataset:** `datasets/processed/stage_0/` (`train=1114`, `val=137`, `test=137`, `unseen_math=242`, `unseen_ood=40`)

| Condition | Best Checkpoint File | Size (Bytes) | SHA-256 | Training Log |
|---|---|---:|---|---|
| A (Dense) | `checkpoints/clyffe-v0.1-A-20261125_best.pt` | `979,566` | `3fac932b1862098eeda541eda031ce23a624976e550191791a60cc80c007f7a1` | `experiments/stage_0/A_dense/metrics.jsonl` |
| B (LittleBit-SVD) | `checkpoints/clyffe-v0.1-B-20261125_best.pt` | `92,479` | `27cdc313b698cf3f75a865bc63d5b74bc1e5d6efdd14b4546404f594f5e24630` | `experiments/stage_0/B_littlebit_svd/metrics.jsonl` |
| C-smoothsign | `checkpoints/clyffe-v0.1-C-smoothsign-20261125_best.pt` | `94,413` | `db5f181d5362be9017f579c29f2efc03242c83adeb6667bb1179462e794127fd` | `experiments/stage_0/C1_littlebit2_smoothsign/metrics.jsonl` |
| C-ste | `checkpoints/clyffe-v0.1-C-ste-20261125_best.pt` | `93,159` | `0ca2c224cf8bcc55f801a05334a3b3f294701836beba21074bd746c37ef08c57` | `experiments/stage_0/C2_littlebit2_ste/metrics.jsonl` |

