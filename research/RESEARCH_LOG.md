# ARC-LB-01 Research Log

## Entry 001 — 2026-11-25 — Phase 1: Hardware Inspection & Research Foundation
- **Status:** Completed
- Inspected local hardware (`documentation/HARDWARE_REPORT.md`), verified `SamsungLabs/LittleBit` (NeurIPS 2025, arXiv:2506.13771; LittleBit-2, ICML 2026, arXiv:2603.00042) and CC BY-NC 4.0 license (`research/RESEARCH_FOUNDATION.md`).

## Entry 002 — 2026-11-25 — Phase 2 & 3: LittleBit-2 Math Engine & Synthetic Validation
- **Status:** Completed
- Implemented all 10+ modules in `src/littlebit/` from first principles.
- All 9 unit tests in `tests/test_littlebit_math.py` and `tests/test_architecture.py` passed (`pytest`, 0.96s).
- Synthetic matrix validation (`scripts/validate_littlebit_math.py`) confirmed Joint-ITQ reduces reconstruction error on both random Gaussian matrices (0.8603 -> 0.8100 relative error at 0.875 BPW) and structured low-rank matrices (0.7490 -> 0.6065 at 0.6458 BPW).

## Entry 003 — 2026-11-25 — Phase 4–8: Tokenizer, Tiny LM, Dataset, Training & Native Inference
- **Status:** Completed
- Built custom digit-aware BPE tokenizer (`src/tokenizer/tokenizer.py`, vocab=512) and 2-layer decoder-only Transformer (`d_model=96, n_heads=4, d_ff=192`, 246,240 dense-equiv params).
- Extended LittleBit-2 to `LittleBitEmbedding` and `lm_head`, achieving **99.81% parameter coverage** and **0.6872 overall effective BPW** (21,152 bytes raw model bits; 55,713 bytes `.pt` checkpoint vs 994,977 bytes dense).
- Generated 5,500-example dataset (4,400 train / 550 val / 550 test) + 500 disjoint unseen math expressions.
- Trained 4 conditions (35 epochs each):
  - `dense` (`ARC-LB-01-INITIAL`): `best_val_loss=0.04246`, `val_acc=0.9870`
  - `littlebit_svd` (`ARC-LB-01-LITTLEBIT`): `best_val_loss=0.13465`, `val_acc=0.9603`
  - `littlebit_joint_itq` (`ARC-LB-01-FINAL`): `best_val_loss=0.15077`, `val_acc=0.9488`
  - `littlebit_direct` (`ARC-LB-01-DIRECT`, Experiment B): `best_val_loss=0.12905`, `val_acc=0.9521`
- **Key Scientific Finding:** Direct LittleBit-2 initialization (`U, V ~ N(0, 0.02)`) outperformed SVD+Joint-ITQ from random dense weights (`0.12905` vs `0.15077` val loss) because random dense matrices have flat Marchenko-Pastur spectra and SVD factors partially saturate SmoothSign gradients, whereas direct small-scale latent init keeps factors inside the active SmoothSign gradient window.
- **Known Limitations:** While identity, science, and out-of-scope rejection work reliably at 0.6872 BPW, exact multi-digit arithmetic generalization on unseen expressions remains weak at 246K sub-1-bit parameters.
