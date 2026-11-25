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

## Entry 004 — 2026-11-25 — Stage 0 Debugging, Diagnosis & Dense Control Gate 1 Recovery
- **Branch:** `debug/stage0-pipeline-diagnosis` (all `v0.1` checkpoints preserved and frozen in `research/debugging/BASELINE_SNAPSHOT.md`).
- **Observed Failure:** Frontend output `"ello 6Hey f nqu th:"` across `clyffe-v0.1-A`, `B`, `C-smoothsign`, `C-ste`.
- **Eight Diagnostic Checks (`research/debugging/01..08`):**
  1. Loss audit (`ln(600)=6.3969`): `A=0.2064`, `B=0.2244`, `C-smoothsign=0.2319`, `C-ste=0.2361`.
  2. Weight hashes & reload: All 4 checkpoints had distinct tensor hashes and logits.
  3. Greedy decoding with `clyffe_v0.1_tokenizer`: Outputs differed across checkpoints, but `A` failed on lowercase `'hello' -> '5'` and `'what is your name' -> 'Myic!'`.
  4. Teacher-forced train accuracy: `A=98.84%`, `B=95.69%`, `C-smoothsign=95.69%`, `C-ste=95.19%`.
  5. 20-example dense overfit test: `loss=0.00498`, `100%` teacher-forced acc by step 50.
  6. Tokenizer round-trip: Exact (`hello -> hello`), no off-by-one deletion; case-sensitive (`Hello=[481]`, `hello=['h','ello']`).
  7. Template & frontend audit: `frontend/server.py` routed `clyffe-v0.1-*` to the stale 512-token `checkpoints/tokenizer` instead of `checkpoints/clyffe_v0.1_tokenizer`.
  8. Dataset audit: `train.jsonl` was imbalanced (`~640` math vs `~176` greetings) and lacked lowercase unpunctuated prompts.
- **Fixes Applied:** Fixed `frontend/server.py` tokenizer routing; added lowercase/unpunctuated greeting & identity paraphrases and balanced arithmetic (`320` rows) in `scripts/build_stage0_clyffe_data.py`.
- **Dense A Retraining (`clyffe-v0.1.1-A-20261125_best.pt`):**
  - `best_val_loss=0.0829` (`Val PPL=1.09`), `In-Scope F1=69.2%`, `False Refusal=0.0%`.
  - Greedy outputs: `'hello' -> 'Hello! How can I help you?'`, `'hi' -> 'Hi! How can I help you?'`, `'who are you' -> 'I am Clyffe, an experimental language model developed under ARC III.'`, `'what is your name' -> 'My name is Clyffe.'`.
- **Gate 1 Result:** **PASS.** Ready to retrain `B`, `C-smoothsign`, and `C-ste` on the corrected pipeline upon approval.

## Entry 005 — 2026-11-25 — Stage 0 v0.1.1 LittleBit Retraining (B, C1, C2)
- **Conditions Retrained:** `B_littlebit_svd_v011`, `C1_littlebit2_smoothsign_v011`, `C2_littlebit2_ste_v011` on `clyffe_v0.1.1_tokenizer` (`vocab=600`).
- **Results:**
  - `B`: `best_val_loss=0.0802` (`PPL=1.08`), `In-Scope F1=70.2%`, `sign_flip_rate=0.002057`
  - `C1 (SmoothSign)`: `best_val_loss=0.0782` (`PPL=1.08`), `In-Scope F1=68.4%`, `sign_flip_rate=0.001538`
  - `C2 (STE)`: `best_val_loss=0.0824` (`PPL=1.09`), `In-Scope F1=68.8%`, `OOD Refusal=75.0%`, `sign_flip_rate=0.002345`
- **Greedy Verification:** All three (`B`, `C1`, `C2`) answered `hello`, `hi`, `who are you`, `what is your name`, `What is gravity?`, and OOD refusal accurately.

## Entry 006 — 2026-11-25 — Transition to General LM Training (`Clyffe v0.2`, `2.03M` params)
- **Capacity Upgrade:** Upgraded from Stage 0 (`600` vocab, `4L, d=64, ctx=96, 203K`) to `Clyffe v0.2` (`4,096` vocab, `6L, d=128, h=4, d_ff=384, ctx=256`, **2,033,280** dense-equiv params, **99.92%** LittleBit-2 coverage at `split_dim=32`, **0.5122 BPW**, `216,347` bytes `.pt`).
- **Human-Written General Corpus (`datasets/processed/general_v02/`):**
  - `wikimedia/wikipedia` (`20231101.simple`): `1,098` train / `95` val articles (`1,257` train / `111` val 256-token blocks = `~322K` pretraining tokens).
  - `databricks/databricks-dolly-15k` + `allenai/ai2_arc` + `openai/gsm8k` + deterministic multi-turn/conversions/math/OOS: `2,231` SFT train / `212` val / `312` held-out test.
- **Stage A LittleBit-2 Gradient & Overfit Check:** All parameters (`U, V, u1, u2, v1, v2` across all 6 layers + `token_emb` + `lm_head`) received non-zero gradients; 16-block loss dropped `8.9875 -> 5.8923` in 40 steps.
- **Pretrain + SFT QAT Results:**
  - FP Warm-Start (4 ep): `val_ppl` `538.5 -> 249.1`
  - Pretrain QAT (6 ep, LittleBit-2 Joint-ITQ): `val_ppl` `309.9 -> 225.49` (surpassed FP warm-start)
  - SFT QAT (14 ep): `best_sft_val_loss = 4.85306`, `val_acc = 18.93%`, `heldout_80_avg_f1 = 8.20%`, `avg_latency = 162.93 ms`, `peak_ram = 1.88 MB`.
- **Honest Scientific Assessment:** While LittleBit-2 QAT stably trained a `2.03M` parameter model at `0.5122 BPW` and learned domain syntax (`"10 kilometres is 10000 metres."`, `"110 + 101 = 110."`), `0.32M` pretraining tokens on CPU is ~125× below the `~40M`-token budget needed for general English fluency and instruction following. Scaling to `~40M` tokens requires installing CUDA PyTorch on the RTX 4050 GPU.
