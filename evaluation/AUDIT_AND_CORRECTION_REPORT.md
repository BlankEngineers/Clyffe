# ARC-LB-01 / Tesla 1.1 — Model Audit & Correction Report

**Organization:** Blank Engineers (ARC III Ecosystem)  
**Audit Date:** 2026-11-25  
**Preserved Baseline Checkpoint (v0.1.0):** `checkpoints/ARC-LB-01-FINAL_best.pt` (`sha256: 3994a6df...`)  
**Corrected Checkpoint (v0.2.0):** `checkpoints/ARC-LB-01-V2_best.pt` (`sha256: d983f61c72696c45003e50b1d72e466e40e9f3d9dd25ee5f112ef98801f27cfe`)

---

## 1. Root Causes Identified in v0.1.0 Audit

1. **Train/Validation/Test Data Leakage & Low Diversity:**
   - `v0.1.0` dataset had `4,400` training rows with only `1,239` unique pairs; `227/550` validation rows and `242/550` test rows overlapped with training.
   - Repeated boilerplate caused phrase leakage (e.g., `"Who built you?"` drifting into `"have sufficient trained knowledge..."` from OOS templates).
2. **Rank-24 Bottleneck on `token_emb` & `lm_head`:**
   - A `512 × 96` vocabulary matrix factorized at `split_dim=24` has at most 24 orthogonal directions, limiting token separation and causing early truncation / subword drift (`"Heree you later!"`).
3. **`int32` Packing Padding Overhead (`split_dim=24` vs `32`):**
   - `pack_binary` packs each row into 32-bit `int32` words. `split_dim=24` was padded to 32 bits per row in `U_packed`, paying 32-bit storage while only using rank 24.
4. **Evaluator Tokenizer Mismatch (`evaluation/inference.py`):**
   - `evaluate_inference` previously defaulted to `checkpoints/tokenizer` (v1) instead of `checkpoints/tokenizer_v2` for `V2` checkpoints. Fixed.

---

## 2. Corrections Implemented

1. **Strictly Disjoint Dataset (`scripts/build_dataset.py`):**
   - Global deduplication across all categories before splitting: `1,831` train / `227` val / `227` test (`0` overlap across splits) + `500` disjoint unseen math expressions.
   - Added complete coverage for all 11 greetings, all 6 identity queries, and diverse OOS templates.
2. **LittleBit-2 Training Upgrade (`ARC-LB-01-V2`, `v0.2.0`):**
   - Preserved all `v0.1.0` checkpoints untouched.
   - Retrained with `LittleBit-2` (`use_itq=True`, `split_dim=32`, `smooth_alpha=15.0`, `60` epochs, `lr=6e-3`), achieving `best_val_loss=0.25793`, `val_token_acc=91.67%` on strictly disjoint validation data.

---

## 3. Model Specification & Storage Audit (Section 8)

| Metric | v0.1.0 (`split_dim=24`) | v0.2.0 (`split_dim=32`) | Explanation |
|---|---:|---:|---|
| **Dense-Equivalent Params** | `246,240` | `246,240` | Total scalar weights in an equivalent dense 2L Transformer (`d=96, h=4, d_ff=192, V=512`). |
| **LittleBit-Covered Params** | `245,760` (`99.81%`) | `245,760` (`99.81%`) | All attention, FFN, `token_emb`, and `lm_head` matrices. Remaining `480` params are 1D `RMSNorm` scales. |
| **Theoretical Raw Bits** | `21,152 B` (`0.6872 BPW`) | `25,280 B` (`0.8213 BPW`) | Unpadded binary factor bits `s(a+b)` + FP16 scale vectors `16(a+b+s)` + FP16 RMSNorms. |
| **Actual Tensor Payload** | `24,532 B` | `26,292 B` | Sum of `numel * element_size` in `state_dict` (includes `int32` 32-bit row-padding & metadata buffers). |
| **Disk `.pt` File Size** | `55,713 B` | `58,247 B` | Actual PyTorch ZIP/pickle file on disk (includes archive headers & config dict). |

---

## 4. Before vs. After Evaluation Results

### Category & Boundary Accuracy (Disjoint Eval)
- **Greetings:** `24.00%` exact / `48.00%` sem (v0.1.0) -> **`93.75%` exact / `97.92%` sem** (v0.2.0)
- **Identity:** `54.00%` / `81.00%` (v0.1.0) -> **`57.14%` exact / `96.43%` sem** (v0.2.0)
- **Conversation:** `63.00%` / `77.00%` (v0.1.0) -> **`94.00%` exact / `96.00%` sem** (v0.2.0)
- **Science:** `38.00%` / `79.00%` (v0.1.0) -> **`75.00%` exact / `92.86%` sem** (v0.2.0)
- **General Knowledge:** `64.00%` / `78.00%` (v0.1.0) -> **`67.86%` exact / `78.57%` sem** (v0.2.0)
- **Out-of-Scope Rejection:** `69.00%` (v0.1.0) -> **`97.00%`** (v0.2.0)
- **Unseen Mathematics (500):** `1.80%` (v0.1.0) -> **`2.00%`** (v0.2.0) — *`37+48->18`, `91-36->3`, `17×9->120`, `225÷15->10` (0/4)*.

### Known Remaining Limitations
1. **Unseen Multi-Digit Arithmetic:** A 2-layer, `d=96` sub-1-bit model (`0.8213 BPW`) without chain-of-thought or larger capacity cannot reliably compute unseen multi-digit arithmetic (`2.00%` on 500 unseen problems).
2. **Minor Greeting/Name Edge Cases:** `'Hey'` mapped to `'Good evening! How can I help you?'` and `'What is your name?'` truncated at `'My name is Tesla 1.?'`.

