# Originality & Prior-Art Analysis — ARC-LB-01

**Organization:** ARC III
**Model:** ARC-LB-01
**Date:** 2026-11-25

---

## 1. Prior-Art Landscape

1. **SamsungLabs/LittleBit (NeurIPS 2025) & LittleBit-2 (ICML 2026):**
   - Developed by Banseok Lee, Dongkyu Kim, Youngcheon You, and Youngmin Kim (Samsung Research).
   - **Scope:** Compresses *pretrained* FP16 LLMs (OPT, Llama-2/3, Phi-4, Qwen2.5/3, Gemma-2/3) into 1.0–0.1 BPW using Knowledge Distillation (`KDTrainer` against a dense teacher model), while excluding `lm_head` and token embeddings.
2. **1-Bit / Ternary From-Scratch LMs (e.g., BitNet b1.58):**
   - Train ternary/1-bit weights directly ($\ge 1$ BPW floor) without low-rank binary latent factorization or Joint-ITQ latent geometry alignment.
3. **Post-Training Quantization (GPTQ, AWQ, GGUF):**
   - Compress pretrained weights to INT4/INT8; unrelated to binary latent factorization.

## 2. Specific Technical Contribution of ARC-LB-01

> *"An independently initialized and trained tiny language model developed using LittleBit-2-based binary latent weight representation, without relying on pretrained language-model weights."*

### Inherited from LittleBit / LittleBit-2
- Low-rank SVD factorization with symmetric singular value splitting.
- Internal Latent Rotation and Joint Iterative Quantization (Joint-ITQ).
- Dual-SVID multi-scale compensation and fused latent scaling.
- `SmoothSign` / `STEBinary` QAT surrogates and 32-bit LSB-first binary packing.

### Original ARC III Contributions
1. **Zero-Pretrained-Weights / Zero-Distillation Pipeline:** Trained purely from random initialization with next-token cross-entropy loss (no teacher LLM).
2. **Full-Model LittleBit-2 Audit (`LittleBitEmbedding` + `LittleBitLinear` `lm_head`):** Covers >99% of dense-equivalent parameters rather than leaving embeddings and `lm_head` in FP16.
3. **Comparative Experimental Evidence (Dense vs. SVD-only vs. Joint-ITQ vs. Direct LittleBit Init):** Empirical study of Marchenko-Pastur random-matrix spectra under LittleBit-2 factorization and QAT.
4. **Native Offline Packed Runtime:** Executes packed `int32` binary factors + FP16 scales directly in factorized form.

