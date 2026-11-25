# Hardware & Feasibility Report — Clyffe (ARC-LB-01)

**Organization:** Blank Engineers (ARC III Ecosystem)  
**Model Name:** Clyffe (Research Code: `ARC-LB-01`)  
**Inspection Date:** 2026-11-25  

---

## 1. Measured Machine Specifications

- **OS & Kernel:** Linux `cachyos-x8664` (`7.2.9-2-cachyos`, `x86_64`)
- **CPU:** 13th Gen Intel(R) Core(TM) i7-13645HX (14 cores / 20 threads)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (**6,141 MiB VRAM**, Driver `615.78.08`)
- **System RAM:** 15 GiB total (~6.9 GiB available at inspection)
- **Storage:** 465 GiB NVMe (`/dev/nvme0n1p2`, 370 GiB free)
- **Python:** `3.12.15` (`.venv`)
- **PyTorch:** `2.15.0+cpu` (`cuda: False` in current `.venv`; CUDA wheels required before Stage 1+)

## 2. Network & Data Access

- **Hugging Face Hub (`https://huggingface.co`):** Confirmed accessible (`HTTP/2 200`).
- **Kaggle:** No `~/.kaggle/kaggle.json` API token found. Not needed for Stage 0–2 since all required human-written corpora are available on Hugging Face.

## 3. Stage Feasibility & Estimated Runtime

| Stage | Target Params | Architecture | Token Budget | Feasible? | Est. Time (per condition) | Notes |
|---|---:|---|---|---|---:|---|
| **Stage 0** | ~0.2M | d64, 4L, vocab ~600 | Curated Q&A | **Yes (CPU)** | ~3–5 min | Proves pipeline, audit table, packed inference, refusal |
| **Stage 1** | ~2M | d128, 6L, vocab ~8k | ~40M tokens | **Yes (GPU)** | ~1.5–2.5 hrs | Requires CUDA PyTorch on RTX 4050 |
| **Stage 2** | ~10M | d256, 8L, vocab ~16k | ~200M tokens | **Yes (GPU)** | ~8–12 hrs | Requires bf16/fp16 + grad accumulation on 6 GiB VRAM |
| **Stage 3** | ~33M | d512, 8L | ~600M tokens | **Marginal** | ~25–40 hrs | Tight on 6 GiB VRAM; requires grad checkpointing |
| **Stage 4** | ~100M | d768, 12L | ~2B tokens | **No** | N/A | Exceeds 6 GiB VRAM & 15 GiB RAM for AdamW states |
