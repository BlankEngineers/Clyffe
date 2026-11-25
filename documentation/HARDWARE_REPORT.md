# Hardware & Environment Report — ARC-LB-01

**Project:** ARC III — ARC-LB-01 (Native LittleBit-Based Tiny Language Model)
**Inspection Date:** 2026-11-25
**Workspace:** `/home/dhanush/Documents/little bit`

---

## 1. Operating System & Kernel

- **Distribution:** CachyOS (Rolling Release, `x86_64`)
- **Kernel:** `Linux cachyos-x8664 7.2.9-2-cachyos #1 SMP PREEMPT_DYNAMIC`

## 2. CPU & System Memory

- **CPU Model:** 13th Gen Intel(R) Core(TM) i7-13645HX
- **Topology:** 14 cores / 20 logical CPUs (threads)
- **System RAM:** 15 GiB total (~4.5 GiB available at inspection time, ~10 GiB in use)
- **Swap:** 15 GiB total (~10 GiB free)

## 3. Storage

- **Filesystem:** `/dev/nvme0n1p2`
- **Capacity:** 465 GiB total, 372 GiB available (16% used)

## 4. GPU & CUDA

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU
- **VRAM:** 6,141 MiB (~6.0 GiB)
- **Driver Version:** `615.78.08`
- **CUDA UMD Version:** `13.4`
- **CUDA Availability:** Hardware and driver present; PyTorch with CUDA support needs to be installed in a dedicated virtual environment.

## 5. Python Environment

- **System Python:** `/usr/bin/python3` -> `Python 3.14.7` (no `torch` installed)
- **Package/Python Manager:** `uv` (`/home/dhanush/.local/bin/uv`)
- **Available `uv` Managed Pythons:**
  - `cpython-3.12.15-linux-x86_64-gnu` (matches LittleBit's recommended Python 3.12)
  - `cpython-3.11.17-linux-x86_64-gnu`
- **Git:** `/usr/bin/git`

## 6. Implications for ARC-LB-01

1. **Compute Budget:** The RTX 4050 Laptop GPU (6 GiB VRAM) is more than sufficient for training and validating a 100K–300K parameter tiny language model from scratch, including SVD/Joint-ITQ initialization and QAT.
2. **Environment Setup (Phase 2):** We should create an isolated `uv` virtual environment using Python 3.12 (`uv venv --python 3.12`) and install PyTorch with CUDA support, `numpy`, `pytest`, and minimal scientific dependencies—without pulling in large pretrained-LLM ecosystems or external AI SDKs.
3. **System RAM Awareness:** With ~4.5 GiB of free RAM at inspection time, we must keep dataset loading and batch sizes lightweight and avoid unnecessary background processes.

