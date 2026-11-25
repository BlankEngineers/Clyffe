# ARC-LB-01 — Native LittleBit-2 Tiny Language Model

**Organization:** ARC III
**Research Title:** Native LittleBit-Based Tiny Language Model

ARC-LB-01 is an independently initialized, independently trained tiny decoder-only language model (~240K dense-equivalent parameters) built from scratch using **LittleBit-2** binary latent factorization, Internal Latent Rotation, Joint Iterative Quantization (Joint-ITQ), Dual-SVID multi-scale compensation, and Quantization-Aware Training (QAT) — **without** pretrained LLM weights or knowledge distillation.

## Project Structure
- `research/` — Research foundation, mathematical method, originality analysis, research log
- `src/littlebit/` — Independent LittleBit-2 computing engine (`factorization.py`, `latent_rotation.py`, `joint_itq.py`, `binary_factors.py`, `scaling.py`, `compensation.py`, `qat.py`, `native_operations.py`, `representation.py`, `reconstruction.py`)
- `src/tokenizer/` — Original digit-aware BPE tokenizer
- `src/architecture/` — Tiny decoder-only Transformer (`model.py`, `embeddings.py`, `attention.py`, `feedforward.py`, `initialization.py`)
- `src/training/` — Reproducible training & QAT pipeline
- `src/inference/` — Native offline packed inference runtime
- `evaluation/` — Automated capability, boundary, compression, and latency evaluation
- `scripts/` — Dataset builder and synthetic math validation

## Reproducibility
```bash
# 1. Generate dataset (~5,500 examples + unseen math test split)
.venv/bin/python scripts/build_dataset.py

# 2. Run unit tests & synthetic LittleBit-2 math validation
.venv/bin/pytest -v
.venv/bin/python scripts/validate_littlebit_math.py

# 3. Train experiments
.venv/bin/python -m src.training.train --mode dense --release_tag ARC-LB-01-INITIAL
.venv/bin/python -m src.training.train --mode littlebit_svd --release_tag ARC-LB-01-LITTLEBIT
.venv/bin/python -m src.training.train --mode littlebit_joint_itq --release_tag ARC-LB-01-FINAL
.venv/bin/python -m src.training.train --mode littlebit_direct --release_tag ARC-LB-01-DIRECT

# 4. Run offline inference & full evaluation
.venv/bin/python run_model.py
.venv/bin/python -m evaluation.final_report checkpoints/ARC-LB-01-FINAL_best.pt
```

## Attribution
LittleBit and LittleBit-2 methods are attributed to Banseok Lee, Dongkyu Kim, Youngcheon You, and Youngmin Kim (Samsung Research, NeurIPS 2025 & ICML 2026, CC BY-NC 4.0).

