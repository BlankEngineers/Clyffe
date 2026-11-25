# Clyffe v0.1.1 (research code ARC-LB-01) — Native LittleBit-2 Tiny Language Model

**Organization:** Blank Engineers (ARC III Ecosystem)  
**Model Name:** Clyffe `v0.1.1` (Research Code: `ARC-LB-01`)  
**Research Title:** Native LittleBit-Based Tiny Language Model  

**Project Leads:**  
This project is done by **Vishnupriyan Sivanesan**, Head of Blank Engineers, and **Chittari Hari Dhanush**, Vice Head of Blank Engineers.  
*This project is part of the **ARC III** ecosystem and is the property of **Blank Engineers**.*

---

**Clyffe** is an independently initialized, independently trained decoder-only language model built from random initialization using **LittleBit-2** binary latent factorization, Internal Latent Rotation, Joint Iterative Quantization (Joint-ITQ), Dual-SVID multi-scale compensation, and Quantization-Aware Training (QAT) — **without** pretrained LLM weights or knowledge distillation.

- **Stage 0 (`Clyffe v0.1.1`):** 4L, `d_model=64`, `n_heads=4`, `d_ff=192`, `ctx=96`, `vocab=600` (~203K dense-equiv params). All learned matrices (including `token_emb` and `lm_head`) are represented in LittleBit-2 form at `1.1727` actual BPW (`94.8 KB` packed checkpoint vs `979.6 KB` dense, `val_loss=0.0782`, `PPL=1.08`).

## Project Structure
- `research/` — Research foundation, mathematical method, originality analysis, research log, and `debugging/` audit reports
- `src/littlebit/` — Independent LittleBit-2 computing engine
- `src/tokenizer/` — Original digit-aware BPE tokenizer
- `src/architecture/` — Decoder-only Transformer (`model.py`, `embeddings.py`, `attention.py`, `feedforward.py`, `initialization.py`)
- `src/training/` — Staged training & QAT pipeline (`stage0_clyffe.py`, `trainer.py`, etc.)
- `src/inference/` — Native offline packed inference runtime
- `frontend/` — Offline local web interface (`index.html`, `server.py`)
- `evaluation/` — Automated capability, boundary, compression, and Stage 0 gate reports (`STAGE_0_REPORT.md`)
- `scripts/` — Dataset builders (`build_stage0_clyffe_data.py`) and synthetic math validation

## Reproducibility (Stage 0 — Clyffe v0.1.1)
```bash
# 1. Build Stage 0 curated dataset (disjoint train/val/test + unseen math + unseen OOD)
PYTHONPATH=. .venv/bin/python scripts/build_stage0_clyffe_data.py

# 2. Run unit tests
PYTHONPATH=. .venv/bin/pytest -v

# 3. Train Clyffe v0.1.1 (Dense A, LittleBit B, LittleBit-2 C1/C2)
PYTHONPATH=. .venv/bin/python -m src.training.stage0_clyffe --version-tag v0.1.1

# 4. Run Clyffe offline CLI or web frontend
PYTHONPATH=. .venv/bin/python run_clyffe.py
PYTHONPATH=. .venv/bin/python frontend/server.py
```

## Attribution & License
- **License:** All Rights Reserved — Blank Engineers (ARC III Ecosystem). Viewing and showcase only; copying, modification, or redistribution is strictly prohibited. See [LICENSE](LICENSE).
- **Method Attribution:** LittleBit and LittleBit-2 methods are attributed to Banseok Lee, Dongkyu Kim, Youngcheon You, and Youngmin Kim (Samsung Research, NeurIPS 2025 & ICML 2026, CC BY-NC 4.0).
