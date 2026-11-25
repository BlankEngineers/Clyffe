# Tesla — Native LittleBit-2 Tiny Language Model

**Organization:** Blank Engineers (ARC III Ecosystem)  
**Model Name:** Tesla (Experimental Identifier: `ARC-LB-01`)  
**Research Title:** Native LittleBit-Based Tiny Language Model  

**Project Leads:**  
This project is done by **Vishnupriyan Sivanesan**, Head of Blank Engineers, and **Chittari Hari Dhanush**, Vice Head of Blank Engineers.  
*This project is part of the **ARC III** ecosystem and is the property of **Blank Engineers**.*

---

Tesla is an independently initialized, independently trained tiny decoder-only language model (~246K dense-equivalent parameters) built from scratch using **LittleBit-2** binary latent factorization, Internal Latent Rotation, Joint Iterative Quantization (Joint-ITQ), Dual-SVID multi-scale compensation, and Quantization-Aware Training (QAT) — **without** pretrained LLM weights or knowledge distillation.

## Project Structure
- `research/` — Research foundation, mathematical method, originality analysis, research log
- `src/littlebit/` — Independent LittleBit-2 computing engine (`factorization.py`, `latent_rotation.py`, `joint_itq.py`, `binary_factors.py`, `scaling.py`, `compensation.py`, `qat.py`, `native_operations.py`, `representation.py`, `reconstruction.py`)
- `src/tokenizer/` — Original digit-aware BPE tokenizer
- `src/architecture/` — Tiny decoder-only Transformer (`model.py`, `embeddings.py`, `attention.py`, `feedforward.py`, `initialization.py`)
- `src/training/` — Reproducible training & QAT pipeline
- `src/inference/` — Native offline packed inference runtime
- `frontend/` — Offline local web interface (`index.html`, `server.py`)
- `evaluation/` — Automated capability, boundary, compression, and latency evaluation
- `scripts/` — Dataset builder and synthetic math validation

## Reproducibility
```bash
# 1. Generate dataset (~5,500 examples + unseen math test split)
.venv/bin/python scripts/build_dataset.py

# 2. Run unit tests & synthetic LittleBit-2 math validation
PYTHONPATH=. .venv/bin/pytest -v
PYTHONPATH=. .venv/bin/python scripts/validate_littlebit_math.py

# 3. Train experiments
PYTHONPATH=. .venv/bin/python -m src.training.train --mode dense --release_tag ARC-LB-01-INITIAL
PYTHONPATH=. .venv/bin/python -m src.training.train --mode littlebit_svd --release_tag ARC-LB-01-LITTLEBIT
PYTHONPATH=. .venv/bin/python -m src.training.train --mode littlebit_joint_itq --release_tag ARC-LB-01-FINAL
PYTHONPATH=. .venv/bin/python -m src.training.train --mode littlebit_direct --release_tag ARC-LB-01-DIRECT

# 4. Run offline inference, frontend & full evaluation
PYTHONPATH=. .venv/bin/python run_model.py
PYTHONPATH=. .venv/bin/python frontend/server.py
PYTHONPATH=. .venv/bin/python -m evaluation.final_report checkpoints/ARC-LB-01-FINAL_best.pt
```

## Attribution & License
- **License:** All Rights Reserved — Blank Engineers (ARC III Ecosystem). Viewing and showcase only; copying, modification, or redistribution is strictly prohibited. See [LICENSE](LICENSE).
- **Method Attribution:** LittleBit and LittleBit-2 methods are attributed to Banseok Lee, Dongkyu Kim, Youngcheon You, and Youngmin Kim (Samsung Research, NeurIPS 2025 & ICML 2026, CC BY-NC 4.0).
