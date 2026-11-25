# Data Sources & Licence Audit — Clyffe (ARC-LB-01)

**Audit Date:** 2026-11-25  
**Policy:** Human-written text only + project-generated deterministic arithmetic/templates. No LLM-generated datasets (e.g., TinyStories, Cosmopedia, UltraChat, Alpaca). Unclear or incompatible licences are skipped.

---

## 1. Hugging Face Candidate Audit

| Dataset | URL | Verified HF Licence | Size Category | What It Is & Human-Written Provenance | Status |
|---|---|---|---|---|---|
| `wikimedia/wikipedia` | https://huggingface.co/datasets/wikimedia/wikipedia | `cc-by-sa-3.0`, `gfdl` | 1M–10M | Simple English & English Wikipedia dumps; human-authored encyclopedia articles. | **Approved (Stage 1+)** |
| `emozilla/pg19` | https://huggingface.co/datasets/emozilla/pg19 | `apache-2.0` | ~28K books | Project Gutenberg public-domain books published before 1919; human-written literature. | **Approved (Stage 1+)** |
| `HuggingFaceFW/fineweb-edu` | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | `odc-by` | >1T tokens | Educational web pages (human-written web text filtered by an educational classifier). | **Approved (Stage 2+, note classifier)** |
| `rajpurkar/squad` | https://huggingface.co/datasets/rajpurkar/squad | `cc-by-sa-4.0` | 10K–100K | Reading comprehension Q&A written by human crowdworkers on Wikipedia passages. | **Approved (Stage 1+ SFT)** |
| `allenai/ai2_arc` | https://huggingface.co/datasets/allenai/ai2_arc | `cc-by-sa-4.0` | 1K–10K | Grade-school science multiple-choice exam questions written by educators. | **Approved (Stage 1+ SFT)** |
| `allenai/openbookqa` | https://huggingface.co/datasets/allenai/openbookqa | `unknown` | 1K–10K | Elementary science Q&A; HF card lists licence as `unknown`. | **SKIPPED (unclear licence)** |
| `allenai/sciq` | https://huggingface.co/datasets/allenai/sciq | `cc-by-nc-3.0` | 10K–100K | Crowdworker-written science exam questions; Non-Commercial (NC) terms. | **Approved (NC-only research)** |
| `google-research-datasets/natural_questions` | https://huggingface.co/datasets/google-research-datasets/natural_questions | `cc-by-sa-3.0` | 100K–1M | Real Google search queries paired with human-annotated Wikipedia answers. | **Approved (Stage 1+ SFT)** |
| `databricks/databricks-dolly-15k` | https://huggingface.co/datasets/databricks/databricks-dolly-15k | `cc-by-sa-3.0` | 10K–100K | 15K instruction/response records written by Databricks human employees. | **Approved (Stage 1+ SFT)** |
| `OpenAssistant/oasst1` | https://huggingface.co/datasets/OpenAssistant/oasst1 | `apache-2.0` | 100K–1M | Human-written, human-annotated assistant conversation trees. | **Approved (Stage 1+ SFT)** |
| `li2017dailydialog/daily_dialog` | https://huggingface.co/datasets/li2017dailydialog/daily_dialog | `cc-by-nc-sa-4.0` | 10K–100K | Human-written daily English dialogues; Non-Commercial ShareAlike (NC-SA). | **Approved (NC-only research)** |
| `bavard/personachat_truecased` | https://huggingface.co/datasets/bavard/personachat_truecased | *Missing on HF card* | 100K–1M | Persona dialogue dataset; no explicit licence tag on HF dataset card. | **SKIPPED (unclear licence)** |
| `openai/gsm8k` | https://huggingface.co/datasets/openai/gsm8k | `mit` | 1K–10K | Grade-school math word problems written by human problem writers. | **Approved (Stage 1+ SFT)** |

---

## 2. Project-Generated Stage 0 & SFT Data (Scripts)

- **Identity, Greetings, Conversation, Name-Memory, General Knowledge, Basic Science Templates (`scripts/build_dataset.py`):** Authored directly in-project with ~50+ unique phrasings per intent, strictly deduplicated across train/val/test (`0` overlap).
- **Arithmetic Generator (`scripts/build_dataset.py`):** Deterministic Python generator for `+`, `-`, `×`, `÷`, and averages, with held-out operand ranges (`unseen_math.jsonl`) for generalization testing.
- **Out-of-Scope / Refusal Set:** Diverse out-of-scope prompts paired with the exact refusal `"I don't have sufficient trained knowledge to answer this question."`, plus a separate held-out OOD topic split for evaluation.

---

## 3. Storage & Manifest

Processed datasets are stored in `datasets/processed/` (alongside `datasets/train/`, `datasets/validation/`, `datasets/test/`) with a SHA-256 and row-count manifest (`datasets/processed/manifest.json`).

