# NLP-F25-Team-1

# 🧩 Fine-Tuning vs. In-Context Learning for Hinglish NER

**Author:** Ashmit Mukherjee

**Duration:** 3 Weeks
**Course:** NLP — Final Project
**Task:** Named Entity Recognition (NER)
**Dataset:** COMI-LINGUA (Hinglish Code-Mixed NER subset)

---

## 🧠 1. Project Overview

This project investigates how **fine-tuned multilingual transformer models** (mBERT and XLM-R) compare against **large language models** (GPT-4o and Claude-3.5-Sonnet) on the task of **Named Entity Recognition (NER)** in **Hinglish (Hindi-English code-mixed)** text.

> **Research Question:**
> Can a smaller, domain-fine-tuned model outperform massive general-purpose LLMs on the specialized, code-mixed NER task?

This is inspired by the COMI-LINGUA benchmark, which highlights that even the strongest LLMs struggle with code-mixed text — especially in cases of **English borrowings written in Devanagari** (e.g., “कोड” for “code”).

---

## 🎯 2. Objective

* Fine-tune multilingual transformer models (mBERT and XLM-R) on the COMI-LINGUA Hinglish NER dataset.
* Evaluate performance on Precision, Recall, and F1-score.
* Compare fine-tuned models against the reported COMI-LINGUA baselines.
* Perform detailed **error analysis** to uncover strengths and weaknesses.

---

## 🧩 3. Scope

**Task:** Named Entity Recognition (NER)
**Dataset:** COMI-LINGUA (Hinglish NER subset)
**Languages:** Roman and Devanagari script
**Entity Tags:** PER, LOC, ORG, MISC (BIO format)
**Evaluation Metric:** F1-score (micro, entity-level)

---

## ⚙️ 4. Methodology

### Step 1: Dataset & Preprocessing

* Acquire COMI-LINGUA NER dataset.
* Tokenize text using Hugging Face tokenizer (WordPiece/BPE).
* Align entity labels with subword tokens (propagate to first subtoken).
* Split into train, dev, and test sets.

### Step 2: Model Fine-Tuning

* Models:

  * `bert-base-multilingual-cased` (mBERT)
  * `xlm-roberta-base`
* Add a token classification head on top.
* Fine-tune using AdamW optimizer:

  * LR: 2e-5 to 5e-5
  * Batch size: 16–32
  * Epochs: 3–5
  * Early stopping on validation F1.

### Step 3: Evaluation

* Use `seqeval` for Precision, Recall, F1 (entity-level).
* Compare to baselines:

  * GPT-4o ≈ **76 F1**
  * Claude-3.5-Sonnet ≈ **84–85 F1**
  * Codeswitch library ≈ **81 F1**
* Report overall and per-entity F1, plus script-specific results (Roman vs. Devanagari).

### Step 4: Error Analysis

* Replicate COMI-LINGUA’s error patterns:

  * English borrowings in Devanagari (e.g., “कोड”).
  * Script-based performance differences.
  * Common tag confusions (e.g., ORG ↔ MISC).
* Include qualitative examples and quantitative breakdowns.

---

## 👥 5. Team Roles and Deliverables

| Member | Role                       | Deliverables                                                          |
| ------ | -------------------------- | --------------------------------------------------------------------- |
| **A**  | Data & Preprocessing       | Scripts for tokenization, alignment, and data splits (`data_prep.py`) |
| **B**  | Model Training             | Fine-tuning pipeline (`train.py`), configs, and checkpoints           |
| **C**  | Evaluation                 | Evaluation script (`evaluate.py`), results tables, plots              |
| **D**  | Error Analysis & Reporting | Analysis notebook, visualizations, final presentation                 |

---

## 📆 6. Timeline (3 Weeks)

| Week       | Tasks                                                                  |
| ---------- | ---------------------------------------------------------------------- |
| **Week 1** | Dataset setup, tokenization, baseline review                           |
| **Week 2** | Fine-tune mBERT and XLM-R, tune hyperparameters                        |
| **Week 3** | Evaluate models, perform error analysis, prepare report & presentation |

---

## 📊 7. Deliverables

1. ✅ Fine-tuning scripts and model configs
2. ✅ Evaluation notebook with metrics and comparison table
3. ✅ COMI-LINGUA baseline comparison (LLMs vs fine-tuned)
4. ✅ Error analysis notebook with visual examples
5. ✅ Final presentation (5–8 slides) summarizing findings

---

## 💡 8. Stretch Goals (If Time Permits)

* Implement LoRA / PEFT fine-tuning on XLM-R for efficiency.
* Conduct few-shot GPT-4o NER experiments for direct comparison.
* Add a token-level script-ID feature (Roman vs. Devanagari) to improve robustness.

---

## 🚀 9. Expected Outcome

By the end of the project, the team will have:

* A **working fine-tuning pipeline** for code-mixed NER.
* Empirical comparison between **fine-tuned** vs **in-context** LLM performance.
* A focused **error analysis** that reveals whether specialized fine-tuning better handles the quirks of Hinglish text.

---

## 🔟 Milestones (Shared)

- **M0 — Repo Ready (Tue, Nov 11):** repo scaffold, issue board, branch rules, data access verified  
- **M1 — Data Pipeline Green (Sat, Nov 15):** tokenization + label alignment + splits reproducible  
- **M2 — First Full Fine-Tune (Thu, Nov 20):** mBERT baseline F1 on dev  
- **M3 — Dual Models Tuned (Mon, Nov 24):** mBERT + XLM-R best checkpoints/configs locked  
- **M4 — Eval + Error Analysis (Thu, Nov 27):** tables, plots, error slices  
- **M5 — Final Package (Mon, Dec 1):** code, results, notebooks, slides

---

## 1️⃣1️⃣ Work Distribution (Owners, Deadlines, DoD)

### A) **Ashmit — Data & Preprocessing Lead** (Nov 11–15)
- **PR-1: Data pipeline** — *Due Thu, Nov 13*  
  - `data_prep.py`: load COMI-LINGUA NER; normalize; optional `script_id`; dedupe  
  - Tokenize + BIO alignment (first subtoken keeps label; others `-100`)  
  - **DoD:** deterministic runs; unit tests for alignment edge cases pass
- **PR-2: Splits + dataset card** — *Due Sat, Nov 15*  
  - Stratify by entity + script; export HF Datasets + JSONL; add `dataset_card.md`  
  - **Artifacts:** `data/processed/{train,dev,test}.jsonl`, `dataset_card.md`  
- **Hand-off:** schemas + 10-row sample to **Harsh** & **Akshith** EOD Nov 15  
- **Stretch:** add `script_id` feature (0=Roman, 1=Devanagari)

### B) **Akshith — Training & MLOps Owner** (Nov 13–24)
- **PR-3: mBERT training pipeline** — *Due Tue, Nov 18*  
  - `train.py` (HF `Trainer`), AdamW, early stop on dev F1, AMP, grad clip, best-F1 ckpt  
  - **DoD:** one full run completes; dev F1 logged to `runs.csv`
- **PR-4: XLM-R + hyperparam sweeps** — *Due Mon, Nov 24*  
  - Sweep LR/epochs/wd; same preprocessing; save best configs  
  - **Artifacts:** `checkpoints/*`, `configs/{mb, xlm-r}.yaml`, `runs.csv` (with seeds/metrics)  
- **Hand-off:** best checkpoints + configs to **Lovnish** EOD Nov 24  
- **Stretch:** LoRA/PEFT flag `--peft` for XLM-R

### C) **Harsh — Baselines & Prompted LLMs** (Nov 15–23)
- **PR-5: LLM baselines harness** — *Due Wed, Nov 19*  
  - Prompt templating (0/1/5-shot, 3 seeds), tag extraction, **BIO repair**  
  - **DoD:** reproducible CSV/JSONL with entity-valid BIO aligned to gold
- **PR-6: Baseline comparison table** — *Due Sun, Nov 23*  
  - Aggregate GPT-4o, Claude-3.5, Codeswitch (reported) + our first mBERT run  
  - **Artifacts:** `eval/baselines.csv`, `figs/baselines_bar.png`, short prompt-sensitivity notes  
- **Hand-off:** JSONL predictions to **Lovnish** EOD Nov 23  
- **Stretch:** prompts targeting Devanagari English borrowings

### D) **Lovnish — Evaluation, Error Analysis & Report** (Nov 18–Dec 1)
- **PR-7: Evaluation suite** — *Due Thu, Nov 20*  
  - `evaluate.py` (`seqeval`): entity-level micro F1, per-entity F1, per-script F1  
  - CI check fails on schema mismatch  
  - **DoD:** `results/{model}/{dev,test}_metrics.json`
- **PR-8: Error analysis** — *Due Thu, Nov 27*  
  - `notebooks/error_analysis.ipynb`: confusions (ORG↔MISC), script slices, 10 curated failures  
  - **Artifacts:** `figs/*`, example tables (gold vs pred)
- **PR-9: Final presentation** — *Due Mon, Dec 1*  
  - **5–8 slides:** setup, methods, results, slices, 3 insights, 2 limitations, 2 next steps  
  - **DoD:** PDF + PPTX committed; figures render on fresh clone

---

## 1️⃣2️⃣ Detailed Task Matrix (ET)

| Date | Task | Owner | Output / DoD |
|---|---|---|---|
| **Tue, Nov 11** | Repo bootstrap, issue board, Makefile | Akshith | `README.md`, `Makefile`, branch rules (**M0**) |
| **Thu, Nov 13** | Data loader + label alignment | **Ashmit** | `src/data_prep.py`, unit tests green |
| **Sat, Nov 15** | Splits + dataset card | **Ashmit** | processed JSONL + `dataset_card.md` (**M1**) |
| **Tue, Nov 18** | mBERT training pipeline | **Akshith** | `src/train.py`, first dev F1 (**M2**) |
| **Wed, Nov 19** | LLM evaluation harness | **Harsh** | prompts + BIO repair + CSV/JSONL preds |
| **Thu, Nov 20** | Eval script (seqeval) | **Lovnish** | `src/evaluate.py`, metrics JSON (**M2 support**) |
| **Sun, Nov 23** | LLM baselines (0/1/5-shot) | **Harsh** | `eval/baselines.csv` |
| **Mon, Nov 24** | XLM-R + sweeps; lock configs | **Akshith** | best ckpts + `configs/*` (**M3**) |
| **Thu, Nov 27** | Error analysis notebook + figs | **Lovnish** | slices, confusions, exemplars (**M4**) |
| **Sat, Nov 29** | Fresh-clone repro dry-run | All | `make eval_small` passes |
| **Mon, Dec 1** | Final slides + packaging | **Lovnish (+ all)** | PDF/PPTX + `results/` (**M5**) |

---

## 1️⃣3️⃣ Interfaces & Hand-offs

- **Data → Training:** columns `tokens`, `labels`, `script_id`; BIO labels; subtokens = `-100`  
- **Training → Eval:** `predictions.jsonl` (token-level BIO + entity spans)  
- **LLMs → Eval:** same schema (after BIO repair)  
- **Eval → Report:** `results_table.csv`, `per_entity_f1.csv`, `per_script_f1.csv`, **10** failure exemplars

---

## 1️⃣4️⃣ Risks & Mitigations

- **GPU time bottleneck:** prioritize mBERT; limit XLM-R sweeps to top 3 configs; use grad accumulation  
- **LLM tag noise:** strict BIO repair + span validation; log and justify any exclusions  
- **Label alignment bugs:** unit tests for multi-subword tokens/punctuation; manual spot-check of 50 sentences (Nov 15–16)

---

## 1️⃣5️⃣ Repo Structure (proposed)

