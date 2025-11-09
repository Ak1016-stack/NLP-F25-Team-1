# NLP-F25-Team-1

# 🧩 Project Proposal: Fine-Tuning vs. In-Context Learning for Hinglish NER

**Team Members:**

* Ashmit
* Akshith
* Harsh
* Lovnish

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
