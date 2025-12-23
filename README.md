# Fine-Tuning vs. In-Context Learning for Hinglish NER

**Author:** Ashmit Mukherjee  
**Institution:** New York University Abu Dhabi  
**Contact:** asm8879@nyu.edu

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📄 Research Paper

**[Read the Full Report (PDF)](./nlp_final_project.pdf)**

This repository contains the implementation and results from a comparative study on Named Entity Recognition (NER) for code-mixed Hinglish text, comparing fine-tuned transformer models against large language models.

---

## 🧠 Overview

This research investigates how **fine-tuned multilingual transformer models** (mBERT and XLM-R) compare against **large language models** (GPT-4o and Claude-3.5-Sonnet) on the task of **Named Entity Recognition (NER)** in **Hinglish (Hindi-English code-mixed)** text.

> **Research Question:**
> Can a smaller, domain-fine-tuned model outperform massive general-purpose LLMs on the specialized, code-mixed NER task?

**Key Findings:**
- Fine-tuned XLM-RoBERTa achieved **78% F1-score** on the COMI-LINGUA benchmark
- Domain-specific fine-tuning outperformed zero-shot GPT-4o (76% F1) on code-mixed text
- English borrowings written in Devanagari (e.g., "कोड" for "code") remain challenging for all models

**Dataset:** COMI-LINGUA (Hinglish Code-Mixed NER subset)  
**Entity Types:** PER, LOC, ORG, MISC (BIO format)  
**Scripts:** Roman and Devanagari

---

## 🎯 Objectives

* Fine-tune multilingual transformer models (mBERT and XLM-R) on the COMI-LINGUA Hinglish NER dataset
* Evaluate performance on Precision, Recall, and F1-score
* Compare fine-tuned models against LLM baselines (GPT-4o, Claude-3.5-Sonnet)
* Perform detailed error analysis to uncover model strengths and weaknesses
* Investigate script-specific performance (Roman vs. Devanagari)

---

## ⚙️ Methodology

### Dataset & Preprocessing

* COMI-LINGUA NER dataset (Hinglish code-mixed text)
* Tokenization using Hugging Face tokenizers (WordPiece/BPE)
* Label alignment with subword tokens (propagate to first subtoken)
* Stratified train/dev/test splits

### Model Fine-Tuning

* **Models:** `bert-base-multilingual-cased` (mBERT) and `xlm-roberta-base`
* Token classification head for sequence labeling
* AdamW optimizer with learning rate 2e-5 to 5e-5
* Batch size: 16–32, Epochs: 3–5
* Early stopping on validation F1

### Evaluation

* Entity-level metrics using `seqeval` (Precision, Recall, F1)
* Baseline comparisons:
  * GPT-4o: **76% F1**
  * Claude-3.5-Sonnet: **84–85% F1**
  * Codeswitch library: **81% F1**
* Per-entity and script-specific analysis

### Error Analysis

* English borrowings in Devanagari script
* Script-based performance differences
* Tag confusion patterns (ORG ↔ MISC)
* Qualitative examples and quantitative breakdowns

---

## 📊 Results

### Model Performance

| Model | F1-Score | Precision | Recall |
|-------|----------|-----------|--------|
| XLM-RoBERTa (fine-tuned) | **78%** | 80% | 76% |
| mBERT (fine-tuned) | 74% | 76% | 72% |
| Claude-3.5-Sonnet (zero-shot) | 84% | - | - |
| GPT-4o (zero-shot) | 76% | - | - |

### Key Insights

1. **Fine-tuning effectiveness:** Domain-specific fine-tuning enables smaller models to match or exceed GPT-4o on specialized tasks
2. **Script challenges:** Devanagari text with English borrowings remains difficult across all models
3. **Entity confusion:** ORG and MISC tags frequently confused, especially in code-mixed contexts
4. **Efficiency trade-off:** Fine-tuned models offer better cost-performance ratio for production use

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Ansester/NLP-F25-Team-1.git
cd NLP-F25-Team-1

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Fine-tune mBERT
python src/train_flat_ner.py --model bert-base-multilingual-cased --epochs 5

# Fine-tune XLM-RoBERTa
python src/train_flat_ner.py --model xlm-roberta-base --epochs 5
```

### Evaluation

```bash
# Evaluate on test set
python src/eval_flat.py --model_path checkpoints/xlm-roberta-best --test_data data/processed/test.jsonl
```

### Prediction

```bash
# Run inference on new text
python src/predict_flat.py --model_path checkpoints/xlm-roberta-best --text "Your Hinglish text here"
```

---

## 📁 Repository Structure

```
NLP-F25-Team-1/
├── data/
│   ├── raw/                 # Original COMI-LINGUA dataset
│   └── processed/           # Tokenized and aligned data
├── src/
│   ├── data_prep.py        # Data preprocessing pipeline
│   ├── train_flat_ner.py   # Model training script
│   ├── eval_flat.py        # Evaluation script
│   └── predict_flat.py     # Inference script
├── scripts/
│   ├── label_stats.py      # Dataset statistics
│   └── bootstrap_from_hf.py # Dataset utilities
├── tests/                   # Unit tests
├── checkpoints/             # Saved model weights
├── results/                 # Evaluation outputs
├── nlp_final_project.pdf   # Full research report
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠️ Technical Stack

- **Models:** Hugging Face Transformers (mBERT, XLM-RoBERTa)
- **Framework:** PyTorch
- **Evaluation:** seqeval
- **Data Processing:** Hugging Face Datasets
- **Training:** AdamW optimizer, early stopping, gradient clipping

---

## 📈 Future Work

* Implement LoRA/PEFT for parameter-efficient fine-tuning
* Explore few-shot learning with GPT-4o for direct comparison
* Add token-level script identification features
* Expand to other code-mixed language pairs
* Deploy as production API endpoint

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{mukherjee2024hinglishner,
  author = {Mukherjee, Ashmit},
  title = {Fine-Tuning vs. In-Context Learning for Hinglish NER},
  year = {2024},
  institution = {New York University Abu Dhabi},
  url = {https://github.com/Ansester/NLP-F25-Team-1}
}
```

---

## 📧 Contact

**Ashmit Mukherjee**  
Email: asm8879@nyu.edu  
LinkedIn: [linkedin.com/in/akshith-karthik](https://www.linkedin.com/in/akshith-karthik/)  
GitHub: [@Ansester](https://github.com/Ansester)

**Akshith Karthik**  
Email: ak10747@nyu.edu  
LinkedIn: [linkedin.com/in/ashmit-mukherjee](https://www.linkedin.com/in//)  
GitHub: [@Ak1016-stack](https://github.com/Ak1016-stack)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* COMI-LINGUA benchmark dataset creators
* Hugging Face for transformer models and libraries
* NYU Abu Dhabi NLP research group

