# ==========================
# Hinglish NER Project Makefile
# ==========================

PYTHON := /opt/anaconda3/bin/python


DATA_DIR := data/processed
RAW := data/raw
MODEL := xlm-roberta-base
OUTPUT := outputs/xlmr
RESULTS := results
ROMAN_ONLY ?= 0
MAJORITY_ROMAN ?= 0
TRANSLIT ?= 1
ROMAN_ASCII ?= 1


# Default target
help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make prep            - Preprocess COMI-LINGUA dataset"
	@echo "  make train           - Fine-tune model"
	@echo "  make evaluate        - Evaluate test split"
	@echo "  make test            - Run unit tests"
	@echo "  make clean           - Remove generated artifacts"

# Environment
install:
	$(PYTHON) -m pip install -U -r requirements.txt

# 1️⃣ Data preprocessing
 prep:
	$(PYTHON) src/data_prep.py \
		--input data/raw \
		--output data/processed \
		--model xlm-roberta-base \
		--add_script_id true

# 2️⃣ Fine-tuning
train:
	$(PYTHON) src/train.py \
		--data_dir $(DATA_DIR) \
		--model $(MODEL) \
		--output_dir $(OUTPUT) \
		--epochs 4 --lr 3e-5 --batch_size 16

# 3️⃣ Evaluation
evaluate:
	$(PYTHON) src/evaluate.py \
		--data_dir $(DATA_DIR) \
		--checkpoint $(OUTPUT) \
		--split test \
		--metrics_out $(RESULTS)/metrics_test.json \
		--write_conll $(RESULTS)/test_predictions.conll

# 4️⃣ Unit tests
test:
	pytest -q

# 5️⃣ Cleanup
clean:
	rm -rf $(DATA_DIR) $(OUTPUT) $(RESULTS) __pycache__ */__pycache__

.PHONY: help install prep train evaluate test clean

stats:
	$(PYTHON) scripts/label_stats.py data/processed


print-python:
	@echo "PYTHON var: $(PYTHON)"
	@command -v $(PYTHON) || true
	@$(PYTHON) -c "import sys; print('sys.executable:', sys.executable)"

print-shell:
	@echo "SHELL: $(SHELL)"

train-xlmr:
	python -m src.train_flat_ner --model_name xlm-roberta-base --output_dir outputs/xlmr-flat

train-mbert:
	python -m src.train_flat_ner --model_name bert-base-multilingual-cased --output_dir outputs/mbert-flat
	
predict-test:
	python -m src.predict_flat_ner --model_dir outputs/xlmr-flat --split test --out_path preds/xlmr_flat_test.jsonl

eval-test:
	python -m scripts.eval_flat --pred_path preds/xlmr_flat_test.jsonl --out_dir eval/xlmr-flat

errors:
	python -m scripts.error_slices --pred_path preds/xlmr_flat_test.jsonl --top_k 30