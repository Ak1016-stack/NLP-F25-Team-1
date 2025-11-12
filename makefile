# ==========================
# Hinglish NER Project Makefile
# ==========================

PYTHON := /opt/anaconda3/bin/python


DATA_DIR := data/processed
RAW := data/raw/comi_lingua.jsonl
MODEL := xlm-roberta-base
OUTPUT := outputs/xlmr
RESULTS := results
ROMAN_ONLY ?= 1

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
		--input $(RAW) \
		--output $(DATA_DIR) \
		--model $(MODEL) \
		--add_script_id true \
		--roman_only $(ROMAN_ONLY) \

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

print-python:
	@echo "PYTHON var: $(PYTHON)"
	@command -v $(PYTHON) || true
	@$(PYTHON) -c "import sys; print('sys.executable:', sys.executable)"

print-shell:
	@echo "SHELL: $(SHELL)"

