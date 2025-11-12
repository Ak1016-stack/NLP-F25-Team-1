#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a saved checkpoint on validation or test split; optionally write CoNLL-style preds.

Example:
  python src/evaluate.py \
    --data_dir data/processed \
    --checkpoint outputs/xlmr \
    --split test \
    --write_conll results/test_predictions.conll \
    --metrics_out results/metrics_test.json
"""

import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
)
from seqeval.metrics import precision_score, recall_score, f1_score

def compute_metrics_builder(id2label):
    def compute_metrics(p):
        predictions, labels = p
        preds = predictions.argmax(-1)

        true_labels, true_preds = [], []
        for pred, lab in zip(preds, labels):
            cur_true_labels, cur_true_preds = [], []
            for p_i, l_i in zip(pred, lab):
                if l_i == -100:
                    continue
                cur_true_labels.append(id2label[int(l_i)])
                cur_true_preds.append(id2label[int(p_i)])
            true_labels.append(cur_true_labels)
            true_preds.append(cur_true_preds)

        prec = precision_score(true_labels, true_preds)
        rec = recall_score(true_labels, true_preds)
        f1 = f1_score(true_labels, true_preds)
        return {"precision": prec, "recall": rec, "f1": f1}
    return compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--metrics_out", type=str, default="")
    parser.add_argument("--write_conll", type=str, default="")
    args = parser.parse_args()

    ds = load_from_disk(args.data_dir)
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found in dataset: {list(ds.keys())}")

    # load model & tokenizer
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)

    # mappings from config
    id2label = model.config.id2label
    # keys may be str indices; ensure int keys
    id2label = {int(k): v for k, v in id2label.items()}

    collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    metrics = trainer.evaluate(ds[args.split])
    print(f"[{args.split.upper()}] {metrics}")

    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    # optional: write CoNLL-style predictions file using tokens if present
    if args.write_conll:
        # we need predictions for the split
        preds_output = trainer.predict(ds[args.split])
        preds = preds_output.predictions.argmax(-1)

        # try to recover tokens from dataset (we saved them in processing JSONL, but
        # HF dataset at training time might not carry them). We'll fall back to decoding.
        tokens_col_present = "tokens" in ds[args.split].column_names

        lines = []
        for i, (pred_row, lab_row) in enumerate(zip(preds, preds_output.label_ids)):
            if tokens_col_present:
                toks = ds[args.split][i]["tokens"]
                # align toks to labels by skipping -100 positions on labels
                j = 0
                for p_i, l_i in zip(pred_row, lab_row):
                    if l_i == -100:
                        continue
                    tok = toks[j] if j < len(toks) else "<UNK>"
                    lines.append(f"{tok} {id2label[int(p_i)]}")
                    j += 1
                lines.append("")  # sentence break
            else:
                # fallback: just write labels per token position (can't perfectly recover subtokens)
                for p_i, l_i in zip(pred_row, lab_row):
                    if l_i == -100:
                        continue
                    lines.append(f"{id2label[int(p_i)]}")
                lines.append("")
        out_path = Path(args.write_conll)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[OK] Wrote predictions to {args.write_conll}")

if __name__ == "__main__":
    main()
