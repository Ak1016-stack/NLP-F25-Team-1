#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune a multilingual model for Hinglish NER.

Example:
  python src/train.py \
    --data_dir data/processed \
    --model xlm-roberta-base \
    --output_dir outputs/xlmr \
    --epochs 4 --lr 3e-5 --batch_size 16
"""

import argparse
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from utils import set_global_seed, infer_label_mappings, to_id_labels


def compute_metrics_builder(id2label):
    def compute_metrics(p):
        predictions, labels = p
        preds = predictions.argmax(-1)

        true_labels = []
        true_preds = []
        for pred, lab in zip(preds, labels):
            cur_true_labels = []
            cur_true_preds = []
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
    parser.add_argument("--data_dir", type=str, required=True, help="Processed dataset folder (save_to_disk)")
    parser.add_argument("--model", type=str, default="xlm-roberta-base")
    parser.add_argument("--output_dir", type=str, default="outputs/model")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    args = parser.parse_args()

    set_global_seed(args.seed)
    data_dir = Path(args.data_dir)
    ds = load_from_disk(str(data_dir))

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Build label mappings from dataset (labels are expected to be tag strings with -100)
    label_list, label2id, id2label = infer_label_mappings(ds)

    # convert labels to IDs for each split
    def convert_example(example):
        example["labels"] = to_id_labels(example["labels"], label2id)
        return example

    ds = ds.map(convert_example, desc="Mapping string labels to IDs")

    # Keep only columns the model expects
    keep_cols = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep_cols])

    num_labels = len(label_list)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label={i: l for i, l in enumerate(label_list)},
        label2id={l: i for i, l in enumerate(label_list)},
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", ds["train"]),
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # final eval on validation (and test if present)
    val_metrics = trainer.evaluate(ds.get("validation", ds["train"]))
    print("[VAL]", val_metrics)

    if "test" in ds:
        test_metrics = trainer.evaluate(ds["test"])
        print("[TEST]", test_metrics)


if __name__ == "__main__":
    main()
