# src/train_flat_ner.py
import os
import argparse
from dataclasses import dataclass

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd

LABEL_LIST = ["O", "PER", "ORG", "LOC", "DATE", "TIME", "HASHTAG", "MENTION"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="xlm-roberta-base")
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--output_dir", type=str, default="outputs/xlmr-flat")
    ap.add_argument("--num_train_epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_cuda", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False,
                    help="Disable CUDA/GPU and use CPU only")
    ap.add_argument("--fp16", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False,
                    help="Enable mixed precision (fp16) training when supported by the GPU")
    ap.add_argument("--use_cpu", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False,
                    help="Alias to force CPU training (preferred over --no_cuda) if set")
    ap.add_argument("--report_to", type=str, default="none",
                    help="Where to report training progress (none, tensorboard, wandb)")
    ap.add_argument("--save_total_limit", type=int, default=3,
                    help="Maximum number of checkpoints to keep")
    ap.add_argument("--dataloader_num_workers", type=int, default=0,
                    help="Number of worker processes for data loading")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of steps to accumulate gradients before backprop")
    return ap.parse_args()


def compute_metrics(pred):
    """
    pred: EvalPrediction from HF Trainer
    """
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    # Flatten but ignore sub-tokens with label -100
    true_labels = []
    pred_labels = []
    for p, l in zip(preds, labels):
        for pi, li in zip(p, l):
            if li == -100:
                continue
            true_labels.append(li)
            pred_labels.append(pi)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )
    acc = accuracy_score(true_labels, pred_labels)

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_precision": w_precision,
        "weighted_recall": w_recall,
        "weighted_f1": w_f1,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load dataset (already tokenized + -100 masking)
    ds = load_from_disk(args.data_dir)
    # Some preprocessing pipelines saved labels as strings (e.g., 'B-PER', 'O', '-100').
    # Convert labels to integer IDs expected by the model (keep -100 masks).
    def _label_strs_to_ids(example):
        labs = example.get("labels") or example.get("ner_tags")
        if labs is None:
            return {"labels": []}
        out = []
        for l in labs:
            # preserve numeric masks encoded as strings
            if isinstance(l, str) and l.strip() == "-100":
                out.append(-100)
                continue
            s = str(l)
            if s.startswith(("B-", "I-")):
                ent = s.split("-", 1)[1]
            else:
                ent = s
            ent = ent.upper()
            # map unknown entities to 'O' (index 0)
            out.append(LABEL2ID.get(ent, LABEL2ID.get("O", 0)))
        return {"labels": out}

    # apply to all splits present
    for split in list(ds.keys()):
        ds[split] = ds[split].map(lambda ex: _label_strs_to_ids(ex))

    train_ds = ds["train"]
    val_ds = ds.get("validation")

    # 2) Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 3) Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        logging_strategy="steps",
        seed=args.seed,
        no_cuda=args.no_cuda,
        use_cpu=args.use_cpu,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=(None if str(args.report_to).lower() in {"none",""} else args.report_to),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    # 4) Save training log as CSV for later:
    log_history = trainer.state.log_history
    df = pd.DataFrame(log_history)
    csv_path = os.path.join(args.output_dir, "training_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved training logs to {csv_path}")


if __name__ == "__main__":
    main()
