# src/predict_flat_ner.py
import os
import argparse
import json

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

LABEL_LIST = ["O", "PER", "ORG", "LOC", "DATE", "TIME", "HASHTAG", "MENTION"]
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Path to fine-tuned model (e.g. outputs/xlmr-flat)")
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out_path", type=str, default="preds/xlmr_flat_test.jsonl")
    ap.add_argument("--batch_size", type=int, default=16)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    ds = load_from_disk(args.data_dir)[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # We'll assume ds already has columns: input_ids, attention_mask, labels, tokens
    def chunk(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]

    with open(args.out_path, "w", encoding="utf-8") as f_out:
        for batch_indices in chunk(list(range(len(ds))), args.batch_size):
            batch = ds.select(batch_indices)

            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            labels = np.array(batch["labels"])

            for i, idx in enumerate(batch_indices):
                example = ds[idx]
                tokens = example["tokens"]
                gold_ids = labels[i]
                pred_ids = preds[i]

                # Filter out sub-tokens where gold label is -100
                gold_flat = []
                pred_flat = []
                for gi, pi in zip(gold_ids, pred_ids):
                    if gi == -100:
                        continue
                    gold_flat.append(ID2LABEL[int(gi)])
                    pred_flat.append(ID2LABEL[int(pi)])

                assert len(tokens) == len(gold_flat), "Tokens and labels mismatch!"

                rec = {
                    "id": int(idx),
                    "tokens": tokens,
                    "gold": gold_flat,
                    "pred": pred_flat,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote predictions to {args.out_path}")


if __name__ == "__main__":
    main()
