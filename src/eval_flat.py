# scripts/eval_flat.py
import argparse
import json
from collections import Counter

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
import matplotlib.pyplot as plt
import pandas as pd

LABEL_LIST = ["O", "PER", "ORG", "LOC", "DATE", "TIME", "HASHTAG", "MENTION"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="eval/xlmr-flat")
    return ap.parse_args()


def load_preds(path):
    gold_all = []
    pred_all = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gold_all.extend(rec["gold"])
            pred_all.extend(rec["pred"])
    return gold_all, pred_all


def plot_confusion_matrix(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Gold",
        xlabel="Predicted",
        title="Token-level Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Optionally annotate counts:
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    import os

    os.makedirs(args.out_dir, exist_ok=True)

    gold_str, pred_str = load_preds(args.pred_path)

    # Convert to integer ids for sklearn confusion_matrix
    label2id = {l: i for i, l in enumerate(LABEL_LIST)}
    gold_ids = np.array([label2id[g] for g in gold_str])
    pred_ids = np.array([label2id[p] for p in pred_str])

    acc = accuracy_score(gold_ids, pred_ids)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold_ids, pred_ids, average="macro", zero_division=0
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        gold_ids, pred_ids, average="weighted", zero_division=0
    )

    print("Overall Metrics:")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Weighted F1:    {w_f1:.4f}")

    # Per-class metrics
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        gold_ids, pred_ids, labels=list(range(len(LABEL_LIST))), zero_division=0
    )
    df = pd.DataFrame(
        {
            "label": LABEL_LIST,
            "precision": per_p,
            "recall": per_r,
            "f1": per_f1,
            "support": support,
        }
    )
    df.to_csv(os.path.join(args.out_dir, "per_class_metrics.csv"), index=False)

    # Full text report
    report = classification_report(
        gold_ids, pred_ids, target_names=LABEL_LIST, zero_division=0
    )
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(gold_ids, pred_ids, labels=list(range(len(LABEL_LIST))))
    plot_confusion_matrix(
        cm,
        LABEL_LIST,
        os.path.join(args.out_dir, "confusion_matrix.png"),
    )
    print(f"Saved confusion matrix and metrics to {args.out_dir}")


if __name__ == "__main__":
    main()
