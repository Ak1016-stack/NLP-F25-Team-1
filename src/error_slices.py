# scripts/error_slices.py
import argparse
import json
from collections import Counter


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=20)
    return ap.parse_args()


def main():
    args = parse_args()

    pair_counts = Counter()
    # also collect some example sentences for each confusion
    examples = {}

    with open(args.pred_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tokens = rec["tokens"]
            gold = rec["gold"]
            pred = rec["pred"]

            for t, g, p in zip(tokens, gold, pred):
                if g != p:
                    key = (g, p)
                    pair_counts[key] += 1
                    # store up to 3 example snippets per pair
                    if key not in examples:
                        examples[key] = []
                    if len(examples[key]) < 3:
                        examples[key].append((t, " ".join(tokens)))

    print(f"Top {args.top_k} gold→pred confusions:")
    for (g, p), cnt in pair_counts.most_common(args.top_k):
        print(f"{g:8s} → {p:8s} : {cnt}")
        for tok, sent in examples[(g, p)]:
            print(f"   [tok={tok}] {sent}")
        print()


if __name__ == "__main__":
    main()
