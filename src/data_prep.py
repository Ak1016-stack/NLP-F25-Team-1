#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preparation for Hinglish NER (COMI-LINGUA subset).

- Loads raw data (JSONL with {"tokens": [...], "ner_tags": [...]})
  or a folder with train/dev/test .jsonl files
- Normalizes tokens and (optionally) adds script_id (0=Roman, 1=Devanagari)
- Deduplicates exact-duplicate token sequences
- Tokenizes with Hugging Face tokenizer and aligns BIO labels to subwords
  (first subtoken gets label; others set to -100)
- Exports HF DatasetDict via save_to_disk + JSONL mirrors + dataset_card.md
- Deterministic behavior via fixed seed

Usage:
  python src/data_prep.py \
    --input data/raw/comi_lingua.jsonl \
    --output data/processed \
    --model xlm-roberta-base \
    --add_script_id true
"""
IGN_TAG = "<IGN>"  # special token to mark ignored sub-tokens; mapped to -100 at training time
import argparse
from html import parser
import json
import os
import random
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List

import numpy as np
import regex as re
from datasets import Dataset, DatasetDict # pyright: ignore[reportMissingImports]
from transformers import AutoTokenizer # pyright: ignore[reportMissingImports]

# Roman-only filtering helpers
def token_has_devanagari(tok: str) -> bool:
    from regex import Regex  # or use global DEVANAGARI_RE if present
    # if DEVANAGARI_RE already exists, just use it; otherwise:
    try:
        pat = DEVANAGARI_RE  # from earlier in your file
    except NameError:
        import regex as re
        pat = re.compile(r"\p{Devanagari}", flags=re.UNICODE)
    return bool(pat.search(tok or ""))

def is_sentence_roman_only(tokens) -> bool:
    return all(not token_has_devanagari(t) for t in tokens)

def is_sentence_majority_roman(tokens) -> bool:
    # optional alternate policy: keep if strictly more roman than devanagari tokens
    dev = sum(1 for t in tokens if token_has_devanagari(t))
    return dev < (len(tokens) - dev)



DEVANAGARI_RE = re.compile(r"\p{Devanagari}", flags=re.UNICODE)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def ascii_lower(s: str) -> str:
    """Lowercase ASCII letters; keep other scripts intact."""
    return "".join(ch.lower() if "A" <= ch <= "Z" or "a" <= ch <= "z" else ch for ch in s)


def normalize_token(tok: str) -> str:
    if tok is None:
        return tok
    tok = unicodedata.normalize("NFC", tok).strip()
    tok = ascii_lower(tok)
    tok = tok.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    return tok


def detect_script_id(word: str) -> int:
    return 1 if DEVANAGARI_RE.search(word or "") else 0


def read_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dedupe_examples(rows: List[Dict]) -> List[Dict]:
    seen = set()
    deduped = []
    for r in rows:
        key = tuple(r.get("tokens", []))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def summarize_split(rows: List[Dict]) -> Dict[str, int]:
    n = len(rows)
    tokens = sum(len(r.get("tokens", [])) for r in rows)
    by_script = {"roman": 0, "devanagari": 0}
    for r in rows:
        sid = [detect_script_id(t) for t in r["tokens"]]
        if sum(sid) >= max(1, len(sid) // 2):
            by_script["devanagari"] += 1
        else:
            by_script["roman"] += 1
    return {"examples": n, "tokens": tokens, **by_script}


def load_raw(input_path: Path) -> Dict[str, List[Dict]]:
    """
    (A) Single JSONL -> {"_all": rows}
    (B) Dir with train/dev/test jsonl -> {"train": [...], "validation": [...], "test": [...]}
    """
    if input_path.is_dir():
        cand = {
            "train": ["train.jsonl", "train.json"],
            "validation": ["dev.jsonl", "valid.jsonl", "dev.json", "valid.json"],
            "test": ["test.jsonl", "test.json"],
        }
        splits = {}
        for split, names in cand.items():
            for name in names:
                p = input_path / name
                if p.exists():
                    splits[split] = read_jsonl(p)
                    break
        if not splits:
            raise ValueError(f"No recognizable JSONL files in {input_path}")
        return splits
    else:
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        rows = read_jsonl(input_path)
        return {"_all": rows}


def random_splits(rows: List[Dict], train_ratio=0.8, dev_ratio=0.1, seed=42) -> Dict[str, List[Dict]]:
    set_seed(seed)
    idx = list(range(len(rows)))
    random.shuffle(idx)
    n = len(idx)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]
    test_idx = idx[n_train + n_dev:]
    to_split = lambda indices: [rows[i] for i in indices]
    return {"train": to_split(train_idx), "validation": to_split(dev_idx), "test": to_split(test_idx)}


def normalize_and_optionally_add_script(rows: List[Dict], add_script_id: bool) -> List[Dict]:
    out = []
    for r in rows:
        toks = [normalize_token(t) for t in r["tokens"]]
        item = {"tokens": toks, "ner_tags": r["ner_tags"]}
        if add_script_id:
            item["script_id"] = [detect_script_id(t) for t in toks]
        out.append(item)
    return out


def align_labels_with_tokenizer(
    examples: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int,
    add_script_id: bool,
) -> List[Dict]:
    """
    Tokenize & align labels.
    - Keep ALL labels as strings to avoid PyArrow mixed-type issues.
    - Use IGN_TAG ("<IGN>") for ignored sub-token positions.
    """
    encoded = []
    for ex in examples:
        tokens = ex["tokens"]
        labels = ex["ner_tags"]
        if len(tokens) != len(labels):
            raise ValueError("tokens and ner_tags length mismatch")

        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=False,
        )
        word_ids = enc.word_ids()

        aligned_labels = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(IGN_TAG)         # was -100
            elif wid != prev_wid:
                aligned_labels.append(labels[wid])      # keep BIO tag string
            else:
                aligned_labels.append(IGN_TAG)          # was -100
            prev_wid = wid

        if add_script_id:
            sid_full = ex["script_id"]
            sid_aligned = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    sid_aligned.append(-100)            # script_ids can stay ints throughout
                elif wid != prev_wid:
                    sid_aligned.append(int(sid_full[wid]))
                else:
                    sid_aligned.append(-100)
                prev_wid = wid
            enc["script_ids"] = sid_aligned

        # Keep originals for inspection
        enc["labels"] = aligned_labels         # <-- now strings only
        enc["tokens"] = tokens
        enc["ner_tags"] = labels
        encoded.append(enc)

    # final sanitation pass: ensure numeric arrays are ints, labels are strings
    def _sanitize_row(r: Dict) -> Dict:
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in r and isinstance(r[key], list):
                r[key] = [int(x) for x in r[key]]
        # labels already strings (BIO tags or "<IGN>")
        if "labels" in r:
            r["labels"] = [str(x) for x in r["labels"]]
        if "script_ids" in r and isinstance(r["script_ids"], list):
            r["script_ids"] = [int(x) for x in r["script_ids"]]
        return r

    return [_sanitize_row(row) for row in encoded]


def to_hf_dataset(encoded_rows: List[Dict]) -> Dataset:
    return Dataset.from_list(encoded_rows)


def save_datasetdict_and_mirrors(ds: DatasetDict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))

    # also provide human-readable mirrors (tokens + ner_tags)
    for split in ds.keys():
        rows = []
        for ex in ds[split]:
            if "tokens" in ex and "ner_tags" in ex:
                rows.append({"tokens": ex["tokens"], "ner_tags": ex["ner_tags"]})
        write_jsonl(out_dir / f"{split}.jsonl", rows)


def write_dataset_card(out_dir: Path, stats: Dict[str, Dict[str, int]], args: argparse.Namespace):
    card = out_dir / "dataset_card.md"
    with card.open("w", encoding="utf-8") as f:
        f.write("# Dataset Card: Hinglish NER (Processed)\n\n")
        f.write("**Source:** COMI-LINGUA (Hinglish Code-Mixed NER subset)\n\n")
        f.write("**Task:** Named Entity Recognition (BIO)\n\n")
        f.write(f"**Tokenizer:** `{args.model}`\n\n")
        f.write(f"**Max length:** {args.max_length}\n\n")
        f.write("## Splits Summary\n\n")
        for split, d in stats.items():
            f.write(f"- **{split}**: examples={d['examples']}, tokens={d['tokens']}, "
                    f"roman_sents={d['roman']}, devanagari_sents={d['devanagari']}\n")
        f.write("\n## Preprocessing\n")
        f.write("- Unicode NFC normalization; ASCII-only lowercasing\n")
        f.write("- Deduplication by exact token sequence\n")
        if args.add_script_id:
            f.write("- Added `script_id` per token (0=Roman, 1=Devanagari)\n")
        f.write("\n## Notes\n")
        f.write("- Labels aligned to first subtoken; others set to -100\n")
        f.write("- Deterministic splits via fixed seed\n")
        if args.roman_only:
            f.write("- Filtered to **Roman-only** sentences (rows with any Devanagari removed)\n")
        elif args.majority_roman:
            f.write("- Filtered to **majority-Roman** sentences (kept if Roman tokens > Devanagari tokens)\n")
def main():
    parser = argparse.ArgumentParser(description="Prepare Hinglish NER data")
    parser.add_argument("--roman_only", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False,
                    help="If true, keep only sentences with no Devanagari chars (pure Roman).")
# (optional) expose majority-roman policy instead of strict
    parser.add_argument("--majority_roman", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False,
                    help="If true, keep sentences with majority Roman tokens (ignored if --roman_only).")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="xlm-roberta-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--add_script_id", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--dedupe", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    input_path = Path(args.input)
    out_dir = Path(args.output)

    raw = load_raw(input_path)
    if "_all" in raw:
        rows = normalize_and_optionally_add_script(raw["_all"], add_script_id=args.add_script_id)
        if args.roman_only:
            rows = [r for r in rows if is_sentence_roman_only(r["tokens"])]
        elif args.majority_roman:
            rows = [r for r in rows if is_sentence_majority_roman(r["tokens"])]
        if args.dedupe:
            rows = dedupe_examples(rows)
        splits = random_splits(rows, train_ratio=args.train_ratio, dev_ratio=args.dev_ratio, seed=args.seed)
    else:
        splits = {}
    for k, v in raw.items():
        nv = normalize_and_optionally_add_script(v, add_script_id=args.add_script_id)
        if args.dedupe:
            nv = dedupe_examples(nv)

    # --- NEW: filter to Roman-only (or majority-roman)
        if args.roman_only:
            nv = [r for r in nv if is_sentence_roman_only(r["tokens"])]
        elif args.majority_roman:
            nv = [r for r in nv if is_sentence_majority_roman(r["tokens"])]
        # ---
            splits[k] = nv

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    encoded_splits = {}
    for split_name, rows in splits.items():
        enc_rows = align_labels_with_tokenizer(rows, tokenizer, max_length=args.max_length, add_script_id=args.add_script_id)
        encoded_splits[split_name] = to_hf_dataset(enc_rows)

    # normalize keys and ensure train/validation
    key_map = {"dev": "validation", "valid": "validation"}
    normalized = {key_map.get(k, k): v for k, v in encoded_splits.items()}

    if "train" not in normalized:
        raise ValueError("Missing 'train' split after processing.")
    if "validation" not in normalized:
        train = normalized["train"]
        n = len(train)
        n_val = max(1, int(0.1 * n))
        val = train.select(range(n_val))
        tr = train.select(range(n_val, n))
        normalized["train"] = tr
        normalized["validation"] = val

    ds = DatasetDict(normalized)
    save_datasetdict_and_mirrors(ds, out_dir)

    stats = {split: summarize_split([{"tokens": ex["tokens"]} for ex in ds[split]]) for split in ds.keys()}
    write_dataset_card(out_dir, stats, args)

    sample_path = out_dir / "sample_rows.jsonl"
    sample_rows = []
    for ex in ds["train"].select(range(min(10, len(ds["train"])))):
        sample_rows.append({"tokens": ex["tokens"], "ner_tags": ex["ner_tags"]})
    write_jsonl(sample_path, sample_rows)

    print(f"[OK] Saved processed dataset to {out_dir}")
    for split in ds.keys():
        print(f"  - {split}: {len(ds[split])} examples")


if __name__ == "__main__":
    main()

