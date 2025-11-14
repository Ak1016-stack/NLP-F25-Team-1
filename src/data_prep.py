#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preparation for Hinglish NER (COMI-LINGUA subset).

- Loads raw data (JSONL with {"tokens": [...], "ner_tags": [...]})
  or a folder with {train, dev/valid, test}.jsonl
- Normalizes tokens; optional script_id (0=Roman, 1=Devanagari)
- Optional transliteration Devanagari -> Roman (IAST)
- Optional filtering to Roman-only or majority-Roman sentences
- Dedupes exact-duplicate token sequences
- Tokenizes with HF tokenizer; aligns BIO labels to subtokens
  (first subtoken keeps label; others are IGN_TAG)
- Exports HF DatasetDict via save_to_disk + JSONL mirrors + dataset_card.md
- Deterministic behavior via fixed seed
"""

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Any
from datasets import Features, Sequence, Value
import numpy as np
import regex as re
from typing import TYPE_CHECKING
# put near the other helpers
import re
# Devanagari block: U+0900–U+097F
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

def token_has_devanagari(tok: str) -> bool:
    return bool(DEVANAGARI_RE.search(tok or ""))

# strip commas, quotes, brackets/braces/whitespace from BOTH ends
_STRIP_WRAP_RE = re.compile(r"""^[\s,'"\]\}\)]+|[\s,'"\]\}\)]+$""")

def _clean_frag(s: str) -> str:
    # remove leading/trailing wrappers repeatedly
    prev = None
    cur = s.strip()
    while prev != cur:
        prev = cur
        cur = _STRIP_WRAP_RE.sub("", cur)
    return cur

# Optional heavy dependencies: import for type checking, and import at runtime
if TYPE_CHECKING:
    # these imports are only for static type checking (Pylance/pyright)
    from datasets import Dataset, DatasetDict  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    # indic_transliteration is optional at runtime; import for type checking to silence Pylance
    try:
        from indic_transliteration import sanscript  # type: ignore
        from indic_transliteration.sanscript import transliterate  # type: ignore
    except Exception:
        pass
else:
    try:
        from datasets import Dataset, DatasetDict
    except Exception:
        Dataset = None  # runtime will error if these are actually needed
        DatasetDict = None
    try:
        from transformers import AutoTokenizer
    except Exception:
        AutoTokenizer = None

# -----------------------------
# Constants
# -----------------------------
IGN_TAG = "<IGN>"  # ignored subtoken marker; mapped to -100 in train.py

# -----------------------------
# Utils
# -----------------------------
import string

_PUNCT_CHARS = set(string.punctuation) | {"…", "’", "“", "”", "–", "—", "•"}  # extend as needed

def is_pure_punct_or_empty(tok: str) -> bool:
    """True if token is empty after strip OR contains no letter/number (only punctuation-like)."""
    if tok is None:
        return True
    t = tok.strip()
    if not t:
        return True
    # keep hashtags/mentions as real tokens in social text
    if t.startswith("#") and len(t) > 1:
        return False
    if t.startswith("@") and len(t) > 1:
        return False
    # if every char is punctuation-like, treat as junk
    return all((ch in _PUNCT_CHARS) or (not ch.isalnum()) for ch in t) and not any(ch.isalnum() for ch in t)

def clean_tokens_and_tags(tokens, tags):
    """Drop tokens that became empty/pure-punct *and* drop the aligned labels at the same indices."""
    keep_tokens, keep_tags = [], []
    for t, y in zip(tokens, tags):
        if is_pure_punct_or_empty(t):
            continue
        keep_tokens.append(t)
        keep_tags.append(y)
    return keep_tokens, keep_tags


import re, json, ast
from typing import Any, List, Optional, Tuple

# Canon map (adjust as you like)
_CANON = {
    "ORGANISATION": "ORG", "ORGANIZATION": "ORG",
    "PERSON": "PER", "PER": "PER",
    "LOCATION": "LOC", "GPE": "LOC", "LOC": "LOC",
    "DATE": "DATE", "TIME": "TIME",
    "MONEY": "MONEY", "PERCENT": "PERCENT",
    "FAC": "FAC", "EVENT": "EVENT",
}
def _canon(ent: Optional[str]) -> str:
    if not ent: return "O"
    u = str(ent).upper()
    return "O" if u in {"O","X","NONE"} else _CANON.get(u, u)

def _safe_json_or_literal(s: str):
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def _stitch_fragments_to_text(frags: List[str]) -> str:
    """
    Join fragment strings into one JSON-like text and tidy common issues.
    """
    # raw join with spaces
    txt = " ".join(x.strip() for x in frags if x and x.strip())
    # Drop extra commas before closing braces/brackets: ", }" -> " }", ", ]" -> " ]"
    txt = re.sub(r",\s*([\}\]])", r"\1", txt)
    # Ensure outer brackets if missing
    if not txt.lstrip().startswith("["):
        txt = "[" + txt
    if not txt.rstrip().endswith("]"):
        txt = txt + "]"
    # Some data uses single quotes; literal_eval can handle it
    return txt

def _extract_pairs_from_obj(obj: Any) -> Optional[List[Tuple[str, str]]]:
    """
    From parsed Python object → list of (word, entity).
    Accepts list[dict] or nested/flat variants.
    """
    if not isinstance(obj, list):
        return None
    out = []
    for x in obj:
        if isinstance(x, dict):
            w = x.get("word")
            e = x.get("entity") or x.get("label") or x.get("tag")
            if isinstance(w, str):
                out.append((w, e if isinstance(e, str) else "O"))
        # tolerate tuples/lists like ("word","entity")
        elif isinstance(x, (tuple, list)) and len(x) >= 2:
            w, e = x[0], x[1]
            out.append((str(w), str(e)))
        else:
            # skip unknown entry
            continue
    return out or None

def _align_words_to_char_tokens(pairs: List[Tuple[str,str]], char_tokens: List[str]) -> Optional[List[str]]:
    """
    Greedy character concatenation to match each word text, and assign BIO tags over that span.
    """
    n = len(char_tokens)
    tags = ["O"] * n
    i = 0
    for w_text, ent in pairs:
        if not w_text:
            continue
        start = i
        built = ""
        while i < n and built != w_text:
            built += char_tokens[i]
            i += 1
        if built != w_text:
            # alignment failed — give up for this row
            return None
        ent = _canon(ent)
        if ent != "O":
            tags[start] = "B-" + ent
            for j in range(start + 1, i):
                tags[j] = "I-" + ent
    return tags

def coerce_ner_tags_aligned_to_chars(raw_tags: Any, char_tokens: List[str]) -> Optional[List[str]]:
    """
    Master coercer: handles list-of-fragments -> stitched text -> parsed list-of-dicts -> BIO over chars.
    Also accepts already-good BIO lists or simple flat entity lists.
    """
    n = len(char_tokens)

    # Already BIO?
    if isinstance(raw_tags, list) and len(raw_tags) == n and all(
        isinstance(x, str) and (x == "O" or x.startswith(("B-","I-"))) for x in raw_tags
    ):
        return raw_tags

    # Simple flat entity list (e.g., ["ORGANISATION", "X", ...]) -> convert to BIO
    if isinstance(raw_tags, list) and len(raw_tags) == n and all(
        isinstance(x, str) for x in raw_tags
    ):
        # Check if all items look like flat entity names (not BIO prefixed, not fragments)
        if all(not x.startswith(("[", "{", '"')) and ("'" not in x or x.count("'") <= 2) 
               for x in raw_tags):
            # Try converting flat entities to BIO
            bio = []
            prev_ent = None
            for ent_raw in raw_tags:
                ent = _canon(ent_raw)
                if ent == "O":
                    bio.append("O")
                    prev_ent = None
                else:
                    if ent != prev_ent:
                        bio.append("B-" + ent)
                    else:
                        bio.append("I-" + ent)
                    prev_ent = ent
            return bio

    # List[str] fragments → stitch → parse
    if isinstance(raw_tags, list) and all(isinstance(x, str) for x in raw_tags):
        txt = _stitch_fragments_to_text(raw_tags)
        obj = _safe_json_or_literal(txt)
        pairs = _extract_pairs_from_obj(obj) if obj is not None else None
        if pairs:
            return _align_words_to_char_tokens(pairs, char_tokens)
        return None

    # Single string possibly encoding the list → parse then recurse
    if isinstance(raw_tags, str) and raw_tags.strip():
        obj = _safe_json_or_literal(raw_tags.strip())
        if isinstance(obj, list):
            return coerce_ner_tags_aligned_to_chars(obj, char_tokens)
        return None

    return None



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def ascii_lower(s):
    """Lowercase ASCII letters only; leave other chars/scripts unchanged.
    Never raise on non-strings."""
    if not isinstance(s, str):
        return s
    return "".join(ch.lower() if ("A" <= ch <= "Z" or "a" <= ch <= "z") else ch for ch in s)

def normalize_token(tok: Any) -> str:
    """NFC normalize, trim, ASCII-only lowercase, strip ZW* chars.
    Handles non-string tokens by stringifying; None -> empty string."""
    if tok is None:
        return ""
    if not isinstance(tok, str):
        tok = str(tok)
    tok = unicodedata.normalize("NFC", tok).strip()
    tok = ascii_lower(tok)
    tok = tok.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    return tok

# (unicodedata already imported above)

def to_ascii(text: str) -> str:
    """Convert text to closest ASCII representation (strip accents/diacritics)."""
    if not isinstance(text, str):
        return text
    # Normalize and remove diacritics
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c) and ord(c) < 128])


def detect_script_id(word: str) -> int:
    return 1 if DEVANAGARI_RE.search(word or "") else 0

def is_sentence_roman_only(tokens: List[str]) -> bool:
    return all(not token_has_devanagari(t) for t in tokens)


def is_sentence_majority_roman(tokens: List[str]) -> bool:
    dev = sum(1 for t in tokens if token_has_devanagari(t))
    return dev < (len(tokens) - dev)


def transliterate_deva_to_roman(token: str) -> str:
    """Devanagari -> IAST (Roman). Falls back to original token if package unavailable."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        return transliterate(token, sanscript.DEVANAGARI, sanscript.IAST)
    except Exception:
        return token


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


def normalize_add_script_and_maybe_transliterate(rows, add_script_id: bool, transliterate_deva: bool=False, roman_ascii: bool=False):
    out = []
    for r in rows:
        toks = [normalize_token(t) for t in r["tokens"]]
        # optional transliteration + ascii strip (your existing logic)
        if transliterate_deva:
            toks = [transliterate_deva_to_roman(t) for t in toks]
        if roman_ascii:
            toks = [to_ascii(t) for t in toks]

        tags = [ _canon(tag) for tag in r["ner_tags"] ]  # your BIO canonicalizer

        # >>> NEW: drop junk tokens and their tags, in-sync <<<
        toks, tags = clean_tokens_and_tags(toks, tags)

        # if after cleaning a sentence is empty or lengths mismatch, skip it
        if not toks or len(toks) != len(tags):
            continue

        item = {"tokens": toks, "ner_tags": tags}
        if add_script_id:
            item["script_id"] = [detect_script_id(t) for t in toks]
        out.append(item)
    return out


IGN_TAG = "<IGN>"

def align_labels_with_tokenizer(
    examples: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int,
    add_script_id: bool,
) -> List[Dict]:
    """
    Tokenize & align labels.
    - ALL label positions are strings: BIO tags or IGN_TAG ("<IGN>").
    - No integers inside `labels`.
    """
    encoded = []
    for ex in examples:
        tokens = ex["tokens"]
        labels = ex["ner_tags"]
        if len(tokens) != len(labels):
            raise ValueError("tokens and ner_tags length mismatch")

        enc_raw = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=False,
        )
        word_ids = enc_raw.word_ids()
        enc = dict(enc_raw)

        aligned_labels: List = []  # mixed str/int: BIO strings or numeric -100 masks
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)             # numeric mask for special tokens
            elif wid != prev_wid:
                aligned_labels.append(str(labels[wid])) # BIO tag string
            else:
                aligned_labels.append(-100)             # numeric mask for subtokens
            prev_wid = wid

        if add_script_id:
            sid_full = ex.get("script_id") or [detect_script_id(t) for t in tokens]
            sid_aligned = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    sid_aligned.append(-100)            # ints are fine for script_ids
                elif wid != prev_wid:
                    sid_aligned.append(int(sid_full[wid]))
                else:
                    sid_aligned.append(-100)
                prev_wid = wid
            enc["script_ids"] = sid_aligned

        # originals (for mirrors)
        enc["labels"] = aligned_labels                  # <-- strings only
        enc["tokens"] = tokens
        enc["ner_tags"] = [str(x) for x in labels]

        # sanitize numeric arrays
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in enc and isinstance(enc[key], list):
                enc[key] = [int(x) for x in enc[key]]

        encoded.append(enc)
    return encoded


def _features_for_rows(has_token_type_ids: bool, has_script_ids: bool) -> Features:
    feats = {
        "input_ids":       Sequence(Value("int64")),
        "attention_mask":  Sequence(Value("int64")),
        # some tokenizers add this; some don't
        "labels":          Sequence(Value("string")),   # <-- force string labels
        "tokens":          Sequence(Value("string")),
        "ner_tags":        Sequence(Value("string")),
    }
    if has_token_type_ids:
        feats["token_type_ids"] = Sequence(Value("int64"))
    if has_script_ids:
        feats["script_ids"] = Sequence(Value("int64"))
    return Features(feats)

def _sanitize_row(row: dict) -> dict:
    # make sure everything matches features
    ints = ("input_ids","attention_mask","token_type_ids","script_ids")
    for k in ints:
        if k in row and isinstance(row[k], list):
            row[k] = [int(x) for x in row[k]]
    for k in ("labels","tokens","ner_tags"):
        if k in row and isinstance(row[k], list):
            row[k] = [str(x) for x in row[k]]          # <-- ensure strings
    return row

def to_hf_dataset(encoded_rows: list) -> Dataset:
    # sanitize all rows first
    rows = [_sanitize_row(r) for r in encoded_rows]
    # detect optional fields to build matching Features
    has_tti = any("token_type_ids" in r for r in rows)
    has_sid = any("script_ids" in r for r in rows)
    feats = _features_for_rows(has_tti, has_sid)
    return Dataset.from_list(rows, features=feats)


def save_datasetdict_and_mirrors(ds: DatasetDict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    # human-readable mirrors (tokens + ner_tags)
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
        if args.transliterate_deva:
            f.write("- Transliteration applied: Devanagari → Roman (IAST)\n")
        f.write("\n## Notes\n")
        f.write("- Labels aligned to first subtoken; others set to IGN_TAG ('<IGN>')\n")
        f.write("- Deterministic splits via fixed seed\n")
        if args.roman_only:
            f.write("- Filtered to **Roman-only** sentences (rows with any Devanagari removed)\n")
        elif args.majority_roman:
            f.write("- Filtered to **majority-Roman** sentences (kept if Roman tokens > Devanagari tokens)\n")


# -----------------------------
# Main (no filtering; always transliterate + ASCII)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare Hinglish NER data (COMI-LINGUA subset)")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="xlm-roberta-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--add_script_id", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    # keep flags but we will ignore roman_only/majority_roman
    parser.add_argument("--roman_only", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    parser.add_argument("--majority_roman", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    parser.add_argument("--dedupe", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    input_path = Path(args.input)
    out_dir = Path(args.output)

    # We force these behaviors:
    FORCE_TRANSLIT = True      # Devanagari -> Roman (IAST)
    FORCE_ASCII = True         # strip diacritics -> plain ASCII
    APPLY_FILTERS = False      # skip roman_only / majority_roman entirely

    raw = load_raw(input_path)
    splits = {}

    # --- Case A: single combined file -> split randomly ---
    if "_all" in raw:
        rows = normalize_add_script_and_maybe_transliterate(
            raw["_all"],
            add_script_id=args.add_script_id,
            transliterate_deva=FORCE_TRANSLIT,
            roman_ascii=FORCE_ASCII,
        )
        if args.dedupe:
            rows = dedupe_examples(rows)
        if not rows:
            raise ValueError("No data remains after preprocessing.")
        splits = random_splits(rows, train_ratio=args.train_ratio, dev_ratio=args.dev_ratio, seed=args.seed)

    # --- Case B: directory with train/dev/test ---
    else:
        for k, v in raw.items():
            nv = normalize_add_script_and_maybe_transliterate(
                v,
                add_script_id=args.add_script_id,
                transliterate_deva=FORCE_TRANSLIT,
                roman_ascii=FORCE_ASCII,
            )
            if args.dedupe:
                nv = dedupe_examples(nv)
            if not nv:
                print(f"[WARN] Split '{k}' is empty after preprocessing; dropping.", file=sys.stderr)
                continue
            splits[k] = nv

    # --- tokenization and alignment ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    encoded_splits = {}
    for split_name, rows in splits.items():
        enc_rows = align_labels_with_tokenizer(
            rows, tokenizer, max_length=args.max_length, add_script_id=args.add_script_id
        )
        if not enc_rows:
            print(f"[WARN] Encoded split '{split_name}' is empty; skipping.", file=sys.stderr)
            continue
        encoded_splits[split_name] = to_hf_dataset(enc_rows)

    # --- normalize keys & ensure validation split exists ---
    key_map = {"dev": "validation", "valid": "validation"}
    normalized = {key_map.get(k, k): v for k, v in encoded_splits.items()}
    if "train" not in normalized or len(normalized["train"]) == 0:
        raise ValueError("Missing non-empty 'train' split after processing.")
    if "validation" not in normalized and len(normalized["train"]) > 1:
        train = normalized["train"]
        n_val = max(1, int(0.1 * len(train)))
        normalized["validation"] = train.select(range(n_val))
        normalized["train"] = train.select(range(n_val, len(train)))

    ds = DatasetDict(normalized)
    save_datasetdict_and_mirrors(ds, out_dir)

    # --- write dataset card + sample preview ---
    stats = {split: summarize_split([{"tokens": ex["tokens"]} for ex in ds[split]]) for split in ds.keys()}
    # build a small Namespace for the dataset card (keeps typing satisfied)
    _a = argparse.Namespace(
        model=args.model,
        max_length=args.max_length,
        add_script_id=args.add_script_id,
        roman_only=False,
        majority_roman=False,
        transliterate_deva=True,
        roman_ascii=True,
    )
    write_dataset_card(out_dir, stats, _a)

    sample_path = out_dir / "sample_rows.jsonl"
    sample_rows = [{"tokens": ex["tokens"], "ner_tags": ex["ner_tags"]}
                   for ex in ds["train"].select(range(min(10, len(ds["train"]))))]
    write_jsonl(sample_path, sample_rows)

    print(f"[OK] Saved processed dataset to {out_dir}")
    for split in ds.keys():
        print(f"  - {split}: {len(ds[split])} examples")

if __name__ == "__main__":
    main()