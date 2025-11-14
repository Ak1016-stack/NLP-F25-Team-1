#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import json, unicodedata, random
from typing import Any, List, Tuple, Optional

from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from datasets.features import Sequence, Value, Features

# ---- options ----
DO_TRANSLIT = True   # Devanagari -> Roman (IAST)
DO_ASCII    = True   # strip diacritics to ASCII

# ---- transliteration helpers ----
try:
    from indic_transliteration.sanscript import transliterate, SCHEMES
    HAVE_INDIC = True
except Exception:
    HAVE_INDIC = False

def has_deva(s: str) -> bool:
    return any('\u0900' <= ch <= '\u097F' for ch in (s or ""))

def deva_to_roman(s: str) -> str:
    if not s: return s
    if HAVE_INDIC and has_deva(s):
        try:
            return transliterate(s, SCHEMES["devanagari"], SCHEMES["iast"])
        except Exception:
            return s
    return s

def to_ascii(s: str) -> str:
    if s is None: return s
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

# ---- detection helpers ----
TOKEN_COL_PREF = ["tokens", "Tokens", "Sentences", "sentence", "Sentence"]
LABEL_COL_PREF = ["ner_tags", "labels", "BIO", "tags", "Annotated by: Annotator 1", "Predicted tags"]

def _maybe_list_of_dicts(val):
    return isinstance(val, list) and val and isinstance(val[0], dict)

def _classlabel_names(ds_split, label_col: str) -> Optional[List[str]]:
    try:
        feat = ds_split.features[label_col]
        # ner_tags is often Sequence(ClassLabel)
        if isinstance(feat, Sequence) and isinstance(feat.feature, ClassLabel):
            return list(feat.feature.names)
        if isinstance(feat, ClassLabel):
            return list(feat.names)
    except Exception:
        pass
    return None

def _extract_tokens_and_tags(ex: dict, ds_split) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    tok_col = next((c for c in TOKEN_COL_PREF if c in ex), None)
    lab_col = next((c for c in LABEL_COL_PREF  if c in ex), None)
    if tok_col is None or lab_col is None:
        return None, None

    toks_raw = ex[tok_col]
    labs_raw = ex[lab_col]
    label_names = _classlabel_names(ds_split, lab_col)
    
    # Parse JSON or Python repr strings (COMI-LINGUA format stores dict-lists as Python repr strings with single quotes)
    if isinstance(toks_raw, str):
        try:
            # Try JSON first
            import json as json_module
            parsed = json_module.loads(toks_raw)
            if isinstance(parsed, list):
                toks_raw = parsed
        except Exception:
            try:
                # Try Python repr via ast.literal_eval
                import ast as ast_module
                parsed = ast_module.literal_eval(toks_raw)
                if isinstance(parsed, list):
                    toks_raw = parsed
            except Exception:
                pass  # leave as string
                
    if isinstance(labs_raw, str):
        try:
            # Try JSON first
            import json as json_module
            parsed = json_module.loads(labs_raw)
            if isinstance(parsed, list):
                labs_raw = parsed
        except Exception:
            try:
                # Try Python repr via ast.literal_eval (COMI-LINGUA uses this)
                import ast as ast_module
                parsed = ast_module.literal_eval(labs_raw)
                if isinstance(parsed, list):
                    labs_raw = parsed
            except Exception:
                pass  # leave as string

    # Case A: tokens already list[str]
    if isinstance(toks_raw, list) and (not toks_raw or isinstance(toks_raw[0], str)):
        tokens = list(map(str, toks_raw))
        if isinstance(labs_raw, list):
            if label_names and labs_raw and isinstance(labs_raw[0], int):
                tags = [label_names[i] for i in labs_raw]
            else:
                tags = [str(x) for x in labs_raw]
        elif isinstance(labs_raw, str):
            tags = labs_raw.strip().split()
        else:
            return None, None
        return (tokens, tags) if len(tokens) == len(tags) else (None, None)

    # Case B: list[dict] with 'word'/'token' and 'entity'/'label'/'tag'
    if _maybe_list_of_dicts(toks_raw):
        tokens = []
        for d in toks_raw:
            if not isinstance(d, dict):
                return None, None
            w = d.get("word", d.get("token"))
            if w is None: 
                return None, None
            tokens.append(str(w))

        # Try to extract tags from dict list (COMI-LINGUA format: [{'word': '...', 'entity': '...'}])
        if _maybe_list_of_dicts(labs_raw):
            tags = []
            for d in labs_raw:
                if not isinstance(d, dict):
                    return None, None
                ent = d.get("entity", d.get("label", d.get("tag", "O")))
                tags.append(str(ent) if ent else "O")
        elif isinstance(labs_raw, list):
            if label_names and labs_raw and isinstance(labs_raw[0], int):
                tags = [label_names[i] for i in labs_raw]
            else:
                tags = [str(x) for x in labs_raw]
        elif isinstance(labs_raw, str):
            tags = labs_raw.strip().split()
        else:
            return None, None
        return (tokens, tags) if len(tokens) == len(tags) else (None, None)

    # Case C: sentence string + dict-list labels  
    # (COMI-LINGUA: whole sentence + [{'word': '...', 'entity': '...'}])
    # Extract tokens from dict-list 'word' fields instead of whitespace-splitting sentence
    if isinstance(toks_raw, str) and _maybe_list_of_dicts(labs_raw):
        # labels are dict-list, extract words from there
        if _maybe_list_of_dicts(labs_raw):
            tokens = []
            tags = []
            for d in labs_raw:
                if not isinstance(d, dict):
                    return None, None
                w = d.get("word", d.get("token"))
                e = d.get("entity", d.get("label", d.get("tag", "O")))
                if w is None:
                    return None, None
                tokens.append(str(w))
                tags.append(str(e) if e else "O")
            return (tokens, tags)
        return None, None

    # Case D: sentence string + space-separated labels  
    # (fallback to simple whitespace split)
    if isinstance(toks_raw, str):
        tokens = toks_raw.strip().split()
        if isinstance(labs_raw, list):
            if label_names and labs_raw and isinstance(labs_raw[0], int):
                tags = [label_names[i] for i in labs_raw]
            else:
                tags = [str(x) for x in labs_raw]
        elif isinstance(labs_raw, str):
            tags = labs_raw.strip().split()
        else:
            return None, None
        return (tokens, tags) if len(tokens) == len(tags) else (None, None)

    return None, None

def _canon_tag(tag: str) -> str:
    if not tag: return "O"
    t = tag.strip()
    if t.lower() == "o": return "O"
    if t.startswith("b-"): t = "B-" + t[2:]
    if t.startswith("i-"): t = "I-" + t[2:]
    return t

def _emit_jsonl(ds_split: Dataset, dst_path: Path) -> int:
    cnt = 0
    with dst_path.open("w", encoding="utf-8") as f:
        for ex in ds_split:
            tokens, tags = _extract_tokens_and_tags(ex, ds_split)
            if tokens is None or tags is None:
                continue

            if DO_TRANSLIT:
                tokens = [deva_to_roman(t) for t in tokens]
            if DO_ASCII:
                tokens = [to_ascii(t) for t in tokens]

            tags = [_canon_tag(t) for t in tags]
            if not tokens or len(tokens) != len(tags):
                continue

            f.write(json.dumps({"tokens": tokens, "ner_tags": tags}, ensure_ascii=False) + "\n")
            cnt += 1
    return cnt

def main():
    random.seed(42)
    ds = load_dataset("LingoIITGN/COMI-LINGUA", "NER")
    available = list(ds.keys())               # e.g., ['train','test'] or more

    # Build validation if missing
    if "validation" not in available and "train" in available:
        full_train: Dataset = ds["train"]
        n = len(full_train)
        n_val = max(1, int(0.1 * n)) if n > 10 else max(1, n//5)  # small-set guard
        idx = list(range(n))
        random.shuffle(idx)
        val_idx = idx[:n_val]
        trn_idx = idx[n_val:]
        ds = DatasetDict({
            "train":      full_train.select(trn_idx),
            "validation": full_train.select(val_idx),
            **({k:v for k,v in ds.items() if k != "train"})
        })

    outdir = Path("data/raw")
    outdir.mkdir(parents=True, exist_ok=True)

    totals = {}
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        path = outdir / f"{split}.jsonl"
        totals[split] = _emit_jsonl(ds[split], path)

    print("Wrote:")
    for k,v in totals.items():
        print(f"  data/raw/{k}.jsonl ({v} rows)")

if __name__ == "__main__":
    main()
