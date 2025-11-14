#!/usr/bin/env python3
import sys, re, unicodedata, collections
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer

BIO_RE = re.compile(r"^(O|[BI]-[A-Za-z][A-Za-z0-9_]*)$")

def has_devanagari(s: str) -> bool:
    return any('\u0900' <= ch <= '\u097F' for ch in s)

def chunk_spans(tokens, tags):
    # reconstruct entity spans from BIO tags for a human check
    spans = []
    cur = None
    for i, t in enumerate(tags):
        if t == "O":
            if cur: spans.append(cur); cur = None
        elif t.startswith("B-"):
            if cur: spans.append(cur)
            cur = {"type": t[2:], "start": i, "end": i+1}
        elif t.startswith("I-"):
            if cur and cur["type"] == t[2:]:
                cur["end"] = i+1
            else:
                # I- without proper B-; treat as new B-
                if cur: spans.append(cur)
                cur = {"type": t[2:], "start": i, "end": i+1}
        else:
            # unexpected tag, dump it as text span
            if cur: spans.append(cur); cur=None
    if cur: spans.append(cur)
    # turn into (type, text) for readability
    out = []
    for sp in spans:
        out.append((sp["type"], " ".join(tokens[sp["start"]:sp["end"]])))
    return out

def main(path="data/processed", model="xlm-roberta-base", show=3):
    ds: DatasetDict = load_from_disk(path)
    print("splits:", {k: len(v) for k,v in ds.items()})

    # 1) structural checks
    for split, d in ds.items():
        bad_len = 0
        bad_bio = 0
        dev_count = 0
        label_counter = collections.Counter()
        token_len = []

        for ex in d:
            toks = ex["tokens"]
            tags = ex["ner_tags"]
            assert isinstance(toks, list) and isinstance(tags, list), "tokens/ner_tags must be lists"
            if len(toks) != len(tags):
                bad_len += 1
            for t in tags:
                if not BIO_RE.match(t):
                    bad_bio += 1
                else:
                    label_counter[t] += 1
            if any(has_devanagari(tok) for tok in toks):
                dev_count += 1
            token_len.append(len(toks))

        print(f"\n[{split}]")
        print("  examples:", len(d))
        print("  bad_length_pairs:", bad_len)
        print("  bad_bio_tags:", bad_bio)
        print("  contains_devanagari_sentences:", dev_count)
        if token_len:
            print("  token_len avg/min/max:",
                  sum(token_len)/len(token_len), min(token_len), max(token_len))
        print("  label freq (top 20):", label_counter.most_common(20))

        # quick peek at a few rows (with reconstructed spans)
        print("  ---- samples ----")
        for i in range(min(show, len(d))):
            ex = d[i]
            toks, tags = ex["tokens"], ex["ner_tags"]
            spans = chunk_spans(toks, tags)
            print(f"   #{i} toks={len(toks)}")
            print("    text:", " ".join(toks)[:200], "...")
            print("    spans:", spans[:8])

    # 2) tokenizer alignment sanity on a tiny sample
    tok = AutoTokenizer.from_pretrained(model)
    sample = ds["train"][:5] if "train" in ds else None
    if sample:
        enc = tok(sample["tokens"], is_split_into_words=True, truncation=True, max_length=256)
        # ensure word_id extraction works
        wid = tok(sample["tokens"][0], is_split_into_words=True).word_ids()
        print("\n[tokenizer sanity] first sample word_ids length:", len(wid))

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    main(p)
