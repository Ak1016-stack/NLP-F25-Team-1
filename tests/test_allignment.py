# -*- coding: utf-8 -*-
import os
import pytest

from transformers import AutoTokenizer
from src.data_prep import align_labels_with_tokenizer

MODEL_NAME = os.environ.get("NER_TEST_MODEL", "xlm-roberta-base")

@pytest.fixture(scope="session")
def tok():
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    except Exception as e:
        pytest.skip(f"Tokenizer download failed ({e}); set NER_TEST_MODEL or ensure internet.")

def _labels_like(tokens, pattern="O"):
    # produce a same-length BIO tag list, with last token as B-MISC for sanity
    labs = [pattern] * len(tokens)
    if tokens:
        labs[-1] = "B-MISC"
    return labs

def test_alignment_basic(tok):
    examples = [
        {"tokens": ["hello", "कोड"], "ner_tags": ["O", "B-MISC"]},
        {"tokens": ["NYUAD", "rocks", "!"], "ner_tags": ["B-ORG", "O", "O"]},
    ]
    enc = align_labels_with_tokenizer(examples, tok, max_length=32, add_script_id=False)
    assert len(enc) == 2
    for row, src in zip(enc, examples):
        # labels should be present and contain at least one real tag string
        assert "labels" in row and any(isinstance(l, str) for l in row["labels"])
        # keep original for inspection
        assert row["tokens"] == src["tokens"]
        assert row["ner_tags"] == src["ner_tags"]

def test_alignment_subwords_and_masking(tok):
    examples = [{"tokens": ["unbelievable", "NYUAD"], "ner_tags": ["O", "B-ORG"]}]
    enc = align_labels_with_tokenizer(examples, tok, max_length=32, add_script_id=False)[0]

    word_ids = tok(examples[0]["tokens"], is_split_into_words=True).word_ids()
    labels = enc["labels"]

    # -100 for special tokens (None), first subtoken keeps label, subsequent subtokens masked
    prev = None
    for wid, lab in zip(word_ids, labels):
        if wid is None:
            assert lab == -100
        else:
            if wid != prev:
                # first subtoken position should carry either "O" or "B-ORG"
                assert lab in ("O", "B-ORG")
            else:
                assert lab == -100
        prev = wid

def test_alignment_with_script_ids(tok):
    examples = [{"tokens": ["code", "कोड", "mix"], "ner_tags": _labels_like(["code","कोड","mix"])}]
    enc = align_labels_with_tokenizer(examples, tok, max_length=32, add_script_id=True)[0]

    assert "script_ids" in enc
    assert len(enc["script_ids"]) == len(enc["labels"])

    # first subtoken of "कोड" should carry a 1 (Devanagari), others -100
    wid = tok(examples[0]["tokens"], is_split_into_words=True).word_ids()
    prev = None
    for i, w in enumerate(wid):
        if w is None:
            assert enc["script_ids"][i] == -100
        elif w == 1 and w != prev:  # token index 1 is "कोड"
            assert enc["script_ids"][i] == 1
        elif w == 1 and w == prev:
            assert enc["script_ids"][i] == -100
        prev = w

def test_mismatched_lengths_raises(tok):
    bad = [{"tokens": ["only", "two"], "ner_tags": ["O"]}]
    with pytest.raises(ValueError):
        align_labels_with_tokenizer(bad, tok, max_length=16, add_script_id=False)
