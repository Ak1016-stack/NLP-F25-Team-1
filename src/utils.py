# -*- coding: utf-8 -*-
from typing import List, Dict, Any
import random
import numpy as np

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def build_label_list_from_dataset(dataset) -> List[str]:
    """
    Extract unique tag strings from dataset['train']['labels'] which may contain tag strings plus -100.
    Ensures 'O' appears first for readability (optional).
    """
    tags = set()
    for ex in dataset["train"]:
        for lab in ex["labels"]:
            if isinstance(lab, str):
                tags.add(lab)
    if not tags:
        # if already ints, try to infer from model config in train.py
        raise ValueError("No string labels found; expected string BIO tags in 'labels'.")
    label_list = sorted(tags)
    if "O" in label_list:
        label_list.remove("O")
        label_list = ["O"] + label_list
    return label_list

def to_id_labels(ex_labels: List[Any], label2id: Dict[str, int]) -> List[int]:
    """
    Convert a sequence of tag strings / -100 to ID sequence.
    """
    converted = []
    for lab in ex_labels:
        if lab == -100:
            converted.append(-100)
        elif isinstance(lab, str):
            converted.append(label2id[lab])
        else:
            converted.append(int(lab))
    return converted

def infer_label_mappings(dataset) -> (List[str], Dict[str,int], Dict[int,str]):
    label_list = build_label_list_from_dataset(dataset)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    return label_list, label2id, id2label
