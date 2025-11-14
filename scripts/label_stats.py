# scripts/label_stats.py
import json, collections, sys
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
for split in ("train","validation","test"):
    p = base / f"{split}.jsonl"
    if not p.exists(): 
        continue
    n=0; tag_counts=collections.Counter()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            obj=json.loads(line); n+=1
            tag_counts.update(t for t in obj["ner_tags"] if t != "O")
    print(f"{split:>10}: {n} examples | non-O tags (top 12): {tag_counts.most_common(12)}")
