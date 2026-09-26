"""Build a flat CampusMate intent dataset from intents.json."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def build_dataset() -> pd.DataFrame:
    """Flatten data/intents.json into a deterministic text,label CSV."""
    random.seed(SEED)
    source = DATA_DIR / "intents.json"
    with source.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows: list[dict[str, str]] = []
    for item in payload["intents"]:
        label = item["tag"]
        for text in item["patterns"]:
            rows.append({"text": str(text).strip(), "label": label})
    random.shuffle(rows)
    df = pd.DataFrame(rows, columns=["text", "label"])
    out_path = DATA_DIR / "dataset.csv"
    df.to_csv(out_path, index=False)
    counts = Counter(df["label"])
    print("Per-class counts:")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")
    print(f"Total size: {len(df)}")
    print(f"Wrote {out_path}")
    return df


if __name__ == "__main__":
    build_dataset()
