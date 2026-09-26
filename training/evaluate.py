"""Evaluate the saved CampusMate checkpoint and draw a confusion matrix."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from training.model import CampusIntentClassifier
from training.train import ARTIFACTS, IntentDataset, prepare_data


def evaluate() -> None:
    """Load artifacts/model.pt, print a report, and save confusion_matrix.png."""
    checkpoint = torch.load(ARTIFACTS / "model.pt", map_location="cpu", weights_only=True)
    labels_payload = json.loads((ARTIFACTS / "label_encoder.json").read_text(encoding="utf-8"))
    vocab = json.loads((ARTIFACTS / "vocab.json").read_text(encoding="utf-8"))
    _, _, (x_test, y_test), _, _ = prepare_data()
    config = dict(checkpoint["config"])
    config.pop("max_len", None)
    model = CampusIntentClassifier(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(IntentDataset(x_test, y_test, vocab), batch_size=64)
    preds, gold = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds.extend(logits.argmax(1).tolist())
            gold.extend(yb.tolist())
    id_to_label = {int(k): v for k, v in labels_payload["id_to_label"].items()}
    names = [id_to_label[i] for i in range(len(id_to_label))]
    print(classification_report(gold, preds, target_names=names, zero_division=0))
    cm = confusion_matrix(gold, preds, labels=list(range(len(names))))
    fig, ax = plt.subplots(figsize=(12, 12))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, xticks_rotation=90, colorbar=False)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "confusion_matrix.png", dpi=170)
    plt.close(fig)
    print(f"Saved {ARTIFACTS / 'confusion_matrix.png'}")


if __name__ == "__main__":
    evaluate()
