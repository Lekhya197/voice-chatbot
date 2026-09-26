"""Train CampusMate intent classifiers and save reproducible artifacts."""
from __future__ import annotations

import json
import os
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.model import CampusIntentClassifier, count_parameters

SEED = 42
MAX_LEN = 24
MIN_FREQ = 1
BATCH_SIZE = 32
ARTIFACTS = ROOT / "artifacts"
DATASET = ROOT / "data" / "dataset.csv"
PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("'", ""))


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def preprocess(text: str) -> str:
    """Lowercase, strip punctuation except apostrophes, and collapse whitespace."""
    text = text.lower().translate(PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text with simple whitespace splitting."""
    return preprocess(text).split()


def build_vocab(texts: Iterable[str], min_freq: int = MIN_FREQ) -> dict[str, int]:
    """Build token vocabulary from training texts only."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in sorted(counts.items()):
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int = MAX_LEN) -> list[int]:
    """Encode text with post-truncation and post-padding."""
    ids = [vocab.get(tok, 1) for tok in tokenize(text)][:max_len]
    return ids + [0] * (max_len - len(ids))


class IntentDataset(Dataset):
    """Tensor dataset for intent classification."""

    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int]) -> None:
        self.x = torch.tensor([encode_text(t, vocab) for t in texts], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def prepare_data(subset_per_class: int | None = None):
    """Load CSV, stratify 70/15/15, and build vocabulary from train split."""
    df = pd.read_csv(DATASET)
    if subset_per_class:
        df = df.groupby("label", group_keys=False).head(subset_per_class).reset_index(drop=True)
    labels = sorted(df["label"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}
    y = df["label"].map(label_to_id).tolist()
    x_train, x_temp, y_train, y_temp = train_test_split(df["text"].tolist(), y, test_size=0.30, stratify=y, random_state=SEED)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)
    vocab = build_vocab(x_train)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), vocab, label_to_id


def run_epoch(model, loader, criterion, optimizer=None, device="cpu") -> tuple[float, float]:
    """Run one training or evaluation epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    losses, preds, gold = [], [], []
    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
            losses.append(loss.item() * len(yb))
            preds.extend(logits.argmax(1).cpu().tolist())
            gold.extend(yb.cpu().tolist())
    return sum(losses) / len(gold), accuracy_score(gold, preds)


def predict_all(model, loader, device="cpu") -> tuple[list[int], list[int]]:
    """Return predicted and gold labels for a loader."""
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            gold.extend(yb.tolist())
    return preds, gold


def train_model(mode: str, epochs: int, patience: int, subset_per_class: int | None = None, save_main: bool = False) -> dict:
    """Train one model variant and optionally save deployable artifacts."""
    set_seed()
    ARTIFACTS.mkdir(exist_ok=True)
    (x_train, y_train), (x_val, y_val), (x_test, y_test), vocab, label_to_id = prepare_data(subset_per_class)
    train_ds, val_ds, test_ds = IntentDataset(x_train, y_train, vocab), IntentDataset(x_val, y_val, vocab), IntentDataset(x_test, y_test, vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    config = {"vocab_size": len(vocab), "num_classes": len(label_to_id), "embedding_dim": 128, "conv_filters": 100, "lstm_hidden": 128, "mode": mode}
    checkpoint_config = {**config, "max_len": MAX_LEN}
    model = CampusIntentClassifier(**config)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_f1, best_state, waited = -1.0, None, 0
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        val_preds, val_gold = predict_all(model, val_loader)
        va_f1 = f1_score(val_gold, val_preds, average="macro", zero_division=0)
        scheduler.step(va_loss)
        for k, v in [("train_loss", tr_loss), ("val_loss", va_loss), ("train_acc", tr_acc), ("val_acc", va_acc), ("val_macro_f1", va_f1)]:
            hist[k].append(v)
        print(f"{mode} epoch {epoch:02d}: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_loss={va_loss:.4f} val_acc={va_acc:.3f} val_macro_f1={va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1, waited = va_f1, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    test_preds, test_gold = predict_all(model, test_loader)
    id_to_label = {v: k for k, v in label_to_id.items()}
    names = [id_to_label[i] for i in range(len(id_to_label))]
    report = classification_report(test_gold, test_preds, target_names=names, output_dict=True, zero_division=0)
    with torch.no_grad():
        sample = test_ds.x[: min(64, len(test_ds))]
        start = time.perf_counter()
        for _ in range(20):
            _ = model(sample)
        latency = ((time.perf_counter() - start) / (20 * len(sample))) * 1000
    result = {"mode": mode, "accuracy": accuracy_score(test_gold, test_preds), "macro_f1": f1_score(test_gold, test_preds, average="macro", zero_division=0), "weighted_f1": f1_score(test_gold, test_preds, average="weighted", zero_division=0), "report": report, "latency_ms_sample_cpu": latency, "params": count_parameters(model), "history": hist, "epochs_ran": len(hist["train_loss"])}
    if save_main:
        (ARTIFACTS / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
        (ARTIFACTS / "label_encoder.json").write_text(json.dumps({"label_to_id": label_to_id, "id_to_label": {str(k): v for k, v in id_to_label.items()}}, indent=2), encoding="utf-8")
        torch.save({"model_state_dict": best_state, "config": checkpoint_config, "label_to_id": label_to_id, "max_len": MAX_LEN, "preprocessing": "lowercase; strip punctuation except apostrophes; whitespace tokenizer"}, ARTIFACTS / "model.pt")
        plt.figure(figsize=(8, 4))
        plt.plot(hist["train_loss"], label="train loss")
        plt.plot(hist["val_loss"], label="val loss")
        plt.plot(hist["train_acc"], label="train acc")
        plt.plot(hist["val_acc"], label="val acc")
        plt.legend(); plt.tight_layout(); plt.savefig(ARTIFACTS / "training_curves.png", dpi=160); plt.close()
    return result


def main() -> None:
    """Train hybrid and ablation models, then save metrics.json."""
    if not DATASET.exists():
        raise SystemExit("Run python data/build_dataset.py first")
    hybrid = train_model("hybrid", epochs=60, patience=8, save_main=True)
    cnn = train_model("cnn", epochs=30, patience=6)
    bilstm = train_model("bilstm", epochs=30, patience=6)
    metrics = {k: hybrid[k] for k in ["accuracy", "macro_f1", "weighted_f1", "latency_ms_sample_cpu", "params", "epochs_ran"]}
    metrics["per_class"] = {k: v for k, v in hybrid["report"].items() if isinstance(v, dict) and k not in {"macro avg", "weighted avg"}}
    metrics["ablation"] = {"cnn_only": {"accuracy": cnn["accuracy"], "macro_f1": cnn["macro_f1"], "params": cnn["params"]}, "bilstm_only": {"accuracy": bilstm["accuracy"], "macro_f1": bilstm["macro_f1"], "params": bilstm["params"]}, "hybrid": {"accuracy": hybrid["accuracy"], "macro_f1": hybrid["macro_f1"], "params": hybrid["params"]}}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
