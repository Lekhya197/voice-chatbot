"""Inference utilities for CampusMate intent classification."""
from __future__ import annotations

import json
import re
import string
import threading
from pathlib import Path
from typing import Any

import torch

from training.model import CampusIntentClassifier

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("'", ""))
_model: CampusIntentClassifier | None = None
_vocab: dict[str, int] | None = None
_id_to_label: dict[int, str] | None = None
_max_len = 24
_lock = threading.Lock()


def preprocess(text: str) -> str:
    """Normalize text exactly as during training."""
    return re.sub(r"\s+", " ", text.lower().translate(PUNCT_TABLE)).strip()


def tokenize(text: str) -> list[str]:
    """Whitespace tokenize normalized text."""
    return preprocess(text).split()


def encode(text: str, vocab: dict[str, int], max_len: int) -> tuple[list[int], list[str]]:
    """Encode text and return padded IDs plus visible tokens."""
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(tok, 1) for tok in tokens]
    return ids + [0] * (max_len - len(ids)), tokens


def artifacts_available() -> bool:
    """Return whether trained model artifacts are present."""
    return all((ARTIFACTS / name).exists() for name in ["model.pt", "vocab.json", "label_encoder.json"])


def load_model() -> None:
    """Lazy-load the saved PyTorch model and metadata."""
    global _model, _vocab, _id_to_label, _max_len
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        if not artifacts_available():
            raise FileNotFoundError("Model artifacts are missing. Run data/build_dataset.py and training/train.py first.")
        checkpoint = torch.load(ARTIFACTS / "model.pt", map_location="cpu", weights_only=True)
        _vocab = json.loads((ARTIFACTS / "vocab.json").read_text(encoding="utf-8"))
        labels_payload = json.loads((ARTIFACTS / "label_encoder.json").read_text(encoding="utf-8"))
        _id_to_label = {int(k): v for k, v in labels_payload["id_to_label"].items()}
        _max_len = int(checkpoint.get("max_len", checkpoint["config"].get("max_len", 24)))
        config = dict(checkpoint["config"])
        config.pop("max_len", None)
        model = CampusIntentClassifier(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _model = model


def model_loaded() -> bool:
    """Return true when the inference singleton is loaded."""
    return _model is not None


def num_intents() -> int:
    """Return the number of known intent classes."""
    if _id_to_label is not None:
        return len(_id_to_label)
    if (ARTIFACTS / "label_encoder.json").exists():
        data = json.loads((ARTIFACTS / "label_encoder.json").read_text(encoding="utf-8"))
        return len(data["id_to_label"])
    return 0


def predict_intent(text: str) -> tuple[str, float, dict[str, float], list[str]]:
    """Predict the intent, confidence, class probabilities, and top attention tokens."""
    load_model()
    assert _model is not None and _vocab is not None and _id_to_label is not None
    ids, tokens = encode(text, _vocab, _max_len)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits, attn = _model(x, return_attention=True)
        probs_tensor = torch.softmax(logits, dim=1).squeeze(0)
    probs = {_id_to_label[i]: float(probs_tensor[i]) for i in range(len(_id_to_label))}
    best_id = int(torch.argmax(probs_tensor).item())
    token_scores = attn.squeeze(0)[: len(tokens)].tolist()
    top_tokens = [tok for tok, _ in sorted(zip(tokens, token_scores), key=lambda p: p[1], reverse=True)[:3]]
    return _id_to_label[best_id], float(probs_tensor[best_id]), probs, top_tokens
