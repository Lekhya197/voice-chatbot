"""Lazy faster-whisper wrapper for server-side ASR fallback."""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

_model = None
_lock = threading.Lock()


def get_model():
    """Load faster-whisper tiny.en once, guarded by a lock."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    return _model


def transcribe_audio_bytes(data: bytes, suffix: str = ".webm") -> str:
    """Transcribe an uploaded audio blob and return plain text."""
    model = get_model()
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        segments, _ = model.transcribe(tmp.name, language="en", beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()
