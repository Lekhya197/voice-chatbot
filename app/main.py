"""FastAPI app for CampusMate."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.asr import transcribe_audio_bytes
from app.inference import artifacts_available, model_loaded, num_intents, predict_intent
from app.responses import CONFIDENCE_THRESHOLD, build_response, fallback_response

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "app" / "static"

app = FastAPI(title="CampusMate", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    """Incoming chat payload."""
    text: str = Field(..., max_length=300)


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    """Return service and artifact status."""
    return {"status": "ok", "model_loaded": model_loaded(), "num_intents": num_intents(), "artifacts_available": artifacts_available()}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, object]:
    """Classify a text query and return the chatbot response."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please provide non-empty text.")
    if len(text) > 300:
        raise HTTPException(status_code=413, detail="Text must be 300 characters or fewer.")
    try:
        intent, confidence, probs, top_tokens = predict_intent(text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    top3 = sorted(({"intent": k, "prob": v} for k, v in probs.items()), key=lambda item: item["prob"], reverse=True)[:3]
    is_fallback = confidence < CONFIDENCE_THRESHOLD
    response = fallback_response() if is_fallback else build_response(intent, text)
    return {"intent": intent, "confidence": confidence, "response": response, "top_tokens": top_tokens, "fallback": is_fallback, "top3": top3}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    """Transcribe uploaded audio using faster-whisper tiny.en."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file must be 15 MB or smaller.")
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    text = transcribe_audio_bytes(data, suffix=suffix)
    return {"text": text, "engine": "faster-whisper-tiny.en"}
