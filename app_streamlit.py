"""Streamlit frontend for CampusMate voice chatbot (clean single-column demo).

Features:
- Single centered column layout
- One recorder (audio_recorder_streamlit)
- One recognized-speech display (hidden until transcript available)
- One chat area using `st.chat_message`
- Sidebar for controls (server ASR toggle, mute, clear chat, description)
- gTTS-based TTS playback via in-memory MP3 and `st.audio`
- Graceful lazy imports for heavy ML deps
"""
from __future__ import annotations

import io
import html
import base64
import streamlit.components.v1 as components
from typing import Optional

import streamlit as st

# Reuse existing project modules where possible
from app.responses import build_response, fallback_response, CONFIDENCE_THRESHOLD
from app.asr import transcribe_audio_bytes

# Layout config
st.set_page_config(page_title="CampusMate", layout="wide")

# --- Helpers -----------------------------------------------------------------


def tts_play_bytes(text: str) -> Optional[bytes]:
    """Return MP3 bytes for `text` using gTTS, or None if gTTS isn't installed.
    Keeps everything in-memory.
    """
    try:
        from gtts import gTTS
    except Exception:
        return None
    buf = io.BytesIO()
    try:
        tts = gTTS(text, lang="en")
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def safe(text: str) -> str:
    return html.escape(text)


# --- Session state init -----------------------------------------------------
if "history" not in st.session_state:
    # list[tuple[user_text, intent, conf_pct, top_tokens, response_text]]
    st.session_state.history = []
if "recognized" not in st.session_state:
    st.session_state.recognized = ""
if "last_audio_len" not in st.session_state:
    st.session_state.last_audio_len = 0


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("CampusMate")
    st.markdown("A demo voice-enabled campus helpdesk chatbot.")
    use_server_asr = st.checkbox("Use server ASR (Whisper)", value=True)
    mute_reply = st.checkbox("Mute voice reply", value=False)
    if st.button("Clear chat"):
        st.session_state.history = []
    st.markdown("---")
    st.caption("Model artifacts must be present in `artifacts/` for inference.")

# persist mute flag so other functions can read current value
st.session_state["mute_reply"] = mute_reply


# --- Main centered column --------------------------------------------------
cols = st.columns([1, 3, 1])
center = cols[1]

with center:
    st.markdown("# CampusMate — Voice Helpdesk")
    st.markdown("_Tap the mic, speak your question, or type below._")

    # container card
    card = st.container()
    with card:
        st.markdown(
            "<div style='padding:16px; border-radius:12px; background:#0b1220;'>", unsafe_allow_html=True)
        st.markdown("<div style='color:#e6f6ff'>", unsafe_allow_html=True)

        # Recorder
        st.markdown("### Tap to speak")
        try:
            from audio_recorder_streamlit import audio_recorder
            recorder_available = True
        except Exception:
            audio_recorder = None
            recorder_available = False

        # The audio_recorder returns bytes or dict; show the widget and process when present
        if recorder_available:
            audio_bytes = audio_recorder()
        else:
            audio_bytes = None
            st.warning(
                "`audio-recorder-streamlit` not installed — install to enable browser recording.")

        # Recording indicator: the component itself usually handles UI.
        # Show transcribing spinner when we detect new audio bytes length change.
        if audio_bytes:
            # extract raw bytes
            if isinstance(audio_bytes, dict) and "audio" in audio_bytes:
                raw = audio_bytes["audio"]
                suffix = audio_bytes.get("format", ".webm")
            else:
                raw = audio_bytes
                suffix = ".webm"

            # simple change-detection to avoid reprocessing same audio repeatedly
            if hasattr(raw, "__len__") and len(raw) != st.session_state.last_audio_len:
                st.session_state.last_audio_len = len(raw)
                with st.spinner("Transcribing..."):
                    try:
                        transcript = transcribe_audio_bytes(raw, suffix=suffix)
                        st.session_state.recognized = transcript or ""
                    except ModuleNotFoundError:
                        st.error(
                            "ASR backend not available. Install faster-whisper to enable transcription.")
                    except Exception as e:
                        st.error(f"Transcription error: {e}")

        # Recognized speech display (only if we have a transcript)
        if st.session_state.recognized:
            st.markdown("**Recognized:**")
            st.markdown(
                f"<div style='padding:10px; background:#0f1724; border-radius:8px; color:#e6f6ff'>{safe(st.session_state.recognized)}</div>", unsafe_allow_html=True)

        st.markdown(
            "<div style='margin-top:8px; color:#9ca3af'>or type your question below</div>", unsafe_allow_html=True)

        # Fallback text input (visible label to avoid accessibility warning)
        typed = st.text_input(
            "Type a campus question...", key="chat_fallback", placeholder="Type a campus question and press Enter")
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Process input (either typed or recognized) when available
    def process_and_respond(text: str) -> None:
        if not text:
            return
        # inference (lazy import)
        try:
            from app.inference import predict_intent
        except ModuleNotFoundError:
            intent = "(no-model)"
            conf = 0.0
            top_tokens = []
            response = fallback_response()
        else:
            try:
                intent, conf, probs, top_tokens = predict_intent(text)
                conf_pct = round(conf * 100, 2)
                is_fallback = conf < CONFIDENCE_THRESHOLD
                response = fallback_response() if is_fallback else build_response(intent, text)
                intent = intent
                conf = conf_pct
            except Exception as e:
                response = fallback_response()
                intent = "(error)"
                conf = 0.0
                top_tokens = []
        # TTS generation (in-memory) unless muted; store bytes with history entry
        tts_bytes = None
        if not st.session_state.get("mute_reply", False):
            # log attempt
            print("[TTS] attempting to generate speech for response")
            st.session_state["last_tts_attempt"] = response
            mp3 = tts_play_bytes(response)
            if mp3:
                tts_bytes = mp3
                print("[TTS] generated MP3 bytes (size=", len(mp3), ")")
                st.session_state["last_tts_status"] = f"ok ({len(mp3)} bytes)"
            else:
                print("[TTS] gTTS not available or generation failed")
                st.session_state["last_tts_status"] = "missing-or-failed"

        # append to history (include tts bytes as 6th element)
        st.session_state.history.append(
            (text, intent, conf, top_tokens, response, tts_bytes))

    # If user typed and pressed Enter (text_input returns new value), process it
    if typed:
        process_and_respond(typed)
        # clear typed input for UX
        st.session_state.chat_fallback = ""

    # Also process recognized transcript if present and not already in history
    if st.session_state.recognized:
        # avoid duplicating if same as last user entry
        last_user = st.session_state.history[-1][0] if st.session_state.history else None
        if st.session_state.recognized != last_user:
            process_and_respond(st.session_state.recognized)

    # Chat history area (use chat_message for cleaner look)
    st.markdown("### Conversation")
    chat_box = st.container()
    with chat_box:
        for entry in st.session_state.history:
            # entry: (user_text, intent, conf, top_tokens, response, tts_bytes)
            if len(entry) == 6:
                user_text, intent, conf, top_tokens, response, tts_bytes = entry
            else:
                user_text, intent, conf, top_tokens, response = entry
                tts_bytes = None

            st.chat_message("user").write(user_text)
            st.chat_message("assistant").write(response)
            # show audio player if TTS bytes exist and not muted
            if tts_bytes and not st.session_state.get("mute_reply", False):
                try:
                    st.audio(tts_bytes, format="audio/mp3")
                except Exception:
                    # fallback: embed base64 HTML audio
                    try:
                        b64 = base64.b64encode(tts_bytes).decode("ascii")
                        audio_html = f"<audio controls src='data:audio/mp3;base64,{b64}'></audio>"
                        components.html(audio_html, height=50)
                    except Exception:
                        st.write("Audio playback not available")

# --- Footer styling ---------------------------------------------------------
st.markdown(
    "<style>"
    "body .stApp { background:#071026; }"
    ".stContainer { max-width:900px; margin: 0 auto; }"
    "</style>",
    unsafe_allow_html=True,
)


if __name__ == "__main__":
    # Streamlit runs the file top-to-bottom; main behavior already executed above.
    pass
