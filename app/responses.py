"""Response templates and fallback handling for CampusMate."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_THRESHOLD = 0.55
_rng = random.Random(42)

FALLBACK_RESPONSE = (
    "I'm not sure I understood. I can help with campus questions such as: "
    "what time does the library close, how do I pay fees, how to connect campus Wi-Fi, "
    "or when is the exam schedule available?"
)


def load_templates() -> dict[str, list[str]]:
    """Load intent response templates from data/intents.json."""
    payload = json.loads((ROOT / "data" / "intents.json").read_text(encoding="utf-8"))
    return {item["tag"]: item["responses"] for item in payload["intents"]}

TEMPLATES = load_templates()


def build_response(intent: str, text: str) -> str:
    """Return a light templated response for a predicted intent."""
    templates = TEMPLATES.get(intent)
    if not templates:
        return FALLBACK_RESPONSE
    response = _rng.choice(templates)
    if intent in {"library_timings", "mess_menu", "transport_bus"} and "today" in text.lower():
        response += " For today-specific changes, please verify the latest campus notice."
    return response


def fallback_response() -> str:
    """Return the graceful fallback response."""
    return FALLBACK_RESPONSE
