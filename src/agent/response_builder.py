"""
Query construction and response message generation.
Separated from orchestrator to keep Agent class under the 200-line limit.
"""
from __future__ import annotations

from typing import Any

_HISTORY_NOISE = {"use your judgment", "not quite right"}


def build_query(user_message: str, session: Any) -> str:
    """Build a retrieval query from active, purified conversational evidence."""
    strong_slots = [
        value for key, value in session.slots.items()
        if session.slot_confidence.get(key, 1.0) >= 0.5
    ]
    weak_slots = [
        value for key, value in session.slots.items()
        if session.slot_confidence.get(key, 1.0) < 0.5
    ]
    strong_text = " ".join(strong_slots)
    slot_text = f"{strong_text} {strong_text} {' '.join(weak_slots)}"
    useful_history = [
        t.get("query_text", t["message"])
        for t in session.history[session.context_start_turn:][-3:]
        if not any(noise in t["message"].lower() for noise in _HISTORY_NOISE)
    ]
    current = session.last_query_text or user_message
    return f"{slot_text} {current} {' '.join(useful_history)}".strip()


def build_message(ask_attribute: str | None, ranked: list[dict[str, Any]]) -> str:
    if ask_attribute:
        return f"Here are some options. Could you tell me your preference for {ask_attribute}?"
    if ranked:
        return "Here are my top picks based on your request."
    return "I couldn't find a match. Could you describe what you're looking for?"
