"""
Query construction and response message generation.
Separated from orchestrator to keep Agent class under the 200-line limit.
"""
from __future__ import annotations

_HISTORY_NOISE = {"use your judgment", "not quite right"}


def build_query(user_message: str, session) -> str:
    """Build retrieval query: slot values (doubled for BM25 weight) + filtered history + message."""
    slot_text = " ".join(session.slots.values())
    useful_history = [
        t["message"] for t in session.history[-3:]
        if not any(noise in t["message"] for noise in _HISTORY_NOISE)
    ]
    return f"{slot_text} {slot_text} {user_message} {' '.join(useful_history)}".strip()


def build_message(ask_attribute: str | None, ranked: list) -> str:
    if ask_attribute:
        return f"Here are some options. Could you tell me your preference for {ask_attribute}?"
    if ranked:
        return "Here are my top picks based on your request."
    return "I couldn't find a match. Could you describe what you're looking for?"
