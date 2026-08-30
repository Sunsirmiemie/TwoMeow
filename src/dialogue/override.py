"""
Intent override and boundary detection from evaluator message patterns.
"""
from __future__ import annotations

import re
from typing import Any

OVERRIDE_RE = re.compile(r"ignore my earlier preference", re.I)
BOUNDARY_RE = re.compile(r"please use your judgment", re.I)


def apply_override(message: str, session: Any) -> bool:
    """Start selective override: replace conflicts and softly retain unrelated evidence."""
    if OVERRIDE_RE.search(message):
        # "Ignore my earlier preference" changes preference evidence, not the
        # stable product class the user is still shopping for.
        use_purification = getattr(session, "use_reply_purification", True)
        category = session.slots.get("category") if use_purification else None
        category_confidence = session.slot_confidence.get("category")
        category_turn = session.slot_turns.get("category")
        if use_purification:
            session.override_snapshot = {
                "slots": dict(session.slots),
                "confidence": dict(session.slot_confidence),
                "turns": dict(session.slot_turns),
            }
        else:
            session.override_snapshot = None
        session.scenario_type = "intent_override"
        session.override_applied = True
        session.slots.clear()
        session.negative_slots.clear()
        session.no_preference_slots.clear()
        session.slot_confidence.clear()
        session.slot_turns.clear()
        if category:
            session.slots["category"] = category
            if category_confidence is not None:
                session.slot_confidence["category"] = category_confidence
            if category_turn is not None:
                session.slot_turns["category"] = category_turn
        if use_purification:
            session.context_start_turn = session.turn_count
        session.last_query_text = ""
        session.asked_attributes.clear()
        session.other_asked = False
        session.last_reply_new_info = True
        session.no_info_streak = 0
        return True
    return False


def detect_boundary(message: str, session: Any) -> bool:
    """Mark boundary flag when boundary phrase detected. Returns True if fired."""
    if BOUNDARY_RE.search(message):
        session.boundary_detected = True
        return True
    return False
