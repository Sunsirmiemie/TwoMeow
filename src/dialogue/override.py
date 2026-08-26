"""
Intent override and boundary detection from evaluator message patterns.
"""
from __future__ import annotations

import re

OVERRIDE_RE = re.compile(r"ignore my earlier preference", re.I)
BOUNDARY_RE = re.compile(r"please use your judgment", re.I)


def apply_override(message: str, session) -> bool:
    """Clear slots and mark override when override phrase detected. Returns True if fired."""
    if OVERRIDE_RE.search(message):
        session.scenario_type = "intent_override"
        session.override_applied = True
        session.slots.clear()
        session.asked_attributes.clear()
        return True
    return False


def detect_boundary(message: str, session) -> bool:
    """Mark boundary flag when boundary phrase detected. Returns True if fired."""
    if BOUNDARY_RE.search(message):
        session.boundary_detected = True
        return True
    return False
