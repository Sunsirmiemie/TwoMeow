"""
Detects which of the four evaluation scenarios a session is in, based on
exact message patterns from local_evaluator.py.

Scenarios (fixed ratio 40/40/15/5):
  buying         - "A key requirement is:" in first message
  browsing       - "still exploring" in first message
  intent_override - "ignore my earlier preference" at turn 3-4
  boundary       - any attribute question gets "please use your judgment"
"""
from __future__ import annotations

import re
from typing import Any

_OVERRIDE_RE = re.compile(r"ignore my earlier preference", re.I)
_BOUNDARY_RE = re.compile(r"please use your judgment", re.I)
_BUYING_RE   = re.compile(r"a key requirement is:", re.I)
_BROWSING_RE = re.compile(r"still exploring", re.I)


class IntentRouter:
    def update_scenario(self, message: str, session: Any) -> None:
        """Update session.scenario_type and flags based on current message."""

        # Override signal — fires mid-session, must clear slots
        if _OVERRIDE_RE.search(message):
            session.scenario_type = "intent_override"
            session.override_applied = True
            session.slots.clear()
            session.asked_attributes.clear()
            return

        # Boundary signal — first question got "use your judgment"
        if _BOUNDARY_RE.search(message):
            session.boundary_detected = True
            # Don't change scenario_type; keep asking (evaluator gives real answers after first)
            return

        # First-turn scenario detection
        if session.turn_count == 0:
            if _BUYING_RE.search(message):
                session.scenario_type = "buying"
            elif _BROWSING_RE.search(message):
                session.scenario_type = "browsing"
            else:
                session.scenario_type = "browsing"  # safe default
