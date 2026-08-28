"""
First-turn scenario classification: buying vs. browsing.
Override and boundary signals are delegated to src/dialogue/override.py.
"""
from __future__ import annotations

import re
from typing import Any

from ..dialogue.override import apply_override, detect_boundary

_BUYING_RE   = re.compile(r"a key requirement is:", re.I)
_BROWSING_RE = re.compile(r"still exploring", re.I)


class IntentRouter:
    def update_scenario(self, message: str, session: Any) -> None:
        """Update session.scenario_type and flags based on the current message."""
        if apply_override(message, session):
            return
        if detect_boundary(message, session):
            return
        if session.turn_count == 0:
            session.scenario_type = "buying" if _BUYING_RE.search(message) else "browsing"
