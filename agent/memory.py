"""
Session memory: tracks scenario state, dialog slots, and turn history.
"""
from __future__ import annotations


class SessionMemory:
    def __init__(self, user_profile: dict):
        self.user_profile = user_profile
        self.slots: dict[str, str] = {}

        # Four-scenario state (buying | browsing | intent_override | boundary | unknown)
        self.scenario_type: str = "unknown"
        self.boundary_detected: bool = False   # saw "please use your judgment"
        self.override_applied: bool = False    # saw "ignore my earlier preference"

        # Track which attributes we've already asked to avoid repeats
        self.asked_attributes: list[str] = []

        self.history: list[dict] = []

    def add_turn(self, message: str, recommendations: list) -> None:
        self.history.append({
            "turn": len(self.history) + 1,
            "message": message,
            "top_asin": recommendations[0]["parent_asin"] if recommendations else None,
        })

    def accumulated_text(self) -> str:
        return " ".join(t["message"] for t in self.history)

    @property
    def turn_count(self) -> int:
        return len(self.history)

    def retrieval_track(self) -> str:
        """Map scenario to retrieval track (buying = hard-filter, browsing = diverse)."""
        if self.scenario_type in ("buying",) or self.override_applied:
            return "buying"
        return "browsing"

    def preference_tags(self) -> list[str]:
        return self.user_profile.get("preference_tags", [])
