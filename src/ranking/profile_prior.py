"""
User profile personalization: boosts candidates matching user preference_tags.
Currently a stub — preference_tags from user_profile are not yet wired into retrieval.
TODO: use preference_tags to re-score or pre-filter candidates in orchestrator.
"""
from __future__ import annotations


def apply_profile_boost(candidates: list[dict], session) -> list[dict]:
    """Stub: returns candidates unchanged until profile integration is implemented."""
    return candidates
