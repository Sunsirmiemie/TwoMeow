"""
Failure analysis: find sessions where the target is at rank 6-10 (reranker opportunity).
Current results show 31 such sessions — enabling the LLM reranker should move these to top-5.
"""
from __future__ import annotations


def find_near_misses(result: dict, lo: int = 6, hi: int = 10) -> list[dict]:
    """Sessions where first hit rank is in [lo, hi] — reranker uplift targets."""
    return [
        s for s in result.get("sessions", [])
        if s.get("first_hit_rank") and lo <= s["first_hit_rank"] <= hi
    ]


def find_complete_misses(result: dict) -> list[dict]:
    """Sessions where target was never in top-10."""
    return [
        s for s in result.get("sessions", [])
        if not s.get("first_hit_rank")
    ]
