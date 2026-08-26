"""
Rerank pool construction: truncates the candidate pool when few slots confirmed.

When the pool is over-general (few confirmed slots, large pool),
passing fewer candidates to the ranker forces convergence rather than
guessing blindly from a vague pool (MD §III / §VII in-scope).
"""
from __future__ import annotations

_FEW_SLOTS_THRESHOLD = 2
_POOL_SIZE_THRESHOLD = 50
_TRUNCATED_SIZE      = 20


def build_rerank_pool(candidates: list[dict], session) -> list[dict]:
    few_slots = len(session.slots) < _FEW_SLOTS_THRESHOLD
    if few_slots and len(candidates) >= _POOL_SIZE_THRESHOLD:
        return candidates[:_TRUNCATED_SIZE]
    return candidates
