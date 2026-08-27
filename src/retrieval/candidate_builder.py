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


def build_rerank_pool(
    candidates: list[dict],
    session,
    few_slots_threshold: int = _FEW_SLOTS_THRESHOLD,
    pool_size_threshold: int = _POOL_SIZE_THRESHOLD,
    truncated_size: int = _TRUNCATED_SIZE,
) -> list[dict]:
    few_slots = len(session.slots) < few_slots_threshold
    if few_slots and len(candidates) >= pool_size_threshold:
        return candidates[:truncated_size]
    return candidates
