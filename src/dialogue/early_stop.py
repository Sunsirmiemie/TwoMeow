"""
Entropy early-stop: halt clarification when remaining information gain drops below τ.
PDF §V defines τ=0.3 (normalized entropy threshold).
The Agent supplies configured thresholds while direct callers retain these defaults.
"""
from __future__ import annotations

from .entropy import MIN_POOL_FOR_DYNAMIC, score_attribute

TAU = 0.3  # entropy threshold from PDF §V


def should_stop(
    candidates: list[dict],
    attr_cache: dict[str, dict],
    remaining_attrs: list[str],
    tau: float = TAU,
    min_pool_for_dynamic: int = MIN_POOL_FOR_DYNAMIC,
) -> bool:
    """Return True when max marginal information gain across remaining attrs < τ."""
    if not candidates or not remaining_attrs:
        return True
    max_score = max(
        score_attribute(a, candidates, attr_cache, min_pool_for_dynamic)
        for a in remaining_attrs
    )
    return max_score < tau
