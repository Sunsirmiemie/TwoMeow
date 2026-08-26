"""
Entropy early-stop: halt clarification when remaining information gain drops below τ.
PDF §V defines τ=0.3 (normalized entropy threshold).
Not yet wired into orchestrator — stub ready for integration.
"""
from __future__ import annotations

from .entropy import score_attribute

TAU = 0.3  # entropy threshold from PDF §V


def should_stop(
    candidates: list[dict],
    attr_cache: dict[str, dict],
    remaining_attrs: list[str],
) -> bool:
    """Return True when max marginal information gain across remaining attrs < τ."""
    if not candidates or not remaining_attrs:
        return True
    max_score = max(score_attribute(a, candidates, attr_cache) for a in remaining_attrs)
    return max_score < TAU
