"""
Entropy early-stop: halt clarification when remaining information gain drops below τ.
PDF §V defines τ=0.3 (normalized entropy threshold).
The Agent supplies configured thresholds while direct callers retain these defaults.
"""
from __future__ import annotations

from .entropy import MIN_POOL_FOR_DYNAMIC, score_attribute

TAU = 0.3  # entropy threshold from PDF §V


def evaluate_stop(
    candidates: list[dict],
    attr_cache: dict[str, dict],
    remaining_attrs: list[str],
    tau: float = TAU,
    min_pool_for_dynamic: int = MIN_POOL_FOR_DYNAMIC,
) -> dict:
    """Return the stop decision and the exact scores used to make it."""
    if not candidates or not remaining_attrs:
        return {
            "triggered": True,
            "scores": {},
            "max_score": None,
            "tau": tau,
        }
    scores = {
        attr: score_attribute(
            attr,
            candidates,
            attr_cache,
            min_pool_for_dynamic,
        )
        for attr in remaining_attrs
    }
    max_score = max(scores.values())
    return {
        "triggered": max_score < tau,
        "scores": scores,
        "max_score": max_score,
        "tau": tau,
    }


def should_stop(
    candidates: list[dict],
    attr_cache: dict[str, dict],
    remaining_attrs: list[str],
    tau: float = TAU,
    min_pool_for_dynamic: int = MIN_POOL_FOR_DYNAMIC,
) -> bool:
    """Return True when max marginal information gain across remaining attrs < τ."""
    return evaluate_stop(
        candidates,
        attr_cache,
        remaining_attrs,
        tau,
        min_pool_for_dynamic,
    )["triggered"]
