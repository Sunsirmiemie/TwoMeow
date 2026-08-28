"""
Dynamic attribute question selector.
Picks argmax(coverage × entropy) over the current candidate pool.
Never asks 'category' or 'brand' — evaluator's classify_constraint() cannot return these.
"""
from __future__ import annotations

from .attribute_stats import SCOREABLE_ATTRS, GLOBAL_ENTROPY
from .entropy import MIN_POOL_FOR_DYNAMIC, score_attribute


class Clarifier:
    def __init__(self, min_pool_for_dynamic: int = MIN_POOL_FOR_DYNAMIC):
        self.min_pool_for_dynamic = min_pool_for_dynamic

    def next_ask(
        self,
        session,
        candidates: list[dict] | None = None,
        attr_cache: dict[str, dict] | None = None,
    ) -> str:
        asked = set(session.asked_attributes)
        known = set(session.slots.keys())
        eligible = [a for a in SCOREABLE_ATTRS if a not in asked and a not in known]

        if not eligible:
            return None

        if candidates and attr_cache:
            scores = {
                a: score_attribute(
                    a,
                    candidates,
                    attr_cache,
                    self.min_pool_for_dynamic,
                )
                for a in eligible
            }
        else:
            # No pool data — use global entropy × assumed 50% coverage
            scores = {a: 0.5 * GLOBAL_ENTROPY.get(a, 0.5) for a in eligible}

        best = max(eligible, key=lambda a: scores[a])
        session.asked_attributes.append(best)
        return best
