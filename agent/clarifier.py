"""
Dynamic attribute selector based on per-pool information gain.

Core insight from PDF analysis of the 50,000-product catalog:
  - NEVER ask 'category' or 'brand': evaluator's classify_constraint() never
    returns these, so asking always yields "no additional preference" — wasted turn.
  - Optimal question = argmax(coverage × normalized_entropy) over current candidate pool.
  - Static priority order is wrong because pool composition changes each turn
    (e.g. in a Dresses pool, use_case beats material; globally, feature beats color).
  - 'other' is a wildcard: evaluator matches any undisclosed constraint, safe fallback.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

# Attributes that evaluator's classify_constraint() CAN return.
# category and brand deliberately excluded (evaluator never classifies them).
SCOREABLE_ATTRS = ["material", "color", "size", "style", "use_case", "feature", "budget"]

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

# Global normalized-entropy weights from PDF Table (全目录 50,000件).
# Used as fallback when candidate pool is too small for reliable dynamic entropy.
_GLOBAL_ENTROPY: dict[str, float] = {
    "material": 0.73,
    "color":    0.77,
    "size":     0.53,
    "style":    0.67,
    "use_case": 0.87,
    "feature":  0.71,
    "budget":   0.79,
}

# Minimum pool size to trust dynamic entropy; fall back to global otherwise.
_MIN_POOL_FOR_DYNAMIC = 10


def _normalized_entropy(values: list[str]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    counts = Counter(values)
    H = -sum((c / n) * math.log2(c / n) for c in counts.values())
    max_H = math.log2(n)
    return H / max_H if max_H > 0 else 0.0


def _score_attribute(
    attr: str,
    candidates: list[dict],
    attr_cache: dict[str, dict],
) -> float:
    """Compute coverage × normalized_entropy for one attribute over the candidate pool."""
    values: list[str] = []
    for c in candidates:
        v = attr_cache.get(c["parent_asin"], {}).get(attr)
        if v:
            values.append(v)

    n = len(candidates)
    coverage = len(values) / n if n > 0 else 0.0

    if len(values) >= _MIN_POOL_FOR_DYNAMIC:
        entropy = _normalized_entropy(values)
    else:
        entropy = _global_entropy(attr, coverage, values)

    return coverage * entropy


def _global_entropy(attr: str, coverage: float, values: list[str]) -> float:
    """Blend global entropy with observed entropy when pool is small."""
    global_h = _GLOBAL_ENTROPY.get(attr, 0.5)
    if not values:
        return global_h
    obs_h = _normalized_entropy(values) if len(values) >= 2 else global_h
    # Weight global entropy more when sample is small
    w = min(len(values) / _MIN_POOL_FOR_DYNAMIC, 1.0)
    return w * obs_h + (1 - w) * global_h


class Clarifier:
    def next_ask(
        self,
        session: Any,
        candidates: list[dict] | None = None,
        attr_cache: dict[str, dict] | None = None,
    ) -> str:
        """
        Pick the most informative attribute to ask about this turn.

        Uses coverage × normalized_entropy on the current candidate pool when
        attr_cache is provided (dynamic mode). Falls back to global entropy
        weights from the PDF when pool is too small or cache is unavailable.

        Never asks 'category' or 'brand' (evaluator cannot classify them).
        Always returns a valid ask_attribute — never None.
        """
        asked = set(session.asked_attributes)
        known = set(session.slots.keys())
        eligible = [a for a in SCOREABLE_ATTRS if a not in asked and a not in known]

        if not eligible:
            # All specific attributes asked or known — use wildcard
            if "other" not in asked:
                session.asked_attributes.append("other")
                return "other"
            return "other"

        if candidates and attr_cache:
            scores = {
                attr: _score_attribute(attr, candidates, attr_cache)
                for attr in eligible
            }
        else:
            # No pool info — use global entropy × assumed 50% coverage
            scores = {attr: 0.5 * _GLOBAL_ENTROPY.get(attr, 0.5) for attr in eligible}

        best = max(eligible, key=lambda a: scores[a])
        session.asked_attributes.append(best)
        return best
