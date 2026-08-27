"""
Information-theoretic attribute scoring for dynamic question selection.
PDF insight: coverage × normalized_entropy per candidate pool beats fixed global order.
"""
from __future__ import annotations

import math
from collections import Counter

from .attribute_stats import GLOBAL_ENTROPY

MIN_POOL_FOR_DYNAMIC = 10  # minimum pool size to trust dynamic entropy


def normalized_entropy(values: list[str]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    counts = Counter(values)
    H = -sum((c / n) * math.log2(c / n) for c in counts.values())
    max_H = math.log2(n)
    return H / max_H if max_H > 0 else 0.0


def blended_entropy(
    attr: str,
    values: list[str],
    min_pool_for_dynamic: int = MIN_POOL_FOR_DYNAMIC,
) -> float:
    """Blend global PDF entropy with observed entropy when pool is too small."""
    global_h = GLOBAL_ENTROPY.get(attr, 0.5)
    if not values:
        return global_h
    obs_h = normalized_entropy(values) if len(values) >= 2 else global_h
    w = min(len(values) / min_pool_for_dynamic, 1.0)
    return w * obs_h + (1 - w) * global_h


def score_attribute(
    attr: str,
    candidates: list[dict],
    attr_cache: dict[str, dict],
    min_pool_for_dynamic: int = MIN_POOL_FOR_DYNAMIC,
) -> float:
    """Compute coverage × normalized_entropy for one attribute over the candidate pool."""
    values: list[str] = [
        v for c in candidates
        if (v := attr_cache.get(c["parent_asin"], {}).get(attr))
    ]
    n = len(candidates)
    coverage = len(values) / n if n > 0 else 0.0
    entropy = (
        normalized_entropy(values)
        if len(values) >= min_pool_for_dynamic
        else blended_entropy(attr, values, min_pool_for_dynamic)
    )
    return coverage * entropy
