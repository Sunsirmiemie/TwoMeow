"""User-profile prior for model-free product reranking."""
from __future__ import annotations

import re

_PROFILE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "comfort": ("comfort", "comfortable"),
    "durability": ("durability", "durable"),
    "fit": ("fit", "fitted", "fitting"),
    "warmth": ("warmth", "warm"),
}


def _tokens(text: str) -> set[str]:
    """Return comparable lowercase word tokens from profile or catalogue text."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _expand_profile_tags(preference_tags: list[str]) -> set[str]:
    """Expand a small, auditable set of common catalog wording variants."""
    tags = _tokens(" ".join(preference_tags))
    return tags | {variant for tag in tags for variant in _PROFILE_SYNONYMS.get(tag, ())}


def apply_profile_boost(
    candidates: list[dict],
    preference_tags: list[str],
    title_lookup: dict[str, str],
    categories: dict[str, list[str]],
    product_meta: dict[str, dict],
    weight: float,
) -> list[dict]:
    """Add a bounded profile-tag overlap prior without mutating retrieval output.

    The signal is deliberately applied only during reranking.  It uses catalog
    title/category text and avoids learned models, external services, and
    profile-based filtering that could remove a relevant product.
    """
    tags = _expand_profile_tags(preference_tags)
    if weight <= 0.0 or not tags:
        return candidates

    boosted: list[dict] = []
    for candidate in candidates:
        asin = candidate["parent_asin"]
        product_tokens = _tokens(
            f"{title_lookup.get(asin, '')} {' '.join(categories.get(asin, []))} "
            f"{product_meta.get(asin, {}).get('profile_text', '')}"
        )
        overlap = len(tags & product_tokens) / len(tags)
        updated = dict(candidate)
        updated["score"] = float(candidate.get("score") or 0.0) + weight * overlap
        boosted.append(updated)
    return boosted
