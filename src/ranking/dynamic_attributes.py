"""Evidence-aware attribute weights and per-product compatibility scores."""
from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_PRIORITY = {
    "category": 1.15,
    "feature": 1.10,
    "material": 1.05,
    "color": 1.00,
    "size": 1.00,
    "style": 1.00,
    "use_case": 1.05,
    "brand": 1.00,
    "budget": 0.90,
}


def _tokens(value: Any) -> set[str]:
    return {
        token.lower() for token in _TOKEN_RE.findall(str(value))
        if len(token) > 1
    }


def dynamic_bm25_field_weights(
    base_weights: Mapping[str, float],
    session: Any | None,
    gain: float = 0.65,
) -> dict[str, float]:
    """Route accumulated evidence to the BM25 fields most able to use it."""
    weights = {key: float(value) for key, value in base_weights.items()}
    if session is None:
        return weights
    slots = session.slots
    confidence = session.slot_confidence

    category_signal = confidence.get("category", 0.75) if slots.get("category") else 0.0
    brand_signal = confidence.get("brand", 0.75) if slots.get("brand") else 0.0
    descriptive = ("material", "color", "size", "style", "use_case", "feature")
    detail_signal = sum(
        confidence.get(attribute, 0.75)
        for attribute in descriptive if slots.get(attribute)
    )
    detail_signal = 1.0 - math.exp(-detail_signal / 2.0)

    weights["title"] *= 1.0 + gain * (0.25 * category_signal + 0.15 * brand_signal)
    weights["categories"] *= 1.0 + gain * category_signal
    weights["features"] *= 1.0 + gain * detail_signal
    weights["description"] *= 1.0 + gain * 0.65 * detail_signal
    weights["details"] *= 1.0 + gain * 0.85 * detail_signal
    weights["store"] *= 1.0 + gain * brand_signal
    return weights


def _catalog_text(
    asin: str,
    titles: Mapping[str, str],
    categories: Mapping[str, list[str]],
    meta: Mapping[str, dict],
) -> str:
    product = meta.get(asin, {})
    return " ".join((
        titles.get(asin, ""),
        " ".join(categories.get(asin, [])),
        str(product.get("attribute_text") or product.get("profile_text") or ""),
        str(product.get("store") or ""),
    ))


def _positive_match(
    attribute: str,
    value: str,
    asin: str,
    attr_cache: Mapping[str, dict],
    titles: Mapping[str, str],
    categories: Mapping[str, list[str]],
    meta: Mapping[str, dict],
) -> float:
    product = meta.get(asin, {})
    if attribute == "budget":
        try:
            budget = float(value)
            price = product.get("price")
            if price is None:
                return 0.25
            return 1.0 if float(price) <= budget else max(0.0, budget / float(price) - 0.25)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0

    if attribute == "category":
        product_tokens = _tokens(" ".join(categories.get(asin, [])))
    else:
        product_tokens = _tokens(_catalog_text(asin, titles, categories, meta))
        cached = attr_cache.get(asin, {}).get(attribute)
        if cached:
            product_tokens |= _tokens(cached)
    wanted = _tokens(value)
    if not wanted:
        return 0.0
    recall = len(wanted & product_tokens) / len(wanted)
    return min(1.0, 1.25 * recall)


def _evidence_attributes(session: Any) -> list[str]:
    keys = set(session.slots) | {
        key for key, values in session.negative_slots.items() if values
    }
    return sorted(keys - set(session.no_preference_slots))


def score_attribute_compatibility(
    candidates: list[dict],
    session: Any,
    attr_cache: Mapping[str, dict],
    titles: Mapping[str, str],
    categories: Mapping[str, list[str]],
    meta: Mapping[str, dict],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return compatibility, exclusion-risk, and learned-at-runtime weights.

    The weights are recomputed from the current candidate pool.  An attribute
    becomes more important when it is explicit, recent, and selective in that
    pool; no future turns or ground-truth labels are consulted.
    """
    attributes = _evidence_attributes(session)
    if not candidates or not attributes:
        return ({item["parent_asin"]: 0.0 for item in candidates}, {}, {})

    raw_weights: dict[str, float] = {}
    for attribute in attributes:
        value = session.slots.get(attribute)
        confidence = session.slot_confidence.get(attribute, 0.80 if value else 0.70)
        age = max(0, session.turn_count + 1 - session.slot_turns.get(attribute, 1))
        recency = max(0.75, 1.0 - 0.04 * age)
        if value:
            matches = [
                _positive_match(
                    attribute, value, item["parent_asin"], attr_cache,
                    titles, categories, meta,
                )
                for item in candidates
            ]
            match_rate = sum(score >= 0.60 for score in matches) / len(matches)
            selectivity = 1.0 - match_rate
        else:
            selectivity = 0.65
        raw_weights[attribute] = (
            confidence * recency * (0.55 + 0.45 * selectivity)
            * _PRIORITY.get(attribute, 1.0)
        )

    total = sum(raw_weights.values()) or 1.0
    weights = {key: value / total for key, value in raw_weights.items()}
    compatibility: dict[str, float] = {}
    violations: dict[str, float] = {}
    for item in candidates:
        asin = item["parent_asin"]
        score = 0.0
        risk = 0.0
        for attribute, weight in weights.items():
            positive = session.slots.get(attribute)
            positive_score = 1.0 if not positive else _positive_match(
                attribute, positive, asin, attr_cache, titles, categories, meta,
            )
            excluded = session.negative_slots.get(attribute, set())
            violates = max((
                _positive_match(
                    attribute, value, asin, attr_cache, titles, categories, meta,
                )
                for value in excluded
            ), default=0.0)
            score += weight * positive_score * (1.0 - violates)
            risk += weight * violates
        compatibility[asin] = score
        violations[asin] = risk
    return compatibility, violations, weights


def rescore_retrieval_candidates(
    candidates: list[dict],
    session: Any,
    attr_cache: Mapping[str, dict],
    titles: Mapping[str, str],
    categories: Mapping[str, list[str]],
    meta: Mapping[str, dict],
    max_evidence_weight: float = 0.52,
) -> list[dict]:
    """Blend BM25 relevance with current per-product attribute compatibility."""
    compatibility, violations, weights = score_attribute_compatibility(
        candidates, session, attr_cache, titles, categories, meta,
    )
    if not weights:
        return candidates
    base_max = max((float(item.get("score") or 0.0) for item in candidates), default=1.0) or 1.0
    evidence_units = sum(session.slot_confidence.get(key, 0.75) for key in session.slots)
    evidence_units += 0.70 * sum(bool(values) for values in session.negative_slots.values())
    evidence_weight = min(max_evidence_weight, 0.18 + 0.10 * evidence_units)

    rescored: list[dict] = []
    for item in candidates:
        asin = item["parent_asin"]
        updated = dict(item)
        updated["lexical_score"] = float(item.get("score") or 0.0)
        updated["attribute_score"] = compatibility[asin]
        updated["negative_violation"] = violations[asin]
        base = float(item.get("score") or 0.0) / base_max
        updated["score"] = (
            (1.0 - evidence_weight) * base
            + evidence_weight * compatibility[asin]
            - evidence_weight * violations[asin]
        )
        rescored.append(updated)
    return sorted(rescored, key=lambda item: item["score"], reverse=True)
