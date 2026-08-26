"""
Attribute regex patterns, global entropy weights, and utility helpers.
All patterns mirror evaluator's classify_constraint() to ensure consistency
(PDF conclusion #1: coverage estimates must reflect what classify_constraint returns).
"""
from __future__ import annotations

import re

# Attributes the evaluator's classify_constraint() can return.
# 'category' and 'brand' excluded — evaluator never classifies them.
SCOREABLE_ATTRS = ["material", "color", "size", "style", "use_case", "feature", "budget"]

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

# Global normalized-entropy weights from PDF Table (全目录 50,000件).
# Used as fallback when candidate pool is too small for reliable dynamic scoring.
GLOBAL_ENTROPY: dict[str, float] = {
    "material": 0.73,
    "color":    0.77,
    "size":     0.53,
    "style":    0.67,
    "use_case": 0.87,
    "feature":  0.71,
    "budget":   0.79,
}

# MUST mirror evaluator's MATERIAL_RE / COLOR_RE exactly
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)
SIZE_RE    = re.compile(r"\b(xs|xxl|xl|small|medium|large|s\b|m\b|l\b)\b", re.I)
STYLE_RE   = re.compile(
    r"\b(casual|formal|vintage|sporty|elegant|bohemian|minimalist|streetwear)\b", re.I
)
USECASE_RE = re.compile(
    r"\b(office|gym|outdoor|beach|wedding|party|work|travel|hiking|running|winter)\b", re.I
)
FEATURE_RE = re.compile(
    r"\b(waterproof|breathable|lightweight|stretchy|slim|oversized|elastic|zip|pocket|lace.up|drawstring)\b",
    re.I,
)


def first_match(pattern: re.Pattern, *texts: str) -> str | None:
    for text in texts:
        m = pattern.search(text)
        if m:
            return m.group(0).lower()
    return None


def budget_bucket(price) -> str | None:
    if price is None:
        return None
    try:
        p = float(price)
        if p < 20:   return "<$20"
        if p < 50:   return "$20-$50"
        if p < 100:  return "$50-$100"
        return ">$100"
    except (ValueError, TypeError):
        return None


def extract_attrs(
    searchable: str,
    features: str,
    details: str,
    price,
) -> dict[str, str | None]:
    """Build attribute dict for one product — used by catalog loader for attr_cache."""
    return {
        "material": first_match(MATERIAL_RE, searchable),
        "color":    first_match(COLOR_RE, searchable),
        "size":     first_match(SIZE_RE, searchable),
        "style":    first_match(STYLE_RE, searchable),
        "use_case": first_match(USECASE_RE, searchable),
        "feature":  first_match(FEATURE_RE, features, details),
        "budget":   budget_bucket(price),
    }
