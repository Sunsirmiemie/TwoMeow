"""Turn user replies into retrieval-safe positive and negative evidence.

The purifier is deliberately rule based: every transformation is auditable,
offline, and derived only from text already supplied by the user.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re


_ATTRIBUTE_NAMES = (
    "material", "color", "size", "style", "use_case", "use case",
    "feature", "budget", "brand", "category",
)
_VALUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "material": re.compile(
        r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|"
        r"denim|linen|suede|velvet)\b", re.I,
    ),
    "color": re.compile(
        r"\b(black|white|blue|red|pink|green|brown|gr[ae]y|purple|yellow|"
        r"orange|navy|beige)\b", re.I,
    ),
    "size": re.compile(
        r"\b(xs|xxl|xl|small|medium|large|extra large|one size|plus size|"
        r"petite|tall|wide|narrow)\b", re.I,
    ),
    "style": re.compile(
        r"\b(casual|formal|vintage|sporty|elegant|bohemian|minimalist|"
        r"streetwear|slim fit|oversized)\b", re.I,
    ),
    "use_case": re.compile(
        r"\b(office|gym|outdoor|beach|wedding|party|daily|work|travel|"
        r"hiking|running|winter)\b", re.I,
    ),
    "feature": re.compile(
        r"\b(waterproof|breathable|lightweight|stretchy|anti-slip|elastic|"
        r"zip|pocket|lace.up|drawstring)\b", re.I,
    ),
}
_NEGATIVE_SUFFIX = re.compile(
    r"(?:\bnot|\bno|\bwithout|\bavoid|\bexclude|\bexcept|\banything but|"
    r"\bdon't want|\bdo not want)\s+(?:\w+\s+){0,3}$",
    re.I,
)
_NO_PREFERENCE = re.compile(
    r"(?:i\s+)?(?:do not|don't|dont|no)\s+(?:have\s+)?(?:an?\s+)?"
    r"(?:additional\s+)?preference\s+(?:for|on)\s+"
    r"(material|color|size|style|use[_ ]case|feature|budget|brand|category)"
    r"[^.;]*[.;]?",
    re.I,
)


@dataclass(frozen=True)
class PurifiedReply:
    """Evidence extracted from one user reply."""

    query_text: str
    excluded_values: dict[str, set[str]] = field(default_factory=dict)
    no_preference_attributes: set[str] = field(default_factory=set)


def purify_reply(text: str) -> PurifiedReply:
    """Remove negated values/no-preference clauses from the positive query.

    Negated values remain available as explicit exclusion evidence.  For
    example, ``not red, blue instead`` becomes positive query text containing
    only ``blue`` and negative evidence ``{"color": {"red"}}``.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    no_preference = {
        match.group(1).lower().replace(" ", "_")
        for match in _NO_PREFERENCE.finditer(normalized)
    }
    removal_spans = [match.span() for match in _NO_PREFERENCE.finditer(normalized)]
    excluded: dict[str, set[str]] = {}

    for attribute, pattern in _VALUE_PATTERNS.items():
        for value_match in pattern.finditer(normalized):
            prefix_start = max(0, value_match.start() - 48)
            prefix = normalized[prefix_start:value_match.start()]
            negative_match = _NEGATIVE_SUFFIX.search(prefix)
            if not negative_match:
                continue
            value = value_match.group(1).lower()
            excluded.setdefault(attribute, set()).add(value)
            removal_spans.append(
                (prefix_start + negative_match.start(), value_match.end())
            )

    cleaned = normalized
    for start, end in sorted(removal_spans, reverse=True):
        cleaned = f"{cleaned[:start]} {cleaned[end:]}"
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")
    return PurifiedReply(cleaned, excluded, no_preference)
