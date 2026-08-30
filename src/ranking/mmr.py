"""Small, deterministic Maximal Marginal Relevance utilities."""
from __future__ import annotations

import re


_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "color", "size", "style", "material", "feature", "use", "case", "budget",
    "brand",
}


def tokenize(text: str) -> set[str]:
    """Tokenize catalogue text for slot coverage and product similarity."""
    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOP
    }


def select_mmr(
    scored: list[tuple[float, dict, set[str]]],
    top_k: int,
    relevance_weight: float,
) -> list[dict]:
    """Greedily select relevance-first results with a duplicate-risk penalty."""
    selected: list[tuple[float, dict, set[str]]] = []
    remaining = scored[:]
    while remaining and len(selected) < top_k:
        if not selected:
            best_index = max(range(len(remaining)), key=lambda i: remaining[i][0])
        else:
            def value(item: tuple[float, dict, set[str]]) -> float:
                relevance, _, tokens = item
                similarity = max(
                    len(tokens & chosen) / (len(tokens | chosen) or 1)
                    for _, _, chosen in selected
                )
                return relevance_weight * relevance - (1.0 - relevance_weight) * similarity

            best_index = max(range(len(remaining)), key=lambda i: value(remaining[i]))
        selected.append(remaining.pop(best_index))
    return [candidate for _, candidate, _ in selected]
