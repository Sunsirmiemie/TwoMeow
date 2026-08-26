"""
Product text extraction for reranker prompts and dense embeddings.
"""
from __future__ import annotations


def product_text(p: dict) -> str:
    """Concatenate title + categories + features[:300] for semantic encoding."""
    title = str(p.get("title") or "")
    cats = p.get("categories") or []
    if isinstance(cats, list):
        cats = " ".join(str(c) for c in cats)
    features = p.get("features") or []
    if isinstance(features, list):
        features = " ".join(str(f) for f in features)
    elif not isinstance(features, str):
        features = str(features)
    return f"{title}. {cats}. {features[:300]}".strip()
