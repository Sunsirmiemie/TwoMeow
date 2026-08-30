"""
Product text extraction for reranker prompts and dense embeddings.
"""
from __future__ import annotations

from typing import Any


def _flatten(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return "" if value is None else str(value)


def product_identity_text(product: dict) -> str:
    """Text naming what the product is: title, taxonomy, and brand/store."""
    return ". ".join(filter(None, (
        _flatten(product.get("title")),
        _flatten(product.get("categories")),
        _flatten(product.get("store")),
    )))


def product_attribute_text(product: dict) -> str:
    """Text describing product properties, kept separate from identity."""
    return ". ".join(filter(None, (
        _flatten(product.get("features")),
        _flatten(product.get("details")),
        _flatten(product.get("description")),
    )))[:1200]


def product_text(p: dict) -> str:
    """Concatenate title + categories + features[:300] for semantic encoding."""
    return f"{product_identity_text(p)}. {product_attribute_text(p)[:300]}".strip()
