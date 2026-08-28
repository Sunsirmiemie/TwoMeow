"""
Catalog loader: builds SQLite FTS5 index, title cache, and attribute cache from JSONL.
The in-memory SQLite connection is owned by CatalogIndex and consumed by BM25Retriever.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from typing import Any

from ..dialogue.attribute_stats import extract_attrs

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "key", "requirement", "matters", "preference", "need", "actually",
    "ignore", "earlier", "just", "also",
}

FIELD_WEIGHTS = {
    "title": 6.0,
    "categories": 4.0,
    "features": 2.5,
    "description": 2.5,
    "store": 1.5,
    "details": 1.0,
}


def normalize(text: Any) -> str:
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    elif isinstance(text, dict):
        text = " ".join(str(v) for v in text.values())
    elif not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFKC", text)


def _normalize_price(value) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) else None


def _safe_int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_query(text: str, max_terms: int = 40) -> str:
    tokens = [
        t.lower() for t in TOKEN_RE.findall(text)
        if len(t) > 1 and t.lower() not in STOPWORDS
    ]
    unique = list(dict.fromkeys(tokens))[:max_terms]
    if not unique:
        return ""
    return " OR ".join(f'"{t}"' for t in unique)


class CatalogIndex:
    """In-memory SQLite FTS5 index + per-ASIN title and attribute caches."""

    def __init__(self, catalog_path: str):
        self.conn: sqlite3.Connection = sqlite3.connect(":memory:")
        self.titles: dict[str, str] = {}
        self.attr_cache: dict[str, dict[str, str | None]] = {}
        self.categories: dict[str, list[str]] = {}
        self.meta: dict[str, dict] = {}
        self._build(catalog_path)

    def _build(self, catalog_path: str) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            CREATE VIRTUAL TABLE products USING fts5(
                parent_asin UNINDEXED,
                title, categories, features, description, store, details,
                price UNINDEXED,
                tokenize='unicode61'
            )
        """)

        batch: list[tuple] = []
        with open(catalog_path, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                asin     = p.get("parent_asin", "")
                details  = p.get("details", {}) or {}
                title    = normalize(p.get("title", ""))
                features = normalize(p.get("features", ""))
                desc     = normalize(p.get("description", ""))
                det      = normalize(details)
                price    = _normalize_price(p.get("price"))

                batch.append((
                    asin, title,
                    normalize(p.get("categories", "")),
                    features, desc,
                    normalize(p.get("store", "")),
                    det, price,
                ))
                self.titles[asin] = title[:120]
                searchable = f"{title} {features} {det} {desc}"
                self.attr_cache[asin] = extract_attrs(searchable, features, det, price)
                self.categories[asin] = [str(value) for value in (p.get("categories") or [])]
                self.meta[asin] = {
                    "price": price,
                    "rating_number": _safe_int(p.get("rating_number")),
                    "average_rating": _safe_float(p.get("average_rating")),
                    # Kept in-memory for the local profile reranker only.
                    "profile_text": f"{title} {features} {det}",
                }

                if len(batch) >= 1000:
                    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", batch)
                    batch.clear()

        if batch:
            cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", batch)
        self.conn.commit()
