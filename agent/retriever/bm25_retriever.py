"""
BM25 retriever via SQLite FTS5.

Also builds an in-memory attribute cache (_attr_cache) used by the Clarifier
to compute per-pool coverage × entropy scores (PDF insight: dynamic scoring
beats fixed ask order).
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "key", "requirement", "matters", "preference", "need", "actually",
    "ignore", "earlier", "please", "just", "also",
}

FIELD_WEIGHTS = {
    "title": 6.0,
    "categories": 4.0,
    "features": 2.5,
    "description": 2.5,
    "store": 1.5,
    "details": 1.0,
}

# Attribute extraction patterns for _attr_cache.
# MUST mirror evaluator's MATERIAL_RE / COLOR_RE exactly so coverage estimates
# reflect what classify_constraint() actually returns (PDF conclusion #1).
_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I
)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)
_SIZE_RE = re.compile(r"\b(xs|xxl|xl|small|medium|large|s\b|m\b|l\b)\b", re.I)
_STYLE_RE = re.compile(
    r"\b(casual|formal|vintage|sporty|elegant|bohemian|minimalist|streetwear)\b", re.I
)
_USECASE_RE = re.compile(
    r"\b(office|gym|outdoor|beach|wedding|party|work|travel|hiking|running|winter)\b", re.I
)
_FEATURE_RE = re.compile(
    r"\b(waterproof|breathable|lightweight|stretchy|slim|oversized|elastic|zip|pocket|lace.up|drawstring)\b", re.I
)


def _budget_bucket(price) -> str | None:
    if price is None:
        return None
    try:
        p = float(price)
        if p < 20:
            return "<$20"
        if p < 50:
            return "$20-$50"
        if p < 100:
            return "$50-$100"
        return ">$100"
    except (ValueError, TypeError):
        return None


def _clean_query(text: str, max_terms: int = 40) -> str:
    tokens = [
        t.lower() for t in TOKEN_RE.findall(text)
        if len(t) > 1 and t.lower() not in STOPWORDS
    ]
    unique = list(dict.fromkeys(tokens))[:max_terms]
    if not unique:
        return ""
    return " OR ".join(f'"{t}"' for t in unique)


def _normalize(text) -> str:
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    elif isinstance(text, dict):
        text = " ".join(str(v) for v in text.values())
    elif not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFKC", text)


def _first_match(pattern: re.Pattern, *texts: str) -> str | None:
    for text in texts:
        m = pattern.search(text)
        if m:
            return m.group(0).lower()
    return None


class BM25Retriever:
    def __init__(self, catalog_path: str):
        self.conn = sqlite3.connect(":memory:")
        # {asin: {attr: value_or_None}} — used by Clarifier for dynamic scoring
        self._attr_cache: dict[str, dict[str, str | None]] = {}
        self._titles: dict[str, str] = {}   # asin → title for LLM reranking
        self._build_index(catalog_path)

    def _build_index(self, catalog_path: str) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            CREATE VIRTUAL TABLE products USING fts5(
                parent_asin UNINDEXED,
                title, categories, features, description, store, details,
                price UNINDEXED,
                tokenize='unicode61'
            )
        """)
        # price stored in UNINDEXED column for budget filtering in _apply_slot_filters()

        batch = []
        with open(catalog_path, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                asin = p.get("parent_asin", "")
                details = p.get("details", {}) or {}
                title = _normalize(p.get("title", ""))
                features = _normalize(p.get("features", ""))
                desc = _normalize(p.get("description", ""))
                det = _normalize(details)
                price = p.get("price")

                batch.append((
                    asin, title,
                    _normalize(p.get("categories", "")),
                    features, desc,
                    _normalize(p.get("store", "")),
                    det, price,
                ))

                # Title cache for LLM reranker
                self._titles[asin] = title[:120]

                # Build attribute cache for Clarifier dynamic scoring
                searchable = f"{title} {features} {det} {desc}"
                self._attr_cache[asin] = {
                    "material": _first_match(_MATERIAL_RE, searchable),
                    "color":    _first_match(_COLOR_RE, searchable),
                    "size":     _first_match(_SIZE_RE, searchable),
                    "style":    _first_match(_STYLE_RE, searchable),
                    "use_case": _first_match(_USECASE_RE, searchable),
                    "feature":  _first_match(_FEATURE_RE, features, det),
                    "budget":   _budget_bucket(price),
                }

                if len(batch) >= 1000:
                    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", batch)
        self.conn.commit()

    def search(self, query: str, slots: dict, intent: str, top_k: int = 100) -> list[dict]:
        expression = _clean_query(query)
        if not expression:
            return []

        weight_str = ", ".join(f"{w}" for w in FIELD_WEIGHTS.values())
        sql = f"""
            SELECT parent_asin, price,
                   bm25(products, 0, {weight_str}) AS score
            FROM products
            WHERE products MATCH ?
            ORDER BY score
            LIMIT ?
        """
        cur = self.conn.cursor()
        try:
            rows = cur.execute(sql, (expression, top_k * 3)).fetchall()
        except sqlite3.OperationalError:
            return []

        results = [
            {"parent_asin": r[0], "price": r[1], "score": -r[2]}
            for r in rows
        ]

        if intent == "buying":
            results = self._apply_slot_filters(results, slots)

        return results[:top_k]

    def _apply_slot_filters(self, results: list[dict], slots: dict) -> list[dict]:
        if "budget" in slots:
            try:
                budget = float(slots["budget"])
                results = [r for r in results if r["price"] is None or r["price"] <= budget]
            except ValueError:
                pass
        return results
