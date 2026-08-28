"""
BM25 retriever using SQLite FTS5.
Reads from a CatalogIndex; exposes title and attr caches for downstream components.
"""
from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping

from .catalog import CatalogIndex, FIELD_WEIGHTS, clean_query


class BM25Retriever:
    def __init__(
        self,
        catalog_path: str,
        field_weights: Mapping[str, float] | None = None,
    ):
        self._index = CatalogIndex(catalog_path)
        self.field_weights = dict(
            FIELD_WEIGHTS if field_weights is None else field_weights
        )

    # ── Cache proxies for HybridRetriever / Ranker ────────────────────────────

    @property
    def _titles(self) -> dict[str, str]:
        return self._index.titles

    @property
    def _attr_cache(self) -> dict[str, dict]:
        return self._index.attr_cache

    @property
    def _categories(self) -> dict[str, list[str]]:
        return self._index.categories

    @property
    def _meta(self) -> dict[str, dict]:
        return self._index.meta

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, slots: dict, intent: str, top_k: int = 100) -> list[dict]:
        expression = clean_query(query)
        if not expression:
            return []

        weight_str = ", ".join(str(w) for w in self.field_weights.values())
        sql = f"""
            SELECT parent_asin, price,
                   bm25(products, 0, {weight_str}) AS score
            FROM products
            WHERE products MATCH ?
            ORDER BY score
            LIMIT ?
        """
        cur = self._index.conn.cursor()
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

    def _buying_expression(self, slots: dict, query: str) -> str:
        """AND query over confirmed slot values (material/color/size/style/use_case/
        feature/brand). Category is applied as a post-filter instead of an FTS phrase,
        because coarse categories are space-joined phrases that break FTS adjacency.
        Falls back to OR when too few terms are known."""
        terms: list[str] = []
        seen: set[str] = set()
        for key in ("material", "color", "size", "style", "use_case", "feature", "brand"):
            value = slots.get(key)
            if not value:
                continue
            value = re.sub(r"\s+", " ", str(value).strip())
            if value and value.lower() not in seen:
                seen.add(value.lower())
                terms.append(f'"{value}"')
        if len(terms) >= 2:
            return " AND ".join(terms[:6])
        return clean_query(query)

    def _apply_slot_filters(self, results: list[dict], slots: dict) -> list[dict]:
        category = slots.get("category")
        if category:
            wanted = {
                token for token in re.findall(r"[a-z0-9]+", str(category).lower())
                if len(token) >= 3
            }
            if wanted:
                filtered = [
                    r for r in results
                    if any(
                        wanted & set(re.findall(r"[a-z0-9]+", str(value).lower()))
                        for value in self._index.categories.get(r["parent_asin"], [])
                    )
                ]
                if filtered:
                    results = filtered
        if "budget" in slots:
            try:
                budget = float(slots["budget"])
                results = [r for r in results if r["price"] is None or r["price"] <= budget]
            except ValueError:
                pass
        return results
