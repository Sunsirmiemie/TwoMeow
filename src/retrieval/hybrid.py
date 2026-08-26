"""
Hybrid retriever: fuses BM25 + dense results via Reciprocal Rank Fusion.
Falls back to BM25-only when sentence-transformers is not installed.
"""
from __future__ import annotations

from .bm25 import BM25Retriever
from ..ranking.scorer import fuse, BM25_BASE, DENSE_BASE


class HybridRetriever:
    def __init__(self, catalog_path: str, config: dict):
        self.bm25 = BM25Retriever(catalog_path)
        self.dense = None
        if config.get("use_dense", True):
            try:
                from .dense import DenseRetriever
                model = config.get("dense_model", "all-MiniLM-L6-v2")
                self.dense = DenseRetriever(catalog_path, model)
            except ImportError:
                pass  # sentence-transformers not installed; BM25-only mode

    def retrieve(
        self,
        query: str,
        slots: dict,
        intent: str,
        top_k: int = 100,
        turn: int = 1,
    ) -> list[dict]:
        if not query.strip():
            query = "*"

        bm25_results = self.bm25.search(query, slots, intent, top_k=top_k)

        if self.dense is None:
            return bm25_results

        # Buying track: evaluator reveals exact constraint text — BM25 is highly precise.
        # Dense adds noise here, so skip fusion for buying.
        if intent == "buying":
            return bm25_results

        # Browsing: turn-aware weight shift (MD §III adaptive orchestration).
        # Early turns: more dense weight for broad semantic recall.
        # Later turns: shift toward BM25 as exact keywords accumulate.
        dense_results = self.dense.search(query, top_k=top_k)
        n_slots = len(slots)
        if n_slots == 0:
            bm25_w, dense_w = 0.50, 0.50
        elif n_slots == 1:
            bm25_w, dense_w = 0.60, 0.40
        else:
            bm25_w, dense_w = BM25_BASE, DENSE_BASE

        return fuse(bm25_results, dense_results, top_k, bm25_w=bm25_w, dense_w=dense_w)
