"""
Hybrid retriever: fuses BM25 + dense results via Reciprocal Rank Fusion.
Falls back to BM25-only when sentence-transformers is not installed.
"""
from __future__ import annotations

from .bm25 import BM25Retriever
from ..ranking.scorer import fuse, BM25_BASE, DENSE_BASE, RRF_K

_BROWSING_WEIGHTS = ((0.50, 0.50), (0.60, 0.40))


class HybridRetriever:
    def __init__(self, catalog_path: str, config: dict):
        self.bm25 = BM25Retriever(catalog_path, config.get("field_weights"))
        self.rrf_k = config.get("rrf_k", RRF_K)
        self.bm25_base = config.get("bm25_base", BM25_BASE)
        self.dense_base = config.get("dense_base", DENSE_BASE)
        self.browsing_weights = tuple(
            tuple(pair) for pair in config.get("browsing_weights", _BROWSING_WEIGHTS)
        )
        self.dense = None
        if config.get("use_dense", True):
            try:
                from .dense import DenseRetriever
                model = config.get("dense_model", "all-MiniLM-L6-v2")
                batch_size = config.get("dense_batch_size", 512)
                self.dense = DenseRetriever(catalog_path, model, batch_size)
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
        bm25_w, dense_w = self.bm25_base, self.dense_base
        if len(slots) < len(self.browsing_weights):
            bm25_w, dense_w = self.browsing_weights[len(slots)]

        return fuse(
            bm25_results,
            dense_results,
            top_k,
            bm25_w=bm25_w,
            dense_w=dense_w,
            rrf_k=self.rrf_k,
        )
