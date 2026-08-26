"""
Hybrid retriever: fuses BM25 + dense results via Reciprocal Rank Fusion (RRF).
Falls back to BM25-only if dense retriever is not available.
"""
from __future__ import annotations

from .bm25_retriever import BM25Retriever


RRF_K = 60  # standard RRF constant

# Base weights — BM25 dominates because evaluator reveals exact constraint text
# that BM25 matches precisely. Dense helps in early turns for broad recall.
_BM25_BASE  = 0.75
_DENSE_BASE = 0.25


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


class HybridRetriever:
    def __init__(self, catalog_path: str, config: dict):
        self.bm25 = BM25Retriever(catalog_path)
        self.dense = None
        if config.get("use_dense", True):
            try:
                from .dense_retriever import DenseRetriever
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

        # Buying track: evaluator reveals exact constraint text each turn,
        # so BM25 keyword matching is highly precise. Dense adds noise.
        # Use BM25-only for buying to preserve MRR.
        if intent == "buying":
            return bm25_results

        # Browsing track: turn-aware hybrid (MD §III adaptive orchestration).
        # Early turns have no confirmed slots → dense helps with broad semantic recall.
        # Later turns have exact keywords → shift weight toward BM25.
        dense_results = self.dense.search(query, top_k=top_k)
        n_slots = len(slots)
        if n_slots == 0:
            bm25_w, dense_w = 0.50, 0.50
        elif n_slots == 1:
            bm25_w, dense_w = 0.60, 0.40
        else:
            bm25_w, dense_w = _BM25_BASE, _DENSE_BASE

        return self._fuse(bm25_results, dense_results, top_k,
                          bm25_w=bm25_w, dense_w=dense_w)

    def _fuse(self, bm25: list[dict], dense: list[dict], top_k: int,
              bm25_w: float = _BM25_BASE, dense_w: float = _DENSE_BASE) -> list[dict]:
        scores: dict[str, float] = {}

        for rank, item in enumerate(bm25):
            asin = item["parent_asin"]
            scores[asin] = scores.get(asin, 0.0) + bm25_w * _rrf_score(rank)

        for rank, item in enumerate(dense):
            asin = item["parent_asin"]
            scores[asin] = scores.get(asin, 0.0) + dense_w * _rrf_score(rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"parent_asin": asin, "score": score} for asin, score in ranked[:top_k]]
