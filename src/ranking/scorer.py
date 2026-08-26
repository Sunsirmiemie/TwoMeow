"""
Reciprocal Rank Fusion (RRF) scorer.
Used by HybridRetriever to fuse BM25 + dense results into a single ranked list.
"""
from __future__ import annotations

RRF_K     = 60    # standard RRF constant
BM25_BASE = 0.75  # BM25 dominates: evaluator reveals exact constraint text each turn
DENSE_BASE = 0.25


def rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def fuse(
    bm25: list[dict],
    dense: list[dict],
    top_k: int,
    bm25_w: float = BM25_BASE,
    dense_w: float = DENSE_BASE,
) -> list[dict]:
    scores: dict[str, float] = {}
    for rank, item in enumerate(bm25):
        asin = item["parent_asin"]
        scores[asin] = scores.get(asin, 0.0) + bm25_w * rrf_score(rank)
    for rank, item in enumerate(dense):
        asin = item["parent_asin"]
        scores[asin] = scores.get(asin, 0.0) + dense_w * rrf_score(rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"parent_asin": asin, "score": score} for asin, score in ranked[:top_k]]
