"""
Reciprocal Rank Fusion (RRF) scorer.
Used by HybridRetriever to fuse BM25 + dense results into a single ranked list.
"""
from __future__ import annotations

RRF_K     = 60    # standard RRF constant
BM25_BASE = 0.75  # BM25 dominates: evaluator reveals exact constraint text each turn
DENSE_BASE = 0.25


def source_confidence(results: list[dict]) -> float:
    """Bounded separation between the head and tail of a current result list."""
    if len(results) < 2:
        return 0.0
    scores = [float(item.get("score") or 0.0) for item in results[:10]]
    high, low = max(scores), min(scores)
    return max(0.0, min(1.0, (high - low) / (abs(high) + 1e-9)))


def adaptive_fusion_weights(
    bm25: list[dict],
    dense: list[dict],
    session,
    fallback: tuple[float, float],
) -> tuple[float, float]:
    """Choose source weights from evidence specificity and current score shape."""
    slot_count = len(getattr(session, "slots", {})) if session is not None else 0
    if slot_count == 0:
        lexical_prior = fallback[0]
    elif slot_count == 1:
        lexical_prior = max(fallback[0], 0.60)
    else:
        lexical_prior = 0.75
    adjustment = 0.10 * (source_confidence(bm25) - source_confidence(dense))
    bm25_weight = max(0.35, min(0.90, lexical_prior + adjustment))
    return bm25_weight, 1.0 - bm25_weight


def rrf_score(rank: int, rrf_k: int = RRF_K) -> float:
    return 1.0 / (rrf_k + rank + 1)


def fuse(
    bm25: list[dict],
    dense: list[dict],
    top_k: int,
    bm25_w: float = BM25_BASE,
    dense_w: float = DENSE_BASE,
    rrf_k: int = RRF_K,
) -> list[dict]:
    scores: dict[str, float] = {}
    for rank, item in enumerate(bm25):
        asin = item["parent_asin"]
        scores[asin] = scores.get(asin, 0.0) + bm25_w * rrf_score(rank, rrf_k)
    for rank, item in enumerate(dense):
        asin = item["parent_asin"]
        scores[asin] = scores.get(asin, 0.0) + dense_w * rrf_score(rank, rrf_k)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"parent_asin": asin, "score": score} for asin, score in ranked[:top_k]]
