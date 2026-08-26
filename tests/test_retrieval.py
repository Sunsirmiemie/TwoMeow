"""Tests for src/retrieval/: catalog loading, BM25 search, RRF fusion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking.scorer import fuse, rrf_score


def test_rrf_score_positive():
    assert rrf_score(0) > 0
    assert rrf_score(0) > rrf_score(10)


def test_fuse_combines_both():
    bm25 = [{"parent_asin": "A"}, {"parent_asin": "B"}]
    dense = [{"parent_asin": "B"}, {"parent_asin": "C"}]
    result = fuse(bm25, dense, top_k=3)
    asins = [r["parent_asin"] for r in result]
    assert "A" in asins
    assert "B" in asins
    assert "C" in asins


def test_fuse_top_k_limits():
    bm25  = [{"parent_asin": f"A{i}"} for i in range(10)]
    dense = [{"parent_asin": f"B{i}"} for i in range(10)]
    result = fuse(bm25, dense, top_k=5)
    assert len(result) == 5


def test_fuse_bm25_only():
    bm25 = [{"parent_asin": "X"}, {"parent_asin": "Y"}]
    result = fuse(bm25, [], top_k=2, bm25_w=1.0, dense_w=0.0)
    assert result[0]["parent_asin"] == "X"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
