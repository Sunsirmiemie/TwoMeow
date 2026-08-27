"""Tests for src/retrieval/: catalog loading, BM25 search, RRF fusion."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking.scorer import fuse, rrf_score
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever


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


def test_budget_search_keeps_products_with_unknown_catalog_prices():
    products = [
        {"parent_asin": "DASH", "title": "Desk lamp", "price": "—"},
        {"parent_asin": "FROM", "title": "Reading lamp", "price": "from 12.99"},
        {"parent_asin": "EXPENSIVE", "title": "Luxury lamp", "price": 75},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        retriever = BM25Retriever(str(catalog_path))

        results = retriever.search(
            "lamp",
            slots={"budget": "50"},
            intent="buying",
            top_k=10,
        )

    assert {result["parent_asin"] for result in results} == {"DASH", "FROM"}
    assert all(result["price"] is None for result in results)


def test_bm25_direct_constructor_preserves_default_field_weights():
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "catalog.jsonl"
        catalog_path.write_text(
            json.dumps({"parent_asin": "A1", "title": "Desk lamp"}) + "\n",
            encoding="utf-8",
        )
        retriever = BM25Retriever(str(catalog_path))

    assert retriever.field_weights == {
        "title": 6.0,
        "categories": 4.0,
        "features": 2.5,
        "description": 2.5,
        "store": 1.5,
        "details": 1.0,
    }


def test_hybrid_retriever_applies_base_weights_and_rrf_override():
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "catalog.jsonl"
        catalog_path.write_text(
            json.dumps({"parent_asin": "BM25", "title": "Desk lamp"}) + "\n",
            encoding="utf-8",
        )
        retriever = HybridRetriever(
            str(catalog_path),
            {
                "use_dense": False,
                "rrf_k": 0,
                "bm25_base": 0.1,
                "dense_base": 0.9,
                "browsing_weights": [],
            },
        )

        class FakeDense:
            def search(self, _query, top_k):
                return [{"parent_asin": "DENSE", "score": 1.0}][:top_k]

        retriever.dense = FakeDense()
        results = retriever.retrieve("lamp", {}, "browsing", top_k=2)

    assert [result["parent_asin"] for result in results] == ["DENSE", "BM25"]
    assert results[0]["score"] == 0.9


def test_hybrid_retriever_applies_slot_aware_browsing_weights():
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "catalog.jsonl"
        catalog_path.write_text(
            json.dumps({"parent_asin": "BM25", "title": "Desk lamp"}) + "\n",
            encoding="utf-8",
        )
        retriever = HybridRetriever(
            str(catalog_path),
            {
                "use_dense": False,
                "browsing_weights": [[0.9, 0.1], [0.1, 0.9]],
            },
        )

        class FakeDense:
            def search(self, _query, top_k):
                return [{"parent_asin": "DENSE", "score": 1.0}][:top_k]

        retriever.dense = FakeDense()
        broad = retriever.retrieve("lamp", {}, "browsing", top_k=2)
        narrowed = retriever.retrieve(
            "lamp",
            {"color": "red"},
            "browsing",
            top_k=2,
        )

    assert broad[0]["parent_asin"] == "BM25"
    assert narrowed[0]["parent_asin"] == "DENSE"


def test_hybrid_retriever_uses_base_weights_after_browsing_weight_stages():
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "catalog.jsonl"
        catalog_path.write_text(
            json.dumps({"parent_asin": "BM25", "title": "Desk lamp"}) + "\n",
            encoding="utf-8",
        )
        retriever = HybridRetriever(
            str(catalog_path),
            {
                "use_dense": False,
                "bm25_base": 0.1,
                "dense_base": 0.9,
            },
        )

        class FakeDense:
            def search(self, _query, top_k):
                return [{"parent_asin": "DENSE", "score": 1.0}][:top_k]

        retriever.dense = FakeDense()
        results = retriever.retrieve(
            "lamp",
            {"color": "red", "material": "cotton"},
            "browsing",
            top_k=2,
        )

    assert results[0]["parent_asin"] == "DENSE"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
