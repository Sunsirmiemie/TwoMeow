"""Tests for src/ranking/reranker.py and src/retrieval/candidate_builder.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking.reranker import Ranker
from src.retrieval.candidate_builder import build_rerank_pool
from src.agent.state import SessionMemory


def _candidates(n: int) -> list[dict]:
    return [{"parent_asin": f"B{i:03d}", "score": float(n - i)} for i in range(n)]


def test_ranker_disabled_passthrough():
    r = Ranker({})
    session = SessionMemory({})
    cands = _candidates(20)
    result = r.rerank(cands, session, top_k=10)
    assert len(result) == 10
    assert result[0]["parent_asin"] == cands[0]["parent_asin"]


def test_ranker_empty_candidates():
    r = Ranker({})
    session = SessionMemory({})
    assert r.rerank([], session, top_k=10) == []


def test_build_rerank_pool_truncates():
    session = SessionMemory({})   # 0 slots → few_slots=True
    cands = _candidates(60)
    pool = build_rerank_pool(cands, session)
    assert len(pool) == 20


def test_build_rerank_pool_no_truncation_with_slots():
    session = SessionMemory({})
    session.slots = {"material": "cotton", "color": "black"}
    cands = _candidates(60)
    pool = build_rerank_pool(cands, session)
    assert len(pool) == 60


def test_build_rerank_pool_small_pool():
    session = SessionMemory({})   # few slots
    cands = _candidates(30)       # below threshold
    pool = build_rerank_pool(cands, session)
    assert len(pool) == 30


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
