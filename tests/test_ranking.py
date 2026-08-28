"""Tests for src/ranking/reranker.py and src/retrieval/candidate_builder.py."""
import sys
from pathlib import Path
from types import SimpleNamespace
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


def test_model_free_rerank_only_considers_retrieval_top_n():
    """A high-scoring item after rank 20 must not enter the final top 10."""
    ranker = Ranker({"use_features": True, "rerank_top_n": 20})
    session = SessionMemory({})
    cands = _candidates(21)
    cands[-1]["score"] = 9999.0

    result = ranker.rerank(cands, session, top_k=10)

    assert len(result) == 10
    assert "B020" not in [item["parent_asin"] for item in result]


def test_profile_prior_promotes_matching_catalogue_text():
    ranker = Ranker(
        {"profile_weight": 1.0, "feature_weights": {"base": 1.0}},
        title_lookup={"B000": "formal dress", "B001": "comfortable running shoes"},
    )
    session = SessionMemory({"preference_tags": ["comfort"]})
    candidates = [
        {"parent_asin": "B000", "score": 1.0},
        {"parent_asin": "B001", "score": 0.9},
    ]

    result = ranker.rerank(candidates, session, top_k=2)

    assert result[0]["parent_asin"] == "B001"


def test_ranker_empty_candidates():
    r = Ranker({})
    session = SessionMemory({})
    assert r.rerank([], session, top_k=10) == []


def test_ranker_reports_token_usage_for_each_call_not_a_running_total():
    responses = [
        SimpleNamespace(
            usage=SimpleNamespace(input_tokens=11, output_tokens=3),
            content=[SimpleNamespace(text='["B001", "B000"]')],
        ),
        SimpleNamespace(
            usage=SimpleNamespace(input_tokens=7, output_tokens=2),
            content=[SimpleNamespace(text='["B000", "B001"]')],
        ),
    ]

    class FakeMessages:
        def create(self, **_kwargs):
            return responses.pop(0)

    fake_client = SimpleNamespace(messages=FakeMessages())
    ranker = Ranker({"use_llm_ranker": True}, client=fake_client)
    session = SessionMemory({})
    candidates = _candidates(2)

    ranker.rerank(candidates, session, top_k=2)
    assert ranker.token_usage == {"prompt_tokens": 11, "completion_tokens": 3}

    ranker.rerank(candidates, session, top_k=2)
    assert ranker.token_usage == {"prompt_tokens": 7, "completion_tokens": 2}


def test_ranker_resets_usage_after_success_when_disabled_or_empty():
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=11, output_tokens=3),
        content=[SimpleNamespace(text='["B001", "B000"]')],
    )

    class FakeMessages:
        def create(self, **_kwargs):
            return response

    ranker = Ranker(
        {"use_llm_ranker": True},
        client=SimpleNamespace(messages=FakeMessages()),
    )
    session = SessionMemory({})
    candidates = _candidates(2)

    ranker.rerank(candidates, session, top_k=2)
    assert ranker.token_usage == {"prompt_tokens": 11, "completion_tokens": 3}

    ranker.enabled = False
    ranker.rerank(candidates, session, top_k=2)
    assert ranker.token_usage == {"prompt_tokens": 0, "completion_tokens": 0}

    ranker.enabled = True
    ranker.rerank(candidates, session, top_k=2)
    ranker.rerank([], session, top_k=2)
    assert ranker.token_usage == {"prompt_tokens": 0, "completion_tokens": 0}


def test_ranker_api_error_reports_zero_usage_and_falls_back():
    class FailingMessages:
        def create(self, **_kwargs):
            raise RuntimeError("service unavailable")

    ranker = Ranker(
        {"use_llm_ranker": True},
        client=SimpleNamespace(messages=FailingMessages()),
    )
    session = SessionMemory({})
    candidates = _candidates(3)

    result = ranker.rerank(candidates, session, top_k=2)

    assert result == candidates[:2]
    assert ranker.token_usage == {"prompt_tokens": 0, "completion_tokens": 0}


def test_ranker_invalid_json_preserves_response_usage_and_falls_back():
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=5, output_tokens=1),
        content=[SimpleNamespace(text="not valid json")],
    )

    class InvalidJsonMessages:
        def create(self, **_kwargs):
            return response

    ranker = Ranker(
        {"use_llm_ranker": True},
        client=SimpleNamespace(messages=InvalidJsonMessages()),
    )
    session = SessionMemory({})
    candidates = _candidates(3)

    result = ranker.rerank(candidates, session, top_k=2)

    assert result == candidates[:2]
    assert ranker.token_usage == {"prompt_tokens": 5, "completion_tokens": 1}


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
