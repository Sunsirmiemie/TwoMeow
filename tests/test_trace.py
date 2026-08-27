"""Public behavior tests for optional per-turn Agent trace data."""
from __future__ import annotations

import json
import inspect
from types import SimpleNamespace

from src.agent.orchestrator import Agent
from src.agent.state import SessionMemory
from src.ranking.reranker import Ranker
from src.retrieval.hybrid import HybridRetriever


def _catalog(tmp_path):
    path = tmp_path / "catalog.jsonl"
    products = [
        {
            "parent_asin": "A1",
            "title": "Cotton desk lamp",
            "features": ["adjustable", "office"],
            "price": 20,
        },
        {
            "parent_asin": "A2",
            "title": "Metal desk lamp",
            "features": ["dimmable", "reading"],
            "price": 30,
        },
    ]
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )
    return path


def _one_turn(agent: Agent) -> dict:
    agent.reset("session", {})
    return agent.respond("session", "desk lamp", turn=1, top_k=10)


def test_trace_disabled_is_absent_and_preserves_response(tmp_path):
    catalog = _catalog(tmp_path)
    implicit = Agent(str(catalog), {"use_dense": False})
    explicit = Agent(
        str(catalog),
        {"use_dense": False, "trace_enabled": False},
    )

    implicit_response = _one_turn(implicit)
    explicit_response = _one_turn(explicit)

    assert implicit.config["trace_enabled"] is False
    assert explicit_response == implicit_response
    assert "debug_trace" not in implicit_response


def test_trace_enabled_adds_stage_snapshots_without_changing_recommendations(tmp_path):
    catalog = _catalog(tmp_path)
    baseline = _one_turn(Agent(str(catalog), {"use_dense": False}))
    traced = _one_turn(
        Agent(
            str(catalog),
            {"use_dense": False, "trace_enabled": True},
        )
    )

    debug = traced.pop("debug_trace")

    assert traced == baseline
    assert debug["schema_version"] == 1
    assert debug["turn"] == 1
    assert debug["retrieval"]["bm25"]["status"] == "applied"
    assert debug["retrieval"]["dense"] == {
        "enabled": False,
        "applied": False,
        "status": "disabled",
        "candidates": [],
    }
    assert debug["retrieval"]["rrf"]["applied"] is False
    assert debug["retrieval"]["rrf"]["status"] == "not_applied"
    assert debug["retrieval"]["rrf"]["candidates"] == []
    assert debug["retrieval"]["output"]["candidates"]

    for stage in (debug["retrieval"]["bm25"], debug["retrieval"]["output"]):
        assert stage["candidates"]
        assert set(stage["candidates"][0]) == {"parent_asin", "rank", "score"}

    assert debug["rerank_pool"]["candidates"]
    assert debug["rerank_pool"]["candidate_count"] == len(
        debug["rerank_pool"]["candidates"]
    )
    assert debug["final"]["candidates"] == [
        {
            "parent_asin": item["parent_asin"],
            "rank": rank,
            "score": item["score"],
        }
        for rank, item in enumerate(traced["recommendations"], start=1)
    ]


def test_agent_trace_interface_has_no_target_or_ground_truth_input(tmp_path):
    catalog = _catalog(tmp_path)
    agent = Agent(
        str(catalog),
        {"use_dense": False, "trace_enabled": True},
    )
    response = _one_turn(agent)

    constructor_parameters = inspect.signature(Agent).parameters
    respond_parameters = inspect.signature(Agent.respond).parameters
    assert "target_asin" not in constructor_parameters
    assert "target_asin" not in respond_parameters
    assert "ground_truth" not in constructor_parameters
    assert "ground_truth" not in respond_parameters

    def keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield str(key).lower()
                yield from keys(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from keys(nested)

    debug_keys = set(keys(response["debug_trace"]))
    assert not any("target" in key or "ground_truth" in key for key in debug_keys)


def test_dialogue_trace_reproduces_the_selected_attribute_and_stop_decision(tmp_path):
    catalog = _catalog(tmp_path)
    agent = Agent(
        str(catalog),
        {
            "use_dense": False,
            "trace_enabled": True,
            "entropy_tau": 0.0,
        },
    )
    response = _one_turn(agent)
    debug = response["debug_trace"]
    dialogue = debug["dialogue"]

    assert dialogue["query"] == debug["retrieval"]["query"]
    assert dialogue["track"] == debug["retrieval"]["track"]
    assert dialogue["slots"] == {}
    assert dialogue["candidate_count"] == len(
        debug["retrieval"]["output"]["candidates"]
    )
    assert dialogue["eligible_attributes"] == dialogue["remaining_attributes"]
    assert set(dialogue["entropy_scores"]) == set(dialogue["remaining_attributes"])
    assert dialogue["max_entropy_score"] == max(dialogue["entropy_scores"].values())
    assert dialogue["tau"] == 0.0
    assert dialogue["early_stop"] == {"enabled": True, "triggered": False}
    assert dialogue["chosen_ask_attribute"] == response["ask_attribute"]
    assert dialogue["question_decision"]["attribute"] == response["ask_attribute"]


def test_disabled_ranker_trace_is_not_attempted_and_never_calls_api_boundary(tmp_path):
    class ExplodingMessages:
        def create(self, **_kwargs):
            raise AssertionError("disabled ranker called the external API boundary")

    candidates = [{"parent_asin": "A1", "score": 1.0}]
    ranker = Ranker(
        {"use_llm_ranker": False},
        client=SimpleNamespace(messages=ExplodingMessages()),
    )

    ranked, ranker_trace = ranker.rerank_with_trace(
        candidates,
        SessionMemory({}),
        top_k=10,
    )

    assert ranked == candidates
    assert ranker_trace["status"] == "disabled"
    assert ranker_trace["attempt_status"] == "not_attempted"
    assert ranker_trace["attempted"] is False
    assert ranker_trace["api_pool"]["candidates"] == [
        {"parent_asin": "A1", "rank": 1, "score": 1.0}
    ]
    assert ranker_trace["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}

    agent_response = _one_turn(
        Agent(
            str(_catalog(tmp_path)),
            {
                "use_dense": False,
                "use_llm_ranker": False,
                "trace_enabled": True,
            },
        )
    )
    assert agent_response["debug_trace"]["ranker"]["status"] == "disabled"
    assert agent_response["debug_trace"]["ranker"]["attempt_status"] == "not_attempted"


def test_dense_and_rrf_trace_capture_the_same_hybrid_output(tmp_path):
    retriever = HybridRetriever(
        str(_catalog(tmp_path)),
        {
            "use_dense": True,
            "browsing_weights": [[0.5, 0.5]],
        },
    )

    class FakeDense:
        def search(self, _query, top_k):
            return [{"parent_asin": "DENSE", "score": 0.9}][:top_k]

    retriever.dense = FakeDense()
    baseline = retriever.retrieve("desk lamp", {}, "browsing", top_k=3)
    traced, debug = retriever.retrieve_with_trace(
        "desk lamp",
        {},
        "browsing",
        top_k=3,
    )

    assert traced == baseline
    assert debug["dense"]["enabled"] is True
    assert debug["dense"]["applied"] is True
    assert debug["dense"]["status"] == "applied"
    assert debug["dense"]["candidates"][0]["parent_asin"] == "DENSE"
    assert debug["rrf"]["applied"] is True
    assert debug["rrf"]["candidates"] == debug["output"]["candidates"]
    json.dumps(debug)


def test_ranker_trace_distinguishes_empty_success_and_fallback():
    session = SessionMemory({})
    candidates = [
        {"parent_asin": "A1", "score": 2.0},
        {"parent_asin": "A2", "score": 1.0},
    ]

    success_response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=5, output_tokens=2),
        content=[SimpleNamespace(text='["A2", "A1"]')],
    )

    class SuccessfulMessages:
        def create(self, **_kwargs):
            return success_response

    success_ranker = Ranker(
        {"use_llm_ranker": True},
        client=SimpleNamespace(messages=SuccessfulMessages()),
    )
    success_output, success_trace = success_ranker.rerank_with_trace(
        candidates,
        session,
        top_k=2,
    )
    assert [item["parent_asin"] for item in success_output] == ["A2", "A1"]
    assert success_trace["status"] == "api_success"
    assert success_trace["attempt_status"] == "api_success"
    assert success_trace["usage"] == {"prompt_tokens": 5, "completion_tokens": 2}

    class FailingMessages:
        def create(self, **_kwargs):
            raise RuntimeError("offline")

    fallback_ranker = Ranker(
        {"use_llm_ranker": True},
        client=SimpleNamespace(messages=FailingMessages()),
    )
    fallback_output, fallback_trace = fallback_ranker.rerank_with_trace(
        candidates,
        session,
        top_k=2,
    )
    assert fallback_output == candidates
    assert fallback_trace["status"] == "fallback"
    assert fallback_trace["attempt_status"] == "fallback"
    assert fallback_trace["error_type"] == "RuntimeError"

    empty_output, empty_trace = fallback_ranker.rerank_with_trace([], session, top_k=2)
    assert empty_output == []
    assert empty_trace["status"] == "empty"
    assert empty_trace["attempt_status"] == "not_attempted"
