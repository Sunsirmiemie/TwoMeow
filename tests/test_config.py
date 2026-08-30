"""Tests for loading the checked-in agent configuration."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config


def test_load_config_flattens_agent_defaults():
    config = load_config()

    assert config == {
        "retrieval_top_k": 100,
        "use_dynamic_attribute_scoring": True,
        "dynamic_field_gain": 0.65,
        "dynamic_attribute_max_weight": 0.52,
        "field_weights": {
            "title": 6.0,
            "categories": 4.0,
            "features": 2.5,
            "description": 2.5,
            "store": 1.5,
            "details": 1.0,
        },
        "rrf_k": 60,
        "bm25_base": 0.75,
        "dense_base": 0.25,
        "browsing_weights": [[0.5, 0.5], [0.6, 0.4]],
        "use_adaptive_fusion": True,
        "use_dense_risk_gate": True,
        "dense_gate_min_bm25": 20,
        "dense_gate_max_confidence": 0.12,
        "use_dense": True,
        "dense_model": "all-MiniLM-L6-v2",
        "dense_batch_size": 512,
        "dense_max_seq_length": 256,
        "dense_device": "auto",
        "dense_query_prefix": "",
        "dense_document_prefix": "",
        "use_field_aware_dense": True,
        "use_llm_ranker": False,
        "use_features": True,
        "ranker_model": "claude-haiku-4-5-20251001",
        "rerank_top_n": 20,
        "mmr_lambda": 1.0,
        "profile_weight": 0.15,
        "use_field_aware_slot_coverage": True,
        "feature_weights": {
            "base": 1.0,
            "attribute": 1.6,
            "negative": 1.4,
            "slot": 0.8,
            "category": 1.0,
            "popularity": 0.15,
            "price": 0.4,
        },
        "entropy_tau": 0.3,
        "use_reply_purification": True,
        "override_carryover_confidence": 0.35,
        "min_pool_for_dynamic": 10,
        "few_slots_threshold": 2,
        "pool_size_threshold": 50,
        "truncated_size": 20,
        "use_dynamic_entropy": True,
        "use_early_stop": True,
        "use_override_detection": True,
    }


def test_agent_merges_flat_overrides_over_yaml_defaults():
    from src.agent.orchestrator import Agent

    product = {"parent_asin": "A1", "title": "Desk lamp", "price": 10}
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "catalog.jsonl"
        catalog_path.write_text(json.dumps(product) + "\n", encoding="utf-8")
        agent = Agent(
            str(catalog_path),
            {"use_dense": False, "rerank_top_n": 7},
        )

    assert agent.config["use_dense"] is False
    assert agent.config["rerank_top_n"] == 7
    assert agent.config["use_early_stop"] is True
    assert agent.config["dense_model"] == "all-MiniLM-L6-v2"


def test_load_config_does_not_share_nested_override_values():
    overrides = {
        "field_weights": {"title": 9.0},
        "browsing_weights": [[0.2, 0.8]],
    }

    config = load_config(overrides)
    config["field_weights"]["title"] = 1.0
    config["browsing_weights"][0][0] = 0.9

    assert overrides == {
        "field_weights": {"title": 9.0},
        "browsing_weights": [[0.2, 0.8]],
    }


def test_agent_applies_retrieval_limit_and_bm25_field_weight_overrides():
    from src.agent.orchestrator import Agent

    products = [
        {"parent_asin": f"A{i}", "title": f"Desk lamp model {i}", "price": 10 + i}
        for i in range(3)
    ]
    weights = {
        "title": 9.0,
        "categories": 4.0,
        "features": 2.5,
        "description": 2.5,
        "store": 1.5,
        "details": 1.0,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        agent = Agent(
            str(catalog_path),
            {
                "use_dense": False,
                "retrieval_top_k": 1,
                "field_weights": weights,
            },
        )
        agent.reset("s1", {})

        response = agent.respond("s1", "desk lamp", turn=1, top_k=10)

    assert len(response["recommendations"]) == 1
    assert agent.retriever.bm25.field_weights == weights


def test_agent_deep_merges_partial_field_weights_without_mutating_caller(tmp_path):
    from src.agent.orchestrator import Agent

    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(
        json.dumps({"parent_asin": "A1", "title": "Desk lamp"}) + "\n",
        encoding="utf-8",
    )
    overrides = {
        "use_dense": False,
        "field_weights": {"title": 9.0},
    }
    expected_weights = {
        "title": 9.0,
        "categories": 4.0,
        "features": 2.5,
        "description": 2.5,
        "store": 1.5,
        "details": 1.0,
    }

    agent = Agent(str(catalog_path), overrides)

    assert agent.config["field_weights"] == expected_weights
    assert agent.retriever.bm25.field_weights == expected_weights
    assert overrides == {
        "use_dense": False,
        "field_weights": {"title": 9.0},
    }


def test_agent_passes_dense_model_and_batch_size_to_encoder(tmp_path, monkeypatch):
    import src.retrieval.dense as dense_module
    from src.agent.orchestrator import Agent

    encode_calls = []
    encoded_texts = []

    class FakeModel:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.device = kwargs.get("device", "cpu")
            self.max_seq_length = None

        def encode(self, texts, **kwargs):
            encoded_texts.append(list(texts))
            encode_calls.append(kwargs)
            return np.ones((len(texts), 2))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeModel),
    )
    monkeypatch.setattr(dense_module, "_CACHE_DIR", tmp_path / "cache")
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(
        json.dumps({"parent_asin": "A1", "title": "Desk lamp"}) + "\n",
        encoding="utf-8",
    )

    agent = Agent(
        str(catalog_path),
        {
            "use_dense": True,
            "dense_model": "fake-encoder",
            "dense_batch_size": 7,
            "dense_max_seq_length": 123,
            "dense_device": "mps",
            "dense_query_prefix": "query: ",
            "dense_document_prefix": "passage: ",
        },
    )
    # Dense is intentionally lazy so a strong BM25 result does not pay model
    # loading/index costs; force creation here to verify encoder configuration.
    agent.retriever._get_dense()

    assert agent.retriever.dense.model_name == "fake-encoder"
    assert agent.retriever.dense.batch_size == 7
    assert agent.retriever.dense.max_seq_length == 123
    assert agent.retriever.dense.model.max_seq_length == 123
    assert agent.retriever.dense.requested_device == "mps"
    assert agent.retriever.dense.device == "mps"
    assert agent.retriever.dense.query_prefix == "query: "
    assert agent.retriever.dense.document_prefix == "passage: "
    assert encode_calls[0]["batch_size"] == 7
    assert len(encode_calls) == 2
    assert all(texts[0].startswith("passage: ") for texts in encoded_texts)
    assert agent.retriever.dense.embeddings is None
    assert agent.retriever.dense.identity_embeddings is not None
    assert agent.retriever.dense.attribute_embeddings is not None

    session = SimpleNamespace(
        slots={"category": "lighting", "color": "black"},
        slot_confidence={"category": 1.0, "color": 1.0},
    )
    agent.retriever.dense.search("desk lamp", top_k=1, session=session)
    assert encoded_texts[-1] == [
        "query: lighting desk lamp",
        "query: black desk lamp",
    ]


def test_agent_applies_rerank_pool_threshold_overrides(tmp_path):
    from src.agent.orchestrator import Agent

    products = [
        {"parent_asin": f"A{i}", "title": f"Desk lamp model {i}"}
        for i in range(5)
    ]
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )
    agent = Agent(
        str(catalog_path),
        {
            "use_dense": False,
            "use_early_stop": False,
            "few_slots_threshold": 99,
            "pool_size_threshold": 1,
            "truncated_size": 2,
        },
    )
    agent.reset("s1", {})

    response = agent.respond("s1", "desk lamp", turn=1, top_k=10)

    # Pool truncation only applies when the LLM reranker is enabled.
    # With the offline feature reranker, the full candidate pool is used.
    assert len(response["recommendations"]) == 5


def test_agent_applies_entropy_threshold_and_dynamic_pool_override(tmp_path):
    from src.agent.orchestrator import Agent

    materials = ["cotton", "leather", "wool", "silk"]
    products = [
        {
            "parent_asin": f"A{i}",
            "title": f"{materials[i % len(materials)]} desk lamp {i}",
        }
        for i in range(20)
    ]
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )
    agent = Agent(
        str(catalog_path),
        {
            "use_dense": False,
            "entropy_tau": 0.9,
            "min_pool_for_dynamic": 3,
        },
    )
    agent.reset("s1", {})

    response = agent.respond("s1", "desk lamp", turn=1, top_k=10)

    assert response["ask_attribute"] == "other"
    assert agent.clarifier.min_pool_for_dynamic == 3


def test_cli_flags_only_override_yaml_when_explicitly_present():
    from run_eval import build_parser as build_local_parser
    from scripts.run_public_eval import build_parser as build_public_parser
    from src.config.cli import agent_overrides

    for build_parser in (build_local_parser, build_public_parser):
        assert agent_overrides(build_parser().parse_args([])) == {}
        assert agent_overrides(build_parser().parse_args(["--no-dense"])) == {
            "use_dense": False,
        }
        assert agent_overrides(build_parser().parse_args(["--llm-rank"])) == {
            "use_llm_ranker": True,
        }


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
