"""Evaluator-owned trace and provenance behavior tests."""
from __future__ import annotations

import json
import csv
import subprocess
import shutil
import uuid
import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator.local_evaluator import evaluate


def _evaluation_inputs(target: str = "TARGET_SECRET"):
    sample = {
        "sample_id": "sample-1",
        "scenario_type": "buying",
        "user_profile": {"preference_tags": ["simple"]},
        "ground_truth": {"parent_asin": target},
        "intent_card": {
            "target_category": "desk lamp",
            "hard_constraints": ["adjustable"],
            "soft_preferences": ["black"],
        },
        "behavior": {"scenario_type": "buying"},
    }
    products = {
        target: {"parent_asin": target, "title": "Adjustable desk lamp"},
    }
    return [sample], {target}, {target: ["Home", "Lamps"]}, products


def _debug_trace(target: str) -> dict:
    candidate = {"parent_asin": target, "rank": 1, "score": 1.0}
    return {
        "schema_version": 1,
        "retrieval": {
            "bm25": {"status": "applied", "candidates": [candidate]},
            "dense": {
                "enabled": False,
                "applied": False,
                "status": "disabled",
                "candidates": [],
            },
            "rrf": {
                "applied": False,
                "status": "not_applied",
                "candidates": [candidate],
            },
        },
        "rerank_pool": {"candidates": [candidate]},
        "ranker": {
            "api_pool": {"candidates": [candidate]},
            "output": {"candidates": [candidate]},
        },
        "final": {"candidates": [candidate]},
        "dialogue": {"chosen_ask_attribute": "feature"},
    }


def test_evaluator_owns_target_and_emits_one_ranked_record_per_turn():
    target = "TARGET_SECRET"
    calls: list[tuple] = []
    records: list[dict] = []

    class RecordingAgent:
        def __init__(self, catalog_path, config):
            calls.append(("init", catalog_path, config))

        def reset(self, session_id, user_profile):
            calls.append(("reset", session_id, user_profile))

        def respond(self, session_id, user_message, turn, top_k):
            calls.append(("respond", session_id, user_message, turn, top_k))
            return {
                "message": "candidate",
                "ask_attribute": "feature",
                "recommendations": [{"parent_asin": target, "score": 1.0}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "debug_trace": _debug_trace(target),
            }

    agent = RecordingAgent("catalog.jsonl", {"trace_enabled": True})
    result = evaluate(
        agent,
        *_evaluation_inputs(target),
        trace_sink=records.append,
        run_id="run-1",
    )

    assert result["recommended_technical_score"] == 1.0
    assert len(records) == 1
    record = records[0]
    assert record["run_id"] == "run-1"
    assert record["sample_id"] == "sample-1"
    assert record["turn"] == 1
    assert record["hit_eligible"] is True
    assert record["target_ranks"] == {
        "bm25": 1,
        "dense": None,
        "rrf": None,
        "rerank_pool": 1,
        "ranker_api_pool": 1,
        "ranker": 1,
        "final": 1,
    }

    # The target may naturally occur in Agent-produced candidates, but it is
    # never supplied to reset/respond or copied into a labeled trace field.
    assert target not in json.dumps(calls)
    assert "target_asin" not in record
    assert "ground_truth" not in record


def test_evaluator_ignores_agent_supplied_candidate_rank_metadata():
    target = "TARGET_SECRET"
    decoy = {"parent_asin": "DECOY", "rank": 1, "score": 2.0}
    adversarial_target = {"parent_asin": target, "rank": 999, "score": 1.0}
    records = []

    class AdversarialRankAgent:
        def reset(self, _session_id, _profile):
            pass

        def respond(self, _session_id, _message, _turn, _top_k):
            candidates = [decoy, adversarial_target]
            return {
                "message": "candidates",
                "ask_attribute": None,
                "recommendations": [
                    {"parent_asin": "DECOY", "score": 2.0},
                    {"parent_asin": target, "score": 1.0},
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "debug_trace": {
                    "retrieval": {
                        "bm25": {"candidates": candidates},
                        "dense": {"applied": False, "candidates": []},
                        "rrf": {"applied": False, "candidates": []},
                    },
                    "rerank_pool": {"candidates": candidates},
                    "ranker": {
                        "api_pool": {"candidates": candidates},
                        "output": {"candidates": candidates},
                    },
                    "final": {"candidates": candidates},
                },
            }

    samples, catalog_ids, categories, products = _evaluation_inputs(target)
    catalog_ids.add("DECOY")
    products["DECOY"] = {"parent_asin": "DECOY", "title": "Decoy"}
    evaluate(
        AdversarialRankAgent(),
        samples,
        catalog_ids,
        categories,
        products,
        trace_sink=records.append,
        run_id="adversarial-rank",
    )

    ranks = records[0]["target_ranks"]
    assert ranks["bm25"] == 2
    assert ranks["rerank_pool"] == 2
    assert ranks["ranker_api_pool"] == 2
    assert ranks["ranker"] == 2
    assert ranks["final"] == 2


def test_trace_artifacts_stream_strict_json_into_unique_run_directories(tmp_path):
    from evaluator.trace_artifacts import TraceArtifacts

    sentinel = tmp_path / "results.json"
    sentinel.write_text("historical", encoding="utf-8")

    first = TraceArtifacts.create(tmp_path)
    second = TraceArtifacts.create(tmp_path)
    try:
        first({"turn": 1, "score": 1.0})
        first({"turn": 2, "score": None})
        with pytest.raises(ValueError):
            first({"turn": 3, "score": float("nan")})
    finally:
        first.close()
        second.close()

    assert first.run_dir != second.run_dir
    assert first.run_dir.parent == tmp_path
    assert second.run_dir.parent == tmp_path
    assert sentinel.read_text(encoding="utf-8") == "historical"
    assert [json.loads(line) for line in first.trace_path.read_text().splitlines()] == [
        {"turn": 1, "score": 1.0},
        {"turn": 2, "score": None},
    ]
    assert {path.name for path in first.run_dir.iterdir()} == {"trace.jsonl"}


def test_failure_report_uses_hit_eligible_turns_and_actual_ranker_api_pool(tmp_path):
    from evaluator.trace_artifacts import FAILURE_COLUMNS, write_failure_report

    trace_path = tmp_path / "trace.jsonl"
    records = [
        # sample-a is BM25-only: target was recalled but fell before API pool.
        {
            "session_id": "session-a",
            "sample_id": "sample-a",
            "hit_eligible": True,
            "target_ranks": {
                "bm25": 42, "dense": None, "rrf": None,
                "rerank_pool": 42, "ranker_api_pool": None,
                "ranker": None, "final": None,
            },
            "agent_debug": {"retrieval": {"rrf": {"applied": False}}},
        },
        # sample-b is fused: a source saw the target but RRF dropped it.
        {
            "session_id": "session-b",
            "sample_id": "sample-b",
            "hit_eligible": True,
            "target_ranks": {
                "bm25": None, "dense": 5, "rrf": None,
                "rerank_pool": None, "ranker_api_pool": None,
                "ranker": None, "final": None,
            },
            "agent_debug": {"retrieval": {"rrf": {"applied": True}}},
        },
        # Pre-override ranks must not contribute to sample-c best ranks.
        {
            "session_id": "session-c",
            "sample_id": "sample-c",
            "hit_eligible": False,
            "target_ranks": {
                "bm25": 1, "dense": None, "rrf": None,
                "rerank_pool": 1, "ranker_api_pool": 1,
                "ranker": 1, "final": 1,
            },
            "agent_debug": {"retrieval": {"rrf": {"applied": False}}},
        },
        {
            "session_id": "session-c",
            "sample_id": "sample-c",
            "hit_eligible": True,
            "target_ranks": {
                "bm25": None, "dense": None, "rrf": None,
                "rerank_pool": None, "ranker_api_pool": None,
                "ranker": None, "final": None,
            },
            "agent_debug": {"retrieval": {"rrf": {"applied": False}}},
        },
        {
            "session_id": "session-hit",
            "sample_id": "sample-hit",
            "hit_eligible": True,
            "target_ranks": {
                "bm25": 3, "dense": None, "rrf": None,
                "rerank_pool": 3, "ranker_api_pool": 3,
                "ranker": 2, "final": 2,
            },
            "agent_debug": {"retrieval": {"rrf": {"applied": False}}},
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    result = {
        "sessions": [
            {"sample_id": "sample-a", "scenario_type": "buying", "hit": False, "first_hit_turn": None, "best_rank": None},
            {"sample_id": "sample-b", "scenario_type": "browsing", "hit": False, "first_hit_turn": None, "best_rank": None},
            {"sample_id": "sample-c", "scenario_type": "intent_override", "hit": False, "first_hit_turn": None, "best_rank": None},
            {"sample_id": "sample-hit", "scenario_type": "buying", "hit": True, "first_hit_turn": 1, "best_rank": 2},
        ]
    }
    report_path = tmp_path / "failure_report.csv"

    write_failure_report(trace_path, result, report_path)

    with report_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0].keys() == dict.fromkeys(FAILURE_COLUMNS).keys()
    by_sample = {row["session_id"]: row for row in rows}
    assert by_sample["session-a"]["bm25_best_rank"] == "42"
    assert by_sample["session-a"]["pool_best_rank"] == ""
    assert by_sample["session-a"]["earliest_failure_stage"] == "pool"
    assert by_sample["session-b"]["dense_best_rank"] == "5"
    assert by_sample["session-b"]["earliest_failure_stage"] == "rrf"
    assert by_sample["session-c"]["bm25_best_rank"] == ""
    assert by_sample["session-c"]["earliest_failure_stage"] == "bm25"
    assert by_sample["session-hit"]["first_hit_rank"] == "2"
    assert by_sample["session-hit"]["earliest_failure_stage"] == ""


def test_manifest_captures_reproducible_inputs_and_is_finalized_atomically(tmp_path):
    from src.agent.orchestrator import Agent
    from evaluator.trace_artifacts import (
        TraceArtifacts,
        build_run_manifest,
        write_manifest_atomic,
    )

    catalog = tmp_path / "catalog.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    catalog.write_text(
        json.dumps({"parent_asin": "A1", "title": "Desk lamp"}) + "\n",
        encoding="utf-8",
    )
    dataset.write_text(json.dumps({"sample_id": "s1"}) + "\n", encoding="utf-8")
    artifacts = TraceArtifacts.create(tmp_path / "runs")
    agent = Agent(
        str(catalog),
        {"use_dense": False, "use_llm_ranker": False, "trace_enabled": True},
    )
    start = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    end = start + timedelta(seconds=1.25)

    manifest = build_run_manifest(
        artifacts=artifacts,
        agent=agent,
        catalog_path=catalog,
        dataset_path=dataset,
        cli_args={"no_dense": True, "artifacts_root": str(tmp_path / "runs")},
        repo_root=".",
        started_at=start,
        ended_at=end,
        status="completed",
    )
    write_manifest_atomic(artifacts.manifest_path, manifest)
    artifacts.close()

    persisted = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert persisted["run_id"] == artifacts.run_id
    assert persisted["status"] == "completed"
    assert persisted["git"]["commit"]
    expected_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        text=True,
    ).strip()
    assert persisted["git"]["branch"] == expected_branch
    assert isinstance(persisted["git"]["dirty"], bool)
    assert persisted["git"]["source_diff_sha256"]
    assert persisted["git_end"]["source_diff_sha256"]
    assert persisted["source_changed_during_run"] is False
    assert persisted["config"] == agent.config
    assert persisted["python"]["implementation"]
    assert persisted["python"]["version"]
    assert persisted["dependencies"]
    assert persisted["catalog"]["sha256"]
    assert persisted["dataset"]["sha256"]
    assert persisted["dense"]["configured_model"] == "all-MiniLM-L6-v2"
    assert persisted["dense"]["enabled"] is False
    assert persisted["dense"]["applied"] is False
    assert persisted["dense"]["embedding_cache_start"] == {
        "path": persisted["dense"]["embedding_cache_start"]["path"],
        "exists": False,
        "sha256": None,
    }
    assert (
        persisted["dense"]["embedding_cache_end"]
        == persisted["dense"]["embedding_cache_start"]
    )
    assert persisted["dense"]["embedding_cache_changed_during_run"] is False
    assert persisted["dense"]["model_cache_end"] == persisted["dense"]["model_cache_start"]
    assert persisted["dense"]["model_cache_changed_during_run"] is False
    assert persisted["dense"]["model_cache_start"]["repository"] == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert set(persisted["dense"]["model_cache_start"]) == {
        "repository", "search_roots", "candidates", "resolution", "exists", "sha256",
    }
    assert persisted["catalog_end"] == persisted["catalog"]
    assert persisted["dataset_end"] == persisted["dataset"]
    assert persisted["catalog_changed_during_run"] is False
    assert persisted["dataset_changed_during_run"] is False
    assert persisted["llm"]["enabled"] is False
    assert persisted["llm"]["applied"] is False
    assert persisted["timing"]["duration_seconds"] == 1.25
    assert datetime.fromisoformat(
        persisted["timing"]["provenance_captured_at_utc"]
    ) >= datetime.fromisoformat(persisted["timing"]["ended_at_utc"])
    assert persisted["artifacts"]["trace"] == str(artifacts.trace_path.resolve())
    assert not list(artifacts.run_dir.glob("*.tmp"))


def test_manifest_recursively_redacts_secret_config_values_without_dropping_keys(tmp_path):
    from evaluator.trace_artifacts import capture_start_provenance
    from src.config import load_config

    catalog = tmp_path / "catalog.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    catalog.write_text(json.dumps({"parent_asin": "A1"}) + "\n", encoding="utf-8")
    dataset.write_text("", encoding="utf-8")
    config = load_config({
        "api_key": "api-secret",
        "nested": {
            "clientSecret": "client-secret",
            "refresh_token": "refresh-secret",
            "password_hash": "password-secret",
            "credential": {"username": "also-sensitive-as-a-whole"},
            "safe_value": "unchanged",
            "max_tokens": 512,
            "tokenizer_model": "unchanged-tokenizer",
        },
        "providers": [
            {"name": "offline", "accessToken": "provider-secret"},
        ],
        "provenance_variants": {
            "private_key": "snake-private",
            "privateKey": "camel-private",
            "PRIVATE_KEY": "upper-private",
            "access_key": "snake-access",
            "accessKey": "camel-access",
            "AWS_ACCESS_KEY_ID": "upper-access",
            "awsAccessKeyId": "camel-aws-access",
            "secret_key": "snake-secret",
            "secretKey": "camel-secret",
            "signing_key": "snake-signing",
            "signingKey": "camel-signing",
            "authorization": "authorization-secret",
            "auth_header": "auth-header-secret",
            "authHeader": "camel-auth-header-secret",
            "bearer": "bearer-secret",
            "public_key": "safe-public-key",
            "publicKey": "safe-camel-public-key",
            "keyboard_layout": "safe-keyboard",
        },
    })

    manifest = capture_start_provenance(
        config=config,
        catalog_path=catalog,
        dataset_path=dataset,
        cli_args={},
        repo_root=Path(__file__).resolve().parent.parent,
        artifacts_root=tmp_path / "runs",
        started_at=datetime.now(timezone.utc),
    )

    stored = manifest["config"]
    assert stored["api_key"] == "[REDACTED]"
    assert stored["nested"]["clientSecret"] == "[REDACTED]"
    assert stored["nested"]["refresh_token"] == "[REDACTED]"
    assert stored["nested"]["password_hash"] == "[REDACTED]"
    assert stored["nested"]["credential"] == "[REDACTED]"
    assert stored["providers"][0]["accessToken"] == "[REDACTED]"
    assert stored["nested"]["safe_value"] == "unchanged"
    assert stored["nested"]["max_tokens"] == 512
    assert stored["nested"]["tokenizer_model"] == "unchanged-tokenizer"
    variants = stored["provenance_variants"]
    for key in (
        "private_key", "privateKey", "PRIVATE_KEY",
        "access_key", "accessKey", "AWS_ACCESS_KEY_ID", "awsAccessKeyId",
        "secret_key", "secretKey", "signing_key", "signingKey",
        "authorization", "auth_header", "authHeader", "bearer",
    ):
        assert variants[key] == "[REDACTED]"
    assert variants["public_key"] == "safe-public-key"
    assert variants["publicKey"] == "safe-camel-public-key"
    assert variants["keyboard_layout"] == "safe-keyboard"
    assert set(stored) == set(config)


def test_trace_runner_is_offline_and_writes_exactly_four_unique_artifacts(tmp_path, monkeypatch):
    from scripts.run_trace_eval import run_trace_evaluation
    from src.ranking.reranker import Ranker

    target = "A1"
    samples, _ids, _categories, products = _evaluation_inputs(target)
    catalog = tmp_path / "catalog.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    catalog.write_text(
        "".join(json.dumps(product) + "\n" for product in products.values()),
        encoding="utf-8",
    )
    dataset.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )

    def explode(_self):
        raise AssertionError("offline trace runner crossed the external API boundary")

    monkeypatch.setattr(Ranker, "_get_client", explode)

    first_dir, first_result = run_trace_evaluation(
        catalog_path=catalog,
        dataset_path=dataset,
        artifacts_root=tmp_path / "runs",
        no_dense=True,
    )
    second_dir, _second_result = run_trace_evaluation(
        catalog_path=catalog,
        dataset_path=dataset,
        artifacts_root=tmp_path / "runs",
        no_dense=True,
    )

    assert first_dir != second_dir
    assert {path.name for path in first_dir.iterdir()} == {
        "results.json", "trace.jsonl", "run_manifest.json", "failure_report.csv",
    }
    assert first_result["reported_token_usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    result_on_disk = json.loads((first_dir / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((first_dir / "run_manifest.json").read_text(encoding="utf-8"))
    trace_lines = (first_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert result_on_disk == first_result
    assert manifest["status"] == "completed"
    assert manifest["offline_enforced"] is True
    assert manifest["config"]["trace_enabled"] is True
    assert manifest["config"]["use_dense"] is False
    assert manifest["config"]["use_llm_ranker"] is False
    assert manifest["llm"]["applied"] is False
    assert len(trace_lines) == 1
    json.loads(trace_lines[0], parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def test_evaluation_metrics_and_sessions_match_with_trace_on_or_off(tmp_path):
    from src.agent.orchestrator import Agent

    target = "A1"
    samples, catalog_ids, categories, products = _evaluation_inputs(target)
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "".join(json.dumps(product) + "\n" for product in products.values()),
        encoding="utf-8",
    )
    baseline = Agent(
        str(catalog),
        {"use_dense": False, "use_llm_ranker": False, "trace_enabled": False},
    )
    traced = Agent(
        str(catalog),
        {"use_dense": False, "use_llm_ranker": False, "trace_enabled": True},
    )
    records = []

    baseline_result = evaluate(
        baseline, samples, catalog_ids, categories, products
    )
    traced_result = evaluate(
        traced,
        samples,
        catalog_ids,
        categories,
        products,
        trace_sink=records.append,
        run_id="parity-run",
    )

    assert traced_result == baseline_result
    assert records


def test_missing_catalog_preserves_original_error_and_leaves_only_failed_manifest(tmp_path):
    from scripts.run_trace_eval import run_trace_evaluation

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")
    runs = tmp_path / "runs"

    with pytest.raises(FileNotFoundError):
        run_trace_evaluation(
            catalog_path=tmp_path / "missing-catalog.jsonl",
            dataset_path=dataset,
            artifacts_root=runs,
            no_dense=True,
        )

    run_dirs = list(runs.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert {path.name for path in run_dir.iterdir()} == {"run_manifest.json"}
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "FileNotFoundError"
    assert manifest["catalog"]["exists"] is False
    assert not list(run_dir.glob("*.tmp"))


def test_provenance_finalization_failure_does_not_mask_agent_init_error(tmp_path, monkeypatch):
    import scripts.run_trace_eval as trace_runner

    catalog = tmp_path / "catalog.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    catalog.write_text(json.dumps({"parent_asin": "A1"}) + "\n", encoding="utf-8")
    dataset.write_text("", encoding="utf-8")

    observed = {}

    class FailingAgent:
        def __init__(self, _catalog, _config):
            observed.update({
                name: os.environ.get(name)
                for name in trace_runner._DENSE_OFFLINE_ENV
            })
            raise RuntimeError("original agent init error")

    def fail_manifest(**_kwargs):
        raise ValueError("secondary provenance failure")

    monkeypatch.setattr(trace_runner, "Agent", FailingAgent)
    monkeypatch.setattr(trace_runner, "finalize_run_manifest", fail_manifest)
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous-value")
    for name in (
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        monkeypatch.delenv(name, raising=False)
    runs = tmp_path / "runs"

    with pytest.raises(RuntimeError, match="original agent init error"):
        trace_runner.run_trace_evaluation(
            catalog_path=catalog,
            dataset_path=dataset,
            artifacts_root=runs,
        )

    manifest = json.loads(
        (next(runs.iterdir()) / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["offline_enforced"] is True
    assert manifest["error"]["type"] == "RuntimeError"
    assert manifest["provenance_error"]["type"] == "ValueError"
    assert observed == dict.fromkeys(trace_runner._DENSE_OFFLINE_ENV, "1")
    assert os.environ["HF_HUB_OFFLINE"] == "previous-value"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "HF_DATASETS_OFFLINE" not in os.environ
    assert "HF_HUB_DISABLE_TELEMETRY" not in os.environ


def test_trace_runner_defaults_to_bm25_only_without_touching_model_or_api(tmp_path, monkeypatch):
    import scripts.run_trace_eval as trace_runner
    from src.ranking.reranker import Ranker
    from src.retrieval.dense import DenseRetriever

    target = "A1"
    samples, _ids, _categories, products = _evaluation_inputs(target)
    catalog = tmp_path / "catalog.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    catalog.write_text(
        "".join(json.dumps(product) + "\n" for product in products.values()),
        encoding="utf-8",
    )
    dataset.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )

    def explode(*_args, **_kwargs):
        raise AssertionError("offline default crossed a model/API boundary")

    monkeypatch.setattr(DenseRetriever, "__init__", explode)
    monkeypatch.setattr(Ranker, "_get_client", explode)
    original_agent = trace_runner.Agent
    observed = {}

    class OfflineCheckingAgent(original_agent):
        def __init__(self, catalog_path, config):
            observed.update({
                name: os.environ.get(name)
                for name in trace_runner._DENSE_OFFLINE_ENV
            })
            super().__init__(catalog_path, config)

    monkeypatch.setattr(trace_runner, "Agent", OfflineCheckingAgent)
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous-value")
    for name in (
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert trace_runner.build_parser().parse_args([]).use_dense is False
    assert trace_runner.build_parser().parse_args(["--no-dense"]).use_dense is False
    assert trace_runner.build_parser().parse_args(["--dense"]).use_dense is True

    run_dir, result = trace_runner.run_trace_evaluation(
        catalog_path=catalog,
        dataset_path=dataset,
        artifacts_root=tmp_path / "runs",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["reported_token_usage"]["total_tokens"] == 0
    assert manifest["config"]["use_dense"] is False
    assert manifest["config"]["use_llm_ranker"] is False
    assert manifest["dense"]["applied"] is False
    assert manifest["llm"]["applied"] is False
    assert manifest["offline_enforced"] is True
    assert observed == dict.fromkeys(trace_runner._DENSE_OFFLINE_ENV, "1")
    assert os.environ["HF_HUB_OFFLINE"] == "previous-value"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "HF_DATASETS_OFFLINE" not in os.environ
    assert "HF_HUB_DISABLE_TELEMETRY" not in os.environ


def test_explicit_dense_fails_locally_before_agent_when_model_cache_is_absent(tmp_path, monkeypatch):
    import scripts.run_trace_eval as trace_runner

    catalog = tmp_path / "catalog.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    catalog.write_text(json.dumps({"parent_asin": "A1"}) + "\n", encoding="utf-8")
    dataset.write_text("", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))
    for name in (
        "SENTENCE_TRANSFORMERS_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)

    class ExplodingAgent:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Agent/model boundary must not run without local cache")

    monkeypatch.setattr(trace_runner, "Agent", ExplodingAgent)
    runs = tmp_path / "runs"

    with pytest.raises(FileNotFoundError, match="not available locally"):
        trace_runner.run_trace_evaluation(
            catalog_path=catalog,
            dataset_path=dataset,
            artifacts_root=runs,
            use_dense=True,
        )

    manifest = json.loads(
        (next(runs.iterdir()) / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "FileNotFoundError"
    assert manifest["offline_enforced"] is True
    assert manifest["dense"]["enabled"] is True
    assert manifest["llm"]["enabled"] is False


def test_explicit_dense_enforces_offline_environment_before_agent_init(tmp_path, monkeypatch):
    import scripts.run_trace_eval as trace_runner

    catalog = tmp_path / "catalog.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    catalog.write_text(json.dumps({"parent_asin": "A1"}) + "\n", encoding="utf-8")
    dataset.write_text("", encoding="utf-8")
    hf_home = tmp_path / "huggingface"
    repository = hf_home / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshot = repository / "snapshots" / "offline-revision"
    snapshot.mkdir(parents=True)
    (snapshot / "weights.bin").write_bytes(b"cached")
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("offline-revision\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous-value")
    for name in (
        "SENTENCE_TRANSFORMERS_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        monkeypatch.delenv(name, raising=False)
    observed = {}

    class OfflineCheckingAgent:
        def __init__(self, _catalog, config):
            observed.update({
                name: os.environ.get(name)
                for name in trace_runner._DENSE_OFFLINE_ENV
            })
            assert config["use_llm_ranker"] is False
            raise RuntimeError("stop after offline-boundary check")

    monkeypatch.setattr(trace_runner, "Agent", OfflineCheckingAgent)
    runs = tmp_path / "runs"

    with pytest.raises(RuntimeError, match="offline-boundary check"):
        trace_runner.run_trace_evaluation(
            catalog_path=catalog,
            dataset_path=dataset,
            artifacts_root=runs,
            use_dense=True,
        )

    assert observed == dict.fromkeys(trace_runner._DENSE_OFFLINE_ENV, "1")
    assert os.environ["HF_HUB_OFFLINE"] == "previous-value"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    manifest = json.loads(
        (next(runs.iterdir()) / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["offline_enforced"] is True
    assert manifest["error"]["type"] == "RuntimeError"


def test_failure_report_keeps_duplicate_sample_ids_separate_by_session(tmp_path):
    from evaluator.trace_artifacts import write_failure_report

    def record(session_id, bm25_rank):
        return {
            "session_id": session_id,
            "sample_id": "duplicate-sample",
            "hit_eligible": True,
            "target_ranks": {
                "bm25": bm25_rank,
                "dense": None,
                "rrf": None,
                "rerank_pool": bm25_rank,
                "ranker_api_pool": None,
                "ranker": None,
                "final": None,
            },
            "agent_debug": {
                "retrieval": {"rrf": {"applied": False}},
            },
            "evaluation_status": {"status": "ok"},
        }

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(record("session-1", None)) + "\n"
        + json.dumps(record("session-2", 3)) + "\n",
        encoding="utf-8",
    )
    result = {
        "sessions": [
            {"sample_id": "duplicate-sample", "scenario_type": "buying", "hit": False, "first_hit_turn": None, "best_rank": None},
            {"sample_id": "duplicate-sample", "scenario_type": "buying", "hit": False, "first_hit_turn": None, "best_rank": None},
        ]
    }
    report = tmp_path / "failure.csv"

    write_failure_report(trace_path, result, report)

    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["session_id"] for row in rows] == ["session-1", "session-2"]
    assert [row["bm25_best_rank"] for row in rows] == ["", "3"]
    assert [row["earliest_failure_stage"] for row in rows] == ["bm25", "pool"]


def test_failure_report_marks_result_session_missing_from_trace_as_instrumentation_error(tmp_path):
    from evaluator.trace_artifacts import write_failure_report

    trace_path = tmp_path / "truncated-trace.jsonl"
    trace_path.write_text(
        json.dumps({
            "session_id": "session-2",
            "sample_id": "sample-2",
            "hit_eligible": True,
            "target_ranks": {},
            "agent_debug": {"retrieval": {"rrf": {"applied": False}}},
            "evaluation_status": {"status": "ok"},
        }) + "\n",
        encoding="utf-8",
    )
    result = {
        "sessions": [
            {
                "sample_id": "sample-1", "scenario_type": "buying",
                "hit": False, "first_hit_turn": None, "best_rank": None,
            },
            {
                "sample_id": "sample-2", "scenario_type": "buying",
                "hit": False, "first_hit_turn": None, "best_rank": None,
            },
        ]
    }
    report = tmp_path / "failure.csv"

    write_failure_report(trace_path, result, report)

    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["session_id"] == "sample-1"
    assert rows[0]["earliest_failure_stage"] == "instrumentation_error"
    assert rows[1]["session_id"] == "session-2"
    assert rows[1]["earliest_failure_stage"] == "bm25"


def test_agent_exception_is_traced_and_classified_as_agent_error(tmp_path):
    from evaluator.trace_artifacts import write_failure_report

    records = []

    class RaisingAgent:
        def reset(self, _session_id, _profile):
            pass

        def respond(self, _session_id, _message, _turn, _top_k):
            raise RuntimeError("agent failed")

    result = evaluate(
        RaisingAgent(),
        *_evaluation_inputs("A1"),
        trace_sink=records.append,
        run_id="failed-agent-run",
    )

    assert records
    assert records[0]["evaluation_status"] == {
        "status": "agent_error",
        "error_type": "RuntimeError",
    }
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    report = tmp_path / "failure.csv"
    write_failure_report(trace, result, report)
    with report.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["earliest_failure_stage"] == "agent_error"


def test_invalid_agent_response_is_explicit_in_trace():
    records = []

    class InvalidAgent:
        def reset(self, _session_id, _profile):
            pass

        def respond(self, _session_id, _message, _turn, _top_k):
            return {"message": 123, "recommendations": "invalid"}

    evaluate(
        InvalidAgent(),
        *_evaluation_inputs("A1"),
        trace_sink=records.append,
        run_id="invalid-agent-run",
    )

    assert records[0]["evaluation_status"] == {"status": "invalid_response"}


def test_non_list_recommendations_make_otherwise_valid_response_invalid(tmp_path):
    from evaluator.trace_artifacts import write_failure_report

    records = []

    class BadRecommendationsAgent:
        def reset(self, _session_id, _profile):
            pass

        def respond(self, _session_id, _message, _turn, _top_k):
            return {
                "message": "valid message",
                "ask_attribute": None,
                "recommendations": "A1",
            }

    result = evaluate(
        BadRecommendationsAgent(),
        *_evaluation_inputs("A1"),
        trace_sink=records.append,
        run_id="bad-recommendations",
    )

    assert records[0]["evaluation_status"] == {"status": "invalid_response"}
    assert result["sessions"][0]["hit"] is False
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    report = tmp_path / "failure.csv"
    write_failure_report(trace, result, report)
    with report.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["earliest_failure_stage"] == "agent_error"


def test_non_string_ask_attribute_invalidates_recommendations():
    records = []

    class BadAskAttributeAgent:
        def reset(self, _session_id, _profile):
            pass

        def respond(self, _session_id, _message, _turn, _top_k):
            return {
                "message": "valid message",
                "ask_attribute": 123,
                "recommendations": [{"parent_asin": "A1"}],
            }

    result = evaluate(
        BadAskAttributeAgent(),
        *_evaluation_inputs("A1"),
        trace_sink=records.append,
        run_id="bad-ask-attribute",
    )

    assert records[0]["evaluation_status"] == {"status": "invalid_response"}
    assert result["sessions"][0]["hit"] is False


def test_missing_debug_trace_is_classified_as_instrumentation_error(tmp_path):
    from evaluator.trace_artifacts import write_failure_report

    records = []

    class MissingTraceAgent:
        def reset(self, _session_id, _profile):
            pass

        def respond(self, _session_id, _message, _turn, _top_k):
            return {
                "message": "valid response without instrumentation",
                "ask_attribute": None,
                "recommendations": [],
            }

    result = evaluate(
        MissingTraceAgent(),
        *_evaluation_inputs("A1"),
        trace_sink=records.append,
        run_id="missing-trace",
    )

    assert records[0]["evaluation_status"] == {"status": "trace_missing"}
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    report = tmp_path / "failure.csv"
    write_failure_report(trace, result, report)
    with report.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["earliest_failure_stage"] == "instrumentation_error"


def test_malformed_stage_candidate_list_is_instrumentation_error():
    records = []

    class MalformedTraceAgent:
        def reset(self, _session_id, _profile):
            pass

        def respond(self, _session_id, _message, _turn, _top_k):
            empty = {"candidates": []}
            return {
                "message": "valid response",
                "ask_attribute": None,
                "recommendations": [],
                "debug_trace": {
                    "retrieval": {
                        "bm25": {"candidates": "not-a-list"},
                        "dense": empty,
                        "rrf": empty,
                    },
                    "rerank_pool": empty,
                    "ranker": {"api_pool": empty, "output": empty},
                    "final": empty,
                },
            }

    evaluate(
        MalformedTraceAgent(),
        *_evaluation_inputs("A1"),
        trace_sink=records.append,
        run_id="malformed-trace",
    )

    assert records[0]["evaluation_status"] == {"status": "instrumentation_error"}


def test_usage_from_original_dict_is_counted_before_invalid_response_fallback():
    class MalformedApiResponseAgent:
        def reset(self, _session_id, _profile):
            pass

        def respond(self, _session_id, _message, _turn, _top_k):
            return {
                "message": "valid message",
                "ask_attribute": None,
                "recommendations": "malformed",
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }

    result = evaluate(
        MalformedApiResponseAgent(),
        *_evaluation_inputs("A1"),
    )

    assert result["reported_token_usage"] == {
        "prompt_tokens": 40,
        "completion_tokens": 20,
        "total_tokens": 60,
    }


def test_source_fingerprint_rejects_artifacts_anywhere_inside_repository():
    from evaluator.trace_artifacts import capture_source_provenance

    repo_root = Path(__file__).resolve().parent.parent

    for artifacts_root in (
        repo_root / "artifacts",
        repo_root / "src" / "trace-runs",
        repo_root / "tests" / "trace-runs",
    ):
        with pytest.raises(ValueError, match="disjoint"):
            capture_source_provenance(repo_root, artifacts_root)


def test_repository_and_git_metadata_are_rejected_before_run_creation(tmp_path):
    from evaluator.trace_artifacts import capture_source_provenance, validate_artifacts_root
    from scripts.run_trace_eval import run_trace_evaluation

    repo_root = Path(__file__).resolve().parent.parent
    catalog = tmp_path / "catalog.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    catalog.write_text(json.dumps({"parent_asin": "A1"}) + "\n", encoding="utf-8")
    dataset.write_text("", encoding="utf-8")
    before = set(repo_root.glob("trace-*"))

    try:
        with pytest.raises(ValueError, match="disjoint"):
            capture_source_provenance(repo_root, repo_root)
        with pytest.raises(ValueError, match="disjoint"):
            capture_source_provenance(repo_root, repo_root.parent)
        with pytest.raises(ValueError, match="disjoint"):
            capture_source_provenance(repo_root, Path(repo_root.anchor))
        with pytest.raises(ValueError, match="disjoint"):
            capture_source_provenance(repo_root, repo_root / "src" / "trace-runs")
        with pytest.raises(ValueError, match="disjoint"):
            capture_source_provenance(repo_root, repo_root / ".git")
        with pytest.raises(ValueError, match="disjoint"):
            capture_source_provenance(repo_root, repo_root / ".git" / "trace-runs")
        assert capture_source_provenance(repo_root, tmp_path)["commit"]
        with pytest.raises(ValueError, match="disjoint"):
            run_trace_evaluation(
                catalog_path=catalog,
                dataset_path=dataset,
                artifacts_root=repo_root,
            )
        after = set(repo_root.glob("trace-*"))
    finally:
        for unexpected in set(repo_root.glob("trace-*")) - before:
            shutil.rmtree(unexpected)

    assert after == before

    git_dir = Path(subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-dir"],
        cwd=repo_root,
        text=True,
    ).strip()).resolve()
    common_dir = Path(subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repo_root,
        text=True,
    ).strip()).resolve()
    for metadata_root in {git_dir, common_dir}:
        with pytest.raises(ValueError, match="Git metadata"):
            validate_artifacts_root(repo_root, metadata_root)


def test_linked_worktree_gitfile_metadata_paths_are_rejected(tmp_path):
    from evaluator.trace_artifacts import validate_artifacts_root

    primary = tmp_path / "primary"
    linked = tmp_path / "linked"
    external = tmp_path / "external"
    subprocess.run(["git", "init", str(primary)], check=True, capture_output=True)
    subprocess.run(
        [
            "git", "-C", str(primary),
            "-c", "user.name=Trace Test",
            "-c", "user.email=trace@example.invalid",
            "commit", "--allow-empty", "-m", "initial",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(primary), "worktree", "add", "--detach", str(linked), "HEAD"],
        check=True,
        capture_output=True,
    )

    git_dir = Path(subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-dir"],
        cwd=linked,
        text=True,
    ).strip()).resolve()
    common_dir = Path(subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=linked,
        text=True,
    ).strip()).resolve()

    assert (linked / ".git").is_file()
    assert validate_artifacts_root(linked, external) == external.resolve()
    for forbidden in (git_dir, git_dir / "runs", common_dir, common_dir / "runs"):
        with pytest.raises(ValueError, match="Git metadata"):
            validate_artifacts_root(linked, forbidden)


def test_artifacts_root_symlinks_are_validated_by_resolved_destination(tmp_path):
    from evaluator.trace_artifacts import validate_artifacts_root

    repo_root = Path(__file__).resolve().parent.parent
    inside_link = tmp_path / "inside-link"
    inside_link.symlink_to(repo_root / "src", target_is_directory=True)
    external_target = tmp_path / "external-target"
    external_target.mkdir()
    external_link = tmp_path / "external-link"
    external_link.symlink_to(external_target, target_is_directory=True)

    with pytest.raises(ValueError, match="disjoint"):
        validate_artifacts_root(repo_root, inside_link)
    assert validate_artifacts_root(repo_root, external_link) == external_target.resolve()


def test_trace_open_error_is_not_masked_when_best_effort_cleanup_fails(tmp_path, monkeypatch):
    from evaluator.trace_artifacts import TraceArtifacts

    original_open = Path.open
    original_rmdir = Path.rmdir
    runs = tmp_path / "runs"

    def fail_trace_open(path, *args, **kwargs):
        if path.name.endswith(".tmp"):
            raise PermissionError("original trace open error")
        return original_open(path, *args, **kwargs)

    def fail_run_cleanup(path):
        if path.parent == runs:
            raise OSError("secondary cleanup error")
        return original_rmdir(path)

    monkeypatch.setattr(Path, "open", fail_trace_open)
    monkeypatch.setattr(Path, "rmdir", fail_run_cleanup)
    try:
        with pytest.raises(PermissionError, match="original trace open error"):
            TraceArtifacts.create(runs)
    finally:
        monkeypatch.setattr(Path, "open", original_open)
        monkeypatch.setattr(Path, "rmdir", original_rmdir)
        if runs.exists():
            shutil.rmtree(runs)


def test_dependency_versions_are_deterministic_and_preserve_normalized_duplicates():
    from evaluator.trace_artifacts import dependency_versions

    distributions = [
        SimpleNamespace(metadata={"Name": "Foo_Bar"}, version="2.0"),
        SimpleNamespace(metadata={"Name": "foo-bar"}, version="1.0"),
        SimpleNamespace(metadata={"Name": "Alpha"}, version="3.0"),
    ]

    versions = dependency_versions(reversed(distributions))

    assert versions == [
        {"name": "Alpha", "normalized_name": "alpha", "version": "3.0"},
        {"name": "foo-bar", "normalized_name": "foo-bar", "version": "1.0"},
        {"name": "Foo_Bar", "normalized_name": "foo-bar", "version": "2.0"},
    ]


def test_dense_provenance_separates_embedding_cache_from_local_model_snapshot(tmp_path, monkeypatch):
    from evaluator.trace_artifacts import dense_provenance
    from src.config import load_config

    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps({"parent_asin": "A1"}) + "\n", encoding="utf-8")
    hf_home = tmp_path / "huggingface"
    repository = hf_home / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshot = repository / "snapshots" / "revision-123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model":"local"}', encoding="utf-8")
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("revision-123\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    for name in (
        "SENTENCE_TRANSFORMERS_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)

    provenance = dense_provenance(
        load_config({"use_dense": False}),
        catalog,
    )

    assert provenance["configured_model"] == "all-MiniLM-L6-v2"
    assert provenance["enabled"] is False
    assert provenance["applied"] is False
    assert provenance["embedding_cache"]["path"].endswith(".npz")
    assert provenance["embedding_cache"]["exists"] is False
    assert provenance["embedding_cache"]["sha256"] is None
    assert provenance["model_cache"]["repository"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert provenance["model_cache"]["resolution"] == {
        "method": "refs_main",
        "confidence": "high",
        "revision": "revision-123",
        "snapshot_path": str(snapshot),
    }
    assert provenance["model_cache"]["candidates"] == [
        {
            "cache_root": str(hf_home / "hub"),
            "revision": "revision-123",
            "snapshot_path": str(snapshot),
            "sha256": provenance["model_cache"]["candidates"][0]["sha256"],
        }
    ]
    assert provenance["model_cache"]["exists"] is True
    assert provenance["model_cache"]["sha256"]


def test_ambiguous_model_cache_hashes_all_candidates_without_claiming_one(tmp_path, monkeypatch):
    from evaluator.trace_artifacts import dense_provenance
    from src.config import load_config

    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps({"parent_asin": "A1"}) + "\n", encoding="utf-8")
    hf_home = tmp_path / "huggingface"
    snapshots = (
        hf_home
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
    )
    for revision in ("revision-a", "revision-b"):
        snapshot = snapshots / revision
        snapshot.mkdir(parents=True)
        (snapshot / "weights.bin").write_bytes(revision.encode())
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    for name in (
        "SENTENCE_TRANSFORMERS_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)

    model_cache = dense_provenance(load_config({"use_dense": False}), catalog)[
        "model_cache"
    ]

    assert [item["revision"] for item in model_cache["candidates"]] == [
        "revision-a",
        "revision-b",
    ]
    assert all(item["sha256"] for item in model_cache["candidates"])
    assert model_cache["resolution"] == {
        "method": "ambiguous",
        "confidence": "none",
        "revision": None,
        "snapshot_path": None,
    }
    assert model_cache["exists"] is True
    assert model_cache["sha256"] is None


def test_local_dense_model_path_is_hashed_directly_at_each_snapshot(tmp_path):
    from evaluator.trace_artifacts import dense_provenance
    from src.config import load_config

    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps({"parent_asin": "A1"}) + "\n", encoding="utf-8")
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    weights = model_dir / "weights.bin"
    weights.write_bytes(b"version-one")
    config = load_config({"use_dense": False, "dense_model": str(model_dir)})

    before = dense_provenance(config, catalog)["model_cache"]
    weights.write_bytes(b"version-two")
    after = dense_provenance(config, catalog)["model_cache"]

    assert before["repository"] is None
    assert before["resolution"] == {
        "method": "local_path",
        "confidence": "high",
        "revision": None,
        "snapshot_path": str(model_dir),
    }
    assert before["candidates"] == [
        {
            "cache_root": str(model_dir.parent),
            "revision": None,
            "snapshot_path": str(model_dir),
            "sha256": before["sha256"],
        }
    ]
    assert before["sha256"] != after["sha256"]


def test_runner_snapshots_inputs_before_agent_initialization(tmp_path, monkeypatch):
    import scripts.run_trace_eval as trace_runner

    catalog = tmp_path / "catalog.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    original = json.dumps({"parent_asin": "A1", "title": "original"}) + "\n"
    catalog.write_text(original, encoding="utf-8")
    dataset.write_text("", encoding="utf-8")
    original_hash = hashlib.sha256(original.encode()).hexdigest()

    class MutatingAgent:
        def __init__(self, catalog_path, _config):
            Path(catalog_path).write_text("mutated\n", encoding="utf-8")
            raise RuntimeError("init failed after mutation")

    monkeypatch.setattr(trace_runner, "Agent", MutatingAgent)
    runs = tmp_path / "runs"

    with pytest.raises(RuntimeError, match="init failed after mutation"):
        trace_runner.run_trace_evaluation(
            catalog_path=catalog,
            dataset_path=dataset,
            artifacts_root=runs,
        )

    run_dir = next(runs.iterdir())
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["catalog"]["sha256"] == original_hash
    assert manifest["catalog_end"]["sha256"] == hashlib.sha256(b"mutated\n").hexdigest()
    assert manifest["catalog_changed_during_run"] is True
    assert manifest["dataset_end"]["sha256"] == manifest["dataset"]["sha256"]
    assert manifest["dataset_changed_during_run"] is False
    assert manifest["status"] == "failed"


def test_manifest_detects_model_cache_created_during_agent_initialization(tmp_path, monkeypatch):
    import scripts.run_trace_eval as trace_runner

    target = "A1"
    samples, _ids, _categories, products = _evaluation_inputs(target)
    catalog = tmp_path / "catalog.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    catalog.write_text(
        "".join(json.dumps(product) + "\n" for product in products.values()),
        encoding="utf-8",
    )
    dataset.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )
    hf_home = tmp_path / "huggingface"
    repository = hf_home / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
    real_agent = trace_runner.Agent

    class CacheCreatingAgent(real_agent):
        def __init__(self, catalog_path, config):
            snapshot = repository / "snapshots" / "created-during-init"
            snapshot.mkdir(parents=True)
            (snapshot / "weights.bin").write_bytes(b"local model weights")
            (repository / "refs").mkdir()
            (repository / "refs" / "main").write_text(
                "created-during-init\n", encoding="utf-8"
            )
            super().__init__(catalog_path, config)

    monkeypatch.setattr(trace_runner, "Agent", CacheCreatingAgent)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    for name in (
        "SENTENCE_TRANSFORMERS_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)

    run_dir, _result = trace_runner.run_trace_evaluation(
        catalog_path=catalog,
        dataset_path=dataset,
        artifacts_root=tmp_path / "runs",
    )

    dense = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))[
        "dense"
    ]
    assert dense["model_cache_start"]["candidates"] == []
    assert [item["revision"] for item in dense["model_cache_end"]["candidates"]] == [
        "created-during-init"
    ]
    assert dense["model_cache_end"]["resolution"]["method"] == "refs_main"
    assert dense["model_cache_changed_during_run"] is True
    assert dense["embedding_cache_changed_during_run"] is False
    assert dense["applied"] is False


def test_atomic_json_rejects_nan_without_leaving_output_or_temporary_file(tmp_path):
    from evaluator.trace_artifacts import write_json_atomic

    output = tmp_path / "results.json"

    with pytest.raises(ValueError):
        write_json_atomic(output, {"score": float("nan")})

    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp"))
