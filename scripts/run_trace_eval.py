#!/usr/bin/env python
"""Run a provenance-rich evaluation in a unique, non-overwriting directory."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from evaluator.trace_artifacts import (
    TraceArtifacts,
    capture_start_provenance,
    finalize_run_manifest,
    write_failure_report,
    write_json_atomic,
    write_manifest_atomic,
    validate_artifacts_root,
)
from src.agent.orchestrator import Agent
from src.config import load_config


_DENSE_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def run_trace_evaluation(
    *,
    catalog_path: str | Path,
    dataset_path: str | Path,
    artifacts_root: str | Path,
    no_dense: bool | None = None,
    use_dense: bool = False,
) -> tuple[Path, dict]:
    """Execute one trace run; LLM ranking is always disabled by this interface."""
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    validated_artifacts_root = validate_artifacts_root(REPO_ROOT, artifacts_root)
    started_at = datetime.now(timezone.utc)
    if no_dense is not None:
        use_dense = not no_dense
    cli_args = {
        "catalog": str(catalog),
        "dataset": str(dataset),
        "artifacts_root": str(validated_artifacts_root),
        "use_dense": use_dense,
    }
    overrides = {
        "trace_enabled": True,
        "use_llm_ranker": False,
        "use_dense": use_dense,
    }
    artifacts = None
    start_provenance = None
    completed = False
    saved_offline_env = {
        name: os.environ.get(name) for name in _DENSE_OFFLINE_ENV
    }
    os.environ.update(_DENSE_OFFLINE_ENV)

    try:
        resolved_config = load_config(overrides)
        artifacts = TraceArtifacts.create(validated_artifacts_root)
        start_provenance = capture_start_provenance(
            config=resolved_config,
            catalog_path=catalog,
            dataset_path=dataset,
            cli_args=cli_args,
            repo_root=REPO_ROOT,
            artifacts_root=artifacts.run_dir.parent,
            started_at=started_at,
            offline_enforced=True,
        )
        if (
            use_dense
            and not start_provenance["dense"]["model_cache_start"]["exists"]
        ):
            raise FileNotFoundError(
                f"Dense model {resolved_config['dense_model']!r} is not available locally"
            )
        agent = Agent(str(catalog), resolved_config)
        samples = load_jsonl(dataset)
        catalog_ids, categories, products = catalog_index(catalog)
        result = evaluate(
            agent,
            samples,
            catalog_ids,
            categories,
            products,
            trace_sink=artifacts,
            run_id=artifacts.run_id,
        )
        artifacts.close()
        write_json_atomic(artifacts.results_path, result)
        write_failure_report(
            artifacts.trace_path,
            result,
            artifacts.failure_report_path,
        )
        ended_at = datetime.now(timezone.utc)
        manifest = finalize_run_manifest(
            start_provenance=start_provenance,
            artifacts=artifacts,
            repo_root=REPO_ROOT,
            artifacts_root=artifacts.run_dir.parent,
            ended_at=ended_at,
            status="completed",
        )
        write_manifest_atomic(artifacts.manifest_path, manifest)
        completed = True
        return artifacts.run_dir, result
    except BaseException as exc:
        if artifacts is None:
            raise
        try:
            artifacts.abort()
        except Exception:
            pass
        ended_at = datetime.now(timezone.utc)
        original_error = {"type": type(exc).__name__, "message": str(exc)}
        try:
            if start_provenance is None:
                manifest = {
                    "schema_version": 1,
                    "run_id": artifacts.run_id,
                    "status": "failed",
                    "offline_enforced": True,
                    "error": original_error,
                    "timing": {
                        "started_at_utc": started_at.isoformat(),
                        "ended_at_utc": ended_at.isoformat(),
                        "duration_seconds": round(
                            (ended_at - started_at).total_seconds(), 6
                        ),
                    },
                }
            else:
                manifest = finalize_run_manifest(
                    start_provenance=start_provenance,
                    artifacts=artifacts,
                    repo_root=REPO_ROOT,
                    artifacts_root=artifacts.run_dir.parent,
                    ended_at=ended_at,
                    status="failed",
                    error=original_error,
                )
            write_manifest_atomic(artifacts.manifest_path, manifest)
        except Exception as provenance_exc:
            fallback = {
                "schema_version": 1,
                "run_id": artifacts.run_id,
                "status": "failed",
                "offline_enforced": True,
                "error": original_error,
                "provenance_error": {"type": type(provenance_exc).__name__},
                "timing": {
                    "started_at_utc": started_at.isoformat(),
                    "ended_at_utc": ended_at.isoformat(),
                    "duration_seconds": round(
                        (ended_at - started_at).total_seconds(), 6
                    ),
                },
            }
            try:
                write_manifest_atomic(artifacts.manifest_path, fallback)
            except Exception:
                pass
        raise
    finally:
        if artifacts is not None and not completed:
            try:
                artifacts.abort()
            except Exception:
                pass
        for name, previous in saved_offline_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run evaluator-owned trace/provenance artifacts without LLM API calls."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--artifacts-root",
        default=str(Path(tempfile.gettempdir()) / "twomeow-trace-runs"),
    )
    dense_group = parser.add_mutually_exclusive_group()
    dense_group.add_argument(
        "--dense",
        action="store_true",
        dest="use_dense",
        help="opt in to the local Dense model/cache (may require local model files)",
    )
    dense_group.add_argument(
        "--no-dense",
        action="store_false",
        dest="use_dense",
        help="explicit alias for the default offline BM25-only mode",
    )
    parser.set_defaults(use_dense=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir, result = run_trace_evaluation(
        catalog_path=args.catalog,
        dataset_path=args.dataset,
        artifacts_root=args.artifacts_root,
        use_dense=args.use_dense,
    )
    summary = {
        key: value
        for key, value in result.items()
        if key not in ("sessions", "scenario_metrics")
    }
    print(json.dumps({"run_dir": str(run_dir.resolve()), **summary}, indent=2))


if __name__ == "__main__":
    main()
