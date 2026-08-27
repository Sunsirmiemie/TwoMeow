"""Streaming evaluation trace and collision-resistant run artifacts."""
from __future__ import annotations

import csv
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


FAILURE_COLUMNS = (
    "session_id",
    "scenario",
    "hit",
    "first_hit_turn",
    "first_hit_rank",
    "bm25_best_rank",
    "dense_best_rank",
    "rrf_best_rank",
    "pool_best_rank",
    "final_best_rank",
    "earliest_failure_stage",
)


def dependency_versions(distributions=None) -> list[dict[str, str]]:
    """Return a stable list without collapsing duplicate normalized names."""
    source = importlib.metadata.distributions() if distributions is None else distributions
    versions = []
    for distribution in source:
        name = distribution.metadata.get("Name")
        if not name:
            continue
        versions.append({
            "name": str(name),
            "normalized_name": re.sub(r"[-_.]+", "-", str(name)).lower(),
            "version": str(distribution.version),
        })
    return sorted(
        versions,
        key=lambda item: (
            item["normalized_name"],
            item["version"],
            item["name"].lower(),
            item["name"],
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SOURCE_PATHS = (
    "src",
    "agent",
    "starter",
    "evaluator",
    "scripts",
    "tests",
    "pyproject.toml",
    "requirements.txt",
    "run_eval.py",
)
_SOURCE_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".txt"}
_REDACTED = "[REDACTED]"


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _overlaps(left: Path, right: Path) -> bool:
    return _inside(left, right) or _inside(right, left)


def _git_metadata_directories(root: Path) -> tuple[Path, ...]:
    directories = []
    for option in ("--git-dir", "--git-common-dir"):
        value = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", option],
            cwd=root,
            text=True,
        ).strip()
        path = Path(value)
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        if resolved not in directories:
            directories.append(resolved)
    return tuple(directories)


def validate_artifacts_root(repo_root: str | Path, artifacts_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    artifacts = Path(artifacts_root).resolve()
    if _overlaps(artifacts, root) or any(
        _overlaps(artifacts, metadata)
        for metadata in _git_metadata_directories(root)
    ):
        raise ValueError(
            "artifacts root must be disjoint from the repository and Git metadata"
        )
    return artifacts


def _source_path(relative: str) -> bool:
    path = Path(relative)
    if path.suffix.lower() not in _SOURCE_SUFFIXES:
        return False
    return path.parts[0] in {"src", "agent", "starter", "evaluator", "scripts", "tests"} or len(path.parts) == 1


def _secret_config_key(key: object) -> bool:
    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", str(key))
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", separated)
    parts = tuple(
        part for part in re.split(r"[^a-z0-9]+", separated.lower()) if part
    )
    adjacent = set(zip(parts, parts[1:]))
    return (
        bool(set(parts) & {
            "secret", "secrets", "token", "password", "passwords",
            "credential", "credentials", "authorization", "bearer",
        })
        or bool(adjacent & {
            ("api", "key"),
            ("private", "key"),
            ("access", "key"),
            ("signing", "key"),
            ("auth", "header"),
        })
        or any(part in {
            "apikey", "privatekey", "accesskey", "signingkey", "authheader",
        } for part in parts)
    )


def _redact_config(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _REDACTED if _secret_config_key(key) else _redact_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_config(item) for item in value)
    return copy.deepcopy(value)


def _status_entries(root: Path) -> list[tuple[str, str]]:
    payload = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
    )
    fields = payload.split(b"\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        status = field[:2].decode("ascii", errors="replace")
        relative = field[3:].decode("utf-8", errors="surrogateescape")
        entries.append((status, relative))
        if "R" in status or "C" in status:
            index += 1
    return entries


def capture_source_provenance(
    repo_root: str | Path,
    artifacts_root: str | Path,
) -> dict:
    """Fingerprint source changes in bounded chunks; artifacts must be external."""
    root = Path(repo_root).resolve()
    validate_artifacts_root(root, artifacts_root)
    entries = _status_entries(root)
    digest = hashlib.sha256()
    process = subprocess.Popen(
        ["git", "diff", "--binary", "HEAD", "--", *_SOURCE_PATHS],
        cwd=root,
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)
    for status, relative in sorted(entries, key=lambda item: item[1]):
        if status != "??" or not _source_path(relative):
            continue
        path = (root / relative).resolve()
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8", errors="surrogateescape") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": bool(entries),
        "source_diff_sha256": digest.hexdigest(),
    }


def _dense_cache_path(catalog_path: Path, model: str) -> Path | None:
    if not catalog_path.is_file():
        return None
    digest = hashlib.sha1()
    with catalog_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    catalog_hash = digest.hexdigest()[:12]
    safe_model = model.replace("/", "_")
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / ".embed_cache" / f"{safe_model}_{catalog_hash}.npz"


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _model_repository(model: str) -> str:
    return model if "/" in model else f"sentence-transformers/{model}"


def _model_cache_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("SENTENCE_TRANSFORMERS_HOME"):
        roots.append(Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("TRANSFORMERS_CACHE"):
        roots.append(Path(os.environ["TRANSFORMERS_CACHE"]))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _model_cache_provenance(model: str) -> dict:
    local_path = Path(model).expanduser()
    if local_path.exists():
        local_path = local_path.resolve()
        local_hash = (
            _sha256_directory(local_path)
            if local_path.is_dir()
            else _sha256(local_path)
        )
        candidate = {
            "cache_root": str(local_path.parent),
            "revision": None,
            "snapshot_path": str(local_path),
            "sha256": local_hash,
        }
        return {
            "repository": None,
            "search_roots": [str(local_path.parent)],
            "candidates": [candidate],
            "resolution": {
                "method": "local_path",
                "confidence": "high",
                "revision": None,
                "snapshot_path": str(local_path),
            },
            "exists": True,
            "sha256": local_hash,
        }
    repository = _model_repository(model)
    directory_name = "models--" + repository.replace("/", "--")
    roots = _model_cache_roots()
    candidates: list[dict] = []
    refs_main_candidates: list[dict] = []
    for root in roots:
        repository_path = root / directory_name
        if not repository_path.is_dir():
            continue
        main_revision = None
        main_ref = repository_path / "refs" / "main"
        if main_ref.is_file():
            main_revision = main_ref.read_text(encoding="utf-8").strip() or None
        snapshots_dir = repository_path / "snapshots"
        snapshots = sorted(
            path for path in snapshots_dir.iterdir() if path.is_dir()
        ) if snapshots_dir.is_dir() else []
        for snapshot in snapshots:
            candidate = {
                "cache_root": str(root),
                "revision": snapshot.name,
                "snapshot_path": str(snapshot),
                "sha256": _sha256_directory(snapshot),
            }
            candidates.append(candidate)
            if snapshot.name == main_revision:
                refs_main_candidates.append(candidate)
    legacy_name = repository.replace("/", "_")
    legacy_roots = []
    if os.environ.get("SENTENCE_TRANSFORMERS_HOME"):
        legacy_roots.append(Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]))
    legacy_roots.append(Path.home() / ".cache" / "torch" / "sentence_transformers")
    for root in legacy_roots:
        snapshot = root.expanduser().resolve() / legacy_name
        if snapshot.is_dir():
            candidates.append({
                "cache_root": str(root.expanduser().resolve()),
                "revision": None,
                "snapshot_path": str(snapshot),
                "sha256": _sha256_directory(snapshot),
            })
    candidates.sort(
        key=lambda item: (
            item["cache_root"],
            item["revision"] or "",
            item["snapshot_path"],
        )
    )
    resolved = None
    if len(refs_main_candidates) == 1:
        resolved = refs_main_candidates[0]
        method, confidence = "refs_main", "high"
    elif refs_main_candidates and len({
        (item["revision"], item["sha256"]) for item in refs_main_candidates
    }) == 1:
        resolved = refs_main_candidates[0]
        method, confidence = "refs_main_consistent", "medium"
    elif refs_main_candidates:
        method, confidence = "ambiguous", "none"
    elif len(candidates) == 1:
        resolved = candidates[0]
        method, confidence = "unique", "medium"
    elif candidates:
        method, confidence = "ambiguous", "none"
    else:
        method, confidence = "absent", "none"
    resolution = {
        "method": method,
        "confidence": confidence,
        "revision": resolved["revision"] if resolved is not None else None,
        "snapshot_path": (
            resolved["snapshot_path"] if resolved is not None else None
        ),
    }
    return {
        "repository": repository,
        "search_roots": [str(root) for root in roots],
        "candidates": candidates,
        "resolution": resolution,
        "exists": bool(candidates),
        "sha256": resolved["sha256"] if resolved is not None else None,
    }


def dense_provenance(config: dict, catalog_path: str | Path) -> dict:
    """Describe embedding and model caches without importing or loading a model."""
    catalog = Path(catalog_path).resolve()
    model = str(config.get("dense_model") or "")
    embedding_path = _dense_cache_path(catalog, model)
    embedding_exists = embedding_path is not None and embedding_path.is_file()
    return {
        "configured_model": model or None,
        "enabled": bool(config.get("use_dense", True)),
        "applied": False,
        "embedding_cache": {
            "path": str(embedding_path) if embedding_path is not None else None,
            "exists": embedding_exists,
            "sha256": _sha256(embedding_path) if embedding_exists else None,
        },
        "model_cache": _model_cache_provenance(model) if model else {
            "repository": None,
            "search_roots": [],
            "candidates": [],
            "resolution": {
                "method": "absent",
                "confidence": "none",
                "revision": None,
                "snapshot_path": None,
            },
            "exists": False,
            "sha256": None,
        },
    }


def _input_provenance(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    exists = resolved.is_file()
    return {
        "path": str(resolved),
        "exists": exists,
        "sha256": _sha256(resolved) if exists else None,
    }


def capture_start_provenance(
    *,
    config: dict,
    catalog_path: str | Path,
    dataset_path: str | Path,
    cli_args: dict,
    repo_root: str | Path,
    artifacts_root: str | Path,
    started_at: datetime,
    offline_enforced: bool = False,
) -> dict:
    """Capture immutable run inputs before Agent/model initialization."""
    dense_start = dense_provenance(config, catalog_path)
    return {
        "schema_version": 1,
        "offline_enforced": offline_enforced,
        "git": capture_source_provenance(repo_root, artifacts_root),
        "config": _redact_config(config),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "dependencies": dependency_versions(),
        "catalog": _input_provenance(catalog_path),
        "dataset": _input_provenance(dataset_path),
        "dense": {
            "configured_model": dense_start["configured_model"],
            "enabled": dense_start["enabled"],
            "applied": False,
            "embedding_cache_start": dense_start["embedding_cache"],
            "model_cache_start": dense_start["model_cache"],
        },
        "llm": {
            "configured_model": config.get("ranker_model"),
            "enabled": bool(config.get("use_llm_ranker", False)),
            "applied": False,
        },
        "timing": {
            "started_at_utc": started_at.astimezone(timezone.utc).isoformat(),
        },
        "cli_args": copy.deepcopy(cli_args),
    }


def finalize_run_manifest(
    *,
    start_provenance: dict,
    artifacts: "TraceArtifacts",
    repo_root: str | Path,
    artifacts_root: str | Path,
    ended_at: datetime,
    status: str,
    error: dict | None = None,
) -> dict:
    """Finalize a start snapshot with runtime state and an end fingerprint."""
    manifest = copy.deepcopy(start_provenance)
    end_git = capture_source_provenance(repo_root, artifacts_root)
    started_at = datetime.fromisoformat(manifest["timing"]["started_at_utc"])
    catalog_end = _input_provenance(manifest["catalog"]["path"])
    dataset_end = _input_provenance(manifest["dataset"]["path"])
    dense_end = dense_provenance(manifest["config"], manifest["catalog"]["path"])
    manifest.update({
        "run_id": artifacts.run_id,
        "status": status,
        "git_end": end_git,
        "source_changed_during_run": (
            end_git["commit"] != manifest["git"]["commit"]
            or end_git["source_diff_sha256"] != manifest["git"]["source_diff_sha256"]
        ),
        "catalog_end": catalog_end,
        "dataset_end": dataset_end,
        "catalog_changed_during_run": (
            catalog_end["exists"] != manifest["catalog"]["exists"]
            or catalog_end["sha256"] != manifest["catalog"]["sha256"]
        ),
        "dataset_changed_during_run": (
            dataset_end["exists"] != manifest["dataset"]["exists"]
            or dataset_end["sha256"] != manifest["dataset"]["sha256"]
        ),
        "artifacts": {
            "run_dir": str(artifacts.run_dir.resolve()),
            "results": str(artifacts.results_path.resolve()),
            "trace": str(artifacts.trace_path.resolve()),
            "manifest": str(artifacts.manifest_path.resolve()),
            "failure_report": str(artifacts.failure_report_path.resolve()),
        },
    })
    manifest["dense"].update({
        "applied": artifacts.dense_applied,
        "embedding_cache_end": dense_end["embedding_cache"],
        "model_cache_end": dense_end["model_cache"],
        "embedding_cache_changed_during_run": (
            dense_end["embedding_cache"]
            != manifest["dense"]["embedding_cache_start"]
        ),
        "model_cache_changed_during_run": (
            dense_end["model_cache"] != manifest["dense"]["model_cache_start"]
        ),
    })
    manifest["llm"]["applied"] = artifacts.llm_applied
    provenance_captured_at = datetime.now(timezone.utc)
    manifest["timing"].update({
        "ended_at_utc": ended_at.astimezone(timezone.utc).isoformat(),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 6),
        "provenance_captured_at_utc": provenance_captured_at.isoformat(),
    })
    if error is not None:
        manifest["error"] = error
    return manifest


def build_run_manifest(
    *,
    artifacts: "TraceArtifacts",
    agent,
    catalog_path: str | Path,
    dataset_path: str | Path,
    cli_args: dict,
    repo_root: str | Path,
    started_at: datetime,
    ended_at: datetime,
    status: str,
    error: dict | None = None,
) -> dict:
    """Compatibility wrapper for callers that already completed initialization."""
    start = capture_start_provenance(
        config=agent.config,
        catalog_path=catalog_path,
        dataset_path=dataset_path,
        cli_args=cli_args,
        repo_root=repo_root,
        artifacts_root=artifacts.run_dir.parent,
        started_at=started_at,
        offline_enforced=False,
    )
    return finalize_run_manifest(
        start_provenance=start,
        artifacts=artifacts,
        repo_root=repo_root,
        artifacts_root=artifacts.run_dir.parent,
        ended_at=ended_at,
        status=status,
        error=error,
    )


@contextmanager
def _atomic_text_file(path: str | Path, *, newline: str | None = None):
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: str | Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with _atomic_text_file(path) as handle:
        handle.write(payload)


def write_manifest_atomic(path: str | Path, manifest: dict) -> None:
    write_json_atomic(path, manifest)


def _best(current: int | None, candidate: object) -> int | None:
    if not isinstance(candidate, int) or candidate < 1:
        return current
    return candidate if current is None else min(current, candidate)


def _failure_stage(aggregate: dict, hit: bool) -> str:
    if hit:
        return ""
    if aggregate.get("instrumentation_error"):
        return "instrumentation_error"
    if aggregate.get("agent_error"):
        return "agent_error"
    if aggregate["fused_applied"]:
        if aggregate["bm25"] is None and aggregate["dense"] is None:
            return "sources"
        if aggregate["rrf"] is None:
            return "rrf"
    elif aggregate["bm25"] is None:
        return "bm25"
    if aggregate["pool"] is None:
        return "pool"
    return "ranker"


def write_failure_report(
    trace_path: str | Path,
    result: dict,
    output_path: str | Path,
) -> None:
    """Stream trace JSONL into one small aggregate row per evaluation session."""
    aggregates: dict[str, dict] = {}
    session_ids_by_sample: dict[str, list[str]] = {}
    with Path(trace_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            session_id = str(record["session_id"])
            if session_id not in aggregates:
                sample_id = str(record["sample_id"])
                session_ids_by_sample.setdefault(sample_id, []).append(session_id)
                aggregates[session_id] = {
                    "session_id": session_id,
                    "sample_id": sample_id,
                    "bm25": None,
                    "dense": None,
                    "rrf": None,
                    "pool": None,
                    "final": None,
                    "fused_applied": False,
                    "agent_error": False,
                    "instrumentation_error": False,
                }
            aggregate = aggregates[session_id]
            if record.get("hit_eligible") is not True:
                continue
            status = (record.get("evaluation_status") or {}).get("status")
            aggregate["agent_error"] = (
                aggregate["agent_error"]
                or status in {"agent_error", "invalid_response"}
            )
            aggregate["instrumentation_error"] = (
                aggregate["instrumentation_error"]
                or status in {"trace_missing", "instrumentation_error"}
            )
            ranks = record.get("target_ranks") or {}
            for name in ("bm25", "dense", "rrf", "final"):
                aggregate[name] = _best(aggregate[name], ranks.get(name))
            aggregate["pool"] = _best(
                aggregate["pool"], ranks.get("ranker_api_pool")
            )
            debug = record.get("agent_debug") or {}
            retrieval = debug.get("retrieval") or {}
            rrf = retrieval.get("rrf") or {}
            aggregate["fused_applied"] = (
                aggregate["fused_applied"] or rrf.get("applied") is True
            )

    output = Path(output_path)
    with _atomic_text_file(output, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS)
        writer.writeheader()
        for session in result.get("sessions", []):
            sample_id = str(session["sample_id"])
            matching_session_ids = session_ids_by_sample.get(sample_id, [])
            aggregate = (
                aggregates[matching_session_ids.pop(0)]
                if matching_session_ids
                else {}
            ) or {
                "session_id": sample_id,
                "sample_id": sample_id,
                "bm25": None,
                "dense": None,
                "rrf": None,
                "pool": None,
                "final": None,
                "fused_applied": False,
                "agent_error": False,
                "instrumentation_error": True,
            }
            hit = bool(session.get("hit"))
            writer.writerow({
                "session_id": aggregate["session_id"],
                "scenario": session["scenario_type"],
                "hit": str(hit).lower(),
                "first_hit_turn": session.get("first_hit_turn"),
                "first_hit_rank": session.get("best_rank"),
                "bm25_best_rank": aggregate["bm25"],
                "dense_best_rank": aggregate["dense"],
                "rrf_best_rank": aggregate["rrf"],
                "pool_best_rank": aggregate["pool"],
                "final_best_rank": aggregate["final"],
                "earliest_failure_stage": _failure_stage(aggregate, hit),
            })


class TraceArtifacts:
    """Own one unique run directory and stream strict JSONL turn records."""

    def __init__(self, run_dir: Path, run_id: str, trace_temp_path: Path, trace_file: TextIO):
        self.run_dir = run_dir
        self.run_id = run_id
        self.trace_path = run_dir / "trace.jsonl"
        self.results_path = run_dir / "results.json"
        self.manifest_path = run_dir / "run_manifest.json"
        self.failure_report_path = run_dir / "failure_report.csv"
        self._trace_temp_path = trace_temp_path
        self._trace_file = trace_file
        self.dense_applied = False
        self.llm_applied = False

    @classmethod
    def create(cls, artifacts_root: str | Path) -> "TraceArtifacts":
        root = Path(artifacts_root)
        root.mkdir(parents=True, exist_ok=True)
        for _attempt in range(10):
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            run_id = f"trace-{timestamp}-{uuid.uuid4().hex[:12]}"
            run_dir = root / run_id
            try:
                run_dir.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            trace_path = run_dir / "trace.jsonl"
            trace_temp_path = run_dir / f".{trace_path.name}.{uuid.uuid4().hex}.tmp"
            try:
                trace_file = trace_temp_path.open("x", encoding="utf-8", buffering=1)
            except BaseException:
                try:
                    run_dir.rmdir()
                except OSError:
                    pass
                raise
            return cls(run_dir, run_id, trace_temp_path, trace_file)
        raise FileExistsError("could not allocate a unique trace run directory")

    def __call__(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self._trace_file.write(line + "\n")
        self._trace_file.flush()
        debug = record.get("agent_debug") or {}
        retrieval = debug.get("retrieval") or {}
        dense = retrieval.get("dense") or {}
        ranker = debug.get("ranker") or {}
        self.dense_applied = self.dense_applied or dense.get("applied") is True
        self.llm_applied = self.llm_applied or ranker.get("status") == "api_success"

    def close(self) -> None:
        if not self._trace_file.closed:
            self._trace_file.flush()
            os.fsync(self._trace_file.fileno())
            self._trace_file.close()
        if self._trace_temp_path.exists():
            os.replace(self._trace_temp_path, self.trace_path)

    def abort(self) -> None:
        if not self._trace_file.closed:
            self._trace_file.close()
        if self._trace_temp_path.exists():
            self._trace_temp_path.unlink()
        for path in (self.trace_path, self.results_path, self.failure_report_path):
            if path.exists():
                path.unlink()
