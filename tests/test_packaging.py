"""Artifact-level checks for the installable wheel."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_wheel_supports_official_and_compatibility_agent_entries(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            ".idea",
            ".pytest_cache",
            ".embed_cache",
            ".venv",
            "__pycache__",
            "*.egg-info",
            "*.zip",
            "build",
            "data",
            "dist",
            "graft",
            "kit_extract",
        ),
    )
    wheel_dir = tmp_path / "wheel"
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
        ],
        cwd=source,
    )
    wheel = next(wheel_dir.glob("*.whl"))

    target = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(target)

    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        '{"parent_asin":"A1","title":"Desk lamp","price":10}\n',
        encoding="utf-8",
    )
    check = f"""
import pathlib
import sys
repo = pathlib.Path({str(ROOT)!r})
target = pathlib.Path({str(target)!r})
sys.path[:] = [
    p for p in sys.path
    if repo != pathlib.Path(p or '.').resolve()
    and repo not in pathlib.Path(p or '.').resolve().parents
]
assert all(repo != pathlib.Path(p or '.').resolve() for p in sys.path)
sys.meta_path[:] = [
    finder for finder in sys.meta_path
    if not (
        "twomeow" in repr(finder).lower()
        and "editable" in repr(finder).lower()
    )
]
sys.path.insert(0, str(target))
import src.agent.orchestrator as orchestrator_module
import starter.agent as starter_module
import agent.agent as compatibility_module
import evaluator as evaluator_package
import evaluator.local_evaluator as evaluator_module
import src.config.loader as config_module
Agent = orchestrator_module.Agent
from starter.agent import Agent as StarterAgent
from agent.agent import Agent as CompatibilityAgent
assert StarterAgent is Agent
assert CompatibilityAgent is Agent
for module in (
    orchestrator_module,
    starter_module,
    compatibility_module,
    evaluator_package,
    evaluator_module,
    config_module,
):
    module_path = pathlib.Path(module.__file__).resolve()
    assert target == module_path or target in module_path.parents
    assert repo != module_path and repo not in module_path.parents
assert not any(
    "twomeow" in repr(finder).lower() and "editable" in repr(finder).lower()
    for finder in sys.meta_path
)
config_yaml = pathlib.Path(config_module.__file__).with_name("default.yaml").resolve()
assert config_yaml.is_file()
assert target in config_yaml.parents
agent = Agent({str(catalog)!r}, {{"use_dense": False, "use_llm_ranker": False}})
assert agent.config["use_dense"] is False
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    _run([sys.executable, "-I", "-c", check], cwd=tmp_path, env=env)
