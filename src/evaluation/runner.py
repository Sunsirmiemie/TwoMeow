"""
Evaluation runner: thin wrapper around the official local_evaluator.
Called by scripts/run_public_eval.py; not meant to be run directly.
"""
from __future__ import annotations

import json
from pathlib import Path


def run(
    catalog_path: str = "data/catalog.jsonl",
    dataset_path: str = "data/public_set.jsonl",
    output_path: str = "results.json",
    config: dict | None = None,
) -> dict:
    from src.agent.orchestrator import Agent
    from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index

    agent = Agent(catalog_path, config or {})
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
