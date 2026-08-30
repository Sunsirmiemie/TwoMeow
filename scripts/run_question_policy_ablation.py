#!/usr/bin/env python
"""Evaluate the current system with question-selection components ablated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src.agent.orchestrator import Agent


ROWS = (
    (
        "global_entropy_no_early_stop",
        {"use_dynamic_entropy": False, "use_early_stop": False},
    ),
    (
        "dynamic_entropy_no_early_stop",
        {"use_dynamic_entropy": True, "use_early_stop": False},
    ),
    (
        "global_entropy_plus_early_stop",
        {"use_dynamic_entropy": False, "use_early_stop": True},
    ),
    (
        "dynamic_entropy_plus_early_stop_current",
        {"use_dynamic_entropy": True, "use_early_stop": True},
    ),
)


def _summary(name: str, result: dict) -> dict:
    keys = (
        "sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency",
        "recommended_technical_score", "reported_token_usage",
        "scenario_metrics",
    )
    return {"name": name, **{key: result[key] for key in keys}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--with-gated-dense",
        action="store_true",
        help="Keep the production Dense gate; default isolates question policy with BM25.",
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    results = []
    for name, question_config in ROWS:
        config = {
            "use_dense": args.with_gated_dense,
            **question_config,
        }
        print(f"Running {name}...")
        result = evaluate(
            Agent(args.catalog, config),
            samples,
            catalog_ids,
            categories,
            products,
        )
        results.append(_summary(name, result))
        print(json.dumps(results[-1], indent=2))

    Path(args.output).write_text(
        json.dumps({"rows": results}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
