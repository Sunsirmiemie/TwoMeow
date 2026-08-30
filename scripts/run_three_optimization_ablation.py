#!/usr/bin/env python
"""Reproducible incremental ablation for the three new optimizations."""
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
        "updated_baseline",
        {
            "use_dense": False,
            "use_reply_purification": False,
            "use_dynamic_attribute_scoring": False,
            "use_field_aware_dense": False,
            "use_adaptive_fusion": False,
        },
    ),
    (
        "plus_reply_purification",
        {
            "use_dense": False,
            "use_reply_purification": True,
            "use_dynamic_attribute_scoring": False,
            "use_field_aware_dense": False,
            "use_adaptive_fusion": False,
        },
    ),
    (
        "plus_dynamic_attributes",
        {
            "use_dense": False,
            "use_reply_purification": True,
            "use_dynamic_attribute_scoring": True,
            "use_field_aware_dense": False,
            "use_adaptive_fusion": False,
        },
    ),
    (
        "plus_field_aware_dense",
        {
            "use_dense": True,
            "use_reply_purification": True,
            "use_dynamic_attribute_scoring": True,
            "use_field_aware_dense": True,
            "use_adaptive_fusion": True,
        },
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
    parser.add_argument("--without-dense", action="store_true")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    rows = ROWS[:-1] if args.without_dense else ROWS
    results = []
    for name, config in rows:
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
        json.dumps({"rows": results}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
