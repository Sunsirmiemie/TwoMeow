#!/usr/bin/env python
"""
Run ablation study: evaluate each config in src/evaluation/ablation.py.
Usage: python scripts/run_ablation.py [--catalog ...] [--dataset ...]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.orchestrator import Agent
from src.evaluation.ablation import ABLATION_CONFIGS
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)

    for cfg in ABLATION_CONFIGS:
        name = cfg.pop("name")
        print(f"\n=== Ablation: {name} ===")
        agent = Agent(args.catalog, cfg)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        summary = {k: v for k, v in result.items() if k not in ("sessions", "scenario_metrics")}
        print(json.dumps(summary, indent=2))
        Path(f"results_ablation_{name}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
