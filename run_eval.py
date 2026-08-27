"""
Local evaluation runner using the official evaluator.
Usage: python run_eval.py [--no-dense] [--llm-rank]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agent.orchestrator import Agent
from src.config.cli import add_agent_flags, agent_overrides
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    return add_agent_flags(parser)


def main():
    args = build_parser().parse_args()
    config = agent_overrides(args)

    print("Loading catalog and building index...")
    agent = Agent(args.catalog, config)
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)

    print(f"Running evaluation on {len(samples)} sessions...")
    result = evaluate(agent, samples, catalog_ids, categories, products)

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in result.items() if k not in ("sessions", "scenario_metrics")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
