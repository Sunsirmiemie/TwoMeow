#!/usr/bin/env python
"""
Run 6-row ablation table: each row adds one more component.
Prints HitRate@10 per scenario in table format.
Usage: python scripts/run_ablation_table.py [--catalog ...] [--dataset ...]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.orchestrator import Agent
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index

CONFIGS = [
    {
        "name":  "Baseline",
        "desc":  "BM25-only  | fixed-ask  | no-override",
        "use_dense": False, "use_dynamic_entropy": False,
        "use_override_detection": False, "use_early_stop": False,
    },
    {
        "name":  "Hybrid",
        "desc":  "+ Dense (BM25+Dense)",
        "use_dense": True,  "use_dynamic_entropy": False,
        "use_override_detection": False, "use_early_stop": False,
    },
    {
        "name":  "+ Entropy",
        "desc":  "+ Dynamic entropy ask",
        "use_dense": True,  "use_dynamic_entropy": True,
        "use_override_detection": False, "use_early_stop": False,
    },
    {
        "name":  "+ Early Stop",
        "desc":  "+ Entropy early-stop τ=0.3",
        "use_dense": True,  "use_dynamic_entropy": True,
        "use_override_detection": False, "use_early_stop": True,
    },
    {
        "name":  "+ Override",
        "desc":  "+ Intent override detection",
        "use_dense": True,  "use_dynamic_entropy": True,
        "use_override_detection": True, "use_early_stop": True,
    },
    {
        "name":  "Final",
        "desc":  "All (no early-stop: maximises HitRate)",
        "use_dense": True,  "use_dynamic_entropy": True,
        "use_override_detection": True, "use_early_stop": False,
    },
]

SCENARIO_ORDER = ["buying", "browsing", "intent_override", "boundary"]
SCENARIO_LABEL = {"buying": "Buying", "browsing": "Browsing",
                  "intent_override": "Override", "boundary": "Boundary"}


def run_config(cfg: dict, catalog: str, dataset: str,
               samples, catalog_ids, categories, products) -> dict:
    agent = Agent(catalog, {k: v for k, v in cfg.items() if k not in ("name", "desc")})
    return evaluate(agent, samples, catalog_ids, categories, products)


def fmt(v: float) -> str:
    return f"{v:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    print("Loading catalog…")
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)

    rows: list[dict] = []

    for cfg in CONFIGS:
        name = cfg["name"]
        desc = cfg["desc"]
        print(f"\n{'='*60}")
        print(f"Running: {name}  ({desc})")
        result = run_config(cfg, args.catalog, args.dataset,
                            samples, catalog_ids, categories, products)
        sm = result.get("scenario_metrics", {})
        row = {
            "name": name,
            "overall_hit": result["hit_rate_at_10"],
            "overall_mrr": result["mrr"],
            "overall_ts":  result["recommended_technical_score"],
            "mttc":        result["mttc"],
        }
        for sc in SCENARIO_ORDER:
            row[sc] = sm.get(sc, {}).get("hit_rate_at_10", 0.0)
        rows.append(row)
        # Interim print
        print(f"  HitRate@10={result['hit_rate_at_10']:.4f}  MRR={result['mrr']:.4f}"
              f"  MTTC={result['mttc']:.2f}  TechScore={result['recommended_technical_score']:.4f}")
        for sc in SCENARIO_ORDER:
            v = sm.get(sc, {})
            print(f"    {SCENARIO_LABEL[sc]:10s}: HR={v.get('hit_rate_at_10',0):.4f}"
                  f"  MRR={v.get('mrr',0):.4f}  MTTC={v.get('mttc',0):.2f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("HitRate@10 per scenario  (ablation table)")
    print("="*80)
    hdr = f"{'System':<14}" + "".join(f"{SCENARIO_LABEL[s]:>10}" for s in SCENARIO_ORDER)
    hdr += f"{'Overall':>10}{'MRR':>8}{'MTTC':>7}{'TechScore':>11}"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        line = f"{row['name']:<14}"
        line += "".join(f"{row[s]:>10.4f}" for s in SCENARIO_ORDER)
        line += f"{row['overall_hit']:>10.4f}{row['overall_mrr']:>8.4f}"
        line += f"{row['mttc']:>7.2f}{row['overall_ts']:>11.4f}"
        print(line)

    # Save results
    out = Path("ablation_results.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
