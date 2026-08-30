#!/usr/bin/env python
"""Build a deterministic evaluator set whose target ASINs are not in public_set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def _load_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--public-set", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exclude-set",
        action="append",
        default=[],
        help="Additional evaluator JSONL whose target ASINs must be excluded.",
    )
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.count <= 0 or args.count % 20:
        raise ValueError("--count must be a positive multiple of 20")

    public = _load_jsonl(args.public_set)
    public_targets = {
        str(sample["ground_truth"]["parent_asin"]) for sample in public
    }
    excluded_targets = set(public_targets)
    for path in args.exclude_set:
        excluded_targets.update(
            str(sample["ground_truth"]["parent_asin"])
            for sample in _load_jsonl(path)
        )
    eligible = [
        product for product in _load_jsonl(args.catalog)
        if str(product.get("parent_asin") or "") not in excluded_targets
        and product.get("title")
        and product.get("categories")
    ]
    if len(eligible) < args.count:
        raise ValueError("not enough eligible unseen products")

    rng = random.Random(args.seed)
    targets = rng.sample(eligible, args.count)
    scenarios = (
        ["buying"] * (args.count * 40 // 100)
        + ["browsing"] * (args.count * 40 // 100)
        + ["intent_override"] * (args.count * 15 // 100)
        + ["boundary"] * (args.count * 5 // 100)
    )
    rng.shuffle(scenarios)
    profiles = [sample.get("user_profile") or {} for sample in public]
    rows = []
    for index, (product, scenario) in enumerate(zip(targets, scenarios, strict=True)):
        rows.append({
            "sample_id": f"unseen_{args.seed}_{index:04d}",
            "scenario_type": scenario,
            "user_profile": profiles[rng.randrange(len(profiles))],
            "ground_truth": {"parent_asin": str(product["parent_asin"])},
        })

    output = Path(args.output)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    assert not ({row["ground_truth"]["parent_asin"] for row in rows} & excluded_targets)
    print(json.dumps({
        "output": str(output),
        "sample_count": len(rows),
        "seed": args.seed,
        "public_target_overlap": 0,
        "all_excluded_target_overlap": 0,
        "scenario_counts": {
            scenario: scenarios.count(scenario) for scenario in sorted(set(scenarios))
        },
    }, indent=2))


if __name__ == "__main__":
    main()
