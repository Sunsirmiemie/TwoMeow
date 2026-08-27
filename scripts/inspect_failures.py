#!/usr/bin/env python
"""
Inspect failure modes in a results.json produced by run_public_eval.py.
Usage: python scripts/inspect_failures.py [--results results.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.failure_analysis import find_near_misses, find_complete_misses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json")
    args = parser.parse_args()

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    near   = find_near_misses(result)
    misses = find_complete_misses(result)

    print(f"Near-misses (rank 6–10): {len(near)}")
    print(f"Complete misses (not in top-10): {len(misses)}")
    if near:
        print("\nSample near-miss sessions:")
        for s in near[:5]:
            print(f"  sample={s.get('sample_id')} best_rank={s.get('best_rank')}")


if __name__ == "__main__":
    main()
