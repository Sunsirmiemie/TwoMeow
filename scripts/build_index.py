#!/usr/bin/env python
"""
Pre-build the field-aware dense embedding index and save it to the cache.
Run this once before eval to avoid embedding delay during evaluation.
Usage: python scripts/build_index.py [--catalog data/catalog.jsonl] [--model all-MiniLM-L6-v2]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.dense import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--model",   default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    print(f"Building dense index for {args.catalog} using {args.model}...")
    DenseRetriever(args.catalog, args.model, use_field_aware=True)
    print("Done. Set TWOMEOW_DENSE_CACHE_DIR to control the cache location.")


if __name__ == "__main__":
    main()
