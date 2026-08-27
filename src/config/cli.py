"""Shared command-line flags that produce explicit flat Agent overrides."""
from __future__ import annotations

import argparse


def add_agent_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--no-dense",
        action="store_const",
        const=False,
        default=None,
        dest="use_dense",
    )
    parser.add_argument(
        "--llm-rank",
        action="store_const",
        const=True,
        default=None,
        dest="use_llm_ranker",
    )
    return parser


def agent_overrides(args: argparse.Namespace) -> dict[str, bool]:
    values = {
        "use_dense": args.use_dense,
        "use_llm_ranker": args.use_llm_ranker,
    }
    return {key: value for key, value in values.items() if value is not None}
