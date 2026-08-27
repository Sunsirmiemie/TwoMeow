"""Small JSON-safe projections used by optional evaluation traces."""
from __future__ import annotations

from typing import Any


def candidate_snapshot(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project ranked candidates without leaking catalog or evaluator internals."""
    return [
        {
            "parent_asin": str(candidate["parent_asin"]),
            "rank": rank,
            "score": (
                float(candidate["score"])
                if candidate.get("score") is not None
                else None
            ),
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]
