"""
Ablation study configs: disable components one-at-a-time to measure individual contributions.
Run via: python scripts/run_ablation.py
"""
from __future__ import annotations

ABLATION_CONFIGS: list[dict] = [
    {"name": "bm25_only",    "use_dense": False, "use_llm_ranker": False},
    {"name": "hybrid",       "use_dense": True,  "use_llm_ranker": False},
    {"name": "bm25_llm",     "use_dense": False, "use_llm_ranker": True},
    {"name": "hybrid_llm",   "use_dense": True,  "use_llm_ranker": True},
]
