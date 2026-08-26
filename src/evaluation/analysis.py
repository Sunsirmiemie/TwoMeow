"""
Score analysis utilities: breakdown by scenario, turn distribution, and rank histogram.
"""
from __future__ import annotations


def summarize(result: dict) -> dict:
    """Strip per-session detail, return top-level metrics."""
    return {k: v for k, v in result.items() if k not in ("sessions", "scenario_metrics")}


def scenario_breakdown(result: dict) -> dict:
    """Return per-scenario HitRate and MRR from evaluate() output."""
    return result.get("scenario_metrics", {})
