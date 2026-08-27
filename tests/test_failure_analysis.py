"""Tests for failure analysis over the official evaluator result schema."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.failure_analysis import find_complete_misses, find_near_misses


def test_failure_analysis_uses_evaluator_session_fields():
    result = {
        "sessions": [
            {"sample_id": "near-6", "hit": True, "best_rank": 6},
            {"sample_id": "near-10", "hit": True, "best_rank": 10},
            {"sample_id": "top-5", "hit": True, "best_rank": 5},
            {"sample_id": "miss", "hit": False, "best_rank": None},
        ]
    }

    assert [s["sample_id"] for s in find_near_misses(result)] == ["near-6", "near-10"]
    assert [s["sample_id"] for s in find_complete_misses(result)] == ["miss"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
