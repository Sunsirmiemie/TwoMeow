"""Tests for src/dialogue/entropy.py and src/dialogue/question_policy.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.state import SessionMemory
from src.dialogue.entropy import normalized_entropy, score_attribute
from src.dialogue.question_policy import Clarifier
from src.dialogue.attribute_stats import SCOREABLE_ATTRS


def test_normalized_entropy_uniform():
    values = ["a", "b", "c", "d"]
    assert normalized_entropy(values) == 1.0


def test_normalized_entropy_single():
    assert normalized_entropy(["a"]) == 0.0


def test_clarifier_always_asks():
    session = SessionMemory({})
    attr = Clarifier().next_ask(session)
    assert attr is not None
    assert attr in set(SCOREABLE_ATTRS) | {"other"}


def test_clarifier_never_asks_category_or_brand():
    session = SessionMemory({})
    c = Clarifier()
    asked = set()
    for _ in range(len(SCOREABLE_ATTRS) + 2):
        asked.add(c.next_ask(session))
    assert "category" not in asked
    assert "brand" not in asked


def test_clarifier_skips_known_slots():
    session = SessionMemory({})
    session.slots = {a: "x" for a in SCOREABLE_ATTRS}
    assert Clarifier().next_ask(session) == "other"


def test_clarifier_skips_already_asked():
    session = SessionMemory({})
    session.asked_attributes = list(SCOREABLE_ATTRS)
    assert Clarifier().next_ask(session) == "other"


def test_clarifier_dynamic_scoring():
    """With pool data, picks attribute with highest coverage×entropy."""
    session = SessionMemory({})
    candidates = [{"parent_asin": f"B{i:03d}"} for i in range(20)]
    use_cases = ["gym", "office", "wedding", "outdoor", "beach"] * 4
    attr_cache = {
        c["parent_asin"]: {
            "material": None, "color": None, "size": None,
            "style": None, "use_case": use_cases[i],
            "feature": None, "budget": None,
        }
        for i, c in enumerate(candidates)
    }
    attr = Clarifier().next_ask(session, candidates, attr_cache)
    assert attr == "use_case"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
