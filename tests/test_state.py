"""Tests for src/agent/state.py: SessionMemory and SlotTracker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.state import SessionMemory, SlotTracker, _parse_constraint_list, _classify_constraint
from src.agent.router import IntentRouter


def test_session_initial_state():
    s = SessionMemory({})
    assert s.scenario_type == "unknown"
    assert s.slots == {}
    assert s.turn_count == 0


def test_parse_matters_is():
    session = SessionMemory({})
    SlotTracker(session).extract_and_update("For that, what matters is: leather; color: black.")
    assert session.slots.get("material") == "leather"
    assert session.slots.get("color") == "black"


def test_parse_key_requirement():
    session = SessionMemory({})
    SlotTracker(session).extract_and_update("I'm looking for shoes. A key requirement is: cotton.")
    assert session.slots.get("material") == "cotton"
    assert session.slots.get("category") == "shoes"


def test_parse_need_is_after_override():
    session = SessionMemory({})
    IntentRouter().update_scenario("Actually, ignore my earlier preference. What I need is: wool.", session)
    SlotTracker(session).extract_and_update("Actually, ignore my earlier preference. What I need is: wool.")
    assert session.slots.get("material") == "wool"


def test_parse_budget():
    constraints = _parse_constraint_list("budget around $75.99")
    assert constraints.get("budget") == "75.99"


def test_classify_constraint():
    assert _classify_constraint("leather") == "material"
    assert _classify_constraint("color: black") == "color"
    assert _classify_constraint("budget around $50") == "budget"
    assert _classify_constraint("slim fit") == "style"
    assert _classify_constraint("hiking boots use") == "use_case"


def test_retrieval_track_buying():
    s = SessionMemory({})
    s.scenario_type = "buying"
    assert s.retrieval_track() == "buying"


def test_retrieval_track_override():
    s = SessionMemory({})
    s.scenario_type = "intent_override"
    s.override_applied = True
    assert s.retrieval_track() == "buying"


def test_retrieval_track_browsing():
    s = SessionMemory({})
    s.scenario_type = "browsing"
    assert s.retrieval_track() == "browsing"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
