"""Tests for src/agent/state.py: SessionMemory and SlotTracker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.state import SessionMemory, SlotTracker, _parse_constraint_list, _classify_constraint
from src.agent.router import IntentRouter
from src.agent.response_builder import build_query


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


def test_negated_value_is_excluded_not_added_to_positive_query():
    session = SessionMemory({})
    changed = SlotTracker(session).extract_and_update(
        "I do not want red; blue instead."
    )

    assert changed is True
    assert session.slots["color"] == "blue"
    assert session.negative_slots["color"] == {"red"}
    assert "red" not in session.last_query_text.lower()


def test_no_preference_removes_stale_slot_and_query_clause():
    session = SessionMemory({})
    SlotTracker(session).extract_and_update("color: red")
    SlotTracker(session).extract_and_update(
        "I don't have a preference for color; please use your judgment."
    )

    assert "color" not in session.slots
    assert "color" in session.no_preference_slots
    assert "preference" not in session.last_query_text.lower()


def test_override_starts_a_fresh_active_history_window():
    session = SessionMemory({})
    session.add_turn("red shoes", [], "red shoes")
    IntentRouter().update_scenario(
        "Actually, ignore my earlier preference. What I need is: wool.", session
    )
    SlotTracker(session).extract_and_update(
        "Actually, ignore my earlier preference. What I need is: wool."
    )

    assert session.context_start_turn == 1
    assert "red" not in session.accumulated_text()
    assert session.slots["material"] == "wool"


def test_override_replaces_conflict_and_softly_retains_unrelated_evidence():
    session = SessionMemory({})
    session.slots = {
        "category": "running shoes",
        "color": "red",
        "style": "casual",
    }
    session.slot_confidence = {"category": 0.9, "color": 1.0, "style": 1.0}
    session.slot_turns = {"category": 1, "color": 1, "style": 1}
    session.add_turn("red casual running shoes", [], "red casual running shoes")
    message = "Actually, ignore my earlier preference. What I need is: color: blue."

    IntentRouter().update_scenario(message, session)
    SlotTracker(session).extract_and_update(message)
    query = build_query(message, session)

    assert session.slots["category"] == "running shoes"
    assert session.slots["color"] == "blue"
    assert session.slot_confidence["color"] == 1.0
    assert session.slots["style"] == "casual"
    assert session.slot_confidence["style"] == 0.35
    assert "red" not in query
    assert query.count("casual") == 1


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
