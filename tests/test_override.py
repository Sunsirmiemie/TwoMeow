"""Tests for src/agent/router.py and src/dialogue/override.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.state import SessionMemory
from src.agent.router import IntentRouter


def test_scenario_buying():
    session = SessionMemory({})
    IntentRouter().update_scenario("I'm looking for shoes. A key requirement is: black leather.", session)
    assert session.scenario_type == "buying"


def test_scenario_browsing():
    session = SessionMemory({})
    IntentRouter().update_scenario("I'm looking for a jacket, but I'm still exploring.", session)
    assert session.scenario_type == "browsing"


def test_scenario_override_clears_slots():
    session = SessionMemory({})
    session.slots = {"color": "red", "size": "L"}
    IntentRouter().update_scenario("Actually, ignore my earlier preference. What I need is: leather.", session)
    assert session.scenario_type == "intent_override"
    assert session.override_applied is True
    assert session.slots == {}


def test_scenario_boundary():
    session = SessionMemory({})
    IntentRouter().update_scenario("I don't have a preference; please use your judgment.", session)
    assert session.boundary_detected is True


def test_override_clears_asked_attributes():
    session = SessionMemory({})
    session.asked_attributes = ["color", "size"]
    IntentRouter().update_scenario("Actually, ignore my earlier preference.", session)
    assert session.asked_attributes == []


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
