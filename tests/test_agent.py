"""
Unit tests covering the four evaluator scenarios and slot parsing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.intent_router import IntentRouter
from agent.slot_tracker import SlotTracker, _parse_constraint_list, _classify_constraint
from agent.clarifier import Clarifier
from agent.memory import SessionMemory


# ── IntentRouter ────────────────────────────────────────────────────────────

def test_scenario_buying():
    session = SessionMemory({})
    IntentRouter().update_scenario(
        "I'm looking for shoes. A key requirement is: black leather.", session
    )
    assert session.scenario_type == "buying"

def test_scenario_browsing():
    session = SessionMemory({})
    IntentRouter().update_scenario(
        "I'm looking for a jacket, but I'm still exploring.", session
    )
    assert session.scenario_type == "browsing"

def test_scenario_override():
    session = SessionMemory({})
    session.slots = {"color": "red", "size": "L"}
    IntentRouter().update_scenario(
        "Actually, ignore my earlier preference. What I need is: leather.", session
    )
    assert session.scenario_type == "intent_override"
    assert session.override_applied is True
    assert session.slots == {}  # slots cleared

def test_scenario_boundary():
    session = SessionMemory({})
    IntentRouter().update_scenario(
        "I don't have a preference for material; please use your judgment.", session
    )
    assert session.boundary_detected is True


# ── SlotTracker ─────────────────────────────────────────────────────────────

def test_parse_matters_is():
    session = SessionMemory({})
    SlotTracker(session).extract_and_update(
        "For that, what matters is: leather; color: black."
    )
    assert session.slots.get("material") == "leather"
    assert session.slots.get("color") == "black"

def test_parse_key_requirement():
    session = SessionMemory({})
    SlotTracker(session).extract_and_update(
        "I'm looking for shoes. A key requirement is: cotton."
    )
    assert session.slots.get("material") == "cotton"
    assert session.slots.get("category") == "shoes"

def test_parse_need_is_after_override():
    session = SessionMemory({})
    # Override already cleared slots (done by IntentRouter)
    IntentRouter().update_scenario(
        "Actually, ignore my earlier preference. What I need is: wool.", session
    )
    SlotTracker(session).extract_and_update(
        "Actually, ignore my earlier preference. What I need is: wool."
    )
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


# ── Clarifier ───────────────────────────────────────────────────────────────

def test_clarifier_always_asks():
    session = SessionMemory({})
    c = Clarifier()
    attr = c.next_ask(session)
    assert attr is not None
    assert attr in {"material", "color", "budget", "feature", "use_case",
                    "style", "size", "brand", "category", "other"}

def test_clarifier_never_asks_category_or_brand():
    """PDF: evaluator's classify_constraint never returns category/brand."""
    session = SessionMemory({})
    c = Clarifier()
    asked = set()
    from agent.clarifier import SCOREABLE_ATTRS
    for _ in range(len(SCOREABLE_ATTRS) + 2):
        attr = c.next_ask(session)
        asked.add(attr)
    assert "category" not in asked
    assert "brand" not in asked

def test_clarifier_uses_scoreable_attrs():
    """Without pool data, clarifier still picks from SCOREABLE_ATTRS."""
    from agent.clarifier import SCOREABLE_ATTRS
    session = SessionMemory({})
    attr = Clarifier().next_ask(session)
    assert attr in set(SCOREABLE_ATTRS) | {"other"}

def test_clarifier_skips_known_slots():
    session = SessionMemory({})
    session.slots = {a: "x" for a in ["material", "color", "size", "style",
                                        "use_case", "feature", "budget"]}
    attr = Clarifier().next_ask(session)
    assert attr == "other"

def test_clarifier_skips_already_asked():
    from agent.clarifier import SCOREABLE_ATTRS
    session = SessionMemory({})
    session.asked_attributes = list(SCOREABLE_ATTRS)
    attr = Clarifier().next_ask(session)
    assert attr == "other"

def test_clarifier_dynamic_scoring():
    """With pool data, clarifier picks attribute with highest coverage×entropy."""
    session = SessionMemory({})
    # Fake candidate pool where use_case is evenly distributed
    candidates = [{"parent_asin": f"B{i:03d}"} for i in range(20)]
    attr_cache = {}
    use_cases = ["gym", "office", "wedding", "outdoor", "beach"] * 4
    for i, c in enumerate(candidates):
        attr_cache[c["parent_asin"]] = {
            "material": None, "color": None, "size": None,
            "style": None, "use_case": use_cases[i],
            "feature": None, "budget": None,
        }
    attr = Clarifier().next_ask(session, candidates, attr_cache)
    assert attr == "use_case"  # 100% coverage, even distribution = highest score

def test_clarifier_fallback_to_other():
    session = SessionMemory({})
    from agent.clarifier import SCOREABLE_ATTRS
    for a in SCOREABLE_ATTRS:
        session.slots[a] = "x"
    attr = Clarifier().next_ask(session)
    assert attr == "other"


# ── Memory ──────────────────────────────────────────────────────────────────

def test_retrieval_track_buying():
    session = SessionMemory({})
    session.scenario_type = "buying"
    assert session.retrieval_track() == "buying"

def test_retrieval_track_override():
    session = SessionMemory({})
    session.scenario_type = "intent_override"
    session.override_applied = True
    assert session.retrieval_track() == "buying"

def test_retrieval_track_browsing():
    session = SessionMemory({})
    session.scenario_type = "browsing"
    assert session.retrieval_track() == "browsing"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
