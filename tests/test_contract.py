"""
API contract tests: verify Agent conforms to the official evaluator interface.
Does NOT load the real catalog — uses a tiny stub to keep tests fast.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.orchestrator import Agent

_STUB_PRODUCT = {
    "parent_asin": "B001TEST",
    "title": "Test Cotton Shirt",
    "categories": ["Clothing"],
    "features": ["lightweight", "breathable"],
    "description": "A comfortable cotton shirt.",
    "store": "TestStore",
    "details": {},
    "price": 29.99,
}


def _make_agent() -> Agent:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(_STUB_PRODUCT) + "\n")
        catalog_path = f.name
    return Agent(catalog_path, {"use_dense": False})


def test_reset_creates_session():
    agent = _make_agent()
    agent.reset("s1", {"preference_tags": []})


def test_respond_returns_required_keys():
    agent = _make_agent()
    agent.reset("s1", {})
    result = agent.respond("s1", "I need a cotton shirt", turn=1, top_k=10)
    assert "message" in result
    assert "ask_attribute" in result
    assert "recommendations" in result


def test_recommendations_have_parent_asin_and_score():
    agent = _make_agent()
    agent.reset("s1", {})
    result = agent.respond("s1", "cotton shirt for office", turn=1, top_k=10)
    for rec in result["recommendations"]:
        assert "parent_asin" in rec
        assert "score" in rec


def test_ask_attribute_is_valid_or_none():
    from src.dialogue.attribute_stats import ALLOWED_ATTRIBUTES
    agent = _make_agent()
    agent.reset("s1", {})
    result = agent.respond("s1", "I'm still exploring options", turn=1, top_k=10)
    attr = result["ask_attribute"]
    assert attr is None or attr in ALLOWED_ATTRIBUTES


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
