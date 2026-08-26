"""
Session state: dialogue slots, scenario flags, and turn history.
Slot parsing mirrors evaluator's classify_constraint() and message templates exactly.
Merges memory (SessionMemory) and slot parsing (SlotTracker) in one responsibility unit.
"""
from __future__ import annotations

import re

# ── Constraint vocabulary (mirrors evaluator constants exactly) ───────────────

MATERIALS = frozenset((
    "cotton", "polyester", "nylon", "leather", "wool",
    "spandex", "silk", "rayon", "fabric",
))
COLORS = frozenset((
    "black", "white", "blue", "red", "pink", "green",
    "brown", "gray", "grey", "purple", "yellow", "orange",
))

_MATTERS_RE     = re.compile(r"what matters is:\s*(.+?)\.?\s*$", re.I)
_REQUIREMENT_RE = re.compile(r"a key requirement is:\s*(.+?)\.?\s*$", re.I)
_NEED_RE        = re.compile(r"what i need is:\s*(.+?)\.?\s*$", re.I)

_FALLBACK_PATTERNS: dict[str, list[str]] = {
    "color":    [r"\b(black|white|red|blue|green|pink|grey|gray|brown|navy|beige|yellow|purple|orange)\b"],
    "size":     [r"\b(xs|s|m|l|xl|xxl|small|medium|large|extra large|\d+[wl]?)\b"],
    "brand":    [r"\b(nike|adidas|levi(?:\'s)?|zara|gucci|puma|reebok|calvin klein|h&m|uniqlo)\b"],
    "budget":   [r"under \$?(\d+)", r"less than \$?(\d+)", r"\$?(\d+)\s*or less", r"budget[^\d]*(\d+)"],
    "material": [r"\b(cotton|leather|polyester|wool|silk|denim|linen|nylon|suede|velvet|spandex|rayon|fabric)\b"],
    "style":    [r"\b(casual|formal|vintage|sporty|elegant|bohemian|minimalist|streetwear)\b"],
    "use_case": [r"\b(office|gym|outdoor|beach|wedding|party|daily|work|travel|hiking|running|winter)\b"],
    "category": [r"\b(shoes|sneakers|boots|sandals|dress|shirt|jacket|pants|jeans|bag|hat|watch|ring|necklace|coat|skirt|shorts)\b"],
    "feature":  [r"\b(waterproof|breathable|lightweight|stretchy|slim fit|oversized|long sleeve|short sleeve|anti-slip)\b"],
}


# ── Slot parsing helpers ──────────────────────────────────────────────────────

def _classify_constraint(value: str) -> str:
    """Mirror evaluator's classify_constraint() to map constraint text → slot name."""
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(m in lowered for m in MATERIALS):
        return "material"
    if any(w in lowered for w in ("color", *COLORS)):
        return "color"
    if any(w in lowered for w in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(w in lowered for w in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(w in lowered for w in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def _extract_value(text: str, slot_type: str) -> str:
    lowered = text.lower().strip()
    if slot_type == "budget":
        m = re.search(r"\$?(\d+(?:\.\d+)?)", text)
        return m.group(1) if m else lowered
    if slot_type == "material":
        for mat in MATERIALS:
            if mat in lowered:
                return mat
        return lowered
    if slot_type == "color":
        m = re.search(r"color[:\s]+(\w+)", lowered)
        if m:
            return m.group(1)
        for c in COLORS:
            if c in lowered:
                return c
        return lowered
    cleaned = re.sub(r"^\w[\w\s]+?:\s*", "", lowered).strip()
    return cleaned or lowered


def _parse_constraint_list(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in re.split(r";\s*", raw.strip(". ")):
        part = part.strip()
        if not part:
            continue
        slot = _classify_constraint(part)
        result[slot] = _extract_value(part, slot)
        for extra_slot, extra_val in _fallback_extract(part).items():
            if extra_slot not in result:
                result[extra_slot] = extra_val
    return result


def _fallback_extract(text: str) -> dict[str, str]:
    extracted: dict[str, str] = {}
    lowered = text.lower()
    for slot, patterns in _FALLBACK_PATTERNS.items():
        for pattern in patterns:
            m = re.search(pattern, lowered)
            if m:
                extracted[slot] = m.group(1) if m.lastindex else m.group(0)
                break
    return extracted


# ── Session state ─────────────────────────────────────────────────────────────

class SessionMemory:
    def __init__(self, user_profile: dict):
        self.user_profile = user_profile
        self.slots: dict[str, str] = {}
        self.scenario_type: str = "unknown"
        self.boundary_detected: bool = False
        self.override_applied: bool = False
        self.asked_attributes: list[str] = []
        self.history: list[dict] = []

    def add_turn(self, message: str, recommendations: list) -> None:
        self.history.append({
            "turn": len(self.history) + 1,
            "message": message,
            "top_asin": recommendations[0]["parent_asin"] if recommendations else None,
        })

    def accumulated_text(self) -> str:
        return " ".join(t["message"] for t in self.history)

    @property
    def turn_count(self) -> int:
        return len(self.history)

    def retrieval_track(self) -> str:
        if self.scenario_type == "buying" or self.override_applied:
            return "buying"
        return "browsing"

    def preference_tags(self) -> list[str]:
        return self.user_profile.get("preference_tags", [])


# ── Slot tracker ──────────────────────────────────────────────────────────────

class SlotTracker:
    """Parses evaluator messages and updates session slots."""

    def __init__(self, session: SessionMemory):
        self.session = session

    def extract_and_update(self, message: str) -> None:
        # IntentRouter has already handled override (slot clear) before this runs.
        m = _MATTERS_RE.search(message)
        if m:
            self.session.slots.update(_parse_constraint_list(m.group(1)))
            return
        m = _NEED_RE.search(message)
        if m:
            self.session.slots.update(_parse_constraint_list(m.group(1)))
            return
        m = _REQUIREMENT_RE.search(message)
        if m:
            self.session.slots.update(_parse_constraint_list(m.group(1)))
            self.session.slots.update(
                {k: v for k, v in _fallback_extract(message).items()
                 if k not in self.session.slots}
            )
            return
        # Generic text — regex fallback, never overwrite confirmed structured slots
        for k, v in _fallback_extract(message).items():
            if k not in self.session.slots:
                self.session.slots[k] = v
