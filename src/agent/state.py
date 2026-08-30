"""
Session state: dialogue slots, scenario flags, and turn history.
Slot parsing mirrors evaluator's classify_constraint() and message templates exactly.
Merges memory (SessionMemory) and slot parsing (SlotTracker) in one responsibility unit.
"""
from __future__ import annotations

import re

from ..dialogue.purification import PurifiedReply, purify_reply
from ..dialogue.override_memory import restore_unrelated_override_evidence
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
_LOOKING_RE     = re.compile(r"looking for\s+([^.;]+?)(?:\s*\.|$)", re.I)

_FALLBACK_PATTERNS: dict[str, list[str]] = {
    "color":    [r"\b(black|white|red|blue|green|pink|grey|gray|brown|navy|beige|yellow|purple|orange)\b"],
    "size":     [
        r"\bsize\s*[:=]?\s*(xs|s|m|l|xl|xxl)\b",
        r"\b(xs|xxl|xl|small|medium|large|extra large|one size|plus size|petite|tall|wide|narrow)\b",
        r"\b((?:xs|s|m|l|xl|xxl)\s*(?:[/\-]\s*(?:xs|s|m|l|xl|xxl)){1,3})\b",
        r"\b(?:us|size)\s*[:=]?\s*(\d{1,2}(?:\.\d)?)\b",
    ],
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
        self.other_asked: bool = False
        self.last_reply_new_info: bool = True
        self.no_info_streak: int = 0
        self.negative_slots: dict[str, set[str]] = {}
        self.no_preference_slots: set[str] = set()
        self.slot_confidence: dict[str, float] = {}
        self.slot_turns: dict[str, int] = {}
        self.context_start_turn: int = 0
        self.last_query_text: str = ""
        self.use_reply_purification: bool = True
        self.override_snapshot: dict | None = None
        self.override_carryover_confidence: float = 0.35

    def add_turn(
        self,
        message: str,
        recommendations: list,
        query_text: str | None = None,
    ) -> None:
        self.history.append({
            "turn": len(self.history) + 1,
            "message": message,
            "query_text": message if query_text is None else query_text,
            "top_asin": recommendations[0]["parent_asin"] if recommendations else None,
        })

    def accumulated_text(self) -> str:
        return " ".join(
            t.get("query_text", t["message"])
            for t in self.history[self.context_start_turn:]
        )

    def constraint_context(self) -> dict:
        """Return a read-only-style snapshot for retrieval and ranking."""
        return {
            "slots": dict(self.slots),
            "negative_slots": {
                key: set(values) for key, values in self.negative_slots.items()
            },
            "no_preference_slots": set(self.no_preference_slots),
            "slot_confidence": dict(self.slot_confidence),
            "slot_turns": dict(self.slot_turns),
            "turn": self.turn_count + 1,
        }

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

    def __init__(self, session: SessionMemory, use_purification: bool = True):
        self.session = session
        self.use_purification = use_purification

    def _record_purified_evidence(self, reply: PurifiedReply) -> None:
        for attribute in reply.no_preference_attributes:
            self.session.no_preference_slots.add(attribute)
            self.session.slots.pop(attribute, None)
            self.session.slot_confidence.pop(attribute, None)
            self.session.slot_turns.pop(attribute, None)
        for attribute, values in reply.excluded_values.items():
            self.session.negative_slots.setdefault(attribute, set()).update(values)

    def _record_positive_evidence(
        self,
        parsed: dict[str, str],
        confidence: float,
    ) -> None:
        current_turn = self.session.turn_count + 1
        for attribute, value in parsed.items():
            self.session.slots[attribute] = value
            self.session.no_preference_slots.discard(attribute)
            self.session.negative_slots.get(attribute, set()).discard(value.lower())
            self.session.slot_confidence[attribute] = confidence
            self.session.slot_turns[attribute] = current_turn

    def extract_and_update(self, message: str) -> bool:
        """Parse a message into slots. Returns True if any slot value changed."""
        before = self.session.constraint_context()
        reply = purify_reply(message) if self.use_purification else PurifiedReply(message)
        self.session.last_query_text = reply.query_text
        self._record_purified_evidence(reply)
        positive_text = reply.query_text

        if "category" not in self.session.slots:
            m = _LOOKING_RE.search(positive_text)
            if m:
                category = re.sub(r"\s+", " ", m.group(1)).strip()
                if category and category.lower() not in ("clothing", "clothing shoes & jewelry"):
                    self._record_positive_evidence({"category": category}, 0.9)

        parsed: dict[str, str]
        confidence = 0.75
        m = _MATTERS_RE.search(positive_text)
        if m:
            parsed = _parse_constraint_list(m.group(1))
            confidence = 1.0
        elif m := _NEED_RE.search(positive_text):
            parsed = _parse_constraint_list(m.group(1))
            confidence = 1.0
        elif m := _REQUIREMENT_RE.search(positive_text):
            parsed = _parse_constraint_list(m.group(1))
            parsed.update({
                key: value for key, value in _fallback_extract(positive_text).items()
                if key not in parsed and key not in self.session.slots
            })
            confidence = 1.0
        else:
            parsed = {
                key: value for key, value in _fallback_extract(positive_text).items()
                if key not in self.session.slots
            }
        self._record_positive_evidence(parsed, confidence)
        restore_unrelated_override_evidence(self.session, parsed, reply)
        return self.session.constraint_context() != before
