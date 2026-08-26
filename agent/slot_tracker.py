"""
Parses user/evaluator messages into dialog slots.

Handles the four evaluator message patterns from local_evaluator.py:
  1. "For that, what matters is: X; Y."          → parse constraints into slots
  2. "A key requirement is: X."                  → buying constraint
  3. "Actually, ignore my earlier preference..."  → handled by IntentRouter (slots already cleared)
  4. Generic text                                → regex extraction fallback
"""
from __future__ import annotations

import re

# Mirror evaluator's MATERIALS constant exactly
MATERIALS = frozenset((
    "cotton", "polyester", "nylon", "leather", "wool",
    "spandex", "silk", "rayon", "fabric",
))

COLORS = frozenset((
    "black", "white", "blue", "red", "pink", "green",
    "brown", "gray", "grey", "purple", "yellow", "orange",
))

# Patterns for evaluator's two structured response types
_MATTERS_RE    = re.compile(r"what matters is:\s*(.+?)\.?\s*$", re.I)
_REQUIREMENT_RE = re.compile(r"a key requirement is:\s*(.+?)\.?\s*$", re.I)
_NEED_RE       = re.compile(r"what i need is:\s*(.+?)\.?\s*$", re.I)

# Fallback regex per slot type for generic text
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
    """Pull the usable value out of a raw constraint string."""
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
    # For style/size/use_case/feature: strip leading "label: " prefix if present
    cleaned = re.sub(r"^\w[\w\s]+?:\s*", "", lowered).strip()
    return cleaned or lowered


def _parse_constraint_list(raw: str) -> dict[str, str]:
    """
    Parse semicolon-separated constraint string into {slot: value} dict.
    Each segment may contain multiple classifiable signals (e.g. "slim fit for gym"
    → style=slim fit AND use_case=gym), so run fallback regex on the full segment
    in addition to the primary classification.
    """
    result: dict[str, str] = {}
    for part in re.split(r";\s*", raw.strip(". ")):
        part = part.strip()
        if not part:
            continue
        # Primary classification
        slot = _classify_constraint(part)
        result[slot] = _extract_value(part, slot)
        # Secondary: run regex fallback to catch any additional signals in the same phrase
        for extra_slot, extra_val in _fallback_extract(part).items():
            if extra_slot not in result:
                result[extra_slot] = extra_val
    return result


def _fallback_extract(text: str) -> dict[str, str]:
    """Generic regex extraction for unstructured user text."""
    extracted: dict[str, str] = {}
    lowered = text.lower()
    for slot, patterns in _FALLBACK_PATTERNS.items():
        for pattern in patterns:
            m = re.search(pattern, lowered)
            if m:
                extracted[slot] = m.group(1) if m.lastindex else m.group(0)
                break
    return extracted


class SlotTracker:
    def __init__(self, session):
        self.session = session

    def extract_and_update(self, message: str) -> None:
        """
        Parse the current message and update session slots.
        IntentRouter has already handled override (slot clear) and boundary flag
        before this method is called.
        """
        # Pattern 1: evaluator reveals constraints — "For that, what matters is: X; Y."
        m = _MATTERS_RE.search(message)
        if m:
            self.session.slots.update(_parse_constraint_list(m.group(1)))
            return

        # Pattern 2: intent override new value — "What I need is: X."
        m = _NEED_RE.search(message)
        if m:
            self.session.slots.update(_parse_constraint_list(m.group(1)))
            return

        # Pattern 3: buying opening — "A key requirement is: X."
        m = _REQUIREMENT_RE.search(message)
        if m:
            self.session.slots.update(_parse_constraint_list(m.group(1)))
            # Also run fallback to catch category from the full message
            self.session.slots.update(
                {k: v for k, v in _fallback_extract(message).items()
                 if k not in self.session.slots}
            )
            return

        # Pattern 4: generic text — use regex fallback
        extracted = _fallback_extract(message)
        # Don't overwrite already-confirmed slots from structured messages
        for k, v in extracted.items():
            if k not in self.session.slots:
                self.session.slots[k] = v
