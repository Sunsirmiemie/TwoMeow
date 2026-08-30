"""Selective evidence carry-over after an explicit intent override."""
from __future__ import annotations

from typing import Any


def restore_unrelated_override_evidence(
    session: Any,
    parsed: dict[str, str],
    reply: Any,
) -> None:
    """Restore non-conflicting old slots as weak, re-askable evidence."""
    snapshot = session.override_snapshot
    if not snapshot:
        return
    blocked = (
        set(parsed)
        | set(reply.no_preference_attributes)
        | set(reply.excluded_values)
        | {"category", "budget"}
    )
    for attribute, value in snapshot["slots"].items():
        if attribute in blocked or attribute in session.slots:
            continue
        session.slots[attribute] = value
        old_confidence = snapshot["confidence"].get(attribute, 0.75)
        session.slot_confidence[attribute] = min(
            session.override_carryover_confidence,
            old_confidence * session.override_carryover_confidence,
        )
        session.slot_turns[attribute] = snapshot["turns"].get(attribute, 1)
    session.override_snapshot = None
