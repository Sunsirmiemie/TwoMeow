"""
Single-session trace for demo purposes.
Shows the full conversation turn-by-turn: user message, agent reply,
top-5 recommendations (with titles), and hit status.

Usage:
    python scripts/trace_session.py                        # default: public_0001
    python scripts/trace_session.py --id public_0042
    python scripts/trace_session.py --scenario browsing    # pick first matching scenario
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── ANSI colours ─────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
RESET  = "\033[0m"

# ── evaluator helpers (duplicated to avoid import side-effects) ───────────────
MAX_TURNS = 10
TOP_K     = 10
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
MATERIALS  = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE    = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def intent_card(product: dict) -> dict:
    title = _clean(str(product.get("title") or "product"))
    candidates = [*_flatten_values(product.get("features")), *_flatten_values(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color    = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(_clean(c) for c in candidates if _clean(c)))
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def behavior_for(scenario: str, card: dict, rng: random.Random) -> dict:
    behavior: dict = {"scenario_type": scenario}
    if scenario == "intent_override":
        hard = card["hard_constraints"]
        soft = card["soft_preferences"]
        old_value = soft[-1] if soft else "I prefer a different style."
        new_value = hard[0] if hard else "Please prioritize the target requirements."
        behavior["override"] = {
            "turn": rng.choice([3, 4]),
            "old_value": old_value,
            "new_value": new_value,
            "message": f"Actually, ignore my earlier preference. What I need is: {new_value}.",
        }
    return behavior


def coarse_category(values: list[str]) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned = [
        part.strip()
        for v in values
        for part in v.split(",")
        if part.strip() and part.strip().lower() not in excluded
    ]
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(m in lowered for m in MATERIALS):
        return "material"
    if any(w in lowered for w in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(w in lowered for w in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(w in lowered for w in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(w in lowered for w in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def initial_message(sample: dict, category: str, disclosed: set[str]) -> str:
    scenario = sample["scenario_type"]
    if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
        constraint = str(sample["intent_card"]["hard_constraints"][0])
        disclosed.add(constraint)
        return f"I'm looking for {category}. A key requirement is: {constraint}."
    if scenario == "intent_override":
        old_value = str(sample["behavior"]["override"]["old_value"])
        return f"I'm looking for {category}. {old_value}"
    return f"I'm looking for {category}, but I'm still exploring."


def customer_reply(sample: dict, ask_attribute: object, disclosed: set[str], boundary_used: bool) -> tuple[str, bool]:
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
        return f"I don't have a preference for {attribute}; please use your judgment.", True
    if not attribute:
        return "Those options are not quite right yet. Ask me about one specific attribute.", boundary_used
    if attribute not in ALLOWED_ATTRIBUTES:
        attribute = "other"
    constraints = [
        *[str(v) for v in sample["intent_card"].get("hard_constraints", [])],
        *[str(v) for v in sample["intent_card"].get("soft_preferences", [])],
    ]
    matches = [v for v in constraints if v not in disclosed and (attribute == "other" or classify_constraint(v) == attribute)][:2]
    if not matches:
        return f"I don't have an additional preference for {attribute}.", boundary_used
    disclosed.update(matches)
    return "For that, what matters is: " + "; ".join(matches) + ".", boundary_used


def normalize_recommendations(payload: object, catalog_ids: set[str]) -> list[str]:
    if not isinstance(payload, list):
        return []
    result, seen = [], set()
    for item in payload:
        asin = item.get("parent_asin", "") if isinstance(item, dict) else item
        asin = str(asin).strip()
        if asin and asin not in seen and asin in catalog_ids:
            seen.add(asin)
            result.append(asin)
        if len(result) >= TOP_K:
            break
    return result


# ── main trace ────────────────────────────────────────────────────────────────

def trace(sample: dict, catalog_ids: set[str], categories: dict, products: dict, titles: dict) -> None:
    from starter.agent import Agent

    target = str(sample["ground_truth"]["parent_asin"])
    target_title = titles.get(target, target)[:70]

    if "intent_card" in sample and "behavior" in sample:
        card, behavior = sample["intent_card"], sample["behavior"]
    else:
        card = intent_card(products[target])
        seed_source = f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}"
        rng = random.Random(seed_source)
        behavior = behavior_for(str(sample["scenario_type"]), card, rng)

    effective = {**sample, "intent_card": card, "behavior": behavior}
    category  = coarse_category(categories.get(target, []))

    print(f"\n{'═'*70}")
    print(f"{BOLD}Session  {CYAN}{sample['sample_id']}{RESET}  │  "
          f"Scenario: {YELLOW}{sample['scenario_type'].upper()}{RESET}")
    print(f"Target   {DIM}{target}{RESET}  {target_title}")
    print(f"{'─'*70}")

    agent = Agent("data/catalog.jsonl")
    session_id = f"trace_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])

    disclosed: set[str] = set()
    boundary_used  = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message   = initial_message(effective, category, disclosed)

    hit_turn: int | None  = None
    best_rank: int | None = None

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n{BOLD}Turn {turn}{RESET}")
        print(f"  {CYAN}User :{RESET} {user_message}")

        try:
            response = agent.respond(session_id, user_message, turn, TOP_K)
        except Exception as exc:
            print(f"  {RED}Agent error: {exc}{RESET}")
            break

        ask_attr = response.get("ask_attribute")
        msg      = response.get("message", "")
        ranked   = normalize_recommendations(response.get("recommendations"), catalog_ids)

        print(f"  {GREEN}Agent:{RESET} {msg}")
        if ask_attr:
            print(f"  {DIM}ask_attribute → {ask_attr}{RESET}")

        print(f"  Top-5 recommendations:")
        for i, asin in enumerate(ranked[:5], 1):
            t = titles.get(asin, asin)[:60]
            hit_marker = f"  {GREEN}◀ TARGET{RESET}" if asin == target else ""
            print(f"    {i}. {DIM}{asin}{RESET}  {t}{hit_marker}")
        if len(ranked) > 5:
            print(f"    {DIM}… +{len(ranked)-5} more{RESET}")

        if override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn  = turn
            print(f"\n  {GREEN}{BOLD}HIT at turn {turn}  rank {best_rank}  MRR={1/best_rank:.4f}{RESET}")
            break

        if turn == MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(effective, ask_attr, disclosed, boundary_used)

    if hit_turn is None:
        print(f"\n  {RED}{BOLD}MISS — target not found in {MAX_TURNS} turns{RESET}")

    print(f"\n{'═'*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace a single evaluation session")
    parser.add_argument("--catalog",  default="data/catalog.jsonl")
    parser.add_argument("--dataset",  default="data/public_set.jsonl")
    parser.add_argument("--id",       default="public_0001", help="sample_id to trace")
    parser.add_argument("--scenario", default=None,
                        help="pick first sample matching this scenario type")
    args = parser.parse_args()

    samples = [json.loads(l) for l in Path(args.dataset).open(encoding="utf-8") if l.strip()]

    catalog_ids: set[str] = set()
    categories:  dict[str, list[str]] = {}
    products:    dict[str, dict] = {}
    titles:      dict[str, str]  = {}
    with Path(args.catalog).open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            asin = str(p["parent_asin"])
            catalog_ids.add(asin)
            categories[asin] = [str(v) for v in (p.get("categories") or [])]
            products[asin]   = p
            titles[asin]     = str(p.get("title") or "")

    if args.scenario:
        sample = next((s for s in samples if s["scenario_type"] == args.scenario), None)
        if sample is None:
            print(f"No sample found with scenario '{args.scenario}'")
            return
    else:
        sample = next((s for s in samples if s["sample_id"] == args.id), None)
        if sample is None:
            print(f"Sample '{args.id}' not found")
            return

    trace(sample, catalog_ids, categories, products, titles)


if __name__ == "__main__":
    main()
