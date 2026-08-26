"""
Main Agent entry point — implements the official reset() / respond() interface.
Orchestrates: intent routing → slot extraction → retrieval → clarification → reranking.

Ablation config flags (all default to current best-known settings):
  use_dense             (bool, True)   – BM25+Dense vs BM25-only
  use_dynamic_entropy   (bool, True)   – pool-aware vs global-entropy clarifier
  use_early_stop        (bool, True)   – halt clarification when gain < TAU=0.3
  use_override_detection(bool, True)   – detect intent override + boundary
"""
from __future__ import annotations

import re

from ..retrieval.hybrid import HybridRetriever
from ..retrieval.candidate_builder import build_rerank_pool
from ..ranking.reranker import Ranker
from ..dialogue.question_policy import Clarifier
from ..dialogue.attribute_stats import SCOREABLE_ATTRS
from ..dialogue.early_stop import should_stop
from .router import IntentRouter
from .state import SessionMemory, SlotTracker
from .response_builder import build_query, build_message

_BUYING_RE = re.compile(r"a key requirement is:", re.I)


class Agent:
    def __init__(self, catalog_path: str, config: dict | None = None):
        self.config    = config or {}
        self.retriever = HybridRetriever(catalog_path, self.config)
        self.router    = IntentRouter()
        self.ranker    = Ranker(self.config, title_lookup=self.retriever.bm25._titles)
        self.clarifier = Clarifier()
        self._sessions: dict[str, SessionMemory] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionMemory(user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        session = self._sessions.get(session_id, SessionMemory({}))

        # 1. Scenario detection (always needed for retrieval track)
        if self.config.get("use_override_detection", True):
            self.router.update_scenario(user_message, session)
        else:
            # Basic buying/browsing only — no override slot-clear, no boundary flag
            if session.turn_count == 0:
                session.scenario_type = "buying" if _BUYING_RE.search(user_message) else "browsing"

        # 2. Extract slots
        SlotTracker(session).extract_and_update(user_message)

        # 3. Retrieve
        query      = build_query(user_message, session)
        track      = session.retrieval_track()
        candidates = self.retriever.retrieve(query, session.slots, track, top_k=100, turn=turn)

        # 4. Attribute selection (dynamic entropy or global-entropy fallback)
        attr_cache   = self.retriever.bm25._attr_cache
        use_dynamic  = self.config.get("use_dynamic_entropy", True)
        cands_arg    = candidates if use_dynamic else None
        cache_arg    = attr_cache if use_dynamic else None

        if self.config.get("use_early_stop", True):
            asked     = set(session.asked_attributes)
            known     = set(session.slots.keys())
            remaining = [a for a in SCOREABLE_ATTRS if a not in asked and a not in known]
            if should_stop(candidates, attr_cache, remaining):
                # Below entropy threshold — wildcard gives any undisclosed constraint
                session.asked_attributes.append("other")
                ask_attribute = "other"
            else:
                ask_attribute = self.clarifier.next_ask(session, cands_arg, cache_arg)
        else:
            ask_attribute = self.clarifier.next_ask(session, cands_arg, cache_arg)

        # 5. Pool truncation + reranking
        rerank_pool = build_rerank_pool(candidates, session)
        ranked      = self.ranker.rerank(rerank_pool, session, top_k=top_k)

        # 6. Record turn
        session.add_turn(user_message, ranked)

        return {
            "message": build_message(ask_attribute, ranked),
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": p["parent_asin"], "score": p["score"]}
                for p in ranked
            ],
            "usage": self.ranker.token_usage,
        }
