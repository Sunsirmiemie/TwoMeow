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

from ..config import load_config
from ..observability import candidate_snapshot
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.candidate_builder import build_rerank_pool
from ..ranking.reranker import Ranker
from ..dialogue.question_policy import Clarifier
from ..dialogue.attribute_stats import SCOREABLE_ATTRS
from ..dialogue.early_stop import evaluate_stop, should_stop
from .router import IntentRouter
from .state import SessionMemory, SlotTracker
from .response_builder import build_query, build_message

_BUYING_RE = re.compile(r"a key requirement is:", re.I)


class Agent:
    def __init__(self, catalog_path: str, config: dict | None = None):
        self.config    = load_config(config)
        self.retrieval_top_k = self.config.get("retrieval_top_k", 100)
        self.retriever = HybridRetriever(catalog_path, self.config)
        self.router    = IntentRouter()
        self.ranker    = Ranker(self.config, title_lookup=self.retriever.bm25._titles)
        self.clarifier = Clarifier(self.config.get("min_pool_for_dynamic", 10))
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
        trace_enabled = self.config["trace_enabled"]
        if trace_enabled:
            candidates, retrieval_trace = self.retriever.retrieve_with_trace(
                query,
                session.slots,
                track,
                top_k=self.retrieval_top_k,
                turn=turn,
            )
        else:
            candidates = self.retriever.retrieve(
                query,
                session.slots,
                track,
                top_k=self.retrieval_top_k,
                turn=turn,
            )

        # 4. Attribute selection (dynamic entropy or global-entropy fallback)
        attr_cache   = self.retriever.bm25._attr_cache
        use_dynamic  = self.config.get("use_dynamic_entropy", True)
        cands_arg    = candidates if use_dynamic else None
        cache_arg    = attr_cache if use_dynamic else None

        asked     = set(session.asked_attributes)
        known     = set(session.slots.keys())
        remaining = [a for a in SCOREABLE_ATTRS if a not in asked and a not in known]
        tau = self.config.get("entropy_tau", 0.3)
        min_pool = self.config.get("min_pool_for_dynamic", 10)
        early_stop_enabled = self.config.get("use_early_stop", True)
        stop_decision = None
        question_decision = None

        if early_stop_enabled:
            if trace_enabled:
                stop_decision = evaluate_stop(
                    candidates,
                    attr_cache,
                    remaining,
                    tau=tau,
                    min_pool_for_dynamic=min_pool,
                )
                stop_triggered = stop_decision["triggered"]
            else:
                stop_triggered = should_stop(
                    candidates,
                    attr_cache,
                    remaining,
                    tau=tau,
                    min_pool_for_dynamic=min_pool,
                )
            if stop_triggered:
                # Below entropy threshold — wildcard gives any undisclosed constraint
                session.asked_attributes.append("other")
                ask_attribute = "other"
                if trace_enabled:
                    question_decision = {
                        "attribute": "other",
                        "eligible_attributes": remaining,
                        "scores": {},
                        "mode": "early_stop",
                    }
            else:
                if trace_enabled:
                    ask_attribute, question_decision = self.clarifier.next_ask_with_trace(
                        session,
                        cands_arg,
                        cache_arg,
                    )
                else:
                    ask_attribute = self.clarifier.next_ask(session, cands_arg, cache_arg)
        else:
            stop_triggered = False
            if trace_enabled:
                stop_decision = evaluate_stop(
                    candidates,
                    attr_cache,
                    remaining,
                    tau=tau,
                    min_pool_for_dynamic=min_pool,
                )
                ask_attribute, question_decision = self.clarifier.next_ask_with_trace(
                    session,
                    cands_arg,
                    cache_arg,
                )
            else:
                ask_attribute = self.clarifier.next_ask(session, cands_arg, cache_arg)

        # 5. Pool truncation + reranking
        rerank_pool = build_rerank_pool(
            candidates,
            session,
            few_slots_threshold=self.config.get("few_slots_threshold", 2),
            pool_size_threshold=self.config.get("pool_size_threshold", 50),
            truncated_size=self.config.get("truncated_size", 20),
        )
        if trace_enabled:
            ranked, ranker_trace = self.ranker.rerank_with_trace(
                rerank_pool,
                session,
                top_k=top_k,
            )
        else:
            ranked = self.ranker.rerank(rerank_pool, session, top_k=top_k)

        # 6. Record turn
        session.add_turn(user_message, ranked)

        response = {
            "message": build_message(ask_attribute, ranked),
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": p["parent_asin"], "score": p["score"]}
                for p in ranked
            ],
            "usage": self.ranker.token_usage,
        }
        if trace_enabled:
            response["debug_trace"] = {
                "schema_version": 1,
                "turn": turn,
                "retrieval": retrieval_trace,
                "dialogue": {
                    "query": query,
                    "track": track,
                    "slots": dict(session.slots),
                    "candidate_count": len(candidates),
                    "eligible_attributes": question_decision["eligible_attributes"],
                    "remaining_attributes": remaining,
                    "entropy_scores": stop_decision["scores"],
                    "max_entropy_score": stop_decision["max_score"],
                    "tau": tau,
                    "early_stop": {
                        "enabled": early_stop_enabled,
                        "triggered": stop_triggered,
                    },
                    "chosen_ask_attribute": ask_attribute,
                    "question_decision": question_decision,
                },
                "rerank_pool": {
                    "configured_truncation_size": self.config["truncated_size"],
                    "input_count": len(candidates),
                    "candidate_count": len(rerank_pool),
                    "truncated": len(rerank_pool) < len(candidates),
                    "candidates": candidate_snapshot(rerank_pool),
                },
                "ranker": ranker_trace,
                "final": {"candidates": candidate_snapshot(ranked)},
            }
        return response
