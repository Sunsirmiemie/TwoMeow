"""
Main Agent entry point — implements the official reset() / respond() interface.
"""
from __future__ import annotations

from .intent_router import IntentRouter
from .slot_tracker import SlotTracker
from .clarifier import Clarifier
from .memory import SessionMemory
from .retriever.hybrid import HybridRetriever
from .ranker import Ranker


class Agent:
    def __init__(self, catalog_path: str, config: dict | None = None):
        self.config = config or {}
        self.retriever = HybridRetriever(catalog_path, self.config)
        self.intent_router = IntentRouter()
        # Pass title lookup so LLM reranker can include product titles in prompt
        self.ranker = Ranker(self.config, title_lookup=self.retriever.bm25._titles)
        self.clarifier = Clarifier()
        self._sessions: dict[str, SessionMemory] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionMemory(user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        session = self._sessions.get(session_id, SessionMemory({}))

        # 1. Detect scenario + update flags (override clears slots here)
        self.intent_router.update_scenario(user_message, session)

        # 2. Extract slots from current message
        SlotTracker(session).extract_and_update(user_message)

        # 3. Build retrieval query: slot values + current message + history
        query = self._build_query(user_message, session)
        track = session.retrieval_track()
        candidates = self.retriever.retrieve(query, session.slots, track, top_k=100, turn=turn)

        # 4. Dynamic truncation (MD §III / §VII in-scope):
        # When the candidate pool is over-general (few confirmed slots, large pool),
        # signal to clarifier that we need a question — which it always provides.
        # The "truncation" here means we pass fewer candidates to the ranker,
        # forcing convergence rather than guessing blindly from a vague pool.
        few_slots = len(session.slots) < 2
        rerank_pool = candidates[:20] if few_slots and len(candidates) >= 50 else candidates

        # 5. Dynamic attribute selection: score against current candidate pool
        #    (PDF: coverage × entropy per pool beats fixed global order)
        attr_cache = self.retriever.bm25._attr_cache
        ask_attribute = self.clarifier.next_ask(session, candidates, attr_cache)

        # 6. Rerank from (possibly truncated) pool
        ranked = self.ranker.rerank(rerank_pool, session, top_k=top_k)

        # 7. Record turn
        session.add_turn(user_message, ranked)

        return {
            "message": self._build_message(ask_attribute, ranked),
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": p["parent_asin"], "score": p["score"]}
                for p in ranked
            ],
            "usage": self.ranker.token_usage,
        }

    def _build_query(self, user_message: str, session: SessionMemory) -> str:
        # Confirmed slot values are highest-confidence — repeat for BM25 weight boost
        slot_text = " ".join(session.slots.values())

        # Last 3 evaluator-revealed messages provide category/constraint context.
        # Exclude generic "no preference" or "not quite right" evaluator filler lines.
        useful_history = [
            t["message"] for t in session.history[-3:]
            if "use your judgment" not in t["message"]
            and "not quite right" not in t["message"]
        ]
        recent_history = " ".join(useful_history)

        return f"{slot_text} {slot_text} {user_message} {recent_history}".strip()

    def _build_message(self, ask_attribute: str | None, ranked: list) -> str:
        if ask_attribute:
            return f"Here are some options. Could you tell me your preference for {ask_attribute}?"
        if ranked:
            return "Here are my top picks based on your request."
        return "I couldn't find a match. Could you describe what you're looking for?"
