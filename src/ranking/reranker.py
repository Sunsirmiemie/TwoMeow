"""
LLM-based listwise reranker via Anthropic API.

NETWORK DEPENDENCY: requires external Claude API when use_llm_ranker=True.
Offline fallback: returns retrieval-score order when API unavailable or disabled.
API key via ANTHROPIC_API_KEY env var — never hardcoded.
"""
from __future__ import annotations

import json
import os

from ..observability import candidate_snapshot

_PROMPT = """\
You are a shopping assistant. Rerank the following products by how likely they match the user's need.

User need (slots confirmed so far): {slots}
Recent conversation: {history}

Products to rerank (index · ASIN · title):
{products}

Return ONLY a JSON array of the ASINs in order from best to worst match.
Example: ["B001XX", "B002YY"]"""


class Ranker:
    def __init__(
        self,
        config: dict,
        title_lookup: dict[str, str] | None = None,
        client=None,
    ):
        self.enabled   = config.get("use_llm_ranker", False)
        self.top_n     = config.get("rerank_top_n", 20)
        self.model     = config.get("ranker_model", "claude-haiku-4-5-20251001")
        self._titles   = title_lookup or {}
        self._client   = client
        self._prompt_tokens     = 0
        self._completion_tokens = 0

    def _get_client(self):
        if self._client is None:
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def rerank(self, candidates: list[dict], session, top_k: int = 10) -> list[dict]:
        ranked, _trace = self._rerank(
            candidates,
            session,
            top_k=top_k,
            with_trace=False,
        )
        return ranked

    def rerank_with_trace(
        self,
        candidates: list[dict],
        session,
        top_k: int = 10,
    ) -> tuple[list[dict], dict]:
        """Return normal ranking output plus API-boundary attempt diagnostics."""
        ranked, trace = self._rerank(
            candidates,
            session,
            top_k=top_k,
            with_trace=True,
        )
        return ranked, trace

    def _rerank(
        self,
        candidates: list[dict],
        session,
        top_k: int,
        with_trace: bool,
    ) -> tuple[list[dict], dict | None]:
        self._prompt_tokens = 0
        self._completion_tokens = 0

        def finish(
            ranked: list[dict],
            status: str,
            attempted: bool,
            api_pool: list[dict] | None = None,
            error_type: str | None = None,
        ) -> tuple[list[dict], dict | None]:
            if not with_trace:
                return ranked, None
            trace = {
                "enabled": self.enabled,
                "model": self.model,
                "top_n": self.top_n,
                "requested_top_k": top_k,
                "status": status,
                "attempted": attempted,
                "attempt_status": status if attempted else "not_attempted",
                "input": {"candidates": candidate_snapshot(candidates)},
                "api_pool": {
                    "configured_limit": self.top_n,
                    "candidates": candidate_snapshot(
                        candidates[: self.top_n] if api_pool is None else api_pool
                    ),
                },
                "output": {"candidates": candidate_snapshot(ranked)},
                "usage": self.token_usage,
            }
            if error_type is not None:
                trace["error_type"] = error_type
            return ranked, trace

        if not self.enabled:
            return finish(candidates[:top_k], "disabled", False)
        if not candidates:
            return finish([], "empty", False)

        pool = candidates[: self.top_n]
        product_lines = "\n".join(
            f"{i+1}. {p['parent_asin']} · {self._titles.get(p['parent_asin'], '(no title)')[:80]}"
            for i, p in enumerate(pool)
        )
        history_str = " | ".join(
            t["message"] for t in session.history[-3:]
            if "use your judgment" not in t["message"]
        )
        prompt = _PROMPT.format(
            slots=json.dumps(session.slots, ensure_ascii=False),
            history=history_str or "(none)",
            products=product_lines,
        )

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            self._prompt_tokens = response.usage.input_tokens
            self._completion_tokens = response.usage.output_tokens

            reranked_asins: list[str] = json.loads(response.content[0].text)
            asin_to_item = {p["parent_asin"]: p for p in pool}
            reranked = [asin_to_item[a] for a in reranked_asins if a in asin_to_item]
            mentioned = set(reranked_asins)
            reranked += [p for p in pool if p["parent_asin"] not in mentioned]
            return finish(reranked[:top_k], "api_success", True, pool)

        except Exception as exc:
            return finish(
                candidates[:top_k],
                "fallback",
                True,
                pool,
                error_type=type(exc).__name__,
            )

    @property
    def token_usage(self) -> dict:
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }
