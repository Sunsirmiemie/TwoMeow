"""
LLM-based reranker: sends top-N candidates (with titles) to Claude for semantic reranking.

NETWORK DEPENDENCY (MD §VII model policy):
  Requires external Claude API when use_llm_ranker=True.
  Offline fallback: returns retrieval-score order when API unavailable or disabled.
  API key via ANTHROPIC_API_KEY env var — never hardcoded.
"""
from __future__ import annotations

import json
import os


_PROMPT = """\
You are a shopping assistant. Rerank the following products by how likely they match the user's need.

User need (slots confirmed so far): {slots}
Recent conversation: {history}

Products to rerank (index · ASIN · title):
{products}

Return ONLY a JSON array of the ASINs in order from best to worst match.
Example: ["B001XX", "B002YY"]"""


class Ranker:
    def __init__(self, config: dict, title_lookup: dict[str, str] | None = None):
        self.enabled = config.get("use_llm_ranker", False)
        self.top_n = config.get("rerank_top_n", 20)
        self.model = config.get("ranker_model", "claude-haiku-4-5-20251001")
        self._titles = title_lookup or {}
        self._client = None
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def _get_client(self):
        if self._client is None:
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def rerank(self, candidates: list[dict], session, top_k: int = 10) -> list[dict]:
        if not self.enabled or not candidates:
            return candidates[:top_k]

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
            self._total_prompt_tokens += response.usage.input_tokens
            self._total_completion_tokens += response.usage.output_tokens

            reranked_asins: list[str] = json.loads(response.content[0].text)
            asin_to_item = {p["parent_asin"]: p for p in pool}
            reranked = [asin_to_item[a] for a in reranked_asins if a in asin_to_item]
            mentioned = set(reranked_asins)
            reranked += [p for p in pool if p["parent_asin"] not in mentioned]
            return reranked[:top_k]

        except Exception:
            # Offline fallback: retrieval-score order
            return candidates[:top_k]

    @property
    def token_usage(self) -> dict:
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
        }
