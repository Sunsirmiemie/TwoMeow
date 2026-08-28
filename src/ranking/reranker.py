"""
LLM-based listwise reranker via Anthropic API.

NETWORK DEPENDENCY: requires external Claude API when use_llm_ranker=True.
Offline fallback: returns retrieval-score order when API unavailable or disabled.
API key via ANTHROPIC_API_KEY env var — never hardcoded.
"""
from __future__ import annotations

import json
import math
import os
import re

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "for", "color", "size", "style", "material", "feature", "use", "case",
    "budget", "brand",
}

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
        categories: dict[str, list[str]] | None = None,
        meta: dict[str, dict] | None = None,
    ):
        self.enabled   = config.get("use_llm_ranker", False)
        self.top_n     = config.get("rerank_top_n", 20)
        self.model     = config.get("ranker_model", "claude-haiku-4-5-20251001")
        self._titles   = title_lookup or {}
        self._categories = categories or {}
        self._meta     = meta or {}
        self._client   = client
        self._prompt_tokens     = 0
        self._completion_tokens = 0
        self.use_features = config.get("use_features", True)
        self.feature_weights = dict(config.get("feature_weights") or {
            "base": 1.0, "slot": 0.5, "category": 0.8,
            "popularity": 0.1, "price": 0.3,
        })

    def _get_client(self):
        if self._client is None:
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def rerank(self, candidates: list[dict], session, top_k: int = 10) -> list[dict]:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        if not candidates:
            return []
        if not self.enabled:
            if self.use_features:
                if session.slots:
                    return self._rerank_features(candidates, session, top_k)
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
            self._prompt_tokens = response.usage.input_tokens
            self._completion_tokens = response.usage.output_tokens

            reranked_asins: list[str] = json.loads(response.content[0].text)
            asin_to_item = {p["parent_asin"]: p for p in pool}
            reranked = [asin_to_item[a] for a in reranked_asins if a in asin_to_item]
            mentioned = set(reranked_asins)
            reranked += [p for p in pool if p["parent_asin"] not in mentioned]
            return reranked[:top_k]

        except Exception:
            if self.use_features:
                if session.slots:
                    return self._rerank_features(candidates, session, top_k)
            return candidates[:top_k]

    def _tokenize(self, text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 1 and token not in _STOP
        }

    def _rerank_features(
        self,
        candidates: list[dict],
        session,
        top_k: int = 10,
    ) -> list[dict]:
        """Model-free feature rerank: base score + slot overlap + category match
        + popularity prior + price proximity. No training, no model weights."""
        slots = session.slots
        slot_tokens = self._tokenize(" ".join(slots.values()))
        category = slots.get("category")
        anchor = None
        if category:
            parts = [part.strip().lower() for part in str(category).split("/") if part.strip()]
            anchor = parts[0] if parts else None
        try:
            budget = float(slots.get("budget"))
        except (TypeError, ValueError):
            budget = None

        base_max = max((float(c.get("score") or 0.0) for c in candidates), default=0.0) or 1.0
        pop_max = max(
            (
                math.log1p(int(self._meta.get(c["parent_asin"], {}).get("rating_number") or 0))
                for c in candidates
            ),
            default=0.0,
        ) or 1.0

        w = self.feature_weights
        scored: list[tuple[float, dict]] = []
        for candidate in candidates:
            asin = candidate["parent_asin"]
            meta = self._meta.get(asin, {})
            cats = self._categories.get(asin, [])
            text_tokens = self._tokenize(
                (self._titles.get(asin, "") or "") + " " + " ".join(cats)
            )
            intersection = len(slot_tokens & text_tokens)
            union = len(slot_tokens | text_tokens) or 1
            jaccard = intersection / union
            category_match = 1.0 if (
                anchor and any(anchor in str(value).lower() for value in cats)
            ) else 0.0
            popularity = math.log1p(int(meta.get("rating_number") or 0)) / pop_max
            price = meta.get("price")
            price_sim = 0.0
            if budget is not None and price is not None:
                price_sim = max(0.0, 1.0 - abs(price - budget) / max(budget, 1.0))
            base_norm = float(candidate.get("score") or 0.0) / base_max
            final = (
                w.get("base", 1.0) * base_norm
                + w.get("slot", 2.2) * jaccard
                + w.get("category", 1.8) * category_match
                + w.get("popularity", 0.3) * popularity
                + w.get("price", 0.6) * price_sim
            )
            scored.append((final, candidate))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in scored[:top_k]]

    @property
    def token_usage(self) -> dict:
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }
