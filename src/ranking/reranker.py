"""Rerank the retrieval top-20 into the final top-10 recommendation list.

The default path is a deterministic, model-free implementation of Maximal
Marginal Relevance (MMR).  It preserves the existing feature relevance score
while avoiding near-duplicate catalogue items in the final list.  The optional
Anthropic path remains supported for backwards compatibility, but is disabled
by the default configuration.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from .profile_prior import apply_profile_boost

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
        # A relevance-first setting: diversity only breaks up very similar
        # products after their relevance has been calculated.
        self.mmr_lambda = float(config.get("mmr_lambda", 0.85))
        self.profile_weight = float(config.get("profile_weight", 0.15))
        self.use_field_aware_slot_coverage = bool(
            config.get("use_field_aware_slot_coverage", True)
        )

    def _get_client(self):
        if self._client is None:
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def rerank(self, candidates: list[dict], session: Any, top_k: int = 10) -> list[dict]:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        if not candidates:
            return []
        # The reranker contract is deliberately local: regardless of the
        # retrieval depth or whether the optional LLM path is active, it sees
        # only the retrieval top-N (20 by default) and returns top-K (10).
        pool = candidates[: self.top_n]
        if not self.enabled:
            if self.use_features:
                return self._rerank_features(pool, session, top_k)
            return pool[:top_k]

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
                return self._rerank_features(pool, session, top_k)
            return pool[:top_k]

    def _tokenize(self, text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 1 and token not in _STOP
        }

    def _effective_mmr_lambda(self) -> float:
        """Cap diversity at the validated risk-aware top-k operating point.

        The configured value is still respected when it is more conservative.
        For larger values, the cap prevents near-duplicate high-score products
        from consuming the final top-10 list.
        """
        return min(self.mmr_lambda, 0.70)

    def _rerank_features(
        self,
        candidates: list[dict],
        session: Any,
        top_k: int = 10,
    ) -> list[dict]:
        """Model-free relevance-first MMR rerank.

        Carbonell & Goldstein (SIGIR 1998) define MMR as a greedy selection of
        the item that maximizes relevance minus its greatest similarity to an
        already selected item.  Here relevance is the existing local feature
        score, and product-product similarity is token-set Jaccard. Confirmed
        slot terms use coverage over title, category, feature, and detail text,
        so explicit user constraints are rewarded consistently. No LLM, learned
        ranker, or network call is used.
        """
        candidates = apply_profile_boost(
            candidates,
            session.preference_tags(),
            self._titles,
            self._categories,
            self._meta,
            self.profile_weight,
        )
        slots = session.slots
        effective_lambda = self._effective_mmr_lambda()
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
        scored: list[tuple[float, dict, set[str]]] = []
        for candidate in candidates:
            asin = candidate["parent_asin"]
            meta = self._meta.get(asin, {})
            cats = self._categories.get(asin, [])
            title_category_tokens = self._tokenize(
                (self._titles.get(asin, "") or "") + " " + " ".join(cats)
            )
            text_tokens = title_category_tokens | self._tokenize(
                str(meta.get("profile_text") or "")
            )
            if self.use_field_aware_slot_coverage:
                slot_score = len(slot_tokens & text_tokens) / (len(slot_tokens) or 1)
            else:
                intersection = len(slot_tokens & title_category_tokens)
                union = len(slot_tokens | title_category_tokens) or 1
                slot_score = intersection / union
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
                + w.get("slot", 2.2) * slot_score
                + w.get("category", 1.8) * category_match
                + w.get("popularity", 0.3) * popularity
                + w.get("price", 0.6) * price_sim
            )
            scored.append((final, candidate, text_tokens))

        # Greedy MMR selection from the fixed retrieval top-20 pool.
        selected: list[tuple[float, dict, set[str]]] = []
        remaining = scored[:]
        while remaining and len(selected) < top_k:
            if not selected:
                best_index = max(range(len(remaining)), key=lambda i: remaining[i][0])
            else:
                def mmr_value(item: tuple[float, dict, set[str]]) -> float:
                    relevance, _, tokens = item
                    max_similarity = max(
                        len(tokens & chosen_tokens) / (len(tokens | chosen_tokens) or 1)
                        for _, _, chosen_tokens in selected
                    )
                    return effective_lambda * relevance - (1.0 - effective_lambda) * max_similarity

                best_index = max(range(len(remaining)), key=lambda i: mmr_value(remaining[i]))
            selected.append(remaining.pop(best_index))

        return [candidate for _, candidate, _ in selected]

    @property
    def token_usage(self) -> dict:
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }
