"""
Dense retriever: sentence-transformers embeddings + numpy cosine similarity.

Runs entirely in-memory (no external vector DB — MD §VII out-of-scope).
First run encodes all 50K products and caches to disk; subsequent runs load cache.

Text strategy: title + categories + first 300 chars of features.
Features are the source of evaluator intent-card hard constraints (PDF insight #3),
so including them gives semantic overlap with the actual query constraints.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


_CACHE_DIR = Path(__file__).parent.parent.parent / ".embed_cache"


def _catalog_hash(catalog_path: str) -> str:
    h = hashlib.sha1(Path(catalog_path).read_bytes()).hexdigest()[:12]
    return h


def _product_text(p: dict) -> str:
    title = str(p.get("title") or "")
    cats = p.get("categories") or []
    if isinstance(cats, list):
        cats = " ".join(str(c) for c in cats)
    features = p.get("features") or []
    if isinstance(features, list):
        features = " ".join(str(f) for f in features)
    elif not isinstance(features, str):
        features = str(features)
    # Title is most important; features contain evaluator-card keywords
    return f"{title}. {cats}. {features[:300]}".strip()


class DenseRetriever:
    def __init__(self, catalog_path: str, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.asins: list[str] = []
        self.embeddings: np.ndarray | None = None
        self._build_or_load(catalog_path)

    def _cache_path(self, catalog_path: str) -> Path:
        safe_model = self.model_name.replace("/", "_")
        h = _catalog_hash(catalog_path)
        return _CACHE_DIR / f"{safe_model}_{h}.npz"

    def _build_or_load(self, catalog_path: str) -> None:
        cache = self._cache_path(catalog_path)
        if cache.exists():
            print(f"Loading dense embeddings from cache: {cache.name}")
            data = np.load(cache, allow_pickle=True)
            self.asins = data["asins"].tolist()
            self.embeddings = data["embeddings"]
            print(f"Loaded {len(self.asins)} embeddings.")
            return

        print(f"Building dense index for {catalog_path} …")
        texts: list[str] = []
        with open(catalog_path, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                self.asins.append(p.get("parent_asin", ""))
                texts.append(_product_text(p))

        self.embeddings = self.model.encode(
            texts,
            batch_size=512,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            asins=np.array(self.asins),
            embeddings=self.embeddings,
        )
        print(f"Saved embeddings to {cache}")

    def search(self, query: str, top_k: int = 100) -> list[dict]:
        if self.embeddings is None or not query.strip():
            return []
        q_emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        # Cosine similarity via dot product (embeddings are L2-normalized)
        scores: np.ndarray = (self.embeddings @ q_emb.T).squeeze()
        # Partial sort: O(n log k) instead of O(n log n)
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [
            {"parent_asin": self.asins[int(i)], "score": float(scores[i])}
            for i in top_idx
        ]
