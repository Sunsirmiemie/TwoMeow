"""
Dense retriever: sentence-transformers embeddings + numpy cosine similarity.
Runs entirely in-memory (no external vector DB — MD §VII out-of-scope).
First run encodes all products and caches to .embed_cache/; subsequent runs load cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..ranking.features import product_text

_CACHE_DIR = Path(__file__).parent.parent.parent / ".embed_cache"


def _catalog_hash(catalog_path: str) -> str:
    return hashlib.sha1(Path(catalog_path).read_bytes()).hexdigest()[:12]


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
        return _CACHE_DIR / f"{safe_model}_{_catalog_hash(catalog_path)}.npz"

    def _build_or_load(self, catalog_path: str) -> None:
        cache = self._cache_path(catalog_path)
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            self.asins = data["asins"].tolist()
            self.embeddings = data["embeddings"]
            return

        texts: list[str] = []
        with open(catalog_path, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                self.asins.append(p.get("parent_asin", ""))
                texts.append(product_text(p))

        self.embeddings = self.model.encode(
            texts,
            batch_size=512,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, asins=np.array(self.asins), embeddings=self.embeddings)

    def search(self, query: str, top_k: int = 100) -> list[dict]:
        if self.embeddings is None or not query.strip():
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores: np.ndarray = (self.embeddings @ q_emb.T).squeeze()
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [
            {"parent_asin": self.asins[int(i)], "score": float(scores[i])}
            for i in top_idx
        ]
