"""
Dense retriever: sentence-transformers embeddings + numpy cosine similarity.
Runs entirely in-memory (no external vector DB — MD §VII out-of-scope).
First run encodes all products and caches to .embed_cache/; subsequent runs load cache.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..ranking.features import (
    product_attribute_text,
    product_identity_text,
    product_text,
)

_CACHE_DIR = Path(__file__).parent.parent.parent / ".embed_cache"
_FIELD_CACHE_VERSION = "field-v2"


def _catalog_hash(catalog_path: str) -> str:
    return hashlib.sha1(Path(catalog_path).read_bytes()).hexdigest()[:12]


class DenseRetriever:
    def __init__(
        self,
        catalog_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 512,
        use_field_aware: bool = False,
        max_seq_length: int = 256,
        device: str = "auto",
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        from sentence_transformers import SentenceTransformer
        model_options = {} if device == "auto" else {"device": device}
        self.model = SentenceTransformer(model_name, **model_options)
        self.model.max_seq_length = max_seq_length
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.field_aware = use_field_aware
        self.requested_device = device
        self.device = str(getattr(self.model, "device", "unknown"))
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.asins: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.identity_embeddings: np.ndarray | None = None
        self.attribute_embeddings: np.ndarray | None = None
        self._build_or_load(catalog_path)

    def _cache_path(self, catalog_path: str) -> Path:
        safe_model = self.model_name.replace("/", "_")
        configured = os.environ.get("TWOMEOW_DENSE_CACHE_DIR")
        cache_dir = Path(configured) if configured else _CACHE_DIR
        version = _FIELD_CACHE_VERSION if self.field_aware else "concat-v1"
        if self.max_seq_length != 256:
            version = f"{version}-seq-{self.max_seq_length}"
        if self.query_prefix or self.document_prefix:
            prefix_key = hashlib.sha1(
                f"{self.query_prefix}\0{self.document_prefix}".encode()
            ).hexdigest()[:8]
            version = f"{version}-prompt-{prefix_key}"
        return cache_dir / f"{safe_model}_{version}_{_catalog_hash(catalog_path)}.npz"

    def _build_or_load(self, catalog_path: str) -> None:
        cache = self._cache_path(catalog_path)
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            self.asins = data["asins"].tolist()
            if self.field_aware:
                self.identity_embeddings = data["identity_embeddings"]
                self.attribute_embeddings = data["attribute_embeddings"]
            else:
                self.embeddings = data["embeddings"]
            return

        texts: list[str] = []
        identity_texts: list[str] = []
        attribute_texts: list[str] = []
        with open(catalog_path, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                self.asins.append(p.get("parent_asin", ""))
                if self.field_aware:
                    identity_texts.append(
                        f"{self.document_prefix}{product_identity_text(p)}"
                    )
                    attribute_texts.append(
                        f"{self.document_prefix}{product_attribute_text(p)}"
                    )
                else:
                    texts.append(f"{self.document_prefix}{product_text(p)}")

        encode_options = {
            "batch_size": self.batch_size,
            "show_progress_bar": True,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.field_aware:
            self.identity_embeddings = self.model.encode(identity_texts, **encode_options)
            self.attribute_embeddings = self.model.encode(attribute_texts, **encode_options)
        else:
            self.embeddings = self.model.encode(texts, **encode_options)
        cache.parent.mkdir(parents=True, exist_ok=True)
        if self.field_aware:
            np.savez_compressed(
                cache,
                asins=np.array(self.asins),
                identity_embeddings=self.identity_embeddings,
                attribute_embeddings=self.attribute_embeddings,
            )
        else:
            np.savez_compressed(cache, asins=np.array(self.asins), embeddings=self.embeddings)

    def _field_weights(self, session) -> tuple[float, float]:
        if session is None:
            return 0.50, 0.50
        specific = sum(
            session.slot_confidence.get(key, 0.75)
            for key in session.slots if key not in {"category", "brand", "budget"}
        )
        attribute_weight = min(0.78, 0.35 + 0.10 * specific)
        if "category" not in session.slots:
            attribute_weight = max(attribute_weight, 0.50)
        return 1.0 - attribute_weight, attribute_weight

    def search(self, query: str, top_k: int = 100, session=None) -> list[dict]:
        if not query.strip():
            return []
        raw_query = query
        if self.field_aware:
            if self.identity_embeddings is None or self.attribute_embeddings is None:
                return []
            category_content = " ".join((
                str(getattr(session, "slots", {}).get("category", "")),
                raw_query,
            )).strip()
            attribute_values = " ".join(
                str(value) for key, value in getattr(session, "slots", {}).items()
                if key not in {"category", "brand"}
            )
            attribute_content = f"{attribute_values} {raw_query}".strip()
            # Retrieval-tuned encoders such as E5/BGE require the instruction
            # at the beginning of every complete query, not in the middle after
            # session-derived category or attribute evidence.
            category_query = f"{self.query_prefix}{category_content}"
            attribute_query = f"{self.query_prefix}{attribute_content}"
            query_embeddings = self.model.encode(
                [category_query, attribute_query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            identity_weight, attribute_weight = self._field_weights(session)
            scores = (
                identity_weight * (self.identity_embeddings @ query_embeddings[0])
                + attribute_weight * (self.attribute_embeddings @ query_embeddings[1])
            )
        else:
            if self.embeddings is None:
                return []
            query = f"{self.query_prefix}{raw_query}"
            q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
            scores = (self.embeddings @ q_emb.T).squeeze()
        limit = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -limit)[-limit:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [
            {"parent_asin": self.asins[int(i)], "score": float(scores[i])}
            for i in top_idx
        ]
