"""
Hybrid retriever: fuses BM25 + dense results via Reciprocal Rank Fusion.
Falls back to BM25-only when sentence-transformers is not installed.
"""
from __future__ import annotations

from .bm25 import BM25Retriever
from ..ranking.scorer import (
    adaptive_fusion_weights,
    fuse,
    source_confidence,
    BM25_BASE,
    DENSE_BASE,
    RRF_K,
)

_BROWSING_WEIGHTS = ((0.50, 0.50), (0.60, 0.40))


class HybridRetriever:
    def __init__(self, catalog_path: str, config: dict):
        self.bm25 = BM25Retriever(
            catalog_path,
            config.get("field_weights"),
            use_dynamic_attribute_scoring=config.get(
                "use_dynamic_attribute_scoring", False
            ),
            dynamic_field_gain=float(config.get("dynamic_field_gain", 0.65)),
            dynamic_attribute_max_weight=float(
                config.get("dynamic_attribute_max_weight", 0.52)
            ),
        )
        self.rrf_k = config.get("rrf_k", RRF_K)
        self.bm25_base = config.get("bm25_base", BM25_BASE)
        self.dense_base = config.get("dense_base", DENSE_BASE)
        self.browsing_weights = tuple(
            tuple(pair) for pair in config.get("browsing_weights", _BROWSING_WEIGHTS)
        )
        self.use_adaptive_fusion = bool(config.get("use_adaptive_fusion", False))
        self.use_dense_risk_gate = bool(config.get("use_dense_risk_gate", False))
        self.dense_gate_min_bm25 = int(config.get("dense_gate_min_bm25", 20))
        self.dense_gate_max_confidence = float(
            config.get("dense_gate_max_confidence", 0.12)
        )
        self._catalog_path = catalog_path
        self._dense_enabled = bool(config.get("use_dense", True))
        self._dense_model = config.get("dense_model", "all-MiniLM-L6-v2")
        self._dense_batch_size = config.get("dense_batch_size", 512)
        self._dense_max_seq_length = int(config.get("dense_max_seq_length", 256))
        self._dense_device = config.get("dense_device", "auto")
        self._dense_query_prefix = config.get("dense_query_prefix", "")
        self._dense_document_prefix = config.get("dense_document_prefix", "")
        self._field_aware_dense = bool(config.get("use_field_aware_dense", False))
        self.dense = None

    def _get_dense(self):
        if self.dense is not None:
            return self.dense
        if not self._dense_enabled:
            return None
        try:
            from .dense import DenseRetriever
            self.dense = DenseRetriever(
                self._catalog_path,
                self._dense_model,
                self._dense_batch_size,
                use_field_aware=self._field_aware_dense,
                max_seq_length=self._dense_max_seq_length,
                device=self._dense_device,
                query_prefix=self._dense_query_prefix,
                document_prefix=self._dense_document_prefix,
            )
        except (ImportError, OSError, RuntimeError):
            self._dense_enabled = False
        return self.dense

    def _should_use_dense(self, bm25_results: list[dict], slots: dict) -> bool:
        if not self.use_dense_risk_gate:
            return True
        if len(bm25_results) < self.dense_gate_min_bm25:
            return True
        broad_without_category = not slots.get("category") and len(slots) <= 1
        return broad_without_category and (
            source_confidence(bm25_results) <= self.dense_gate_max_confidence
        )

    def retrieve(
        self,
        query: str,
        slots: dict,
        intent: str,
        top_k: int = 100,
        turn: int = 1,
        session=None,
    ) -> list[dict]:
        if not query.strip():
            query = "*"

        bm25_results = self.bm25.search(
            query, slots, intent, top_k=top_k, session=session,
        )

        if self.dense is None and not self._dense_enabled:
            return bm25_results

        # Buying track: evaluator reveals exact constraint text — BM25 is highly precise.
        # Dense adds noise here, so skip fusion for buying.
        if intent == "buying":
            return bm25_results

        if not self._should_use_dense(bm25_results, slots):
            return bm25_results

        dense = self._get_dense()
        if dense is None:
            return bm25_results

        # Browsing: turn-aware weight shift (MD §III adaptive orchestration).
        # Early turns: more dense weight for broad semantic recall.
        # Later turns: shift toward BM25 as exact keywords accumulate.
        if getattr(dense, "field_aware", False):
            dense_results = dense.search(query, top_k=top_k, session=session)
        else:
            dense_results = dense.search(query, top_k=top_k)
        bm25_w, dense_w = self.bm25_base, self.dense_base
        if len(slots) < len(self.browsing_weights):
            bm25_w, dense_w = self.browsing_weights[len(slots)]
        if self.use_adaptive_fusion:
            bm25_w, dense_w = adaptive_fusion_weights(
                bm25_results,
                dense_results,
                session,
                (bm25_w, dense_w),
            )

        return fuse(
            bm25_results,
            dense_results,
            top_k,
            bm25_w=bm25_w,
            dense_w=dense_w,
            rrf_k=self.rrf_k,
        )
