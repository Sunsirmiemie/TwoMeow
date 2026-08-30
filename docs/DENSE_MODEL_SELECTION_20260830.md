# Dense model selection — 2026-08-30

## Decision

Keep `all-MiniLM-L6-v2` as the submitted default. Do not switch the production path to E5, BGE, or MPNet, and do not force Dense globally.

## Controlled MPS comparison

All rows used Apple MPS (`mps:0`), field-aware identity/attribute embeddings, `max_seq_length=256`, fixed random environment, BM25-only Buying queries, and forced Dense for the remaining scenarios. Model weights were selected on the 400-sample development validation set; the 200-sample public set was used only for confirmation.

| Encoder | Dimensions | Best tested Dense weight | Validation score | Public score |
|---|---:|---:|---:|---:|
| MiniLM | 384 | 0.05 | 0.679823 | 0.771714 |
| BGE-small-v1.5 | 384 | 0.10 | 0.683344 | 0.773181 |
| MPNet-base-v2 | 768 | 0.10 | 0.681397 | 0.767663 |
| E5-small-v2 | 384 | 0.30 | **0.685906** | **0.774071** |

E5 exceeds MiniLM by only `0.006083` on development validation and `0.002357` on public confirmation. MPNet's 768 dimensions are operationally viable with prebuilt indexes, but did not improve ranking quality.

The current MiniLM configuration with the Dense risk gate scores `0.767773` on development validation and `0.852039` on public. The much larger difference comes from avoiding the forced RRF path, not from the encoder choice.

## Why the model is not changed

1. The encoder-to-encoder differences are too small to justify larger caches, additional input conventions, or a model migration.
2. An empty-Dense control that still passes BM25 through RRF scores only `0.677106`; RRF currently replaces the raw BM25 score consumed by the downstream reranker.
3. Until fusion preserves `bm25_raw_score` and exposes semantic similarity as a separate bounded feature, changing the encoder would optimize a secondary effect.
4. Keeping MiniLM minimizes submission size and latency while preserving the existing offline fallback behavior.

This branch adds no model weights, Hugging Face cache files, MPNet/E5 indexes, or local evaluation outputs. The MiniLM `.npz` cache already tracked by the repository baseline is left unchanged.
