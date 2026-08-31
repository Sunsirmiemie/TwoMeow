# System Architecture

## Overview

TwoMeow is a pure in-memory, no-external-database multi-turn conversational e-commerce search Agent, implementing the official `reset() / respond()` interface.

```
User message (turn N)
      │
      ▼
┌─────────────────────────────────────────────────┐
│  src/agent/orchestrator.py  (Agent.respond)      │
│                                                 │
│  ① IntentRouter   Scene classification + Override detection │
│  ② SlotTracker    Structured slot extraction             │
│  ③ HybridRetriever Retrieval (BM25 + Dense + RRF)       │
│  ④ Clarifier      Dynamic entropy attribute selection + Early Stop │
│  ⑤ Ranker         top-20 → field-aware local reranking → top-10 │
│  ⑥ Return result  recommendations + ask_attr            │
└─────────────────────────────────────────────────┘
      │
      ▼
Evaluator determines next turn's user message
```

## Module Dependency Graph

```
dialogue/attribute_stats.py   ← Base layer (no internal deps)
agent/state.py                ← Base layer

dialogue/entropy.py           ← attribute_stats
dialogue/override.py          ← no internal deps
dialogue/question_policy.py   ← entropy + attribute_stats
dialogue/early_stop.py        ← entropy

agent/router.py               ← dialogue/override
agent/response_builder.py     ← no internal deps

ranking/scorer.py             ← no internal deps
ranking/features.py           ← no internal deps
ranking/profile_prior.py       ← no internal deps
ranking/reranker.py           ← profile_prior

retrieval/catalog.py          ← dialogue/attribute_stats
retrieval/bm25.py             ← retrieval/catalog
retrieval/dense.py            ← ranking/features
retrieval/hybrid.py           ← retrieval/bm25 + retrieval/dense + ranking/scorer

agent/orchestrator.py         ← all upper layers
```

Dependency direction is one-way: `dialogue/ranking → retrieval → agent`, no circular imports.

## Layer Responsibilities

### Layer 1: Dialogue State (`src/agent/state.py`)

**SessionMemory** — single-session memory:
- `slots: dict[str, str]` — confirmed constraints (material/color/size/style/use_case/feature/budget)
- `scenario_type` — buying / browsing / intent_override / boundary
- `asked_attributes` — already-asked attributes (prevents repetition)
- `history` — recent message history (for query construction)

**SlotTracker** — parsing rules (priority from high to low):
1. `"what matters is: X; Y"` → structured multi-slot parsing
2. `"a key requirement is: X"` → single-slot buyer constraint
3. `"what I need is: X"` → new constraint post-Override
4. Free text → regex fallback (does not override existing slots)

Slot classifier mirrors the evaluator's `classify_constraint()` exactly, ensuring consistency.

### Layer 2: Scene Routing (`src/agent/router.py` + `src/dialogue/override.py`)

| Signal | Pattern | Action |
|------|------|------|
| Intent Override | `"ignore my earlier preference"` | Clear slots + asked_attributes, switch to buying track |
| Boundary | `"please use your judgment"` | Set flag, skip this attribute, continue |
| Buying | Turn 0 + `"a key requirement is:"` | scenario_type = buying |
| Browsing | Turn 0 + other | scenario_type = browsing |

### Layer 3: Retrieval (`src/retrieval/`)

**BM25 Retrieval** (SQLite FTS5, fully in-memory)

Field weights:
```
title(6.0) > categories(4.0) > features(2.5) = description(2.5) > store(1.5) > details(1.0)
```

Query construction: `slot_text×2 + current message + last 3 turns of history` (double slot text amplifies BM25 exact-match weight)

Buying-specific: appends `price ≤ budget` hard filter.

**Dense Retrieval** (sentence-transformers all-MiniLM-L6-v2)

Field-separated encoding strategy (field-aware Dense):
- Identity vector: `title + categories + store`
- Attribute vector: `features + details + description`

Query is likewise split into category/identity query and attribute query; the two similarity scores are weighted and fused at runtime without concatenating vectors. The more specific attributes are known, the higher the attribute similarity weight; when the category is clear, identity similarity weight is higher.

50K products are cached as `.embed_cache/*.npz` after first encoding and loaded directly thereafter.

**Risk gate**: skips Dense when BM25 candidates are sufficient (≥20 with stable category); Buying track always skips Dense. Automatically falls back to BM25 if Dense is unavailable.

cosine similarity via dot product (equivalent to L2-normalized dot product) + partial sort O(n log k).

**RRF Fusion** (Browsing track only, after Dense risk gate passes)

| Confirmed slots | BM25 weight | Dense weight | Rationale |
|-----------|-----------|-----------|------|
| 0 | 50% | 50% | No constraints, semantic recall as fallback |
| 1 | 60% | 40% | Few constraints, still need semantic |
| ≥2 | 75% | 25% | Many constraints, exact match preferred |

`score = w_bm25 × 1/(60+rank_bm25) + w_dense × 1/(60+rank_dense)`

Buying track skips Dense, uses BM25-only directly (evaluator provides precise constraint text each turn; Dense introduces noise here).

### Layer 4: Question Decision (`src/dialogue/`)

**Information Entropy Scoring**

For each remaining attribute in the candidate pool:
```
score(attr) = coverage(attr) × entropy(attr)

coverage = products with this attribute value / total candidate pool
entropy  = normalized information entropy (blended with global entropy when pool < 10)
```

Global entropy sourced from: PDF statistics over 50,000 products (use_case=0.87 > budget=0.79 > color=0.77 > …)

**Early Stop (τ=0.3)**

```python
if max(score(attr) for attr in remaining) < 0.3:
    ask "other"  # wildcard, evaluator returns any remaining constraint
else:
    ask argmax(score)  # attribute with highest information gain
```

When the candidate pool has converged (attribute distributions become uniform), continuing to ask specific attributes only yields "I don't have an additional preference" (empty turns). Early Stop reduces MTTC from 5.1 to 3.4, and HitRate from 0.800 to 0.890.

### Layer 5: Ranking (`src/ranking/`)

**Candidate pool truncation**: if slots < 2 and candidates ≥ 50 → take top-20 (prevents guessing from an overly fuzzy pool when few slots are confirmed)

**Field-aware local reranking** (default):
- Fixes retrieval top-20 as input, outputs top-10;
- Relevance score composed of BM25/RRF scores, category, price, popularity, and confirmed slot exact coverage rate;
- Coverage check verifies whether a product's `title/categories/features/details` satisfies confirmed constraints;
- No model training, no network or LLM calls. LLM path retained as optional compatible branch, not the default evaluation approach.

## Hyperparameter Summary

All centralized in `src/config/default.yaml`; no magic numbers in code.

| Parameter | Value | Description |
|------|-----|------|
| RRF K | 60 | Standard RRF constant |
| BM25 base weight | 0.75 | Browsing ≥2 slots |
| Dense base weight | 0.25 | Same |
| entropy τ | 0.3 | Early Stop threshold |
| min pool for dynamic | 10 | Below this, blend with global entropy |
| rerank top_n | 20 | Local reranking candidate count |
| use_field_aware_slot_coverage | true | Confirmed slot field coverage toggle |
| truncated pool size | 20 | Truncation when few slots |

## Historical System Ablation Results (public set 200 sessions, no LLM reranking)

| System | Buying | Browsing | Override | Boundary | Overall | MRR | MTTC | TechScore |
|------|--------|----------|----------|----------|---------|-----|------|-----------|
| Official Baseline | - | - | - | - | 0.125 | 0.068 | 9.81 | 0.107 |
| BM25-only | 0.663 | 0.725 | 0.767 | 0.900 | 0.715 | 0.464 | 6.36 | 0.590 |
| + Dense (Hybrid) | 0.663 | 0.725 | 0.767 | 0.900 | 0.715 | 0.427 | 6.38 | 0.578 |
| + Entropy | 0.738 | 0.838 | 0.833 | 0.900 | 0.800 | 0.459 | 5.08 | 0.656 |
| + Early Stop | 0.888 | 0.900 | 0.833 | 0.900 | 0.885 | 0.536 | 3.44 | 0.755 |
| + Override | 0.888 | 0.900 | 0.867 | 0.900 | 0.890 | 0.555 | 3.40 | **0.763** |

The above table documents historical dialogue strategy gains (up to the pre-rerank stage).

**Current version (after three optimizations, 2026-08-30) public set results**: HitRate@10=0.955, MRR=0.706129, MTTC=2.865, TechnicalScore=**0.852039**. See `docs/THREE_OPTIMIZATIONS_20260829.md`.

**Generalization warning**: In the second 400-ASIN blind test with zero overlap with the public set (seed 20260830), TechnicalScore=0.750553, below the original BM25's 0.760657. The public set gains cannot be considered proven generalization; the three current optimizations should still be treated as experimental.
