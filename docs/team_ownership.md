# Team Ownership and Module Assignments

## Module Owners

| Module | Files | Owner | Notes |
|------|------|--------|------|
| Scene routing & state management | `src/agent/router.py` `src/agent/state.py` `src/dialogue/override.py` | TBD | Intent classification, slot extraction, Override reset |
| BM25 retrieval | `src/retrieval/catalog.py` `src/retrieval/bm25.py` | TBD | SQLite FTS5 index, field weights, slot_filter |
| Dense retrieval | `src/retrieval/dense.py` | TBD | sentence-transformers embeddings, caching strategy |
| Hybrid retrieval & fusion | `src/retrieval/hybrid.py` `src/ranking/scorer.py` | TBD | RRF fusion, Buying/Browsing track weights |
| Clarification strategy | `src/dialogue/entropy.py` `src/dialogue/question_policy.py` `src/dialogue/early_stop.py` | TBD | Dynamic entropy scoring, Early Stop τ=0.3 |
| Ranker | `src/ranking/reranker.py` `src/ranking/profile_prior.py` | TBD | Local field-aware reranking, user profile prior, risk-aware MMR; LLM path optional only |
| Evaluation & analysis | `src/evaluation/` `scripts/` | TBD | Ablation experiments, failure analysis, script maintenance |
| Orchestration & interface | `src/agent/orchestrator.py` `starter/agent.py` | TBD | Main flow, official interface compatibility |

## Code Standards (from directory structure guidelines)

- Single business Python file should stay within **200–250 lines**
- One file carries one clear responsibility
- No "super agent.py" files exceeding 500 lines
- No circular imports between modules
- All key hyperparameters go into `src/config/default.yaml`
- All public functions must have type hints
- Core strategies have a short docstring
- No experimental logic scattered in business code (isolate in `scripts/`)
- Official evaluator files treated as external dependency, do not refactor

## Tasks to Claim

The following are follow-up optimizations; the user profile prior is already wired into the current rerank, it is no longer a stub:

| Task | File | Priority |
|------|------|--------|
| slot_filter extension (material/color hard filtering) | `src/retrieval/bm25.py` | High |
| Profile prior robustness validation on private set | `src/ranking/profile_prior.py` | Medium |
| Offline cross-encoder reranking | `src/ranking/reranker.py` | Medium (finals) |
| τ hyperparameter sweep | `src/dialogue/early_stop.py` | Low |

## Pre-submission Checklist

- [ ] `starter/agent.py` correctly points to `src.agent.orchestrator.Agent`
- [ ] `.env.example` exists, `.env` not committed
- [ ] `data/catalog.jsonl` in `.gitignore` (large file)
- [ ] All tests pass: `python tests/test_*.py`
- [ ] `results_rerank_risk_aware_mmr_profile.json` records the current rerank scheme scores
- [ ] `docs/` documentation is in sync with the actual code
