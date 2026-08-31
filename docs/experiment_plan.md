# Experiment Plan and Results Log

> Historical note: the following "completed experiments" are preserved for tracing the entropy, early-stop, and Override system improvements.
> **Current version (after three optimizations, 2026-08-30)** public set results: HitRate@10=0.955, MRR=0.706129, MTTC=2.865, TechnicalScore=**0.852039** (`docs/THREE_OPTIMIZATIONS_20260829.md`).
> **Generalization warning**: The 400-ASIN blind test with zero overlap with the public set (seed 20260830) scored TechnicalScore=0.750553, below the original BM25's 0.760657; the three current optimizations should still be treated as experimental.

## Completed Experiments (Ablation Analysis)

### Experiment Setup

- Dataset: `data/public_set.jsonl`, 200 sessions (Buying 80 / Browsing 80 / Override 30 / Boundary 10)
- Product catalog: 50,000 clothing items
- No LLM reranking (offline mode, `use_llm_ranker=False`)
- Run script: `scripts/run_ablation_table.py`

### Results Summary

| Config | Buying HR | Browsing HR | Override HR | Boundary HR | Overall HR | MRR | MTTC | TechScore |
|------|-----------|-------------|-------------|-------------|------------|-----|------|-----------|
| Baseline (BM25 only) | 0.663 | 0.725 | 0.767 | 0.900 | 0.715 | 0.464 | 6.36 | 0.590 |
| + Hybrid (Dense) | 0.663 | 0.725 | 0.767 | 0.900 | 0.715 | 0.427 | 6.38 | 0.578 |
| + Entropy | 0.738 | 0.838 | 0.833 | 0.900 | 0.800 | 0.459 | 5.08 | 0.656 |
| + Early Stop (τ=0.3) | 0.888 | 0.900 | 0.833 | 0.900 | 0.885 | 0.536 | 3.44 | 0.755 |
| + Override Detection | 0.888 | 0.900 | 0.867 | 0.900 | 0.890 | 0.555 | 3.40 | **0.763** |
| Final (no Early Stop) | 0.738 | 0.838 | 0.767 | 0.900 | 0.790 | 0.473 | 5.12 | 0.655 |

**Second ablation round (three optimizations, 2026-08-30; see `docs/THREE_OPTIMIZATIONS_20260829.md`)**

| Config | Overall HR | MRR | MTTC | TechScore |
|------|------------|-----|------|-----------|
| Updated repo original retest (Rerank + MMR baseline) | 0.905 | 0.706437 | 3.535 | 0.813731 |
| + Negation-aware response purification | 0.880 | 0.664893 | 3.660 | 0.786268 |
| + Dynamic attribute weights (strict Override, historical experiment) | 0.935 | 0.688629 | 2.995 | 0.834189 |
| Final: selective Override + Dense risk gate | **0.955** | 0.706129 | **2.865** | **0.852039** |
| Final: selective Override + forced field Dense | 0.920 | 0.546383 | 3.165 | 0.780615 |

### Key Findings

1. **Dense alone is harmful**: Without better question selection, Dense introduces noise reducing MRR (0.464→0.427).

2. **Entropy is the largest single-step gain**: +0.078 TechScore. Dynamic coverage×entropy makes each turn ask the most informative attribute, MTTC from 6.36→5.08.

3. **Early Stop is the largest surprise gain**: +0.099 TechScore, MTTC from 5.08→3.44. Once the candidate pool converges, asking specific attributes yields empty replies; asking `other` (wildcard) directly gets the evaluator to proactively disclose constraints, eliminating wasted turns.

4. **Override Detection precisely fixes Override scenario**: Override HR 0.833→0.867, MRR 0.601→0.722, no impact on other scenarios.

5. **Current "Final" config (no Early Stop) is suboptimal**: Early Stop is now default `True`, optimal TechScore=0.763.

---

## Experiments to Explore

### Short-term (qualifying stage)

#### E1: τ Hyperparameter Sweep
Early Stop threshold is currently fixed at τ=0.3 (derived from PDF theoretical analysis).

| τ | Expected impact |
|----|---------|
| 0.1 | More aggressive stopping, lower MTTC, may lose HitRate |
| 0.3 | Current optimal (experimentally validated) |
| 0.5 | More conservative, more questions, slightly higher MTTC but HitRate may be more stable |

Run: modify `TAU` in `src/dialogue/early_stop.py`, execute `scripts/run_public_eval.py`.

#### E2: LLM Reranker (not adopted)
> Current approach does not use LLM reranking. Default `use_llm_ranker=False`, 200-session public set result is zero tokens, pure local run (TechScore=0.852164). LLM path retained as optional compatible branch, but prompt information content is limited (only title), expected benefit is marginal, not evaluating for now.

#### E3: slot_filter Extension
> **Partially complete**: Dynamic attribute weights already apply compatibility reranking for material/color etc. at the top-300→100 stage (`src/ranking/dynamic_attributes.py`); BM25 `_apply_slot_filters()` still only hard-filters budget. Adding material/color hard filtering can further improve Buying scenario HitRate.

File: `src/retrieval/bm25.py:_apply_slot_filters()`

#### E4: User Profile Integration
**Current implementation**: `profile_prior.py` reads `preference_tags` at rerank stage and applies bounded local bonus scoring; this profile prior is retained in the current approach.

### Medium-term (finals stage)

#### E5: Offline Reranker (when cross-network is disabled)
Alternative to LLM reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~25MB, CPU inference).

Requires modification: add cross-encoder fallback branch in `src/ranking/reranker.py`.

#### E6: BM25 slot_filter Extended to All Attributes
Buying scenario HR=0.888, still ~11% miss. Partial cause is material/color slots not hard-filtered.

#### E7: Query Rewriting
> **Completed**: Post-Override selective overwrite implemented via `src/dialogue/purification.py` + `override_carryover_confidence=0.35`; old slots demoted to low-confidence weak evidence, old raw text not re-added to queries.

---

## Experiment Management Standards

- All new experiment results saved as separate `results_*.json`, not overwriting original `results.json`; current rerank result is `results_rerank_risk_aware_mmr_profile.json`.
- Create a separate yaml under `src/config/experiments/` before modifying hyperparameters
- Each experiment appends a row to the results summary table in this file
