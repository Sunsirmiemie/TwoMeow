# Reranker Update: Profile Prior + Field-Aware + Risk-Constrained MMR

## Objective

Perform local reranking of candidate products from the retrieval stage, outputting the final top-10 products, without using large models, without modifying retrieval, and without modifying the official evaluator. The goal is to reduce redundant near-duplicate products occupying the top-10 while preserving relevance signals.

## Current Approach

Reranking is in `src/ranking/reranker.py`, processes only existing candidates without filtering or adding new products. The flow is:

1. **Profile prior**: Extracts preference words from the session's `preference_tags`, computes overlap with product title, category, and profile text token sets; this score is only a bounded small bonus, does not exclude any candidate.
2. **Field-aware constraint coverage**: Matches confirmed slot words against product title, category, features, and details text. Explicitly expressed colors, materials, categories etc. get coverage scores whenever they appear in any field, avoiding missing valid products by only checking the title.
3. **Relevance feature fusion**: Uses raw BM25/RRF score as primary, also fusing slot coverage, category hit, rating-count popularity, and budget price proximity. All weights configured uniformly in `src/config/default.yaml` under `ranking.feature_weights`.
4. **Risk-constrained MMR for top-10 selection**: First computes each candidate's score based on the above relevance, then iteratively selects products that are "highly relevant and not redundant with already-selected products." Inter-product similarity uses token-set Jaccard; effective `mmr_lambda` is capped at 0.70 to prevent near-duplicate products from filling all top-10 positions.

The entire path is local rules and term computation: no network calls, no model training, zero token usage. LLM reranking interface is retained for compatibility but was not used in this public set experiment.

## Inspiration Paper

The core inspiration for this approach comes from Puthiya Parambath, Vijayakumar, and Chawla:

> **Risk Aware Ranking for Top-k Recommendations** (2019)
> https://arxiv.org/abs/1904.05325

The paper discusses how Top-k recommendations should not only optimize individual product prediction scores: the final list should also control uncertainty and redundancy risk. Rather than replicating the paper's learned risk model, this uses an auditable local approximation: preserving relevance-dominant feature scores and applying MMR penalty on similarity to already-selected products. This ensures the top-10 results retain opportunity to hit the target while preventing near-duplicate products from crowding the list.

## Configuration and Reproduction

Key parameters in `src/config/default.yaml`:

| Parameter | Current value | Purpose |
| --- | ---: | --- |
| `rerank_top_n` | 20 | Rerank top-20 candidates from retrieval |
| `mmr_lambda` | 1.0 (effective cap 0.70) | Controls relevance vs. deduplication balance |
| `profile_weight` | 0.15 | Bounded bonus for profile prior |
| `use_field_aware_slot_coverage` | true | Enable title/category/feature/details field coverage |
| `feature_weights` | see config | Fusion weights for each relevance feature |

Run from project root:

```bash
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python scripts/run_public_eval.py --output results_rerank_risk_aware_mmr_profile.json
```

## Public Set Results

Run against 200 official public sessions with official evaluator, results written to `results_rerank_risk_aware_mmr_profile.json`:

| Metric | Current reranking result |
| --- | ---: |
| Hit Rate@10 | 0.905 |
| MRR | 0.706437 |
| MTTC | 3.535 |
| Efficiency | 0.7465 |
| TechnicalScore | 0.813731 |
| LLM tokens | 0 |

This result is consistent across repeated runs under the same fixed environment variables. Evaluation results should be based on this JSON file and the official evaluator's actual output.

## Scope of Changes

- Main logic: `src/ranking/reranker.py`
- Profile prior: `src/ranking/profile_prior.py`
- Parameter config: `src/config/default.yaml`
- Result file: `results_rerank_risk_aware_mmr_profile.json`

Official `evaluator/` was not refactored or modified.
