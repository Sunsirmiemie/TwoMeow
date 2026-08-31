# Three Optimizations, Ablation, and Compliance Notes (2026-08-29)

## Baseline and Scope

This round re-implements from scratch using the latest remote commit `53a2905` as the sole baseline; no optimization code was copied from old working trees. The official entry contract `starter.agent.Agent` is maintained; the frozen product catalog is read-only; `evaluator/local_evaluator.py` and `data/public_set.jsonl` are unmodified.

Original public set retest (200 sessions, BM25-only, `PYTHONHASHSEED=0`): HitRate@10 `0.905`, MRR `0.706437`, MTTC `3.535`, TechnicalScore `0.813731`.

## Optimization 1: Negation-Aware Response Purification

`src/dialogue/purification.py` splits each turn's answer into three categories of observed evidence:

- Positive constraints, e.g. `blue`;
- Exclusion constraints, e.g. `not red`;
- No preference, e.g. `I don't have a preference for color`.

Example: `I do not want red; blue instead.` no longer sends both `red` and `blue` into retrieval. The retrieval text retains only `blue instead`, while the session records `negative_slots.color={red}`. Subsequent product scoring applies an exclusion penalty to red products, rather than treating the negation word as a positive keyword.

Intent Override uses selective overwrite: stable categories are retained; old values for the same attribute as the new answer are replaced directly; attributes explicitly negated or stated as no-preference are deleted; other old attributes are demoted to `0.35` confidence, only enter the query once and can still be re-confirmed by subsequent questions. Old raw text is not re-added to queries. This process does not use the next turn's message or hidden labels.

## Optimization 2: Dynamic Attribute Weights Through 50,000→100→10

Attribute weights are not fixed constants, nor are they just adjusting the overall BM25/Dense ratio. Each turn computes for each current attribute `a`:

`raw_weight(a) = evidence_confidence × recency × (0.55 + 0.45 × pool_selectivity) × attribute_prior`

Then normalized over all valid attributes for this turn. Here:

- `evidence_confidence` distinguishes structured explicit answers from generalized regex extraction;
- `recency` slightly decays very old conditions but does not forget them;
- `pool_selectivity` checks how many products in the current candidate pool already satisfy that value — attributes that better discriminate products get higher weight;
- `attribute_prior` is only a small, auditable prior; the final weight is recomputed from current accumulated evidence and the candidate pool.

Dynamic results are applied in two stages:

1. **50,000→100**: accumulated category information increases BM25 `categories/title` field weights; material, color, size, style, use-case, feature information increases `features/details/description` field weights. BM25 retrieves the top 300 results, then reranks by per-product attribute compatibility, and truncates to top 100.
2. **top-20→10**: the relevance component of the existing risk-aware MMR is replaced with the current dynamic compatibility score, and an exclusion risk penalty is applied to products matching negative constraints; MMR's deduplication responsibility remains unchanged.

For example, when only `category=running shoes` is known, category weight dominates; when the user then adds `waterproof`, if only a few products among the top 300 are waterproof, `feature` selectivity rises and it automatically gains higher weight; after adding `not red`, red products receive exclusion risk. The entire process only looks at information already provided before the current turn.

## Optimization 3: Field-Separated Dense + Risk-Gated Adaptive Fusion

Dense no longer encodes one product vector by concatenating `title + category + features`; instead two separate indexes are built:

- Identity vector: `title + categories + store`;
- Attribute vector: `features + details + description`.

Queries are likewise split into category/identity query and attribute query. The more specific attributes are known, the higher the attribute similarity weight; when the category is clear, identity similarity weight is higher. The two similarity scores are weighted at runtime without concatenating vectors.

The original commit, after actually loading the original Dense, scored `0.742997` TechnicalScore on the public set, below its BM25-only `0.813731`; the current selective-overwrite version with forced field Dense scores `0.780615`, also below the risk-gated default path's `0.852039`. Therefore an auditable risk gate is applied by default:

- Buying always uses exact BM25;
- When BM25 candidates are no fewer than 20 and a stable category exists, Dense is not called;
- Dense is only lazily loaded when lexical candidates are insufficient, or for broad queries with no category and very low BM25 confidence;
- When Dense cannot be installed, the model is not in offline cache, or loading fails, automatically falls back to BM25.

After the gate opens Dense, the RRF source weights are determined jointly by the accumulated slot count and the score separation between the two result lists in this turn. There is no prediction of next turn's question gain, and no use of future scores unavailable in the real competition.

## Complete Ablation Results

| Config | HitRate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| Updated repo original retest | 0.905 | 0.706437 | 3.535 | 0.813731 |
| + Response purification | 0.880 | 0.664893 | 3.660 | 0.786268 |
| + Dynamic attributes (strict Override, historical experiment) | 0.935 | 0.688629 | 2.995 | 0.834189 |
| Final: selective Override + Dense gate | **0.955** | 0.706129 | **2.865** | **0.852039** |
| Final: selective Override + forced field Dense | 0.920 | 0.546383 | 3.165 | 0.780615 |

The primary reason response purification alone decreased on the public set: the public evaluator does not generate true exclusion answers like `not red`, and the soft preferences before Intent Override happen to come from the target product. Selective overwrite reduces information loss from full clearing while avoiding re-using old raw text as strong queries. Dense was put behind the risk gate based on the forced comparison results, rather than sacrificing scores for the formality of using a model.

## Unseen Products and Overfitting Audit

First generated 400 development validation samples from the frozen catalog with zero overlap with public target ASINs (seed `20260829`). This set was used to discover issues with strict Override, so it cannot serve as the final blind test. After switching to selective overwrite, it moved from the original BM25's `0.766886` to `0.767773`.

Then froze code and parameters, and generated a second 400-ASIN final blind test (seed `20260830`) with zero overlap with both the public set and development validation set:

| Final blind test config | HitRate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| Original BM25 | **0.875** | **0.587855** | 3.660 | **0.760657** |
| Current selective-overwrite version | 0.8725 | 0.552678 | **3.575** | 0.750553 |

Conclusion: the current version has slightly better efficiency on the final blind test, but HitRate, MRR, and overall score did not exceed the original. Therefore the public set `0.852039` is only a public set result and cannot be claimed as proven generalization. This project retains this negative result; future parameter selection should be done on training/validation folds, with evaluation on a new frozen test set only once.

## Competition Compliance Check

- Official evaluator, public set, starter entry, or frozen catalog were not modified;
- No RL policy training, base model fine-tuning, or LLM-generated product ASINs;
- Dense uses local `all-MiniLM-L6-v2` inference, no training;
- Vector index is local `.npz`, similarity computation entirely in memory, no external vector database;
- Default path does not call LLM, public set token usage is 0;
- All recommendations consist of valid `parent_asin` from the frozen catalog;
- Decisions use only current and historical observed information, not future questions, future gains, or ground truth.

## Reproduction

```bash
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m pytest -q

python scripts/run_public_eval.py \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output /tmp/final-optimized.json

python scripts/run_public_eval.py \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --force-dense \
  --output /tmp/forced-dense.json

python scripts/run_three_optimization_ablation.py \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output /tmp/three-optimization-ablation.json
```
