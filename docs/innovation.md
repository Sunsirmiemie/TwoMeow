# Innovation Description

This document corresponds to the competition judging dimension "Innovation & Problem Insight" (20% weight), explaining TwoMeow's core technical innovations and deep understanding of the problem.

## Innovation 1: Entropy-Driven Dynamic Clarification Strategy

**Problem**: Static attribute priority ordering (fixed order: material → color → size → ...) ignores how information is distributed differently across dimensions in different product pools. For example, in a candidate pool of all-cotton dresses, asking about material yields no new information.

**Innovation**: Each dialogue turn, computes the information gain of each attribute in real-time based on the current candidate pool:

```
score(attr) = coverage(attr) × normalized_entropy(attr)
```

- `coverage`: proportion of candidate pool products that have a value for this attribute (measures "answerability" of the question)
- `entropy`: normalized information entropy of the attribute value distribution (measures "information content" of the answer)

When the candidate pool is small (<10), blended with global entropy from PDF analysis of 50,000 products, avoiding small-sample noise.

**Effect**: TechScore +0.066 (BM25-only 0.590 → + Entropy 0.656), MTTC reduced from 6.36 to 5.08.

## Innovation 2: Entropy-Threshold Early Stop (τ=0.3)

**Problem**: When the candidate pool is already highly focused (all attribute entropies are low), continuing to ask specific attributes only yields "I don't have an additional preference" — wasting turns and inflating MTTC.

**Innovation**: Introduces an entropy-threshold early stop mechanism:

```
if max(score(attr) for attr in remaining) < τ=0.3:
    ask "other"  # wildcard question: let the evaluator choose any undisclosed constraint
```

When the information gain of all remaining attributes is below the threshold, instead of guessing which attribute to ask, use a wildcard to let the evaluator proactively choose the most relevant constraint to answer. This converts "wasted questions" into "effective information acquisition."

**Effect**: TechScore +0.099 (largest single-step gain), MTTC reduced from 5.08 to 3.44, HitRate increased from 0.800 to 0.885. This is experimental validation consistent with the theoretical prediction (PDF identified τ=0.3 as optimal).

## Innovation 3: Four-Scenario-Aware Adaptive Retrieval Routing

**Problem**: Buying and Browsing scenarios have fundamentally different retrieval needs — Buying users have precise hard constraints requiring keyword exact matching; Browsing users have vague intent requiring semantic recall as fallback.

**Innovation**: Dynamically adjusts retrieval strategy based on scenario and current slot count:

| Scenario/slot state | BM25 | Dense | Rationale |
|------------|------|-------|------|
| Buying | 100% | 0% | Evaluator provides precise constraint text each turn, BM25 hits directly |
| Browsing, 0 slots | 50% | 50% | No constraints, semantic generalization as fallback |
| Browsing, 1 slot | 60% | 40% | Few constraints, still need semantic supplement |
| Browsing, ≥2 slots | 75% | 25% | Enough constraints, exact match preferred |

Ablation experiments found: adding Dense into RRF for Buying actually reduces MRR (0.464→0.427), so the Buying track completely skips Dense — a counterintuitive finding discovered from data.

## Innovation 4: Atomic State Reset on Intent Override

**Problem**: In the Intent Override scenario (15% of sessions), users suddenly say "ignore my earlier preference" at turn 3/4. If old slots are not cleared, old constraints pollute the new query, resulting in completely wrong retrieval results.

**Innovation**: Uses regex to precisely match the evaluator's fixed template `"ignore my earlier preference"`, atomically:
1. Clear all confirmed slots (`slots.clear()`)
2. Clear the asked-attributes list (`asked_attributes.clear()`)
3. Reset to buying track (new constraints are typically hard constraints)

Key design: SlotTracker runs after IntentRouter, so it can immediately parse `"What I need is: X"` as a new slot without interference from old slots.

**Effect**: Override scenario HR 0.833→0.867, MRR 0.601→0.722.

## Innovation 5: Double-Weighted Query Construction

**Problem**: BM25 treats all terms equally, while "confirmed slot values" (e.g., "leather" explicitly stated by the user) have higher retrieval priority than noise words in history messages.

**Innovation**: Actively repeats slot value text when constructing queries:

```python
query = f"{slot_text} {slot_text} {user_message} {filtered_history}"
```

Repeating once doubles BM25 TF score, equivalent to applying 2× weight on confirmed constraints, without modifying the FTS5 implementation. History messages are also noise-filtered (skipping evaluator fill phrases like "use your judgment" / "not quite right").

## Innovation 6: Negation-Aware Response Purification

**Problem**: When a user answers "not red; blue instead", sending both `red` and `blue` into retrieval makes `red` a positive BM25 term, recalling products the user doesn't want.

**Innovation**: `src/dialogue/purification.py` splits each turn's answer into three evidence categories: positive constraints, exclusion constraints, and no-preference. Only the positive part is kept in retrieval text; exclusion constraints are recorded in `negative_slots`, and downstream product scoring applies an exclusion penalty to matching items. On Intent Override, selective overwrite is used: conflicting attributes are replaced directly, other old attributes are demoted to 0.35 confidence weak evidence, and old raw text is not re-added to queries.

## Innovation 7: Candidate-Pool-Aware Dynamic Product Attribute Weights

**Problem**: Fixed field weights (title/features/...) cannot self-adapt as accumulated evidence changes; the two truncation stages 50,000→100 and top-20→10 each have static weights.

**Innovation**: Each turn computes a dynamic weight for each confirmed attribute `a`:

```
raw_weight(a) = evidence_confidence × recency × (0.55 + 0.45 × pool_selectivity) × attribute_prior
```

- `pool_selectivity`: the lower the proportion of candidate pool products satisfying this attribute value, the more discriminating the attribute and the higher its weight;
- Normalized and applied in both stages: BM25 field weights (50,000→100) and MMR relevance score (top-20→10).

**Effect**: In dynamic attribute single-item experiments, historical TechScore improved from 0.814 to 0.834, MTTC reduced from 3.54 to 3.00.

## Innovation 8: Field-Separated Dense + Risk-Gated Adaptive Fusion

**Problem**: Concatenating title+categories+features and encoding them into one vector makes the same vector carry both "what the product is" and "what properties the product has," mixing semantics and weakening retrieval precision. On the public set, forced Dense (original) TechScore 0.743, actually lower than BM25-only's 0.814.

**Innovation**:
- **Field separation**: Identity vector (title+categories+store) and attribute vector (features+details+description) are encoded and queried separately; the more specific attributes are known, the higher the attribute similarity weight.
- **Risk gate**: Buying scenario always skips Dense; when BM25 candidates are sufficient (≥20 with stable category), Dense is not called; only lazily loads Dense when lexical recall is weak. Automatically falls back to BM25 when Dense is unavailable.

**Effect**: Default path (risk gate) on public set TechScore=0.852039, better than forced field Dense's 0.780615.

---

## Combining Technical Innovation with Business Value

The above innovations together form an adaptive strategy that "converges while conversing": the system does not mechanically go through all questions, but rather evaluates in real-time "what else needs to be known" — confidently recommending when information is sufficient, and precisely asking when information is lacking.

This closely mirrors real e-commerce scenarios: a good sales assistant doesn't follow a fixed question script, but rather judges from customer responses how much the candidate set has narrowed, deciding whether to keep asking or to recommend directly.
