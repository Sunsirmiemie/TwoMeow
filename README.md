# TwoMeow — TechJam 2026 Conversational Search Agent

Multi-turn conversational e-commerce search Agent, competition entry: TechJam 2026 Conversational E-Commerce Search Challenge.

**Current default results** (public set 200 sessions, `PYTHONHASHSEED=0`, BM25-only, no LLM): HitRate@10 = **0.955**, MRR = **0.706546**, MTTC = **2.865**, Efficiency = **0.8135**, TechnicalScore = **0.852164**, tokens = **0**. Relative to the updated repository's original public set results (0.905 / 0.706437 / 3.535 / 0.813731), HitRate improved by 5 percentage points, MTTC reduced by 0.67 turns, TechnicalScore improved by 0.038433.

**Generalization warning**: In a second completely isolated 400-ASIN final blind test, the current version scores TechnicalScore `0.750553`, below the original BM25's `0.760657`. Therefore the public set gains above cannot be claimed as proven generalization; the three current optimizations should still be treated as experimental.

This round re-implements three optimizations on top of the existing risk-aware MMR: negation-aware response purification, candidate-pool-aware dynamic product attribute scoring, and field-separated Dense with risk gating. For design, ablation, and compliance boundaries see [`docs/THREE_OPTIMIZATIONS_20260829.md`](docs/THREE_OPTIMIZATIONS_20260829.md).

**Best historical record in repository** (public set 200 sessions, see `results_optimal.json`): HitRate@10 = **0.890**, MRR = **0.555**, TechScore = **0.763**. The project's original notes recorded this experiment as no-LLM reranking, but the JSON itself does not include sufficient provenance to independently verify the run configuration; Dense/LLM was not re-run in this round either.

**This round's reproducible BM25-only baseline**: HitRate@10 = **0.865**, MRR = **0.553**, MTTC = **3.650**, TechScore = **0.745**. Full scope, commands and change notes in [`docs/BASELINE_STABILIZATION.md`](docs/BASELINE_STABILIZATION.md).

Official Baseline TechScore = 0.107.

---

## Quick Start

### Environment Setup

Supports Python `>=3.10,<3.13`. Install based on your use case:

```bash
python -m venv .venv
source .venv/bin/activate

pip install .              # Core: BM25 + config loading
pip install '.[dense]'     # Optional: Dense retrieval
pip install '.[test]'      # Optional: testing, build & artifact validation
```

`requirements.txt` is the full aggregate dependency for Dense, LLM, and dev/test tools; run `pip install -r requirements.txt` when you need everything at once.

### Local Evaluation

```bash
# Standard evaluation (recommended): BM25-only, fully reproducible, zero tokens
PYTHONHASHSEED=0 python scripts/run_public_eval.py --no-dense --output /tmp/twomeow-result.json

# Default config evaluation: Dense risk gate enabled (first run downloads sentence-transformers model)
python scripts/run_public_eval.py --output /tmp/twomeow-default.json

# Diagnostic: force bypass risk gate (noticeably reduces score on public set)
python scripts/run_public_eval.py --force-dense --output /tmp/twomeow-forced-dense.json
```

**`results.json` convention**: `results.json` holds both the current default script output and the best result achieved so far. When a new run produces a higher TechnicalScore than the existing `results.json`, overwrite it with the new result. Use named files (e.g. `results_*.json`) for all other experiment outputs so that `results.json` always reflects the best reproducible result on the public set.

### Ablation Experiments

```bash
# Note: the current script writes or overwrites ablation_results.json in the repo root.
# It is not a safe read-only verification process; for read-only verification use the /tmp eval commands above or pytest.
# Run the full 6-row ablation table
python scripts/run_ablation_table.py

# Analyze failed sessions
python scripts/inspect_failures.py --results results.json

# Pre-build Dense index (avoid waiting on first evaluation run)
python scripts/build_index.py
```

### Running Tests

```bash
python -m pytest -q
```

---

## Project Structure

```
TwoMeow/
├── src/
│   ├── agent/
│   │   ├── orchestrator.py      # Agent main entry (reset / respond)
│   │   ├── router.py            # Scene classification (buying/browsing/override/boundary)
│   │   ├── state.py             # SessionMemory + SlotTracker
│   │   └── response_builder.py  # Query construction + message generation
│   ├── retrieval/
│   │   ├── catalog.py           # SQLite FTS5 index + attribute cache
│   │   ├── bm25.py              # BM25 retrieval + slot hard filtering
│   │   ├── dense.py             # Dense retrieval (all-MiniLM-L6-v2)
│   │   ├── hybrid.py            # BM25 + Dense + RRF fusion
│   │   └── candidate_builder.py # Candidate pool truncation strategy
│   ├── dialogue/
│   │   ├── attribute_stats.py   # Attribute regex patterns + global entropy statistics
│   │   ├── entropy.py           # coverage × entropy scoring
│   │   ├── purification.py      # Negation/retraction/no-preference response purification
│   │   ├── question_policy.py   # Clarifier (dynamic attribute selection)
│   │   ├── early_stop.py        # Entropy-threshold early stop (τ=0.3)
│   │   └── override.py          # Intent Override / Boundary detection
│   ├── ranking/
│   │   ├── scorer.py            # RRF fusion algorithm
│   │   ├── dynamic_attributes.py # Dynamic attribute weights and product compatibility
│   │   ├── mmr.py               # MMR deduplication selection
│   │   ├── reranker.py          # Local field-aware top-20 → top-10 reranking (LLM-compatible optional path)
│   │   ├── features.py          # Product text extraction
│   │   └── profile_prior.py     # User profile prior (local, deterministic)
│   ├── evaluation/
│   │   ├── runner.py            # Evaluation wrapper
│   │   ├── analysis.py          # Score analysis
│   │   ├── failure_analysis.py  # Failed session analysis
│   │   └── ablation.py          # Ablation configuration
│   └── config/
│       └── default.yaml         # Main tunable Agent parameters for this round
├── scripts/
│   ├── run_public_eval.py       # Main evaluation script
│   ├── run_ablation_table.py    # 6-config ablation table
│   ├── build_index.py           # Pre-build Dense index
│   └── inspect_failures.py      # Failure analysis
├── tests/                       # Per-module unit tests
├── docs/
│   ├── architecture.md          # System architecture details
│   ├── BASELINE_STABILIZATION.md # Baseline stabilization changes and validation record
│   ├── experiment_plan.md       # Experiment records and plan
│   ├── innovation.md            # Innovation description
│   └── team_ownership.md        # Team ownership
├── data/
│   ├── catalog.jsonl            # 50,000 product catalog
│   └── public_set.jsonl         # 200 official public evaluation sessions
├── evaluator/                   # Official evaluator (do not modify)
├── starter/agent.py             # Official interface entry (re-export)
├── requirements.txt
└── .env.example
```

---

## System Flow

Each `respond()` call executes 6 steps:

```
① Scene classification  → buying / browsing / override / boundary
② Slot extraction       → {material: "cotton", color: "blue", ...}
③ Retrieval             → Dynamic-field BM25 → dynamic product attribute reranking → top-100
                          ↘ Field-separated Dense + adaptive RRF enabled only when BM25 is weak
④ Question decision     → coverage×entropy scoring → Early Stop(τ=0.3) → ask_attribute
⑤ Ranking               → Top-20 candidates → local field-aware reranking → top-10
⑥ Return                → {recommendations, ask_attribute, message}
```

**Retrieval strategy**:
- Buying: Dynamic-field BM25 + price filter + per-product dynamic attribute scoring
- Browsing: First assess BM25 candidate count and confidence; field-separated Dense enabled only when lexical recall is insufficient
- When Dense is active, identity fields and attribute fields are encoded separately; weights shift with accumulated evidence; BM25/Dense fusion weights also reference current result confidence

**Core innovations**:
- Response purification: Distinguishes positive constraints, negation constraints, and "no preference"; replaces conflicting attributes on intent override, other old attributes serve only as low-confidence weak evidence
- Dynamic product attributes: Attribute weights determined jointly by evidence confidence, recency, and current candidate pool selectivity; applied in both 50,000→100 and top-20→10 stages
- Risk-gated field Dense: Does not concatenate product vectors; semantic recall only intervenes when BM25 is weak, automatically falls back on offline failure
- Dynamic entropy attribute selection: Real-time scoring against candidate pool each turn, selects attribute with highest coverage×entropy to ask
- Early Stop (τ=0.3): Switches to wildcard question when candidate pool converges, eliminating wasted turns, MTTC reduced from 5.1 to 3.4
- Intent override reset: Precisely detects Override signal, atomically clears old slots, rebuilds search context
- Field-aware reranking: Advances exact-match products by coverage rate of confirmed slots in title/categories/features/details

See `docs/architecture.md`, `docs/innovation.md` and `docs/BASELINE_STABILIZATION.md` for details.

---

## Configuration

The main tunable Agent parameters for this round are in `src/config/default.yaml`, overridable via the flat `config` dict in `Agent(catalog_path, config)`:

```python
agent = Agent("data/catalog.jsonl", {
    "use_dense": True,            # BM25+Dense hybrid retrieval
    "use_dense_risk_gate": True,  # Skip Dense noise in strong-BM25 scenarios
    "use_field_aware_dense": True,# Identity/attribute fields encoded separately
    "dense_device": "auto",      # MPS experiments can explicitly set "mps"
    "dense_max_seq_length": 256,  # Unified context length for fair multi-model comparison
    "dense_query_prefix": "",    # Configurable official prefix for BGE/E5 retrieval models
    "dense_document_prefix": "",
    "use_reply_purification": True,
    "use_dynamic_attribute_scoring": True,
    "use_dynamic_entropy": True,  # Dynamic entropy attribute selection
    "use_early_stop": True,       # Entropy-threshold early stop
    "use_override_detection": True,
    "use_llm_ranker": False,      # Optional, disabled by default
    "ranker_model": "claude-haiku-4-5-20251001",
    "use_field_aware_slot_coverage": True,
})
```

### Dense Model Selection

In fixed-seed MPS comparisons, E5/BGE/MPNet show minimal gains over MiniLM, and all four models are significantly weaker than the current risk-gated path when forced into the RRF path. Therefore the submitted version continues using the smaller and lower-latency `all-MiniLM-L6-v2`, enabling Dense only when BM25 lexical recall is weak. Model selection data and boundaries in `docs/DENSE_MODEL_SELECTION_20260830.md`.

---

## Offline Mode (Competition Network Ban)

The default approach (BM25-only) is fully offline, with no network dependencies and no model downloads required. It reranks top-20 candidates using field-aware confirmed constraint coverage, profile prior, and risk-aware MMR.

When Dense is enabled: if the model/index is unavailable it automatically falls back to BM25; the default risk gate uses lazy loading, so strong-BM25 scenarios do not trigger Dense model loading.

---

## Dependencies

Core dependencies are only `numpy` and `PyYAML`; `sentence-transformers` (Dense), `anthropic` (LLM), and test/build tools are installed via `pyproject.toml` extras groups. `requirements.txt` provides the complete aggregate environment.

No external vector database; all data structures maintained in memory.

---

## Data

- `data/catalog.jsonl`: 50,000 Amazon clothing products, fields per competition specification
- `data/public_set.jsonl`: 200 official public evaluation sessions (with ground_truth), scenario ratio 40/40/15/5
- Evaluation session dialogues are dynamically generated at runtime by `evaluator/local_evaluator.py`, not pre-recorded
