# TwoMeow Baseline Stabilization

> Historical stabilization record (2026-08-27): this file does not describe subsequent rerank changes. `results.json` is retained as the original downloaded legacy result; the current rerank result is in `results_rerank_risk_aware_mmr_profile.json`.

## 1. Status and Scope

- Working branch: `stabilize-baseline`
- Reference commit: `64a7db54d5185a835fa8a850f14b2f22cb1d900e`
- Record date: 2026-08-27
- Current status: changes saved in local workspace only, not yet committed or pushed
- This round's goal: converge the entry point, configuration, key data boundaries, evaluation analysis, and installation/test process into a single trustworthy and reproducible baseline

This round did not overwrite `results.json` or `results_optimal.json`, and did not claim to have re-validated the Dense or real LLM paths. Application code changes are minimal and backward-compatible; the old entry point still works.

## 2. Why This Stabilization Was Needed

The original baseline had several issues affecting reproducibility and follow-up optimization:

1. The repository had two `Agent` implementations; different entry points could execute different logic.
2. Product prices could be numeric or strings like `"—"` or `"from 12.99"`; budget-filtered searches would directly compare strings with floats.
3. `default.yaml` appeared to be a unified config source, but multiple hardcoded values remained at runtime, and CLI defaults would unconditionally override YAML.
4. LLM token usage was a cumulative process value, while the evaluator needed per-response usage.
5. Failure analysis read a non-existent `first_hit_rank`, inconsistent with the evaluator's actual `hit` / `best_rank` output.
6. No formal Python version range, optional dependency groups, or post-install artifact tests.
7. README presented historical best results as "current scores" without distinguishing historical artifacts from this round's reproducible results.

Therefore, this round first fixes the foundational questions — "are results trustworthy, is there a single entry point, does configuration actually take effect, can the environment be installed" — before continuing with retrieval and dialogue strategy optimization.

## 3. Specific Changes

### 3.1 Single Agent Implementation and Compatible Entry Points

`src.agent.orchestrator.Agent` is now the sole implementation:

- `starter.agent.Agent` continues to re-export this class, satisfying the official evaluation entry point;
- `agent.agent.Agent` is changed to a compatible re-export, no longer maintaining a second implementation;
- `run_eval.py` and `scripts/run_public_eval.py` use the canonical Agent directly;
- Tests assert that all three public entry points point to the same class object.

This preserves old import paths while eliminating implementation drift.

### 3.2 Price Normalization and Budget Behavior

Catalog loading uniformly handles prices:

- Convertible and finite values (e.g., `29.99`, `"29.99"`) are converted to `float`;
- Missing values, unparseable strings, `NaN`, and infinite values are converted to `None`;
- Budget filtering in the buying scenario retains only products where `price <= budget` for known prices;
- Unknown prices are kept in candidates, avoiding loss of potential hits due to catalog field irregularities.

This is a conservative strategy: `"from 12.99"` is currently not guessed as 12.99, but treated as unknown price.

### 3.3 YAML Configuration Actually Enters Runtime

Added `src.config.load_config()`, reads the nested `src/config/default.yaml`, then maps to the flat config used by Agent. Current wiring covers:

- Retrieval counts and BM25 field weights;
- RRF constant, base fusion weights, and browsing weights varying by slot count (0 slots 50/50, 1 slot 60/40, ≥2 slots use base weight 75/25);
- Dense toggle, model name, and batch size;
- LLM ranker toggle, model name, and rerank top-N;
- Entropy threshold, dynamic entropy minimum candidate pool;
- Rerank pool slot count, candidate count, and truncation threshold;
- Dynamic entropy, early stop, and override detection toggles.

Merge semantics:

1. Each call reads from YAML and deep-copies the default values;
2. Caller passes a flat override; explicit values take final precedence;
3. `field_weights` supports partial deep merge — overriding only `title` will not clear other fields;
4. Other lists or mappings are also copied before use, preventing reverse modification of caller objects;
5. CLI only generates overrides when the user explicitly provides `--no-dense` or `--llm-rank`; unprovided parameters retain YAML defaults.

### 3.4 Per-Response Token Usage

`Ranker` resets usage to zero at the start of each `rerank()` call, recording only the input/output tokens of that API response:

- Two consecutive calls do not accumulate;
- Returns 0/0 when LLM is disabled, candidates are empty, or API fails before receiving a response;
- When a response is received but content cannot be parsed, retains this response's usage and falls back to original retrieval order;
- Tests verified via injected fake client, no real network access.

### 3.5 Failure Analysis Aligned to Evaluator Schema

Failure analysis now uses the official evaluator's actual fields:

- Near miss: `hit is True` and `best_rank` within a specified closed interval, default 6–10;
- Complete miss: `hit is False`;
- `scripts/inspect_failures.py` outputs `sample_id` and `best_rank`.

### 3.6 Packaging, Dependencies, and Repository Hygiene

- Added `pyproject.toml`, supports Python `>=3.10,<3.13`;
- Core dependencies: `numpy>=1.26,<3`, `PyYAML>=6,<7`;
- Optional extras: `dense`, `llm`, `test`;
- `requirements.txt` is the full aggregate environment for Dense, LLM, testing and build tools;
- wheel includes `src/config/default.yaml`, and provides canonical, starter, and legacy entry points;
- Added `.gitignore` blocking newly added `graft/`, Python/pytest caches, virtual environments, `.env`, `.idea/`, `.embed_cache/`, `*.egg-info/`, `build/`, `dist/`, specified catalog copies, and `*.zip` from being tracked.

Note: these rules only affect currently-untracked files matching the above patterns; they will not automatically untrack existing large files, catalog, zip, or IDE files in the repository.

### 3.7 Test Reinforcement

New or extended test coverage:

- Non-numeric prices and budget filtering;
- Canonical Agent entry point consistency;
- YAML defaults, flat overrides, partial field weight merges, and caller object isolation;
- Whether config values actually reach BM25, Dense, RRF, dialogue entropy threshold, and rerank pool;
- CLI generates only explicit overrides;
- Per-response token usage, disabled/empty candidates/API exception/invalid JSON fallback;
- `hit` / `best_rank` failure analysis;
- Isolated import/construction from the wheel unpacked to a temp directory: entry points, evaluator module, and YAML artifact are available.

## 4. RED → GREEN Record

This round followed a behavior-contract-first approach: failing tests were written before implementing minimal fixes:

| Contract | RED behavior | GREEN result |
| --- | --- | --- |
| Non-numeric prices can participate in budget-filtered search | String price vs. float budget triggers `TypeError` | Invalid prices normalized to `None`, unknown prices retained |
| Failure analysis reads official schema | `first_hit_rank` doesn't exist, hit/miss classification wrong | Uses `hit` and `best_rank` |
| All public entry points use same Agent | Legacy and canonical are different classes | All three entry points re-export canonical class |
| YAML is runtime default configuration | No public loader, multiple defaults not reaching components | Loader, CLI, and components fully wired |
| Token usage belongs to single response | Second call includes first call's tokens | Each call independently resets and records |
| Installed artifact can run standalone | No standard project metadata and YAML artifact constraint | Isolated import/construction test from wheel passes |

## 5. Verification Results

### 5.1 Automated Tests and Artifact Isolation

Independent verification ran the full suite on three supported Python minor versions:

| Python | Result |
| --- | --- |
| 3.10 | `74 passed` |
| 3.11 | `74 passed` |
| 3.12 | `74 passed` |

Unified command:

```bash
python -m pytest -q
```

The artifact test builds a wheel, unpacks it to a temp directory, removes the repo path and editable-import finder, then verifies:

- `src.agent.orchestrator.Agent`, `starter.agent.Agent`, `agent.agent.Agent` all come from this wheel and are identical;
- `evaluator.local_evaluator` can be imported from the artifact;
- `default.yaml` is packaged;
- A tiny catalog can create an Agent.

Also passes:

- `git diff --check`;
- `results.json` and `results_optimal.json` byte-identical relative to the reference commit;
- Evaluation output written to `/tmp`, not written to repository result files; repository root `build/` and `*.egg-info/` generated during verification are cleaned up precisely after checking, leaving no root `build/`, `dist/`, or `*.egg-info/`.

### 5.2 This Round's Reproducible BM25-only Public Set Baseline

To avoid overwriting repository results, output must be explicitly written to a temp directory:

```bash
python scripts/run_public_eval.py \
  --no-dense \
  --output /tmp/twomeow_bm25_stabilize_20260827.json
```

Re-ran 200 public sessions on 2026-08-27 with Python 3.12.13, PyYAML 6.0.3, NumPy 2.4.6:

| Metric | Value |
| --- | ---: |
| sample_count | 200 |
| HitRate@10 | 0.865 |
| MRR | 0.552575 |
| MTTC | 3.65 |
| Efficiency | 0.735 |
| Recommended Technical Score | 0.745273 |
| Prompt / Completion Tokens | 0 / 0 |

This result only validates the BM25-only path; sentence-transformers was not loaded, no LLM was called.

### 5.3 Historical Result Files in Repository

Both files retained as-is:

| File | Role | HitRate@10 | MRR | MTTC | TechScore |
| --- | --- | ---: | ---: | ---: | ---: |
| `results_optimal.json` | Existing historical best record | 0.890 | 0.554536 | 3.405 | 0.763261 |
| `results.json` | Earlier, non-canonical legacy result, no longer current baseline | 0.790 | 0.471744 | 5.12 | 0.654123 |

The historical best file was not newly generated in this round. The project's original notes recorded it as a no-LLM experiment, but the JSON itself does not contain sufficient run configuration provenance to independently verify this; the fresh, reproducible, and clearly-scoped numbers for this round are the BM25-only baseline in the previous section.

## 6. Known Limitations and Deferred Work

1. **Price semantics remain conservative**: `"from 12.99"` is treated as unknown, without parsing currency, ranges, or starting prices.
2. **Config has wiring and merging but no schema validation**: invalid types, negative thresholds, missing fields, or wrong weight lengths may only surface downstream.
3. **Old `agent/` module tree retained**: entry points are unified, but auxiliary modules from the old implementation have not been deleted, to reduce compatibility risk in this round.
4. **Tracked large files not yet cleaned**: catalog copies, Dense cache, participant kit zip, and `.idea` files remain in Git history/index; new ignore rules will not automatically untrack them.
5. **Dense strict offline behavior not verified**: sentence-transformers may attempt downloads when the model is not cached; no full Dense evaluation was run in this round.
6. **Real LLM path not verified**: only protocol, usage, and fallback verified with fake client; Anthropic API was not called.
7. **Dependencies not locked to hash level**: version ranges are installable, but transitive dependencies resolved at different times may vary.
8. **BM25 queries remain OR-combined**: recall is broad; slot value repetition does not create additional term frequency weight after deduplication; needs separate experiments with AND/phrase/per-field strategies.
9. **Entropy formula vs. paper scope still needs calibration**: current normalization denominator uses candidate value count; needs comparison experiment with attribute value domain scope.
10. **Early stop `other` strategy still needs validation**: asking `other` under low information gain is a heuristic design; whether it outperforms stopping questions or returning results needs per-scenario ablation confirmation.

## 7. Next Steps

In priority order:

1. **Establish a single official baseline process**: run BM25-only and offline-loadable Dense separately in a fixed environment, save commands, dependency snapshot, model provenance, and new result files; clarify which results can be published.
2. **Add config validation and error messages**: validate weights, thresholds, paths, models, and boolean toggles at startup, avoiding silent degradation.
3. **Run retrieval experiments on 200 public sessions**: prioritize comparing BM25 OR/AND/phrase, slot value field weighting, and unknown-price strategies; all experiment output to new files, cannot overwrite historical artifacts.
4. **Calibrate dialogue strategy scope**: run per-scenario ablation on entropy normalization and `other` early-stop, focusing on the MTTC vs. HitRate tradeoff.
5. **Complete repository size reduction**: after confirming data distribution requirements, untrack IDE files, caches, duplicate catalogs, and zip; if large files must be retained, switch to release artifacts or LFS.
6. **Generate dependency lock and offline install packages**: generate auditable lock files for Python 3.10–3.12, and verify network-free installation and model loading.
7. **Delete old module tree last**: first announce canonical import externally, then remove no-longer-used legacy auxiliary modules.

## 8. Current Git Status

At the time this document was completed, the following boundaries remained:

- Branch created: `stabilize-baseline`;
- No commit created;
- No remote branch set or pushed;
- `results.json` and `results_optimal.json` unchanged;
- This round's recorded BM25-only output saved to `/tmp/twomeow_bm25_stabilize_20260827.json`, never written to in-repository result files.
