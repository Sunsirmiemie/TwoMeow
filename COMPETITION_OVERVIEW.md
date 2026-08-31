# TechJam 2026 Conversational E-Commerce Search Competition — Complete Overview

> Source: Official competition package (problem statement, `docs/`, `evaluator/local_evaluator.py`, `data/public_set.jsonl`)
> Compiled: 2026-08-26

## I. What Is This Competition?

TechJam 2026 Conversational E-Commerce Search Challenge — a conversational e-commerce search Agent competition.

The task in one sentence: **Build an AI shopping assistant that, within at most 10 turns of dialogue, guides the user to find the product they truly want to buy through questions and recommendations** (the target product is hidden — the Agent cannot see it).

The organizers build an evaluation environment with "simulated users + hidden answers" based on Amazon Reviews 2023: each session corresponds to a product from a real purchase record; the evaluator plays the role of a customer; each turn the Agent may ask one question and give a batch of recommendations, until the target is hit or 10 turns are exhausted.

## II. Data

| Data | Description |
|---|---|
| Product catalog | Frozen 50,000 products, Clothing_Shoes_and_Jewelry category from Amazon Reviews 2023 |
| Visible fields | parent_asin, title, features, description, price, categories, details, average_rating, rating_number, store |
| Scoring field | **Only parent_asin** |
| Public set | 200 labeled development sessions, for local debugging |
| Private set | 800 sessions, held by organizers, used for final scoring |
| Session composition | Anonymous user profiles + simulated dialogue (**not real shopping dialogues**, simulated by the evaluator per hidden intent cards) |
| User profile | purchase_frequency, average_prior_rating, rating_style, preference_tags, summary |

Key rule: the catalog is **strictly read-only** — forbidden to modify or inject new product IDs; users and target products in the public set and private set do not overlap.

## III. Task and Four Pillars

**I. Core Architecture: Intent Routing and Hybrid Pipeline**

- Dual-track routing: quickly identify Buying (wants to buy) → high-precision filtering track for locking in hard constraints; Browsing (just looking) → diverse dense retrieval track for cross-category matching;
- Pipeline: multi-source retrieval (keywords + category + vector similarity) → LLM semantic ranking.

**II. Dialogue Strategy: Multi-turn Scenario Evolution**

- Dynamic state machine: information accumulation (incremental slot filling) + intent mutation (slot erasure and rewriting);
- Active guidance: triggers retrieval truncation when candidate pool is too large, proactively generates structured clarification questions.

**III. Self-Evolution: Dynamic Context Programming**

- Runtime adaptation: personalizes context distillation from dialogue history, updating short-term session state and long-term user profile;
- Adaptive orchestration: reorders workflow and strategies at runtime.

**IV. Evaluation Matrix**

- Coverage (Hit Rate@K): catalog recall at retrieval stage;
- Precision (MRR / Top-K Hit Rate): precision of ranking the target product to the top;
- Efficiency (MTTC): average turns to find the target, penalizing excess dialogue burden.

## IV. Agent Interface (The Only Service Contract to Implement)

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

Key points:

- `ask_attribute` can only be one of 10: `category / material / color / size / style / brand / budget / feature / use_case / other`, or `null`;
- Recommendation list up to 100, **only the first 10 valid, non-duplicate parent_asins in the catalog are scored**;
- Recommendations must be sorted from best to worst (affects MRR);
- `usage` reports token counts (not part of core scoring, serves as feasibility indicator);
- Exceptions, invalid output, timeout → treated as misses.

## V. Evaluator Mechanics (The Key to Winning)

The evaluator = a simulated user that answers questions based on a "hidden intent card."

1. The evaluator generates an "intent card" (hard constraints + soft preferences) from the target product metadata (title, features, details, material, color, price), hidden from the Agent;
2. Sends the first message based on scenario;
3. Each turn the Agent returns `message + ask_attribute + recommendations`;
4. The evaluator checks if the top-10 contains the target product — **if it does, the session ends immediately as a hit**;
5. If not, answers based on the asked attribute: picks undisclosed constraints from the intent card matching that attribute (up to 2); if no match, replies "I don't have any additional preferences";
6. 10 turns without a hit → loss.

**Four scenarios (fixed ratio 40/40/15/5):**

| Scenario | Ratio | First message | Key mechanism |
|---|---|---|---|
| Buying | 40% | Reveals category + one hard constraint upfront | Fast filtering to lock in |
| Browsing | 40% | "I'm just browsing" | Ask questions to uncover soft preferences |
| Intent Override | 15% | Gives one preference | At turn 3 or 4 suddenly changes mind; old slots must reset |
| Boundary | 5% | Normal opening | First question about any attribute is always answered "use your judgment" — asking is wasting turns |

Mechanism details:

- **Hit ends the session** → recommendations should be returned every turn;
- **Intent Override sessions cannot hit before the change** → Agents fixated on first-turn preferences will fail.

## VI. Metrics and Scoring Formula

```text
HitRate@10  = successful sessions / total
MRR         = average(1 / target product rank), 0 for misses
MTTC        = average(turn of first hit), 11 for misses
Efficiency  = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
```

Baseline (official weak BM25):

| Metric | Value |
|---|---|
| Hit Rate@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |
| Efficiency | 0.119 |
| **TechnicalScore** | **0.10671** |

Metrics reported per scenario — if one scenario collapses it will show up in per-scenario breakdown.

## VII. Rules and Constraints

**In scope**: intent detection modules; heterogeneous retrieval routing (weights / dynamic truncation / slot decay); runtime adaptive memory layer; LLM ranking stage prompts or local scoring logic.

**Out of scope**: UI/UX development; base model training or full fine-tuning; external vector database clusters (**must run entirely in memory**); multimodal (text only).

**Hard limits**:

- Maximum 10 turns per session, forced termination beyond that and the session scores zero;
- Catalog is read-only, structural changes or fake ASINs forbidden;
- Input is pre-cleaned; catalog/prices/category tree are static; sessions are isolated with no concurrent pressure.

**Model policy**: any legal LLM API or local model may be used during development; **final scoring may ban network access** — must declare network dependencies and describe offline fallback; API keys only via environment variables, strictly forbidden to commit to repository.

## VIII. Resources

- Frozen catalog of 50,000 products (download via GitHub Release, with SHA256 checksum);
- 200 public development sessions;
- Weak BM25 starter Agent (Python standard library);
- Deterministic local evaluator (`evaluator/local_evaluator.py`);
- Agent API contract, evaluation configuration, baseline results, submission rules documentation.

## IX. Deliverables

1. **Devpost written description**: how the approach solves the problem, development tools, APIs, libraries and frameworks, datasets;
2. **Public GitHub repository**: well-structured, commented code; README includes project overview, installation instructions, reproduction steps, limitations, team contributions;
3. **Demo video**: end-to-end demonstration (backend/NLP track may use API calls or inference analysis demo), publicly available on YouTube with link on Devpost; no unauthorized third-party trademarks or copyrighted content.

Submission rules also require: `Agent` entry file + dependency list + installation instructions + brief technical report (methods, model selection, limitations) + latency/token/cost disclosure.

## X. Judging Criteria

| Criterion | Weight | What's assessed |
|---|---|---|
| Technical Execution | 35% | Code structure, architecture, API/model usage, reliable demo |
| Innovation & Problem Insight | 20% | Originality, problem understanding depth, whether approach hits the core |
| Impact & Relevance | 20% | Value for real users/scenarios, beyond the competition itself |
| Feasibility & Practicality | 15% | Reasonable resource use, deployable architecture, no empty talk |
| Presentation & Communication | 10% | Final presentation narrative and Q&A depth |

Note: **TechnicalScore is only part of the judging** — Innovation, Impact, and delivery quality account for more than half.

## XI. Practical Implications

1. **Return top-10 every turn** (hit ends the session, recommend early to win early);
2. **Asking the right attribute = directly extracting constraints from target product metadata** (asking material may directly get "leather");
3. **Don't ask in Boundary, reset on Override, guide in Browsing**;
4. The evaluator source code is in the local repo; any strategy hypothesis can be verified with `evaluator/local_evaluator.py`.
