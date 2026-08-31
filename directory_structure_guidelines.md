# Directory Structure Guidelines (modifiable)
techjam-conversational-search/  
│  
├── data/  
│   ├── catalog.jsonl  
│   └── public_set.jsonl  
│  
├── docs/  
│   ├── architecture.md  
│   ├── experiment_plan.md  
│   ├── innovation.md  
│   └── team_ownership.md  
│  
├── src/  
│   ├── agent/  
│   │   ├── orchestrator.py  
│   │   ├── router.py  
│   │   ├── state.py  
│   │   └── response_builder.py  
│   │  
│   ├── retrieval/  
│   │   ├── catalog.py  
│   │   ├── bm25.py  
│   │   ├── dense.py  
│   │   ├── hybrid.py  
│   │   └── candidate_builder.py  
│   │  
│   ├── dialogue/  
│   │   ├── attribute_stats.py  
│   │   ├── entropy.py  
│   │   ├── question_policy.py  
│   │   ├── early_stop.py  
│   │   └── override.py  
│   │  
│   ├── ranking/  
│   │   ├── features.py  
│   │   ├── scorer.py  
│   │   ├── profile_prior.py  
│   │   └── reranker.py  
│   │  
│   ├── evaluation/  
│   │   ├── runner.py  
│   │   ├── analysis.py  
│   │   ├── ablation.py  
│   │   └── failure_analysis.py  
│   │  
│   └── config/  
│       ├── default.yaml  
│       └── experiments/  
│  
├── tests/  
│   ├── test_state.py  
│   ├── test_override.py  
│   ├── test_entropy.py  
│   ├── test_retrieval.py  
│   ├── test_ranking.py  
│   └── test_contract.py  
│  
├── scripts/  
│   ├── build_index.py  
│   ├── run_public_eval.py  
│   ├── run_ablation.py  
│   └── inspect_failures.py  
│  
├── starter/  
│   └── agent.py  
│  
├── evaluator/  
│   └── [official files — DO NOT MODIFY]  
│  
├── README.md  
├── requirements.txt  
└── .env.example  

## File Size and Code Standards

To support five-person collaboration:
* Individual business Python files should stay within 200–250 lines.
* One file carries one clear responsibility.
* No "super agent.py" files exceeding 500 lines.
* No circular imports between modules.
* All key hyperparameters go into config.
* All public functions must have type hints.
* Core strategies must have a short docstring.
* No experimental logic scattered in business code.
* All experiments must be reproducible via scripts.
* Official evaluator files are treated as external dependencies and must not be refactored.
