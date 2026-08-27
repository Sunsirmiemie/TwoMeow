"""Load the nested YAML file into the agent's stable, flat config interface."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).with_name("default.yaml")
_AGENT_KEYS = {
    ("retrieval", "top_k"): "retrieval_top_k",
    ("retrieval", "field_weights"): "field_weights",
    ("fusion", "rrf_k"): "rrf_k",
    ("fusion", "bm25_base"): "bm25_base",
    ("fusion", "dense_base"): "dense_base",
    ("fusion", "browsing_weights"): "browsing_weights",
    ("dense", "use_dense"): "use_dense",
    ("dense", "model"): "dense_model",
    ("dense", "batch_size"): "dense_batch_size",
    ("ranking", "use_llm_ranker"): "use_llm_ranker",
    ("ranking", "ranker_model"): "ranker_model",
    ("ranking", "rerank_top_n"): "rerank_top_n",
    ("observability", "trace_enabled"): "trace_enabled",
    ("dialogue", "entropy_tau"): "entropy_tau",
    ("dialogue", "min_pool_for_dynamic"): "min_pool_for_dynamic",
    ("dialogue", "few_slots_threshold"): "few_slots_threshold",
    ("dialogue", "pool_size_threshold"): "pool_size_threshold",
    ("dialogue", "truncated_size"): "truncated_size",
    ("dialogue", "use_dynamic_entropy"): "use_dynamic_entropy",
    ("dialogue", "use_early_stop"): "use_early_stop",
    ("dialogue", "use_override_detection"): "use_override_detection",
}


def load_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return flat agent defaults, with explicit flat overrides applied last."""
    nested = yaml.safe_load(_DEFAULT_PATH.read_text(encoding="utf-8")) or {}
    config = {
        target: deepcopy(nested[section][source])
        for (section, source), target in _AGENT_KEYS.items()
    }
    if overrides:
        override_values = deepcopy(dict(overrides))
        if "field_weights" in override_values:
            config["field_weights"].update(override_values.pop("field_weights"))
        config.update(override_values)
    return config
