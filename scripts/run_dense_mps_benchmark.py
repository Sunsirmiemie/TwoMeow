#!/usr/bin/env python
"""Benchmark field-aware Dense encoders and fixed RRF weights on Apple MPS.

Buying deliberately remains BM25-only, matching the production route.  Dense is
forced for the other tracks so the experiment measures model and source weight
instead of being hidden by the production risk gate.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter


_REPRODUCIBLE_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
if any(os.environ.get(key) != value for key, value in _REPRODUCIBLE_ENV.items()):
    stable_environment = dict(os.environ)
    stable_environment.update(_REPRODUCIBLE_ENV)
    os.execve(sys.executable, [sys.executable, *sys.argv], stable_environment)

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src.agent.orchestrator import Agent


MODEL_SPECS = {
    "all-MiniLM-L6-v2": {
        "model": "all-MiniLM-L6-v2",
        "batch_size": 512,
        "query_prefix": "",
        "document_prefix": "",
    },
    "all-mpnet-base-v2": {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "batch_size": 128,
        "query_prefix": "",
        "document_prefix": "",
    },
    "bge-small-en-v1.5": {
        "model": "BAAI/bge-small-en-v1.5",
        "batch_size": 256,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
    },
    "e5-small-v2": {
        "model": "intfloat/e5-small-v2",
        "batch_size": 128,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
}


def _weights(value: str) -> list[float]:
    parsed = [float(item) for item in value.split(",")]
    if not parsed or any(item < 0.0 or item >= 1.0 for item in parsed):
        raise argparse.ArgumentTypeError("weights must be comma-separated values in [0, 1)")
    return parsed


def _check_device(device: str) -> None:
    if device != "mps":
        return
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")


def _summary(result: dict) -> dict:
    keys = (
        "sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency",
        "recommended_technical_score", "reported_token_usage",
        "scenario_metrics",
    )
    return {key: result[key] for key in keys}


def _write_output(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class _EmptyDense:
    """Keep the RRF code path while contributing no Dense products."""

    field_aware = True

    def search(self, query: str, top_k: int = 100, session=None) -> list[dict]:
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--hf-home")
    parser.add_argument("--dense-cache")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--weights", type=_weights, default=_weights("0.05,0.10,0.20,0.30"))
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_SPECS),
        help="Repeat to benchmark a subset; defaults to all four models.",
    )
    args = parser.parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = str(Path(args.hf_home).resolve())
    if args.dense_cache:
        os.environ["TWOMEOW_DENSE_CACHE_DIR"] = str(
            Path(args.dense_cache).resolve()
        )
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _check_device(args.device)

    selected_models = args.model or list(MODEL_SPECS)
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    output_path = Path(args.output)
    payload = {
        "protocol": {
            "device_requested": args.device,
            "field_aware_dense": True,
            "dense_risk_gate": False,
            "adaptive_fusion": False,
            "buying_dense": False,
            "weights_are_dense_rrf_fraction": True,
            "dataset": args.dataset,
            "sample_count": len(samples),
            "hf_home": os.environ.get("HF_HOME"),
            "dense_cache": os.environ.get("TWOMEOW_DENSE_CACHE_DIR"),
            "reproducible_environment": _REPRODUCIBLE_ENV,
        },
        "rows": [],
    }

    for model_key in selected_models:
        spec = MODEL_SPECS[model_key]
        config = {
            "use_dense": True,
            "use_dense_risk_gate": False,
            "use_adaptive_fusion": False,
            "browsing_weights": [],
            "use_field_aware_dense": True,
            "dense_model": spec["model"],
            "dense_batch_size": spec["batch_size"],
            "dense_max_seq_length": 256,
            "dense_device": args.device,
            "dense_query_prefix": spec["query_prefix"],
            "dense_document_prefix": spec["document_prefix"],
        }
        agent = Agent(args.catalog, config)
        build_started = perf_counter()
        dense = agent.retriever._get_dense()
        build_seconds = perf_counter() - build_started
        if dense is None:
            raise RuntimeError(f"failed to initialize {model_key}")
        if args.device != "auto" and not dense.device.startswith(args.device):
            raise RuntimeError(
                f"{model_key} requested {args.device}, actual device was {dense.device}"
            )
        embeddings = dense.identity_embeddings
        dimension = int(embeddings.shape[1]) if embeddings is not None else None

        for dense_weight in args.weights:
            agent.retriever.bm25_base = 1.0 - dense_weight
            agent.retriever.dense_base = dense_weight
            agent.retriever.browsing_weights = ()
            indexed_dense = agent.retriever.dense
            if dense_weight == 0.0:
                agent.retriever.dense = _EmptyDense()
            started = perf_counter()
            result = evaluate(
                agent,
                samples,
                catalog_ids,
                categories,
                products,
            )
            agent.retriever.dense = indexed_dense
            row = {
                "model_key": model_key,
                "model": spec["model"],
                "device": dense.device,
                "dimension": dimension,
                "batch_size": spec["batch_size"],
                "max_seq_length": dense.max_seq_length,
                "query_prefix": spec["query_prefix"],
                "document_prefix": spec["document_prefix"],
                "bm25_weight": round(1.0 - dense_weight, 6),
                "dense_weight": round(dense_weight, 6),
                "rrf_empty_dense_control": dense_weight == 0.0,
                "index_load_or_build_seconds": round(build_seconds, 3),
                "evaluation_seconds": round(perf_counter() - started, 3),
                **_summary(result),
            }
            payload["rows"].append(row)
            _write_output(output_path, payload)
            print(json.dumps(row, indent=2))
        del dense
        del agent
        gc.collect()
        if args.device == "mps":
            import torch
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
