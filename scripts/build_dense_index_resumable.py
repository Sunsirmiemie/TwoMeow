#!/usr/bin/env python
"""Build field-aware Dense caches in resumable chunks.

This is intended for large MPS indexes. Each completed chunk is written as a
standalone .npy file, so an interrupted run resumes at the first missing chunk
instead of re-encoding the whole catalogue.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


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

from src.ranking.features import product_attribute_text, product_identity_text
from src.retrieval.dense import _catalog_hash


MODEL_SPECS = {
    "all-mpnet-base-v2": {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "batch_size": 64,
        "query_prefix": "",
        "document_prefix": "",
    },
    "e5-small-v2": {
        "model": "intfloat/e5-small-v2",
        "batch_size": 64,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
}


def _cache_path(
    cache_dir: Path,
    catalog: Path,
    model: str,
    query_prefix: str,
    document_prefix: str,
    max_seq_length: int,
) -> Path:
    safe_model = model.replace("/", "_")
    version = "field-v2"
    if max_seq_length != 256:
        version = f"{version}-seq-{max_seq_length}"
    if query_prefix or document_prefix:
        prompt_key = hashlib.sha1(
            f"{query_prefix}\0{document_prefix}".encode()
        ).hexdigest()[:8]
        version = f"{version}-prompt-{prompt_key}"
    return cache_dir / f"{safe_model}_{version}_{_catalog_hash(str(catalog))}.npz"


def _load_catalog(catalog: Path, document_prefix: str):
    asins: list[str] = []
    identity_texts: list[str] = []
    attribute_texts: list[str] = []
    with catalog.open("r", encoding="utf-8") as source:
        for line in source:
            product = json.loads(line)
            asins.append(product.get("parent_asin", ""))
            identity_texts.append(
                f"{document_prefix}{product_identity_text(product)}"
            )
            attribute_texts.append(
                f"{document_prefix}{product_attribute_text(product)}"
            )
    return asins, identity_texts, attribute_texts


def _encode_field(
    model,
    texts: list[str],
    field: str,
    parts_dir: Path,
    chunk_size: int,
    batch_size: int,
    device: str,
) -> list[Path]:
    part_paths: list[Path] = []
    total_chunks = (len(texts) + chunk_size - 1) // chunk_size
    for chunk_index, start in enumerate(range(0, len(texts), chunk_size)):
        stop = min(start + chunk_size, len(texts))
        part_path = parts_dir / f"{field}-{chunk_index:05d}.npy"
        expected_rows = stop - start
        if part_path.exists():
            existing = np.load(part_path, mmap_mode="r")
            if existing.ndim == 2 and existing.shape[0] == expected_rows:
                print(
                    f"[{field}] {chunk_index + 1}/{total_chunks} resume "
                    f"{tuple(existing.shape)}",
                    flush=True,
                )
                part_paths.append(part_path)
                continue
            raise RuntimeError(f"invalid checkpoint shape: {part_path}")

        started = perf_counter()
        encoded = model.encode(
            texts[start:stop],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        np.save(part_path, encoded)
        print(
            f"[{field}] {chunk_index + 1}/{total_chunks} {tuple(encoded.shape)} "
            f"{perf_counter() - started:.2f}s",
            flush=True,
        )
        part_paths.append(part_path)
        del encoded
        gc.collect()
        if device == "mps":
            import torch
            torch.mps.synchronize()
            torch.mps.empty_cache()
    return part_paths


def _combine(parts: list[Path]) -> np.ndarray:
    return np.concatenate([np.load(path, mmap_mode="r") for path in parts], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()

    if args.chunk_size <= 0 or (args.batch_size is not None and args.batch_size <= 0):
        parser.error("chunk and batch sizes must be positive")

    if args.device == "mps":
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")

    from sentence_transformers import SentenceTransformer

    catalog = Path(args.catalog).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[args.model]
    batch_size = args.batch_size or spec["batch_size"]
    target = _cache_path(
        cache_dir,
        catalog,
        spec["model"],
        spec["query_prefix"],
        spec["document_prefix"],
        args.max_seq_length,
    )
    if target.exists():
        print(f"cache already exists: {target}", flush=True)
        return

    parts_dir = Path(f"{target}.parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    asins, identity_texts, attribute_texts = _load_catalog(
        catalog, spec["document_prefix"]
    )
    manifest = {
        "catalog": str(catalog),
        "catalog_hash": _catalog_hash(str(catalog)),
        "model_key": args.model,
        "model": spec["model"],
        "device_requested": args.device,
        "query_prefix": spec["query_prefix"],
        "document_prefix": spec["document_prefix"],
        "max_seq_length": args.max_seq_length,
        "chunk_size": args.chunk_size,
        "row_count": len(asins),
        "target": str(target),
        "reproducible_environment": _REPRODUCIBLE_ENV,
    }
    manifest_path = parts_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_protocol = {
            key: value for key, value in previous.items()
            if key not in {"batch_size", "batch_size_history"}
        }
        if previous_protocol != manifest:
            raise RuntimeError(f"checkpoint manifest mismatch: {manifest_path}")
        batch_history = previous.get("batch_size_history")
        if batch_history is None:
            batch_history = [previous.get("batch_size", batch_size)]
    else:
        batch_history = []
    if batch_size not in batch_history:
        batch_history.append(batch_size)
    manifest["batch_size_history"] = batch_history
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    model = SentenceTransformer(spec["model"], device=args.device)
    model.max_seq_length = args.max_seq_length
    actual_device = str(model.device)
    if args.device != "auto" and not actual_device.startswith(args.device):
        raise RuntimeError(
            f"requested device {args.device}, but model uses {actual_device}"
        )
    print(
        f"model={spec['model']} device={actual_device} rows={len(asins)} "
        f"batch={batch_size} chunk={args.chunk_size}",
        flush=True,
    )

    identity_parts = _encode_field(
        model,
        identity_texts,
        "identity",
        parts_dir,
        args.chunk_size,
        batch_size,
        args.device,
    )
    attribute_parts = _encode_field(
        model,
        attribute_texts,
        "attribute",
        parts_dir,
        args.chunk_size,
        batch_size,
        args.device,
    )
    identity_embeddings = _combine(identity_parts)
    attribute_embeddings = _combine(attribute_parts)
    if len(identity_embeddings) != len(asins) or len(attribute_embeddings) != len(asins):
        raise RuntimeError("combined embedding row count does not match catalogue")

    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        asins=np.asarray(asins),
        identity_embeddings=identity_embeddings,
        attribute_embeddings=attribute_embeddings,
    )
    os.replace(temporary, target)
    print(
        f"completed cache={target} identity={identity_embeddings.shape} "
        f"attribute={attribute_embeddings.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
