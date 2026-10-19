# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Checkpoint identity for reproducible Qwen2.5 KV experiments, including shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SHAPES = {"1.5b": (1536, 12, 2, 28), "7b": (3584, 28, 4, 28), "14b": (5120, 40, 8, 48)}


def read_profile(model: Path) -> dict[str, Any]:
    config = json.loads((model / "config.json").read_text())
    values = tuple(
        int(config[k]) for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "num_hidden_layers")
    )
    if config.get("model_type") != "qwen2" or values not in SHAPES.values():
        raise ValueError(f"Expected Qwen2.5 1.5B/7B/14B config, got {values}")
    if config.get("use_sliding_window", False) or config.get("rope_scaling"):
        raise ValueError("The fixed experiment protocol requires full attention and unscaled RoPE")
    hidden, heads, kv_heads, layers = values
    if hidden // heads != 128 or config.get("head_dim", 128) != 128:
        raise ValueError("The mixed kernel supports head_dim=128")
    return {
        "size": next(size for size, shape in SHAPES.items() if shape == values),
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "head_dim": 128,
        "num_layers": layers,
        "max_position_embeddings": int(config["max_position_embeddings"]),
        "tie_word_embeddings": bool(config.get("tie_word_embeddings", False)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_provenance(model: Path) -> dict[str, Any]:
    profile = read_profile(model)
    index = model / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    if not names or any(Path(name).name != name for name in names):
        raise ValueError("Invalid checkpoint shard paths")
    files = {name: {"sha256": sha256_file(model / name), "size_bytes": (model / name).stat().st_size} for name in names}
    identity = (
        files[names[0]]["sha256"]
        if len(names) == 1
        else hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    )
    metadata = {
        p.name: sha256_file(p)
        for p in model.iterdir()
        if p.is_file()
        and p.name
        in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "model.safetensors.index.json",
        )
    }
    return {
        "local_model_path": str(model.resolve()),
        "profile": profile,
        "metadata_sha256": metadata,
        "weights": {
            "sha256": identity,
            "files": files,
            "identity_scheme": "file_sha256" if len(names) == 1 else "sorted_shard_manifest_sha256",
        },
    }
