# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU tests for Qwen2.5 checkpoint profiles and shard provenance."""

import json
from pathlib import Path

import pytest

from tools.qwen25_experiment_model import checkpoint_provenance, read_profile


@pytest.mark.parametrize(
    ("size", "hidden_size", "heads", "kv_heads", "layers", "group_size"),
    [
        ("7b", 3584, 28, 4, 28, 7),
        ("14b", 5120, 40, 8, 48, 5),
    ],
)
def test_qwen25_model_profiles(
    tmp_path: Path, size: str, hidden_size: int, heads: int, kv_heads: int, layers: int, group_size: int
) -> None:
    model = tmp_path / size
    model.mkdir()
    config = {
        "model_type": "qwen2",
        "hidden_size": hidden_size,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "num_hidden_layers": layers,
        "max_position_embeddings": 32768,
    }
    (model / "config.json").write_text(json.dumps(config))

    profile = read_profile(model)

    assert profile["size"] == size
    assert profile["num_heads"] == heads
    assert profile["num_kv_heads"] == kv_heads
    assert profile["num_heads"] // profile["num_kv_heads"] == group_size
    assert profile["head_dim"] == 128


def test_checkpoint_provenance_tracks_shard_content(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    config = {
        "model_type": "qwen2",
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "num_hidden_layers": 28,
        "max_position_embeddings": 32768,
    }
    (model / "config.json").write_text(json.dumps(config))
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for name, contents in zip(shard_names, (b"first-shard", b"second-shard")):
        (model / name).write_bytes(contents)
    index = {"weight_map": {"layer.0.weight": shard_names[0], "layer.1.weight": shard_names[1]}}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))

    before = checkpoint_provenance(model)
    (model / shard_names[1]).write_bytes(b"changed-second-shard")
    after = checkpoint_provenance(model)

    assert before["weights"]["identity_scheme"] == "sorted_shard_manifest_sha256"
    assert before["weights"]["sha256"] != after["weights"]["sha256"]
    assert before["weights"]["files"][shard_names[1]]["sha256"] != after["weights"]["files"][shard_names[1]]["sha256"]
