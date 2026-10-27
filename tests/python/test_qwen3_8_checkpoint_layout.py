# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Load the official 27B checkpoint's shapes on meta, without weight values/GPU.

This checks real CUDA-layer construction and checkpoint-name/shape mapping.
The row-weight packing boundary is a metadata-only fixture; no operator math,
RoPE correctness, GPU deployment or quality result follows from this test.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest
import torch

_PYTHON_ROOT = Path(__file__).parents[2] / "xllm/python"
_CUDA_PACKAGE = types.ModuleType("xllm.python.kernels_cuda")
_CUDA_PACKAGE.__path__ = [str(_PYTHON_ROOT / "kernels_cuda")]
sys.modules.setdefault("xllm.python.kernels_cuda", _CUDA_PACKAGE)

from xllm.python import kernels  # noqa: E402
from xllm.python.layers.attention import Attention  # noqa: E402
from xllm.python.layers.cuda.qwen3_5.decoder_layer import CudaQwen3_5DecoderLayer  # noqa: E402
from xllm.python.layers.cuda.qwen3_5.gated_delta_net import CudaQwen3_5GatedDeltaNet  # noqa: E402
from xllm.python.models import qwen3_5  # noqa: E402


class _MetaCheckpoint:
    def __init__(self, tensors: list[dict]) -> None:
        self._shapes = {row["name"]: row["shape"] for row in tensors}
        self.loaded: set[str] = set()

    def has(self, name: str) -> bool:
        return name in self._shapes

    def get_tensor(self, name: str) -> torch.Tensor:
        self.loaded.add(name)
        return torch.empty(self._shapes[name], dtype=torch.bfloat16, device="meta")


def test_official_27B_text_weights_map_to_native_hybrid_layers_on_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/qwen3_8_27b_checkpoint_shapes.json").read_text())
    config = dict(fixture["text_config"])
    rope = config["rope_parameters"]
    config.update(
        dtype="bfloat16",
        device="meta",
        tp_size=1,
        world_size=1,
        moe_tp_size=1,
        rope_theta=rope["rope_theta"],
        partial_rotary_factor=rope["partial_rotary_factor"],
    )
    monkeypatch.setattr(qwen3_5, "get_qwen3_5_decoder_layer_class", lambda _device: CudaQwen3_5DecoderLayer)
    monkeypatch.setattr(kernels, "resolve_gdn_prefill_backend", lambda: "flashinfer", raising=False)
    monkeypatch.setattr(kernels, "prepare_row_parallel_weight", lambda weight: (weight, False), raising=False)
    model = qwen3_5.Qwen3_5ForCausalLM(config)
    state = _MetaCheckpoint(fixture["tensors"])
    model.load_weights([state], tp_rank=0, tp_size=1)
    expected = {
        row["name"]
        for row in fixture["tensors"]
        if row["name"].startswith("model.language_model.") or row["name"] == "lm_head.weight"
    }
    assert len(expected) == 851 and state.loaded == expected
    assert all(tensor.device.type == "meta" for tensor in model.parameters())
    full = [layer.layer_id for layer in model.modules() if isinstance(layer, Attention)]
    assert full == list(range(3, 64, 4))
    assert sum(isinstance(layer, CudaQwen3_5GatedDeltaNet) for layer in model.modules()) == 48
    assert (model.cfg.n_heads, model.cfg.n_kv_heads, model.cfg.head_dim) == (24, 4, 256)
    assert not any("visual" in name or name.startswith("mtp.") for name in state.loaded)
