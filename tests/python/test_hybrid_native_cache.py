# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU regression checks for native paged KV beside recurrent hybrid state."""

from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.native_kv import _full_attention_layers, cache_allocation
from xllm.python.attention.quantized import KVCacheCodec
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.executor import ModelExecutor, _create_attention_backend


def _hybrid_caches(quantized: bool = False) -> list[LayerCache]:
    caches = []
    for layer_id in range(4):
        if layer_id % 2 == 0:
            caches.append(LayerCache(None, None, conv=torch.full((3, 8, 3), 2.0), ssm=torch.full((3, 2, 4, 4), 3.0)))
            continue
        shape = (2, 4, 2, 8)
        caches.append(
            LayerCache(
                torch.ones(shape, dtype=torch.uint8 if quantized else torch.bfloat16),
                torch.ones(shape, dtype=torch.uint8 if quantized else torch.bfloat16),
                key_scale=torch.ones(shape[:-1]) if quantized else None,
                value_scale=torch.ones(shape[:-1]) if quantized else None,
            )
        )
    return caches


def _backend_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, types.ModuleType]:
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__version__ = "0.6.18.post1"
    decode = types.ModuleType("flashinfer.decode")
    decode.fast_decode_plan = MagicMock()
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", decode)
    return (
        importlib.import_module("xllm.python.attention.flashinfer"),
        importlib.import_module("xllm.python.attention.flashinfer_fp8"),
    )


def _backend(module: types.ModuleType, quantized: bool = False) -> object:
    backend_type = module.FlashInferFP8Backend if quantized else module.FlashInferBackend
    backend = object.__new__(backend_type)
    backend._kv_caches = []
    backend._graph_decode_wrappers = {}
    backend._decode_impl = "xqa"
    backend.dtype = torch.bfloat16
    backend.num_kv_heads = 2
    if quantized:
        backend._codec = KVCacheCodec("fp8_e4m3", 8)
        backend._cache_dtype = torch.float8_e4m3fn
        backend._fp8_caches = {}
        backend._decode_workspace = torch.zeros(1, dtype=torch.uint8)
    return backend


def test_allocation_keeps_full_layer_ids_and_accounts_for_linear_state_separately() -> None:
    caches = _hybrid_caches()
    report = json.loads(cache_allocation(caches, "bf16"))
    assert report["full_attention_layer_ids"] == [1, 3]
    assert [layer["layer_id"] for layer in report["layers"]] == [1, 3]
    assert [layer["layer_id"] for layer in report["linear_state_layers"]] == [0, 2]
    assert report["resident_bytes"] == report["full_attention_resident_bytes"] == 1024
    assert report["linear_state_resident_bytes"] == 2 * (3 * 8 * 3 + 3 * 2 * 4 * 4) * 4
    assert report["total_resident_bytes"] == 1024 + report["linear_state_resident_bytes"]
    assert report["token_capacity"] == 8


@pytest.mark.parametrize("replacement", ["missing_value", "different_pages", "different_dtype"])
def test_invalid_full_cache_is_rejected_before_binding(replacement: str) -> None:
    caches = _hybrid_caches()
    cache = caches[3]
    if replacement == "missing_value":
        caches[3] = replace(cache, value=None)
    elif replacement == "different_pages":
        caches[3] = replace(cache, key=cache.key[:1], value=cache.value[:1])
    else:
        caches[3] = replace(cache, value=cache.value.float())
    with pytest.raises(ValueError):
        _full_attention_layers(caches)


@pytest.mark.parametrize("quantized", [False, True])
def test_xqa_initializes_only_full_kv_and_rebinding_preserves_live_state(
    monkeypatch: pytest.MonkeyPatch, quantized: bool
) -> None:
    base, fp8 = _backend_modules(monkeypatch)
    backend = _backend(fp8 if quantized else base, quantized)
    caches = _hybrid_caches(quantized)
    backend.bind_kv_caches(caches)
    assert backend._kv_caches[3] is caches[3]
    assert torch.count_nonzero(caches[1].key) == 0
    assert torch.all(caches[0].conv == 2) and torch.all(caches[2].ssm == 3)
    caches[1].value.fill_(9)
    backend.bind_kv_caches(list(caches))
    assert torch.all(caches[1].value == 9)
    if quantized:
        assert set(backend._fp8_caches) == {1, 3}
        for layer_id, (key, value) in backend._fp8_caches.items():
            assert key.dtype == value.dtype == torch.float8_e4m3fn
            assert key.data_ptr() == caches[layer_id].key.data_ptr()
            assert value.data_ptr() == caches[layer_id].value.data_ptr()


@pytest.mark.parametrize("slot", ["conv", "ssm", "key_scale", "value_scale", "key"])
def test_xqa_rejects_replacement_of_any_bound_cache_slot(monkeypatch: pytest.MonkeyPatch, slot: str) -> None:
    _, fp8 = _backend_modules(monkeypatch)
    backend = _backend(fp8, True)
    caches = _hybrid_caches(True)
    backend.bind_kv_caches(caches)
    changed = list(caches)
    layer_id = 0 if slot in ("conv", "ssm") else 1
    changed[layer_id] = replace(caches[layer_id], **{slot: getattr(caches[layer_id], slot).clone()})
    with pytest.raises(RuntimeError, match="replace XQA resident storage"):
        backend.bind_kv_caches(changed)
    assert backend._kv_caches[layer_id] is caches[layer_id]


def test_executor_rejects_changed_recurrent_state_before_touching_runners() -> None:
    executor = object.__new__(ModelExecutor)
    executor.model = nn.Sequential(Attention(4, 2, 8, 8**-0.5, 0, layer_id=3))
    executor.attention_backend = MagicMock()
    executor.attention_backend.native_kv_runtime_info = None
    executor.eager_runner = MagicMock()
    executor.decode_graph_runner = None
    executor.inductor_runner = None
    executor._kv_bound = False
    caches = _hybrid_caches()
    executor.bind_kv_caches(caches)
    changed = list(caches)
    changed[0] = replace(caches[0], ssm=caches[0].ssm.clone())
    with pytest.raises(RuntimeError, match="linear state"):
        executor.bind_kv_caches(changed)
    executor.attention_backend.bind_kv_caches.assert_called_once()
    executor.eager_runner.bind_layer_caches.assert_called_once()


def test_only_native_flashinfer_admits_hybrid_quantized_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("xllm.python.attention.flashinfer_fp8")
    module.FlashInferFP8Backend = MagicMock()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    executor_module = importlib.import_module("xllm.python.model_executor.executor")
    monkeypatch.setattr(executor_module.current_platform, "is_cuda", lambda: True)
    monkeypatch.setenv("XLLM_QUANTIZED_BACKEND", "flashinfer")
    monkeypatch.delenv("XLLM_FLASHINFER_NATIVE_KV", raising=False)
    config = {"kv_cache_dtype": "fp8_e4m3", "full_attention_interval": 4}
    layer = Attention(4, 2, 8, 8**-0.5, 0, layer_id=3)
    result = _create_attention_backend(layer, torch.device("cuda"), torch.bfloat16, config)
    assert result is module.FlashInferFP8Backend.return_value
    monkeypatch.setenv("XLLM_QUANTIZED_BACKEND", "triton")
    with pytest.raises(ValueError, match="ordinary MHA/GQA"):
        _create_attention_backend(layer, torch.device("cuda"), torch.bfloat16, config)
