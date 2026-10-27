# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU format/dispatch tests; actual NVFP4 numerics and Graph need a GPU gate."""

from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.native_kv import cache_allocation
from xllm.python.attention.native_profile import NativeFlashInferConfig
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.executor import _create_attention_backend


def _module(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, MagicMock]:
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__version__ = "0.7.2.post1"
    page = types.ModuleType("flashinfer.page")
    writer = MagicMock()
    page.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping = writer
    flashinfer.page = page
    decode = types.ModuleType("flashinfer.decode")
    decode.fast_decode_plan = MagicMock()
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.page", page)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", decode)
    module = importlib.import_module("xllm.python.attention.flashinfer_nvfp4")
    # Explicitly bind the fixture when another test already imported the module.
    monkeypatch.setattr(module, "flashinfer", flashinfer)
    return module, writer


def _caches(dim: int = 256) -> list[LayerCache]:
    shape = (2, 4, 2, dim // 2)
    sf_shape = (*shape[:-1], dim // 16)
    return [
        LayerCache(None, None, conv=torch.full((2, 8, 3), 3.0), ssm=torch.full((2, 4, 8, 8), 5.0)),
        LayerCache(
            torch.zeros(shape, dtype=torch.uint8),
            torch.zeros(shape, dtype=torch.uint8),
            key_scale=torch.zeros(sf_shape, dtype=torch.uint8),
            value_scale=torch.zeros(sf_shape, dtype=torch.uint8),
        ),
    ]


def _backend(module: types.ModuleType, caches: list[LayerCache]) -> object:
    backend = object.__new__(module.FlashInferNVFP4Backend)
    backend.head_dim, backend.num_heads, backend.num_kv_heads = 256, 12, 2
    backend.dtype = torch.bfloat16
    backend._decode_workspace = torch.zeros(1, dtype=torch.uint8)
    backend._decode_impl = "flashinfer"
    backend._kv_caches, backend._graph_decode_wrappers = [], {}
    backend._fixed_scales = (0.125, 0.0625)
    backend._global_scales = tuple(torch.tensor([s]) for s in backend._fixed_scales)
    backend._nvfp4_pages, backend._nvfp4_scales = {}, {}
    backend.bind_kv_caches(caches)
    return backend


def test_NVFP4_actual_resident_bytes_include_block_scales_and_separate_GDN() -> None:
    caches = _caches()
    report = json.loads(cache_allocation(caches, "nvfp4"))
    assert report["full_attention_layer_ids"] == [1]
    assert report["layers"][0]["bytes_per_token"] == 2 * 2 * (128 + 16)
    assert report["full_attention_resident_bytes"] == 8 * 576
    assert report["linear_state_resident_bytes"] == (2 * 8 * 3 + 2 * 4 * 8 * 8) * 4
    assert report["total_resident_bytes"] == report["resident_bytes"] + report["linear_state_resident_bytes"]


@pytest.mark.parametrize("slot", ["key_scale", "value_scale"])
@pytest.mark.parametrize("bad", ["missing", "FP32", "per_head", "noncontiguous"])
def test_rejects_incompatible_NVFP4_scale_layout_before_binding(
    monkeypatch: pytest.MonkeyPatch, slot: str, bad: str
) -> None:
    module, _ = _module(monkeypatch)
    caches = _caches()
    tensor = getattr(caches[1], slot)
    replacement = {
        "missing": None,
        "FP32": tensor.float(),
        "per_head": tensor[..., 0],
        "noncontiguous": tensor.transpose(0, 1),
    }[bad]
    caches[1] = replace(caches[1], **{slot: replacement})
    with pytest.raises(ValueError):
        _backend(module, caches)
    assert torch.all(caches[0].ssm == 5)


def test_direct_writer_receives_resident_packed_pages_scale_views_and_persistent_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, writer = _module(monkeypatch)
    caches = _caches()
    backend = _backend(module, caches)
    key = torch.ones(3, 2, 256, dtype=torch.bfloat16)
    value = torch.full_like(key, 2)
    slots = torch.tensor([3, -1, 7], dtype=torch.int32)
    backend.write_cache(key, value, slots, 1)
    args, kwargs = writer.call_args
    assert args[0] is key and args[1] is value and args[2] is slots
    assert args[3] == (caches[1].key, caches[1].value)
    assert args[4][0].dtype == torch.float8_e4m3fn
    assert args[4][0].data_ptr() == caches[1].key_scale.data_ptr()
    assert args[4][1].data_ptr() == caches[1].value_scale.data_ptr()
    assert args[5] is backend._global_scales[0] and args[6] is backend._global_scales[1]
    assert kwargs == {"kv_layout": "NHD"}


@pytest.mark.parametrize("prefill", [False, True])
def test_FA2_attention_consumes_compressed_resident_pages_and_scales_directly(
    monkeypatch: pytest.MonkeyPatch, prefill: bool
) -> None:
    module, _ = _module(monkeypatch)
    caches = _caches()
    backend = _backend(module, caches)
    backend._metadata = SimpleNamespace(is_prefill=prefill, is_chunked_prefill=False)
    query = torch.zeros(3, 12, 256, dtype=torch.bfloat16)
    wrapper = MagicMock()
    wrapper.run.return_value = query
    backend._prefill_paged_wrapper = backend._active_decode_wrapper = wrapper
    result = backend.execute_attention(query, SimpleNamespace(layer_id=1))
    assert result.shape == (3, 12 * 256)
    args, kwargs = wrapper.run.call_args
    assert args[0].dtype == torch.bfloat16
    assert args[1][0] is caches[1].key and args[1][1] is caches[1].value
    assert kwargs["kv_cache_sf"] is backend._nvfp4_scales[1]
    assert kwargs["k_scale"] == 0.125 and kwargs["v_scale"] == 0.0625


def test_explicit_native_configuration_has_no_implicit_FA3_or_XQA_NVFP4_fallback() -> None:
    assert NativeFlashInferConfig().resolve(torch.uint8)["planner"] == "fa2"
    with pytest.raises(ValueError):
        NativeFlashInferConfig("fa3").resolve(torch.uint8)
    with pytest.raises(ValueError):
        NativeFlashInferConfig("fa2", "xqa").resolve(torch.uint8)
    assert NativeFlashInferConfig("fa3", "xqa").resolve(torch.bfloat16)["query_fp8"] is False


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 0, -1, 1e-50, 1e50])
def test_NVFP4_rejects_invalid_global_scales_before_allocating(monkeypatch: pytest.MonkeyPatch, scale: float) -> None:
    module, _ = _module(monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    with pytest.raises(ValueError, match="positive normal FP32"):
        module.FlashInferNVFP4Backend(12, 2, 256, 256**-0.5, 0, torch.device("cuda"), torch.bfloat16, scale, 1.0)


def test_configured_NVFP4_factory_needs_no_experiment_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = importlib.import_module("xllm.python.model_executor.executor")
    monkeypatch.setattr(executor.current_platform, "is_cuda", lambda: True)
    for name in ("XLLM_QUANTIZED_BACKEND", "XLLM_FLASHINFER_NATIVE_KV", "XLLM_FP8_K_SCALE", "XLLM_FP8_V_SCALE"):
        monkeypatch.delenv(name, raising=False)
    module = types.ModuleType("xllm.python.attention.flashinfer_nvfp4")
    module.FlashInferNVFP4Backend = MagicMock()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = {
        "model_type": "qwen3_5_text",
        "full_attention_interval": 4,
        "kv_cache_dtype": "nvfp4",
        "kv_cache_backend": "flashinfer",
        "kv_cache_k_scale": 0.25,
        "kv_cache_v_scale": 0.125,
        "python_graph_backend": "cudagraphs",
    }
    result = _create_attention_backend(
        Attention(12, 2, 256, 256**-0.5, 0, layer_id=3), torch.device("cuda"), torch.bfloat16, config
    )
    assert result is module.FlashInferNVFP4Backend.return_value
    assert module.FlashInferNVFP4Backend.call_args.kwargs["key_scale"] == 0.25
    assert module.FlashInferNVFP4Backend.call_args.kwargs["value_scale"] == 0.125


def test_explicit_native_config_refuses_other_platforms_and_NVFP4_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = importlib.import_module("xllm.python.model_executor.executor")
    layer = Attention(12, 2, 256, 256**-0.5, 0, layer_id=3)
    monkeypatch.setattr(executor.current_platform, "is_cuda", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        _create_attention_backend(layer, torch.device("cpu"), torch.bfloat16, {"kv_cache_backend": "flashinfer"})
    monkeypatch.setattr(executor.current_platform, "is_cuda", lambda: True)
    with pytest.raises(ValueError, match="NVFP4 requires native FlashInfer"):
        _create_attention_backend(
            layer, torch.device("cuda"), torch.bfloat16, {"kv_cache_dtype": "nvfp4", "kv_cache_backend": "reference"}
        )
