# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""CPU policy/dispatch checks for explicit modern native production candidates."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from xllm.python.attention.native_profile import NativeFlashInferConfig, resolve_native_profile


@pytest.mark.parametrize("maximum", [65536, 131072, 262144])
def test_native_XQA_declares_new_model_context_without_legacy_32K_limit(maximum: int) -> None:
    from xllm.python.attention.xqa_decode import validate_xqa_lengths

    config = NativeFlashInferConfig("fa3", "xqa", max_context_tokens=maximum)
    assert config.resolve(torch.bfloat16)["decode_impl"] == "xqa"
    validate_xqa_lengths([0, maximum // 64], [64], 64, config.max_context_tokens)
    with pytest.raises(ValueError, match="exceeds"):
        validate_xqa_lengths([0, maximum // 64 + 1], [1], 64, config.max_context_tokens)


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5, 2**31])
def test_native_context_declaration_rejects_invalid_token_bounds(maximum: object) -> None:
    with pytest.raises(ValueError, match="positive int32"):
        NativeFlashInferConfig(max_context_tokens=maximum).resolve(torch.bfloat16)


def test_historical_default_and_unmodified_default_are_preserved() -> None:
    assert resolve_native_profile({}, True, True)["planner"] == "fa2"
    assert resolve_native_profile({}, False, False)["planner"] == "auto"


def test_bf16_fa3_prefill_xqa_decode_and_its_real_operator_admission() -> None:
    from tools.check_native_fa3_operator import _operator_profile

    environment = {"XLLM_FLASHINFER_NATIVE_PLANNER": "fa3", "XLLM_FLASHINFER_NATIVE_DECODE": "xqa"}
    profile = resolve_native_profile(environment, True, False)
    assert profile["planner"] == "fa3" and profile["decode_impl"] == "xqa"
    assert profile["query_fp8"] is False
    assert _operator_profile("bf16", environment) == profile
    # The unsupported BF16-Q/FP8-KV FA3 prefill remains refused before execution.
    with pytest.raises(ValueError):
        _operator_profile("fp8_e4m3", environment)
    with pytest.raises(ValueError):
        _operator_profile("bf16", {"XLLM_FLASHINFER_NATIVE_DECODE": "xqa"})


def test_bf16_xqa_dispatch_keeps_native_bf16_pages_and_unit_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_fi = types.ModuleType("flashinfer")
    fake_fi.__version__ = "0.6.18.post1"
    fake_decode = types.ModuleType("flashinfer.decode")
    fake_decode.fast_decode_plan = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flashinfer", fake_fi)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", fake_decode)
    from xllm.python.attention.backend import LayerCache

    module = importlib.import_module("xllm.python.attention.flashinfer")
    backend = module.FlashInferBackend.__new__(module.FlashInferBackend)
    backend.num_heads, backend.head_dim, backend.scale = 2, 8, 8 ** -.5
    backend._decode_impl = "xqa"
    backend._metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False)
    cache = LayerCache(torch.zeros((1, 128, 1, 8), dtype=torch.bfloat16),
                       torch.zeros((1, 128, 1, 8), dtype=torch.bfloat16))
    backend._kv_caches = [cache]
    backend._decode_workspace = torch.zeros(1, dtype=torch.uint8)

    class Plan:
        def run(self, query: torch.Tensor, pages: tuple, workspace: torch.Tensor, *scales: float) -> torch.Tensor:
            assert query.dtype == torch.bfloat16
            assert pages[0] is cache.key and pages[1] is cache.value
            assert workspace is backend._decode_workspace
            assert scales == (8 ** -.5, 1.0, 1.0)
            return query.clone()

    backend._active_xqa_plan = Plan()
    query = torch.ones(3, 2, 8, dtype=torch.bfloat16)
    layer = SimpleNamespace(layer_id=0, num_heads=2, head_dim=8)
    torch.testing.assert_close(backend.execute_attention(query, layer), query.view(3, 16))


def test_fa3_bf16_and_explicit_fp8_query_are_supported_profiles() -> None:
    assert resolve_native_profile({"XLLM_FLASHINFER_NATIVE_PLANNER": "fa3"}, True, False)["query_fp8"] is False
    profile = resolve_native_profile({"XLLM_FLASHINFER_NATIVE_PLANNER": "fa3", "XLLM_FLASHINFER_FP8_QUERY": "1",
                                      "XLLM_FLASHINFER_FP8_Q_SCALE": "0.5"}, True, True)
    assert profile["query_fp8"] is True and profile["query_scale"] == .5
    assert profile["scope"] == "production_fa3"


@pytest.mark.parametrize("native,fp8,environment", [
    (False, True, {"XLLM_FLASHINFER_NATIVE_PLANNER": "fa3"}),
    (True, True, {"XLLM_FLASHINFER_NATIVE_PLANNER": "auto"}),
    (True, True, {"XLLM_FLASHINFER_NATIVE_PLANNER": "fa3"}),
    (True, False, {"XLLM_FLASHINFER_NATIVE_PLANNER": "fa3", "XLLM_FLASHINFER_FP8_QUERY": "1"}),
    (True, True, {"XLLM_FLASHINFER_FP8_QUERY": "1"}),
    (True, True, {"XLLM_FLASHINFER_FP8_QUERY": "yes"}),
    (True, True, {"XLLM_FLASHINFER_FP8_Q_SCALE": "1"}),
])
def test_unsupported_profiles_refuse_instead_of_falling_back(native: bool, fp8: bool, environment: dict) -> None:
    with pytest.raises(ValueError):
        resolve_native_profile(environment, native, fp8)


@pytest.mark.parametrize("scale", ["nan", "inf", "0", "-1", "1e-50", "1e50"])
def test_query_scale_is_a_finite_normal_positive_fp32_scalar(scale: str) -> None:
    with pytest.raises(ValueError):
        resolve_native_profile({"XLLM_FLASHINFER_NATIVE_PLANNER": "fa3", "XLLM_FLASHINFER_FP8_QUERY": "1",
                                "XLLM_FLASHINFER_FP8_Q_SCALE": scale}, True, True)


def test_fa3_fp8_dispatch_converts_query_passes_scale_and_returns_model_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real xLLM dispatch with CPU tensors; the CUDA conversion and FI kernel are explicit fixtures.
    fake_fi = types.ModuleType("flashinfer")
    fake_fi.__version__ = "0.6.18.post1"
    fake_decode = types.ModuleType("flashinfer.decode")
    fake_decode.fast_decode_plan = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flashinfer", fake_fi)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", fake_decode)
    # Existing tests may have previously imported the native backend with real/fake FI.
    module = importlib.import_module("xllm.python.attention.flashinfer_fp8")
    backend = module.FlashInferFP8Backend.__new__(module.FlashInferFP8Backend)
    backend.num_heads, backend.head_dim = 2, 8
    backend._fixed_scales = (.125, .0625)
    backend._fp8_query_enabled, backend._fp8_query_scale = True, .5
    backend._decode_impl = "flashinfer"
    backend._metadata = SimpleNamespace(is_prefill=True, is_chunked_prefill=False)
    backend._fp8_caches = [(torch.zeros(1), torch.zeros(1))]
    calls = []

    def _convert(query: torch.Tensor, scale: float) -> torch.Tensor:
        calls.append((query.clone(), scale))
        return (query / scale).to(torch.float8_e4m3fn)

    class Wrapper:
        def run(self, query: torch.Tensor, pages: tuple, **kwargs: Any) -> torch.Tensor:
            assert query.dtype == torch.float8_e4m3fn
            assert kwargs == {"q_scale": .5, "k_scale": .125, "v_scale": .0625}
            return query.to(torch.bfloat16) * .5

    monkeypatch.setattr(module, "quantize_query_fp8", _convert)
    backend._prefill_paged_wrapper = Wrapper()
    query = torch.ones(3, 2, 8, dtype=torch.bfloat16)
    output = backend.execute_attention(query, SimpleNamespace(layer_id=0))
    assert len(calls) == 1
    assert output.dtype == torch.bfloat16 and output.shape == (3, 16)
    torch.testing.assert_close(output, query.view(3, 16))


def test_operator_acceptance_reference_uses_the_causal_context_offset() -> None:
    from tools.check_native_fa3_operator import _reference
    query = torch.zeros(2, 3, 2, 8)
    key = torch.zeros(2, 5, 1, 8)
    value = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1).expand(2, 5, 1, 8)
    result = _reference(query, key, value, 3).view(2, 3, 2, 8)
    expected = torch.tensor([1.0, 1.5, 2.0]).view(1, 3, 1, 1).expand(2, 3, 2, 8)
    torch.testing.assert_close(result, expected)


def test_operator_replan_cannot_mutate_the_reference_or_later_padded_page_mapping() -> None:
    from tools.check_native_fa3_operator import _paging_metadata
    pages = torch.tensor([[2, 0, 4], [3, 1, 5]], dtype=torch.int32)
    reference = pages.clone()
    metadata = _paging_metadata(pages, torch.zeros(2, dtype=torch.int32), 257, 1, 128)
    metadata.paged_kv_indices[:4].copy_(pages[:, :2].flatten())
    torch.testing.assert_close(pages, reference)
    assert pages[:, :2].flatten().tolist() == [2, 0, 3, 1]
    assert metadata.paged_kv_indices.data_ptr() != pages.data_ptr()


def test_xqa_dispatch_passes_bf16_query_and_resident_alias_without_external_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_fi = types.ModuleType("flashinfer")
    fake_fi.__version__ = "0.6.18.post1"
    fake_decode = types.ModuleType("flashinfer.decode")
    fake_decode.fast_decode_plan = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flashinfer", fake_fi)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", fake_decode)
    module = importlib.import_module("xllm.python.attention.flashinfer_fp8")
    backend = module.FlashInferFP8Backend.__new__(module.FlashInferFP8Backend)
    backend.num_heads, backend.head_dim, backend.scale = 2, 8, 8 ** -.5
    backend._decode_impl = "xqa"
    backend._metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False)
    backend._fp8_caches = [(torch.zeros(1), torch.zeros(1))]
    backend._fixed_scales = (.125, .0625)
    backend._decode_workspace = torch.zeros(1, dtype=torch.uint8)
    calls = []

    class Plan:
        def run(self, query: torch.Tensor, pages: tuple, workspace: torch.Tensor, *scales: float) -> torch.Tensor:
            assert query.dtype == torch.bfloat16
            assert pages is backend._fp8_caches[0]
            assert workspace is backend._decode_workspace
            calls.append(scales)
            return query.clone()

    def forbidden(*args: object) -> object:
        raise AssertionError("XQA must not call the separate FA3 query converter")

    backend._active_xqa_plan = Plan()
    monkeypatch.setattr(module, "quantize_query_fp8", forbidden)
    query = torch.ones(3, 2, 8, dtype=torch.bfloat16)
    assert backend.execute_attention(query, SimpleNamespace(layer_id=0)).shape == (3, 16)
    assert calls == [(8 ** -.5, .125, .0625)]
