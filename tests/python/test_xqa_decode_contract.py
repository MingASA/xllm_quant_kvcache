# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU admission/metadata failures; real XQA and graph correctness need SM90."""

import pytest
import torch
import types
import sys

from tools.native_kv_quality_contract import native_environment
from xllm.python.attention.native_profile import resolve_native_profile
from xllm.python.attention.xqa_decode import XQADecodePlan, validate_xqa_lengths


def test_real_partial_pages_and_empty_native_graph_padding() -> None:
    validate_xqa_lengths([0, 3, 5, 5], [1, 2, 1], 128, 32768)
    validate_xqa_lengths([0, 256], [128], 128, 32768)


@pytest.mark.parametrize("indptr,last", [
    ([1, 2], [1]), ([0, 1, 2], [1]), ([0, -1], [1]),
    ([0, 1], [0]), ([0, 1], [129]), ([0, 257], [1]),
    ([0, 0], [2]), ([False, 1], [1]), ([0, 1.5], [1]),
])
def test_bad_page_plan_refused_before_device_update(indptr: list, last: list) -> None:
    with pytest.raises(ValueError):
        validate_xqa_lengths(indptr, last, 128, 32768)


def test_explicit_xqa_keeps_prefill_query_bf16_and_declares_its_profile() -> None:
    env = native_environment("fp8_e4m3", .125, .0625, planner="fa2", decode="xqa")
    profile = resolve_native_profile(env, True, True)
    assert profile["decode_impl"] == "xqa" and profile["scope"] == "production_xqa"
    assert profile["query_fp8"] is False
    assert "XLLM_FLASHINFER_FP8_QUERY" not in env
    # This says nothing about the SM90 kernel's internal FP8 conversion.


@pytest.mark.parametrize("env,native", [
    ({"XLLM_FLASHINFER_NATIVE_DECODE": "xqa"}, False),
    ({"XLLM_FLASHINFER_NATIVE_DECODE": "xqa", "XLLM_FLASHINFER_NATIVE_PLANNER": "fa3"}, True),
    ({"XLLM_FLASHINFER_NATIVE_DECODE": "xqa", "XLLM_FLASHINFER_FP8_QUERY": "1"}, True),
    ({"XLLM_FLASHINFER_NATIVE_DECODE": "other"}, True),
])
def test_unsupported_xqa_selection_never_falls_back(env: dict, native: bool) -> None:
    with pytest.raises(ValueError):
        resolve_native_profile(env, native, True)


def test_xqa_raw_pointer_boundary_materializes_strided_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = XQADecodePlan.__new__(XQADecodePlan)
    plan._metadata_addresses, plan._batch_size, plan._maximum = (1, 2, 3), 2, 32768
    plan._table = torch.zeros((2, 256), dtype=torch.int32)
    plan._lengths = torch.ones(2, dtype=torch.uint32)
    query = torch.randn(2, 56, 128, dtype=torch.bfloat16)[:, :40]
    assert not query.is_contiguous()
    pages = (torch.zeros(1), torch.zeros(1))
    decode = types.ModuleType("flashinfer.decode")

    def fixture(query_arg: torch.Tensor, cache_arg: tuple, *args: object, **kwargs: object) -> torch.Tensor:
        assert query_arg.is_contiguous() and cache_arg is pages
        assert kwargs["enable_pdl"] is True
        torch.testing.assert_close(query_arg, query)
        return query_arg

    decode.xqa_batch_decode_with_kv_cache = fixture
    monkeypatch.setitem(sys.modules, "flashinfer.decode", decode)
    plan.run(query, pages, torch.zeros(1, dtype=torch.uint8), 128 ** -.5, .125, .0625)
