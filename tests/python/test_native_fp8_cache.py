# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resident-page ownership on CPU; writer/query numerics and replay on SM90."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from .test_native_kv_prefill_dispatch_cpu import _load_method


def test_fp8_aliases_preserve_owner_and_whole_page_copy() -> None:
    bind = _load_method("bind_kv_caches", globals_={"torch": torch, "logger": Mock(), "json": json})
    owner_k = torch.zeros((3, 128, 4, 256), dtype=torch.uint8)
    owner_v = torch.zeros_like(owner_k)
    linear = SimpleNamespace(key=None, value=None)
    caches = [linear, SimpleNamespace(key=owner_k, value=owner_v)]
    backend = SimpleNamespace(head_dim=256, num_kv_heads=4, _decode_workspace=torch.empty(1),
                              native_kv_runtime_info=lambda: {})
    bind(backend, caches)
    key, value = backend._pages[1]
    assert key.dtype == value.dtype == torch.float8_e4m3fn
    assert key.data_ptr() == owner_k.data_ptr()
    assert value.data_ptr() == owner_v.data_ptr()
    assert backend._kv_caches is caches
    assert 0 not in backend._pages
    owner_k[0].fill_(17)
    owner_v[0].fill_(29)
    owner_k[2].copy_(owner_k[0])
    owner_v[2].copy_(owner_v[0])
    assert torch.equal(key[2].view(torch.uint8), owner_k[0])
    assert torch.equal(value[2].view(torch.uint8), owner_v[0])
    assert not owner_k[1].any()


@pytest.mark.parametrize("kind", ["dtype", "shape", "strides", "missing_value"])
def test_fp8_binding_rejects_wrong_physical_storage(kind: str) -> None:
    bind = _load_method("bind_kv_caches", globals_={"torch": torch})
    key = torch.zeros((2, 128, 4, 256), dtype=torch.uint8)
    if kind == "dtype":
        key = key.to(torch.bfloat16)
    elif kind == "shape":
        key = key[..., :128].contiguous()
    elif kind == "strides":
        key = key.transpose(0, 1)
    value = None if kind == "missing_value" else key.clone()
    backend = SimpleNamespace(head_dim=256, num_kv_heads=4, _decode_workspace=torch.empty(1))
    with pytest.raises(ValueError, match="contiguous resident byte pages"):
        bind(backend, [SimpleNamespace(key=key, value=value)])


@pytest.mark.parametrize("cache_dtype", ["nvfp4", "mxfp4", "fp8_e5m2"])
def test_pr_backend_rejects_other_formats(cache_dtype: str) -> None:
    constructor = _load_method("__init__")
    with pytest.raises(ValueError, match="fp8_e4m3"):
        constructor(object(), 24, 4, 256, 0.0625, 0, torch.device("cuda"),
                    torch.bfloat16, cache_dtype)


def test_fp8_rejects_unqualified_virtual_pages() -> None:
    import math

    constructor = _load_method("__init__", globals_={"torch": torch, "math": math})
    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), pytest.raises(
        ValueError, match="VMM is not qualified"
    ):
        constructor(object(), 24, 4, 256, 0.0625, 0, torch.device("cuda"),
                    torch.bfloat16, "fp8_e4m3", virtual_memory=True)


_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.mark.skipif(not _SM90, reason="Native FP8 integration targets SM90")
@pytest.mark.parametrize("graph", [False, True])
def test_fp8_writer_matches_independent_cast_and_replays(graph: bool) -> None:
    from xllm.python.attention.native_kv_writer import write_fp8_kv

    key = torch.randn((5, 4, 512), device="cuda", dtype=torch.bfloat16)[..., ::2]
    value = torch.randn_like(key) * 100
    slots = torch.tensor([0, 127, 128, 255, -1], dtype=torch.int32, device="cuda")
    owners = tuple(torch.zeros((3, 128, 4, 256), dtype=torch.uint8, device="cuda") for _ in range(2))
    pages = tuple(tensor.view(torch.float8_e4m3fn) for tensor in owners)
    scales = (0.125, 0.5)
    write_fp8_kv(key, value, slots, pages, scales)
    torch.cuda.synchronize()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            write_fp8_kv(key, value, slots, pages, scales)
        key.mul_(2)
        value.mul_(0.5)
        captured.replay()
    for source, owner, scale in zip((key, value), owners, scales):
        expected = torch.zeros_like(owner).view(-1, 4, 256)
        quantized = (source.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        expected[slots[:-1].long()] = quantized[:-1]
        assert torch.equal(owner, expected.view_as(owner))


@pytest.mark.skipif(not _SM90, reason="Native FP8 integration targets SM90")
def test_query_conversion_preserves_strided_values() -> None:
    from xllm.python.attention.query_quantization import quantize_query_fp8

    query = torch.randn((17, 24, 512), dtype=torch.bfloat16, device="cuda")[..., ::2]
    actual = quantize_query_fp8(query, 1.0)
    expected = query.float().clamp(-448, 448).to(torch.float8_e4m3fn)
    assert actual.is_contiguous()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
