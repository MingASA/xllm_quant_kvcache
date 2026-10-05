# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the FlashInfer attention backend."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("FlashInfer tests require CUDA", allow_module_level=True)
pytest.importorskip("flashinfer", reason="FlashInfer is not installed")

# conftest.py stands in for xllm.python, whose import would bind the active
# platform's kernel package and reach for operators from the C++ binary.
from xllm.python.attention.flashinfer import _should_use_tensor_core_decode


def test_tensor_core_decode_for_large_gqa_groups():
    assert _should_use_tensor_core_decode(torch.bfloat16, 24, 4)
    assert _should_use_tensor_core_decode(torch.float16, 32, 8)


def test_cuda_core_decode_for_small_gqa_groups_or_float32():
    assert not _should_use_tensor_core_decode(torch.bfloat16, 8, 4)
    assert not _should_use_tensor_core_decode(torch.float32, 24, 4)


@pytest.mark.parametrize("cache_dtype", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("queries", [1, 16, 129])
def test_native_fp8_paged_attention(cache_dtype, queries):
    """Actual FP8 pages, shuffled slots, partial tails, and scalar scaling."""
    from tools.benchmark_kv_cache import _attention_metadata, _dense
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.flashinfer_fp8 import FlashInferFP8Backend
    from xllm.python.attention.quantized import KVCacheCodec

    torch.manual_seed(17)
    device = torch.device("cuda")
    length, page_size, dim, batch = 129, 128, 128, 2
    heads, kv_heads = 12, 2
    shape = (4, page_size, kv_heads, dim)
    cache = LayerCache(
        torch.zeros(shape, device=device, dtype=torch.uint8),
        torch.zeros(shape, device=device, dtype=torch.uint8),
        key_scale=torch.ones(shape[:-1], device=device),
        value_scale=torch.ones(shape[:-1], device=device),
    )
    backend = FlashInferFP8Backend(
        heads,
        kv_heads,
        dim,
        dim**-0.5,
        -1,
        device,
        torch.bfloat16,
        cache_dtype,
        key_scale=0.5,
        value_scale=2.0,
    )
    backend.bind_kv_caches([cache])
    pages = torch.tensor([[2, 0], [3, 1]], device=device, dtype=torch.int32)
    positions = torch.arange(length, device=device)
    slots = (pages[:, positions // page_size] * page_size + positions % page_size).flatten()
    k = torch.randn(batch, length, kv_heads, dim, device=device, dtype=torch.bfloat16) * 0.2
    v = torch.randn_like(k)
    q = torch.randn(batch, queries, heads, dim, device=device, dtype=torch.bfloat16)
    backend.write_cache(k.flatten(0, 1), v.flatten(0, 1), slots, 0)
    # Poison unwritten page tails: native attention must honor last_page_len.
    for payload in (cache.key, cache.value):
        payload[pages[:, -1].long(), 1:] = 255
    for scales in (cache.key_scale, cache.value_scale):
        scales[pages[:, -1].long(), 1:] = float("nan")
    # Padding must leave bytes and scale metadata untouched.
    before = cache.key.clone()
    backend.write_cache(k[0, :1], v[0, :1], torch.tensor([-1], device=device), 0)
    assert torch.equal(cache.key, before)
    assert backend._fp8_caches[0][0].data_ptr() == cache.key.data_ptr()
    metadata = _attention_metadata(pages, slots.view(batch, length)[:, -queries:].flatten(), length, queries, page_size)
    # Exercise initial prefill as well as chunked prefill and decode.
    if queries == length:
        metadata.is_prefill = True
        metadata.is_chunked_prefill = False
    backend.prepare(metadata)
    layer = SimpleNamespace(layer_id=0, num_heads=heads, num_kv_heads=kv_heads, head_dim=dim)
    actual = backend.execute(
        q.flatten(0, 1),
        k[:, -queries:].flatten(0, 1),
        v[:, -queries:].flatten(0, 1),
        layer,
    )
    codec = KVCacheCodec(cache_dtype, dim)
    decoded = [
        codec.decode(payload, scale)[pages.long()].reshape(batch, -1, kv_heads, dim)[:, :length]
        for payload, scale in ((cache.key, cache.key_scale), (cache.value, cache.value_scale))
    ]
    expected = _dense(q.float(), *decoded)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.005)
    assert torch.all(cache.key_scale.flatten(0, 1)[slots.long()] == 0.5)
    assert torch.all(cache.value_scale.flatten(0, 1)[slots.long()] == 2.0)
    with pytest.raises(ValueError, match="eager"):
        backend.prepare(metadata, graph_mode=True)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_native_fp8_rejects_invalid_scalar_scale(scale):
    from xllm.python.attention.flashinfer_fp8 import FlashInferFP8Backend

    with pytest.raises(ValueError, match="finite and positive"):
        FlashInferFP8Backend(12, 2, 128, 128**-0.5, -1, torch.device("cuda"), torch.bfloat16, "fp8", scale)
