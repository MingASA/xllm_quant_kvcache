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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from xllm.python.attention.backend import LayerCache, normalize_layer_caches
from xllm.python.attention.quantized import KVCacheCodec, QuantizedPagedAttentionBackend, write_quantized_kv

_DTYPES = ("int8", "fp8", "fp8_e4m3", "fp8_e5m2", "int4")
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _cache(codec: KVCacheCodec, device: str = "cpu", page_size: int = 4) -> LayerCache:
    shape = (8, page_size, 2, codec.storage_dim)
    return LayerCache(
        key=torch.zeros(shape, dtype=codec.storage_dtype, device=device),
        value=torch.zeros(shape, dtype=codec.storage_dtype, device=device),
        key_scale=torch.ones(shape[:-1], dtype=torch.float32, device=device),
        value_scale=torch.ones(shape[:-1], dtype=torch.float32, device=device),
    )


def _metadata(
    pages: list[list[int]], query_lengths: list[int], context_lengths: list[int], page_size: int, device: str
) -> SimpleNamespace:
    slots = [
        row[pos // page_size] * page_size + pos % page_size
        for row, q_len, kv_len in zip(pages, query_lengths, context_lengths)
        for pos in range(kv_len - q_len, kv_len)
    ]
    return SimpleNamespace(
        block_table=torch.tensor(pages, dtype=torch.int32, device=device),
        q_seq_lens_host=torch.tensor(query_lengths, dtype=torch.int32),
        kv_seq_lens_host_values=context_lengths,
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
    )


def _layer(head_dim: int, window: int = -1, num_heads: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=2,
        layer_id=0,
        scale=head_dim**-0.5,
        sliding_window=window,
        causal=True,
    )


def _dense_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, window: int = -1) -> torch.Tensor:
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1).transpose(0, 1).float()
    value = value.repeat_interleave(groups, dim=1).transpose(0, 1).float()
    positions = torch.arange(query.shape[0], device=query.device) + key.shape[1] - query.shape[0]
    key_positions = torch.arange(key.shape[1], device=query.device)
    mask = key_positions[None] <= positions[:, None]
    if window > 0:
        mask &= key_positions[None] > positions[:, None] - window
    return (
        F.scaled_dot_product_attention(query.transpose(0, 1).float(), key, value, attn_mask=mask)
        .transpose(0, 1)
        .to(query.dtype)
        .flatten(1)
    )


@pytest.mark.parametrize("cache_dtype", _DTYPES)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("head_dim", (1, 9, 128))
def test_codec_roundtrip_and_zero(cache_dtype: str, device: str, head_dim: int) -> None:
    torch.manual_seed(17)
    codec = KVCacheCodec(cache_dtype, head_dim)
    values = torch.randn(7, 2, head_dim, device=device)
    values[0].zero_()
    values[1] *= 100
    payload, scales = codec.encode(values)
    restored = codec.decode(payload, scales)
    assert payload.element_size() == 1
    assert payload.shape[-1] == ((head_dim + 1) // 2 if cache_dtype == "int4" else head_dim)
    assert torch.equal(restored[0], values[0])
    assert torch.equal(scales[0], torch.ones_like(scales[0]))
    bound = {"int8": 0.5 / 127, "int4": 0.5 / 7, "fp8_e5m2": 0.125}.get(cache_dtype, 0.063)
    assert torch.all((values - restored).abs() <= values.abs().amax(-1, keepdim=True) * bound + 1e-5)


def test_int4_nibble_order_sign_and_odd_padding() -> None:
    codec = KVCacheCodec("int4", 5)
    values = torch.tensor([[-7.0, -1.0, 0.0, 1.0, 7.0]])
    payload, scale = codec.encode(values)
    assert payload.tolist() == [[0xF9, 0x10, 0x07]]
    torch.testing.assert_close(codec.decode(payload, scale), values)


@pytest.mark.parametrize(
    "dtype,bound,one,max_bits", [("fp8_e4m3", 448.0, 0x38, 0x7E), ("fp8_e5m2", 57344.0, 0x3C, 0x7B)]
)
def test_fp8_stores_float_encoding(dtype: str, bound: float, one: int, max_bits: int) -> None:
    codec = KVCacheCodec(dtype, 5)
    values = torch.tensor([[0.0, 1.0, -2.0, bound, -bound]])
    payload, scale = codec.encode(values)
    assert payload.tolist() == [[0, one, 0xC0, max_bits, max_bits | 0x80]]
    torch.testing.assert_close(codec.decode(payload, scale), values)


@pytest.mark.parametrize("cache_dtype", _DTYPES)
def test_storage_budget_includes_scales(cache_dtype: str) -> None:
    codec = KVCacheCodec(cache_dtype, 128)
    cache = _cache(codec)
    actual = sum(t.numel() * t.element_size() for t in (cache.key, cache.value, cache.key_scale, cache.value_scale))
    expected = 8 * 4 * 2 * 2 * (codec.storage_dim + 4)
    assert actual == expected


@pytest.mark.parametrize("cache_dtype", _DTYPES)
@pytest.mark.parametrize("device", _DEVICES)
def test_scatter_padding_and_reused_slot(cache_dtype: str, device: str) -> None:
    codec = KVCacheCodec(cache_dtype, 9)
    cache = _cache(codec, device)
    key = torch.randn(3, 2, 9, device=device)
    slots = torch.tensor([7, -1, 2], device=device)
    write_quantized_kv(cache, key, -key, slots, codec)
    encoded, scales = codec.encode(key[[2, 0]])
    assert torch.equal(cache.key.flatten(0, 1)[[2, 7]], encoded)
    assert torch.equal(cache.key_scale.flatten(0, 1)[[2, 7]], scales)
    assert torch.count_nonzero(cache.key.flatten(0, 1)[-1]) == 0
    key[0].zero_()
    write_quantized_kv(cache, key, key, slots, codec)
    assert torch.count_nonzero(cache.key.flatten(0, 1)[7]) == 0
    assert torch.equal(cache.key_scale.flatten(0, 1)[7], torch.ones(2, device=device))


@pytest.mark.parametrize("cache_dtype", _DTYPES)
@pytest.mark.parametrize("window", (-1, 3))
@pytest.mark.parametrize("device", _DEVICES)
def test_prefill_chunked_prefill_and_decode(cache_dtype: str, window: int, device: str) -> None:
    torch.manual_seed(42)
    codec = KVCacheCodec(cache_dtype, 9)
    cache = _cache(codec, device)
    backend = QuantizedPagedAttentionBackend(cache_dtype, 9, 2)
    backend.bind_kv_caches([cache])
    key, value = torch.randn(2, 11, 2, 9, device=device)
    query = torch.randn(11, 4, 9, device=device)
    decoded_key = codec.decode(*codec.encode(key))
    decoded_value = codec.decode(*codec.encode(value))
    start = 0
    for end in (5, 8, 9, 11):
        backend.prepare(_metadata([[4, 1, 5]], [end - start], [end], 4, device))
        output = backend.execute(query[start:end], key[start:end], value[start:end], _layer(9, window))
        expected = _dense_attention(query[start:end], decoded_key[:end], decoded_value[:end], window)
        torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
        start = end


@pytest.mark.parametrize("cache_dtype", _DTYPES)
def test_shared_prefix_and_independent_batch_tails(cache_dtype: str) -> None:
    torch.manual_seed(123)
    codec = KVCacheCodec(cache_dtype, 8)
    cache = _cache(codec)
    backend = QuantizedPagedAttentionBackend(cache_dtype, 8, 2)
    backend.bind_kv_caches([cache])
    key, value = torch.randn(2, 7, 2, 8)
    query = torch.randn(7, 4, 8)
    backend.prepare(_metadata([[3]], [4], [4], 4, "cpu"))
    backend.execute(query[:4], key[:4], value[:4], _layer(8))
    prefix_data = cache.key[3].clone()
    prefix_scales = cache.key_scale[3].clone()
    backend.prepare(_metadata([[3, 1], [3, 6]], [1, 2], [5, 6], 4, "cpu"))
    output = backend.execute(query[4:], key[4:], value[4:], _layer(8))
    for begin, end in ((4, 5), (5, 7)):
        indices = list(range(4)) + list(range(begin, end))
        expected = _dense_attention(
            query[begin:end], codec.decode(*codec.encode(key[indices])), codec.decode(*codec.encode(value[indices]))
        )
        torch.testing.assert_close(output[begin - 4 : end - 4], expected, atol=2e-6, rtol=2e-5)
    assert torch.equal(cache.key[3], prefix_data)
    assert torch.equal(cache.key_scale[3], prefix_scales)


@pytest.mark.parametrize("query_dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_query_tiling_matches_dense_attention(query_dtype: torch.dtype) -> None:
    torch.manual_seed(18)
    codec = KVCacheCodec("int8", 8)
    cache = _cache(codec, page_size=16)
    backend = QuantizedPagedAttentionBackend("int8", 8, 2)
    backend.bind_kv_caches([cache])
    query = torch.randn(70, 2, 8).to(query_dtype)
    key, value = torch.randn(2, 70, 2, 8).to(query_dtype)
    backend.prepare(_metadata([[6, 1, 5, 2, 7]], [70], [70], 16, "cpu"))
    output = backend.execute(query, key, value, _layer(8, num_heads=2))
    expected = _dense_attention(query, codec.decode(*codec.encode(key)), codec.decode(*codec.encode(value)))
    torch.testing.assert_close(output, expected, atol=1e-3 if query_dtype != torch.float32 else 2e-6, rtol=1e-3)


@pytest.mark.parametrize("bad_slots", ([0, 0], [-2, 0], [32, 0]))
def test_bad_writes_fail_before_cache_mutation(bad_slots: list[int]) -> None:
    codec = KVCacheCodec("int8", 8)
    cache = _cache(codec)
    values = torch.randn(2, 2, 8)
    with pytest.raises(ValueError):
        write_quantized_kv(cache, values, values, torch.tensor(bad_slots), codec)
    assert torch.count_nonzero(cache.key) == 0


@pytest.mark.parametrize("invalid_value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_values_fail_before_cache_mutation(invalid_value: float) -> None:
    codec = KVCacheCodec("fp8", 8)
    cache = _cache(codec)
    key = torch.ones(1, 2, 8)
    value = torch.full_like(key, invalid_value)
    with pytest.raises(ValueError, match="finite"):
        write_quantized_kv(cache, key, value, torch.tensor([0]), codec)
    assert torch.count_nonzero(cache.key) == 0


def test_metadata_graph_and_missing_scale_rejected() -> None:
    codec = KVCacheCodec("int8", 8)
    backend = QuantizedPagedAttentionBackend("int8", 8, 2)
    cache = _cache(codec)
    with pytest.raises(ValueError, match="scales"):
        backend.bind_kv_caches([LayerCache(cache.key, cache.value)])
    backend.bind_kv_caches([cache])
    metadata = _metadata([[2]], [1], [1], 4, "cpu")
    with pytest.raises(ValueError, match="eager"):
        backend.prepare(metadata, graph_mode=True)
    metadata.slot_mapping[0] = 0
    with pytest.raises(ValueError, match="appended"):
        backend.prepare(metadata)
    assert torch.count_nonzero(cache.key) == 0


def test_layer_cache_tuple_compatibility() -> None:
    cache = _cache(KVCacheCodec("int8", 8))
    legacy = normalize_layer_caches([(cache.key, cache.value)])[0]
    assert legacy.key_scale is None
    extended = (cache.key, cache.value) + (None,) * 9 + (cache.key_scale, cache.value_scale)
    normalized = normalize_layer_caches([extended])[0]
    assert normalized.key_scale is cache.key_scale
    assert normalized.value_scale is cache.value_scale


@pytest.mark.parametrize("dtype", ("auto", "float8", "INT8", "int3"))
def test_unknown_quantized_dtype_rejected(dtype: str) -> None:
    with pytest.raises(ValueError, match="dtype"):
        KVCacheCodec(dtype, 8)
