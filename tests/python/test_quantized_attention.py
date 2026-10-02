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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("queries", (1, 16))
@pytest.mark.parametrize("length", (19, 1025))
@pytest.mark.parametrize("window", (-1, 3))
def test_bf16_same_framework_paged_attention(queries: int, length: int, window: int) -> None:
    from xllm.python.attention.quantized_triton import quantized_paged_attention, write_bf16_kv

    torch.manual_seed(2026)
    page_size, dim = 4, 9
    key = torch.randn(length, 2, dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query = torch.randn(queries, 4, dim, device="cuda", dtype=torch.bfloat16)
    blocks = (length + page_size - 1) // page_size
    pages = torch.randperm(blocks, device="cuda", dtype=torch.int32)[None]
    positions = torch.arange(length, device="cuda")
    slots = pages[0, positions // page_size] * page_size + positions % page_size
    cache_key = torch.zeros(blocks, page_size, 2, dim, device="cuda", dtype=torch.bfloat16)
    cache_value = torch.zeros_like(cache_key)
    write_bf16_kv(key, value, slots, cache_key, cache_value)
    actual = quantized_paged_attention(
        query,
        cache_key,
        cache_value,
        cache_key,
        cache_value,
        pages,
        torch.zeros(queries, device="cuda", dtype=torch.int32),
        torch.arange(length - queries, length, device="cuda", dtype=torch.int32),
        torch.tensor([length], device="cuda", dtype=torch.int32),
        "bf16",
        dim**-0.5,
        window,
        True,
        torch.tensor([0, queries], device="cuda", dtype=torch.int32),
        queries,
    )
    torch.testing.assert_close(actual.flatten(1), _dense_attention(query, key, value, window), rtol=0.02, atol=0.005)


def _assert_prefill_gqa_group_and_tail_matches_dense(
    cache_dtype: str, gqa_group: int, query_tokens: int, window: int
) -> None:
    from xllm.python.attention.quantized_triton import quantized_paged_attention, write_bf16_kv

    torch.manual_seed(880 + gqa_group * 31 + query_tokens * 3 + window)
    page_size, dim, kv_heads = 8, 32, 2
    query_heads = kv_heads * gqa_group
    context_length = query_tokens + 7
    num_blocks = (context_length + page_size - 1) // page_size
    codec = KVCacheCodec("int8", dim) if cache_dtype == "int8" else None
    storage_dtype = codec.storage_dtype if codec is not None else torch.bfloat16
    storage_dim = codec.storage_dim if codec is not None else dim
    cache_shape = (num_blocks, page_size, kv_heads, storage_dim)
    cache_key = torch.zeros(cache_shape, dtype=storage_dtype, device="cuda")
    cache_value = torch.zeros_like(cache_key)
    key_scale = torch.ones(cache_shape[:-1], dtype=torch.float32, device="cuda")
    value_scale = torch.ones_like(key_scale)

    key = torch.randn(context_length, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query = torch.randn(query_tokens, query_heads, dim, device="cuda", dtype=torch.bfloat16)
    pages = torch.randperm(num_blocks, device="cuda", dtype=torch.int32)[None]
    positions = torch.arange(context_length, device="cuda")
    slots = pages[0, positions // page_size].to(torch.int64) * page_size + positions % page_size
    if codec is None:
        write_bf16_kv(key, value, slots, cache_key, cache_value)
        decoded_key, decoded_value = key, value
        attention_format = "bf16"
    else:
        cache = LayerCache(key=cache_key, value=cache_value, key_scale=key_scale, value_scale=value_scale)
        write_quantized_kv(cache, key, value, slots, codec)
        decoded_key = codec.decode(*codec.encode(key))
        decoded_value = codec.decode(*codec.encode(value))
        attention_format = "int8"

    actual = quantized_paged_attention(
        query,
        cache_key,
        cache_value,
        key_scale,
        value_scale,
        pages,
        torch.zeros(query_tokens, device="cuda", dtype=torch.int32),
        torch.arange(context_length - query_tokens, context_length, device="cuda", dtype=torch.int32),
        torch.tensor([context_length], device="cuda", dtype=torch.int32),
        attention_format,
        dim**-0.5,
        window,
        True,
        torch.tensor([0, query_tokens], device="cuda", dtype=torch.int32),
        query_tokens,
    )
    expected = _dense_attention(query, decoded_key, decoded_value, window)
    torch.testing.assert_close(actual.flatten(1), expected, rtol=0.02, atol=0.005)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("cache_dtype", ("bf16", "int8"))
@pytest.mark.parametrize("gqa_group", (1, 2, 4, 6))
@pytest.mark.parametrize("query_tokens", (3, 16, 19))
@pytest.mark.parametrize("window", (-1, 5))
def test_prefill_gqa_group_and_tail_matches_dense(
    cache_dtype: str, gqa_group: int, query_tokens: int, window: int
) -> None:
    _assert_prefill_gqa_group_and_tail_matches_dense(cache_dtype, gqa_group, query_tokens, window)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("cache_dtype", ("bf16", "int8"))
def test_prefill_gqa_group8_and_group3(cache_dtype: str) -> None:
    _assert_prefill_gqa_group_and_tail_matches_dense(cache_dtype, 8, 19, 3)
    _assert_prefill_gqa_group_and_tail_matches_dense(cache_dtype, 3, 19, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("page_size", (16, 128))
@pytest.mark.parametrize("window", (-1, 5))
def test_int8_mixed_gqa6_prefill_and_decode(page_size: int, window: int) -> None:
    torch.manual_seed(611)
    dim, context_length = 128, 145
    query_lengths = [65, 1, 7]
    blocks_per_sequence = (context_length + page_size - 1) // page_size
    num_blocks = blocks_per_sequence * len(query_lengths)
    codec = KVCacheCodec("int8", dim)
    shape = (num_blocks, page_size, 2, dim)
    cache = LayerCache(
        key=torch.zeros(shape, dtype=torch.int8, device="cuda"),
        value=torch.zeros(shape, dtype=torch.int8, device="cuda"),
        # Unwritten tail slots must not contaminate attention with NaNs.
        key_scale=torch.full(shape[:-1], float("nan"), device="cuda"),
        value_scale=torch.full(shape[:-1], float("nan"), device="cuda"),
    )
    pages = torch.randperm(num_blocks).reshape(len(query_lengths), blocks_per_sequence).tolist()
    backend = QuantizedPagedAttentionBackend("int8", dim, 2)
    backend.bind_kv_caches([cache])
    queries, expected = [], []
    for row, query_length in zip(pages, query_lengths):
        key = torch.randn(context_length, 2, dim, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        query = torch.randn(query_length, 12, dim, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(context_length, device="cuda")
        device_pages = torch.tensor(row, device="cuda")
        slots = device_pages[positions // page_size] * page_size + positions % page_size
        write_quantized_kv(cache, key, value, slots, codec)
        queries.append(query)
        expected.append(
            _dense_attention(query, codec.decode(*codec.encode(key)), codec.decode(*codec.encode(value)), window)
        )
    backend.prepare(_metadata(pages, query_lengths, [context_length] * len(query_lengths), page_size, "cuda"))
    actual = backend.execute_attention(torch.cat(queries), _layer(dim, window, num_heads=12))
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0.02, atol=0.005)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("window", (-1, 3))
def test_int8_split_decode_with_empty_partitions(window: int) -> None:
    torch.manual_seed(51)
    codec = KVCacheCodec("int8", 9)
    cache = _cache(codec, "cuda", page_size=128)
    backend = QuantizedPagedAttentionBackend("int8", 9, 2)
    backend.bind_kv_caches([cache])
    key, value = torch.randn(2, 17, 2, 9, device="cuda")
    query = torch.randn(1, 4, 9, device="cuda")
    slots = torch.arange(17, device="cuda")
    write_quantized_kv(cache, key, value, slots, codec)
    # Table capacity triggers splitting, while only its first page is valid.
    backend.prepare(_metadata([[0, 1, 2, 3, 4, 5, 6, 7]], [1], [17], 128, "cuda"))
    actual = backend.execute_attention(query, _layer(9, window))
    expected = _dense_attention(query, codec.decode(*codec.encode(key)), codec.decode(*codec.encode(value)), window)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_int8_prefill_split_window_with_empty_partitions() -> None:
    torch.manual_seed(52)
    codec = KVCacheCodec("int8", 16)
    cache = _cache(codec, "cuda", page_size=128)
    backend = QuantizedPagedAttentionBackend("int8", 16, 2)
    backend.bind_kv_caches([cache])
    key, value = torch.randn(2, 17, 2, 16, device="cuda")
    query = torch.randn(3, 4, 16, device="cuda", dtype=torch.bfloat16)
    write_quantized_kv(cache, key, value, torch.arange(17, device="cuda"), codec)
    backend.prepare(_metadata([list(range(8))], [3], [17], 128, "cuda"))

    actual = backend.execute(query, key[14:], value[14:], _layer(16, window=3))
    decoded_key = codec.decode(*codec.encode(key))
    decoded_value = codec.decode(*codec.encode(value))
    expected = _dense_attention(query, decoded_key, decoded_value, window=3)

    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)


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


def _cumulative_lengths(lengths: tuple[int, ...]) -> list[int]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return offsets


def _execute_batch_with_lengths(
    query_lengths: tuple[int, ...],
    context_lengths: tuple[int, ...],
    pages: tuple[tuple[int, ...], ...],
    *,
    cumulative: bool,
) -> torch.Tensor:
    torch.manual_seed(20261001)
    page_size, head_dim = 4, 8
    total_query_tokens = sum(query_lengths)
    codec = KVCacheCodec("int8", head_dim)
    cache = _cache(codec, page_size=page_size)
    backend = QuantizedPagedAttentionBackend("int8", head_dim, 2)
    backend.bind_kv_caches([cache])
    metadata = _metadata(
        [list(row) for row in pages],
        list(query_lengths),
        list(context_lengths),
        page_size,
        "cpu",
    )
    if cumulative:
        metadata.q_seq_lens_host = torch.tensor(_cumulative_lengths(query_lengths), dtype=torch.int32)
        metadata.kv_seq_lens_host_values = _cumulative_lengths(context_lengths)
    backend.prepare(metadata)

    key, value = torch.randn(2, total_query_tokens, 2, head_dim)
    query = torch.randn(total_query_tokens, 4, head_dim)
    return backend.execute(query, key, value, _layer(head_dim))


@pytest.mark.parametrize(
    ("query_lengths", "context_lengths", "pages"),
    [
        ((3,), (3,), ((0, 1, 2),)),
        ((2, 3), (2, 3), ((0, 1, 2), (3, 4, 5))),
        ((2, 1), (5, 7), ((0, 1, 2), (3, 4, 5))),
        ((1, 1), (6, 9), ((0, 1, 2), (3, 4, 5))),
    ],
    ids=("batch1-prefill", "batch2-prefill", "batch2-chunked-prefill", "batch2-decode"),
)
def test_cumulative_host_lengths_match_per_sequence_lengths(
    query_lengths: tuple[int, ...],
    context_lengths: tuple[int, ...],
    pages: tuple[tuple[int, ...], ...],
) -> None:
    per_sequence_output = _execute_batch_with_lengths(query_lengths, context_lengths, pages, cumulative=False)
    cumulative_output = _execute_batch_with_lengths(query_lengths, context_lengths, pages, cumulative=True)

    torch.testing.assert_close(cumulative_output, per_sequence_output, rtol=0, atol=0)


@pytest.mark.parametrize("invalid_field", ("query", "context"))
def test_invalid_cumulative_host_length_count_is_rejected(invalid_field: str) -> None:
    codec = KVCacheCodec("int8", 8)
    backend = QuantizedPagedAttentionBackend("int8", 8, 2)
    backend.bind_kv_caches([_cache(codec, page_size=4)])
    metadata = _metadata([[0]], [1], [1], 4, "cpu")
    if invalid_field == "query":
        metadata.q_seq_lens_host = torch.tensor([0, 1, 2], dtype=torch.int32)
    else:
        metadata.kv_seq_lens_host_values = [0, 1, 2]

    with pytest.raises(ValueError, match="batch metadata does not match"):
        backend.prepare(metadata)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_int8_writer_matches_bf16_codec_on_rounding_ties() -> None:
    codec = KVCacheCodec("int8", 8)
    cache = _cache(codec, "cuda")
    key_head = torch.tensor([[127.0, 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5]], device="cuda", dtype=torch.bfloat16)
    value_head = torch.tensor([[127.0, -0.5, -1.5, -2.5, -3.5, 0.5, 1.5, 2.5]], device="cuda", dtype=torch.bfloat16)
    key, value = key_head[:, None, :].expand(-1, 2, -1), value_head[:, None, :].expand(-1, 2, -1)
    slots = torch.tensor([3], device="cuda")

    write_quantized_kv(cache, key, value, slots, codec)
    expected_key, expected_key_scale = codec.encode(key)
    expected_value, expected_value_scale = codec.encode(value)

    torch.cuda.synchronize()
    assert torch.equal(cache.key.flatten(0, 1)[3], expected_key[0])
    assert torch.equal(cache.key_scale.flatten(0, 1)[3], expected_key_scale[0])
    assert torch.equal(cache.value.flatten(0, 1)[3], expected_value[0])
    assert torch.equal(cache.value_scale.flatten(0, 1)[3], expected_value_scale[0])


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
@pytest.mark.parametrize("device", _DEVICES)
def test_shared_prefix_and_independent_batch_tails(cache_dtype: str, device: str) -> None:
    torch.manual_seed(123)
    codec = KVCacheCodec(cache_dtype, 8)
    cache = _cache(codec, device)
    backend = QuantizedPagedAttentionBackend(cache_dtype, 8, 2)
    backend.bind_kv_caches([cache])
    key, value = torch.randn(2, 7, 2, 8, device=device)
    query = torch.randn(7, 4, 8, device=device)
    backend.prepare(_metadata([[3]], [4], [4], 4, device))
    backend.execute(query[:4], key[:4], value[:4], _layer(8))
    prefix_data = cache.key[3].clone()
    prefix_scales = cache.key_scale[3].clone()
    backend.prepare(_metadata([[3, 1], [3, 6]], [1, 2], [5, 6], 4, device))
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
