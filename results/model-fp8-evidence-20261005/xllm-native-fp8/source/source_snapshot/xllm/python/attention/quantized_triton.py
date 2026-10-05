# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused Triton kernels for quantized, paged KV cache operations."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_FORMAT_INT8 = 0
_FORMAT_FP8_E4M3 = 1
_FORMAT_FP8_E5M2 = 2
_FORMAT_INT4 = 3


@triton.jit
def _scatter_bf16_kv_kernel(
    key_ptr,
    value_ptr,
    slots_ptr,
    key_cache_ptr,
    value_cache_ptr,
    num_tokens,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    cache_capacity: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head = tl.program_id(0)
    token = token_head // num_kv_heads
    head = token_head % num_kv_heads
    slot = tl.load(slots_ptr + token)
    if token < num_tokens and slot >= 0 and slot < cache_capacity:
        dims = tl.arange(0, BLOCK_D)
        input_base = (token * num_kv_heads + head) * head_dim
        output_base = (slot * num_kv_heads + head) * head_dim
        key_values = tl.load(key_ptr + input_base + dims, mask=dims < head_dim, other=0)
        value_values = tl.load(value_ptr + input_base + dims, mask=dims < head_dim, other=0)
        tl.store(key_cache_ptr + output_base + dims, key_values, mask=dims < head_dim)
        tl.store(value_cache_ptr + output_base + dims, value_values, mask=dims < head_dim)


@triton.jit
def _round_to_nearest_even(values):
    lower = tl.floor(values)
    fraction = values - lower
    lower_int = lower.to(tl.int32)
    round_up = (fraction > 0.5) | ((fraction == 0.5) & ((lower_int & 1) != 0))
    return tl.where(round_up, lower + 1.0, lower)


@triton.jit
def _quantize_and_scatter_kv_kernel(
    key_ptr,
    value_ptr,
    slots_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_scale_ptr,
    value_scale_ptr,
    num_tokens,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    storage_dim: tl.constexpr,
    cache_capacity: tl.constexpr,
    FORMAT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_STORAGE: tl.constexpr,
    FIXED_KEY_SCALE: tl.constexpr,
    FIXED_VALUE_SCALE: tl.constexpr,
):
    token_head = tl.program_id(0)
    token = token_head // num_kv_heads
    head = token_head % num_kv_heads
    slot = tl.load(slots_ptr + token)
    if token < num_tokens and slot >= 0 and slot < cache_capacity:
        dims = tl.arange(0, BLOCK_D)
        input_base = (token * num_kv_heads + head) * head_dim
        key_values = tl.load(key_ptr + input_base + dims, mask=dims < head_dim, other=0).to(tl.float32)
        value_values = tl.load(value_ptr + input_base + dims, mask=dims < head_dim, other=0).to(tl.float32)

        if FORMAT == 0:
            bound = 127.0
        elif FORMAT == 1:
            bound = 448.0
        elif FORMAT == 2:
            bound = 57344.0
        else:
            bound = 7.0

        key_max = tl.max(tl.abs(key_values), axis=0)
        value_max = tl.max(tl.abs(value_values), axis=0)
        key_scale = tl.where(key_max == 0.0, 1.0, tl.maximum(key_max / bound, 1.1754943508222875e-38))
        value_scale = tl.where(value_max == 0.0, 1.0, tl.maximum(value_max / bound, 1.1754943508222875e-38))
        if FIXED_KEY_SCALE > 0:
            key_scale = FIXED_KEY_SCALE
            value_scale = FIXED_VALUE_SCALE
        if FORMAT == 0:
            # Approximate reciprocal division can flip half-integer ties for
            # BF16 inputs. Match the codec's FP32 nearest-even division.
            key_normalized = tl.clamp(tl.div_rn(key_values, key_scale), -bound, bound)
            value_normalized = tl.clamp(tl.div_rn(value_values, value_scale), -bound, bound)
        else:
            key_normalized = tl.clamp(key_values / key_scale, -bound, bound)
            value_normalized = tl.clamp(value_values / value_scale, -bound, bound)

        output_base = (slot * num_kv_heads + head) * storage_dim
        scale_index = slot * num_kv_heads + head
        if FORMAT == 0:
            key_quantized = _round_to_nearest_even(key_normalized).to(tl.int8)
            value_quantized = _round_to_nearest_even(value_normalized).to(tl.int8)
            tl.store(key_cache_ptr + output_base + dims, key_quantized, mask=dims < head_dim)
            tl.store(value_cache_ptr + output_base + dims, value_quantized, mask=dims < head_dim)
        elif FORMAT == 1:
            key_quantized = key_normalized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            value_quantized = value_normalized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            tl.store(key_cache_ptr + output_base + dims, key_quantized, mask=dims < head_dim)
            tl.store(value_cache_ptr + output_base + dims, value_quantized, mask=dims < head_dim)
        elif FORMAT == 2:
            key_quantized = key_normalized.to(tl.float8e5).to(tl.uint8, bitcast=True)
            value_quantized = value_normalized.to(tl.float8e5).to(tl.uint8, bitcast=True)
            tl.store(key_cache_ptr + output_base + dims, key_quantized, mask=dims < head_dim)
            tl.store(value_cache_ptr + output_base + dims, value_quantized, mask=dims < head_dim)
        else:
            storage_offsets = tl.arange(0, BLOCK_STORAGE)
            first_dim = storage_offsets * 2
            second_dim = first_dim + 1
            safe_first_dim = tl.minimum(first_dim, head_dim - 1)
            safe_second_dim = tl.minimum(second_dim, head_dim - 1)
            key_first = _round_to_nearest_even(tl.gather(key_normalized, safe_first_dim, 0)).to(tl.int32)
            key_second = _round_to_nearest_even(tl.gather(key_normalized, safe_second_dim, 0)).to(tl.int32)
            value_first = _round_to_nearest_even(tl.gather(value_normalized, safe_first_dim, 0)).to(tl.int32)
            value_second = _round_to_nearest_even(tl.gather(value_normalized, safe_second_dim, 0)).to(tl.int32)
            key_first = tl.where(first_dim < head_dim, key_first, 0)
            key_second = tl.where(second_dim < head_dim, key_second, 0)
            value_first = tl.where(first_dim < head_dim, value_first, 0)
            value_second = tl.where(second_dim < head_dim, value_second, 0)
            key_packed = (key_first & 15) | ((key_second & 15) << 4)
            value_packed = (value_first & 15) | ((value_second & 15) << 4)
            tl.store(
                key_cache_ptr + output_base + storage_offsets,
                key_packed.to(tl.uint8),
                mask=storage_offsets < storage_dim,
            )
            tl.store(
                value_cache_ptr + output_base + storage_offsets,
                value_packed.to(tl.uint8),
                mask=storage_offsets < storage_dim,
            )

        tl.store(key_scale_ptr + scale_index, key_scale)
        tl.store(value_scale_ptr + scale_index, value_scale)


@triton.jit
def _decode_cache_values(
    cache_ptr,
    scale_ptr,
    block_id,
    page_offsets,
    kv_head,
    dims,
    num_kv_heads: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    storage_dim: tl.constexpr,
    FORMAT: tl.constexpr,
    valid_tokens=None,
):
    physical_slots = block_id * page_size + page_offsets
    valid_offsets = page_offsets < page_size
    if valid_tokens is not None:
        valid_offsets = valid_offsets & valid_tokens
    payload_mask = valid_offsets[:, None] & (dims[None, :] < head_dim)
    scale_offsets = physical_slots * num_kv_heads + kv_head
    if FORMAT == 4:
        scales = tl.full((page_offsets.shape[0],), 1.0, tl.float32)
    else:
        scales = tl.load(scale_ptr + scale_offsets, mask=valid_offsets, other=1.0).to(tl.float32)
    if FORMAT == 0 or FORMAT == 4:
        payload_offsets = (physical_slots[:, None] * num_kv_heads + kv_head) * storage_dim + dims[None, :]
        payload = tl.load(cache_ptr + payload_offsets, mask=payload_mask, other=0).to(tl.float32)
    elif FORMAT == 1:
        payload_offsets = (physical_slots[:, None] * num_kv_heads + kv_head) * storage_dim + dims[None, :]
        payload = (
            tl.load(cache_ptr + payload_offsets, mask=payload_mask, other=0)
            .to(tl.uint8)
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )
    elif FORMAT == 2:
        payload_offsets = (physical_slots[:, None] * num_kv_heads + kv_head) * storage_dim + dims[None, :]
        payload = (
            tl.load(cache_ptr + payload_offsets, mask=payload_mask, other=0)
            .to(tl.uint8)
            .to(tl.float8e5, bitcast=True)
            .to(tl.float32)
        )
    else:
        payload_offsets = (physical_slots[:, None] * num_kv_heads + kv_head) * storage_dim + dims[None, :] // 2
        packed = tl.load(cache_ptr + payload_offsets, mask=payload_mask, other=0).to(tl.uint8).to(tl.int32)
        nibble = tl.where(dims[None, :] % 2 == 0, packed & 15, packed >> 4)
        payload = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
    return payload * scales[:, None]


@triton.jit
def _quantized_paged_attention_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_scale_ptr,
    value_scale_ptr,
    block_table_ptr,
    query_to_sequence_ptr,
    query_positions_ptr,
    context_lengths_ptr,
    output_ptr,
    partial_lse_ptr,
    num_query_tokens,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    storage_dim: tl.constexpr,
    page_size: tl.constexpr,
    block_table_stride: tl.constexpr,
    scale: tl.constexpr,
    sliding_window: tl.constexpr,
    CAUSAL: tl.constexpr,
    FORMAT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    query_token = tl.program_id(0)
    query_head = tl.program_id(1)
    split = tl.program_id(2)
    if query_token < num_query_tokens:
        sequence = tl.load(query_to_sequence_ptr + query_token).to(tl.int32)
        query_position = tl.load(query_positions_ptr + query_token).to(tl.int32)
        context_length = tl.load(context_lengths_ptr + sequence).to(tl.int32)
        kv_head = query_head // (num_heads // num_kv_heads)
        dims = tl.arange(0, BLOCK_D)
        query_offsets = (query_token * num_heads + query_head) * head_dim + dims
        query = tl.load(query_ptr + query_offsets, mask=dims < head_dim, other=0).to(tl.float32)
        running_max = -float("inf")
        denominator = 0.0
        accumulator = tl.full((BLOCK_D,), 0.0, tl.float32)
        num_pages = tl.cdiv(context_length, page_size)

        pages_per_split = tl.cdiv(num_pages, NUM_SPLITS)
        for page_index in range(split * pages_per_split, tl.minimum((split + 1) * pages_per_split, num_pages)):
            block_id = tl.load(block_table_ptr + sequence * block_table_stride + page_index).to(tl.int32)
            for page_start in range(0, page_size, BLOCK_N):
                page_offsets = page_start + tl.arange(0, BLOCK_N)
                key_positions = page_index * page_size + page_offsets
                valid = (page_offsets < page_size) & (key_positions < context_length)
                if CAUSAL:
                    valid = valid & (key_positions <= query_position)
                if sliding_window > 0:
                    valid = valid & (key_positions > query_position - sliding_window)

                key = _decode_cache_values(
                    key_cache_ptr,
                    key_scale_ptr,
                    block_id,
                    page_offsets,
                    kv_head,
                    dims,
                    num_kv_heads,
                    page_size,
                    head_dim,
                    storage_dim,
                    FORMAT,
                )
                value = _decode_cache_values(
                    value_cache_ptr,
                    value_scale_ptr,
                    block_id,
                    page_offsets,
                    kv_head,
                    dims,
                    num_kv_heads,
                    page_size,
                    head_dim,
                    storage_dim,
                    FORMAT,
                )
                scores = tl.sum(key * query[None, :], axis=1) * scale
                scores = tl.where(valid, scores, -float("inf"))
                tile_max = tl.max(scores, axis=0)
                next_max = tl.maximum(running_max, tile_max)
                safe_max = tl.where(next_max > -float("inf"), next_max, 0.0)
                correction = tl.exp(running_max - safe_max)
                weights = tl.exp(scores - safe_max)
                accumulator = accumulator * correction + tl.sum(weights[:, None] * value, axis=0)
                denominator = denominator * correction + tl.sum(weights, axis=0)
                running_max = next_max

        result = accumulator / tl.maximum(denominator, 1e-30)
        output_offsets = ((query_token * num_heads + query_head) * NUM_SPLITS + split) * head_dim + dims
        tl.store(output_ptr + output_offsets, result, mask=dims < head_dim)
        if NUM_SPLITS > 1:
            lse = tl.where(denominator > 0, running_max + tl.log(denominator), -float("inf"))
            tl.store(partial_lse_ptr + (query_token * num_heads + query_head) * NUM_SPLITS + split, lse)


@triton.jit
def _merge_attention_splits_kernel(
    partial_ptr,
    lse_ptr,
    output_ptr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    splits = tl.arange(0, NUM_SPLITS)
    dims = tl.arange(0, BLOCK_D)
    lse = tl.load(lse_ptr + row * NUM_SPLITS + splits)
    weights = tl.exp(lse - tl.max(lse, 0))
    weights = weights / tl.sum(weights, 0)
    partial = tl.load(
        partial_ptr + (row * NUM_SPLITS + splits[:, None]) * HEAD_DIM + dims[None, :],
        mask=dims[None, :] < HEAD_DIM,
        other=0,
    )
    result = tl.sum(partial * weights[:, None], 0)
    tl.store(output_ptr + row * HEAD_DIM + dims, result, mask=dims < HEAD_DIM)


@triton.jit
def _quantized_paged_prefill_attention_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_scale_ptr,
    value_scale_ptr,
    block_table_ptr,
    query_positions_ptr,
    query_offsets_ptr,
    context_lengths_ptr,
    output_ptr,
    partial_lse_ptr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    storage_dim: tl.constexpr,
    page_size: tl.constexpr,
    block_table_stride: tl.constexpr,
    scale: tl.constexpr,
    sliding_window: tl.constexpr,
    CAUSAL: tl.constexpr,
    FORMAT: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GQA_GROUPS: tl.constexpr,
    LOW_PRECISION_DOT: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    query_tile = tl.program_id(0)
    head_group = tl.program_id(1)
    sequence = tl.program_id(2) // NUM_SPLITS
    split = tl.program_id(2) % NUM_SPLITS
    query_start = tl.load(query_offsets_ptr + sequence).to(tl.int32)
    query_end = tl.load(query_offsets_ptr + sequence + 1).to(tl.int32)
    query_length = query_end - query_start
    grouped_rows = query_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
    query_local = grouped_rows // GQA_GROUPS
    query_head = head_group * GQA_GROUPS + grouped_rows % GQA_GROUPS
    query_tokens = query_start + query_local
    query_valid = query_local < query_length
    query_positions = tl.load(query_positions_ptr + query_tokens, mask=query_valid, other=0).to(tl.int32)
    context_length = tl.load(context_lengths_ptr + sequence).to(tl.int32)
    kv_head = head_group // (num_heads // num_kv_heads // GQA_GROUPS)
    dims = tl.arange(0, BLOCK_D)
    query_offsets = (query_tokens[:, None] * num_heads + query_head[:, None]) * head_dim + dims[None, :]
    query = tl.load(
        query_ptr + query_offsets,
        mask=query_valid[:, None] & (dims[None, :] < head_dim),
        other=0,
    ).to(tl.float32)

    running_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    denominator = tl.full((BLOCK_Q,), 0.0, tl.float32)
    accumulator = tl.full((BLOCK_Q, BLOCK_D), 0.0, tl.float32)
    # A rectangular mixed-batch grid includes tiles beyond short queries.
    # Such tiles must not read the cache or perform attention work.
    has_queries = tl.sum(query_valid.to(tl.int32), axis=0) > 0
    key_end = context_length
    if CAUSAL:
        key_end = tl.minimum(key_end, tl.max(query_positions, axis=0) + 1)
    key_end = tl.where(has_queries, key_end, 0)
    key_begin = 0
    if sliding_window > 0:
        first_position = tl.min(tl.where(query_valid, query_positions, 2147483647), axis=0)
        key_begin = tl.where(has_queries, tl.maximum(first_position - sliding_window + 1, 0), 0)
    first_tile = key_begin // BLOCK_N
    num_tiles = tl.cdiv(key_end, BLOCK_N) - first_tile
    tiles_per_split = tl.cdiv(num_tiles, NUM_SPLITS)
    for tile_index in range(
        first_tile + split * tiles_per_split,
        first_tile + tl.minimum((split + 1) * tiles_per_split, num_tiles),
    ):
        key_positions = tile_index * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_keys = (key_positions >= key_begin) & (key_positions < key_end)
        page_indices = key_positions // page_size
        block_ids = tl.load(
            block_table_ptr + sequence * block_table_stride + page_indices,
            mask=valid_keys,
            other=0,
        ).to(tl.int32)
        page_offsets = key_positions % page_size
        valid = query_valid[:, None] & valid_keys[None, :]
        if CAUSAL:
            valid = valid & (key_positions[None, :] <= query_positions[:, None])
        if sliding_window > 0:
            valid = valid & (key_positions[None, :] > query_positions[:, None] - sliding_window)

        key = _decode_cache_values(
            key_cache_ptr,
            key_scale_ptr,
            block_ids,
            page_offsets,
            kv_head,
            dims,
            num_kv_heads,
            page_size,
            head_dim,
            storage_dim,
            FORMAT,
            valid_keys,
        )
        value = _decode_cache_values(
            value_cache_ptr,
            value_scale_ptr,
            block_ids,
            page_offsets,
            kv_head,
            dims,
            num_kv_heads,
            page_size,
            head_dim,
            storage_dim,
            FORMAT,
            valid_keys,
        )
        if LOW_PRECISION_DOT:
            # Preserve the FP32 decoded-cache precision with BF16 Tensor Cores.
            key_hi = key.to(tl.bfloat16)
            key_lo = (key - key_hi.to(tl.float32)).to(tl.bfloat16)
            scores = (
                tl.dot(query.to(tl.bfloat16), tl.trans(key_hi)) + tl.dot(query.to(tl.bfloat16), tl.trans(key_lo))
            ) * scale
        else:
            scores = tl.dot(query, tl.trans(key), input_precision="tf32x3") * scale
        scores = tl.where(valid, scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, tile_max)
        safe_max = tl.where(next_max > -float("inf"), next_max, 0.0)
        correction = tl.exp(running_max - safe_max)
        weights = tl.exp(scores - safe_max[:, None])
        if LOW_PRECISION_DOT:
            weight_hi = weights.to(tl.bfloat16)
            weight_lo = (weights - weight_hi.to(tl.float32)).to(tl.bfloat16)
            value_hi = value.to(tl.bfloat16)
            value_lo = (value - value_hi.to(tl.float32)).to(tl.bfloat16)
            weighted_values = tl.dot(weight_hi, value_hi) + tl.dot(weight_lo, value_hi) + tl.dot(weight_hi, value_lo)
        else:
            weighted_values = tl.dot(weights, value, input_precision="tf32x3")
        accumulator = accumulator * correction[:, None] + weighted_values
        denominator = denominator * correction + tl.sum(weights, axis=1)
        running_max = next_max

    result = accumulator / tl.maximum(denominator[:, None], 1e-30)
    output_rows = query_tokens * num_heads + query_head
    output_offsets = (output_rows[:, None] * NUM_SPLITS + split) * head_dim + dims[None, :]
    tl.store(
        output_ptr + output_offsets,
        result,
        mask=query_valid[:, None] & (dims[None, :] < head_dim),
    )
    if NUM_SPLITS > 1:
        lse = tl.where(denominator > 0, running_max + tl.log(denominator), -float("inf"))
        tl.store(partial_lse_ptr + output_rows * NUM_SPLITS + split, lse, mask=query_valid)


def _format_code(cache_dtype: str) -> int:
    formats = {
        "int8": _FORMAT_INT8,
        "fp8_e4m3": _FORMAT_FP8_E4M3,
        "fp8_e5m2": _FORMAT_FP8_E5M2,
        "int4": _FORMAT_INT4,
    }
    try:
        return formats[cache_dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported quantized KV cache dtype: {cache_dtype}") from error


def write_quantized_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    cache_dtype: str,
    head_dim: int,
    fixed_scales: tuple[float, float] | None = None,
) -> None:
    """Quantize K/V vectors and scatter payloads/scales in one GPU launch."""
    if fixed_scales is not None:
        if (
            cache_dtype not in ("fp8_e4m3", "fp8_e5m2")
            or not all(math.isfinite(scale) and scale > 0 for scale in fixed_scales)
            or len(fixed_scales) != 2
        ):
            raise ValueError("Fixed scales require FP8 and two finite positive scalar scales")
    num_tokens, num_kv_heads, _ = key.shape
    storage_dim = key_cache.shape[-1]
    capacity = key_cache.shape[0] * key_cache.shape[1]
    block_d = triton.next_power_of_2(head_dim)
    _quantize_and_scatter_kv_kernel[(num_tokens * num_kv_heads,)](
        key,
        value,
        slots,
        key_cache,
        value_cache,
        key_scale,
        value_scale,
        num_tokens,
        num_kv_heads,
        head_dim,
        storage_dim,
        capacity,
        _format_code(cache_dtype),
        block_d,
        triton.next_power_of_2(storage_dim),
        fixed_scales[0] if fixed_scales is not None else 0.0,
        fixed_scales[1] if fixed_scales is not None else 0.0,
        num_warps=4,
    )


def write_bf16_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> None:
    """Scatter BF16 K/V by slot in one GPU launch for standalone benchmarks."""
    num_tokens, num_kv_heads, head_dim = key.shape
    cache_capacity = key_cache.shape[0] * key_cache.shape[1]
    _scatter_bf16_kv_kernel[(num_tokens * num_kv_heads,)](
        key,
        value,
        slots,
        key_cache,
        value_cache,
        num_tokens,
        num_kv_heads,
        head_dim,
        cache_capacity,
        triton.next_power_of_2(head_dim),
        num_warps=4,
    )


def quantized_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    block_table: torch.Tensor,
    query_to_sequence: torch.Tensor,
    query_positions: torch.Tensor,
    context_lengths: torch.Tensor,
    cache_dtype: str,
    scale: float,
    sliding_window: int,
    causal: bool,
    query_offsets: torch.Tensor | None = None,
    max_query_tokens: int | None = None,
) -> torch.Tensor:
    """Attend directly over paged quantized cache, dequantizing on read."""
    num_query_tokens, num_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[2]
    page_size = key_cache.shape[1]
    storage_dim = key_cache.shape[-1]
    output = torch.empty_like(query)
    block_d = triton.next_power_of_2(head_dim)
    max_query_tokens = num_query_tokens if max_query_tokens is None else max_query_tokens
    if max_query_tokens <= 1:
        # Host shape only: no per-step device-to-host context read.
        num_splits = 8 if block_table.shape[1] * page_size >= 1024 else 1
        partial = (
            torch.empty((num_query_tokens, num_heads, num_splits, head_dim), device=query.device, dtype=torch.float32)
            if num_splits > 1
            else output
        )
        partial_lse = (
            torch.empty((num_query_tokens, num_heads, num_splits), device=query.device, dtype=torch.float32)
            if num_splits > 1
            else output
        )
        _quantized_paged_attention_kernel[(num_query_tokens, num_heads, num_splits)](
            query,
            key_cache,
            value_cache,
            key_scale,
            value_scale,
            block_table,
            query_to_sequence,
            query_positions,
            context_lengths,
            partial,
            partial_lse,
            num_query_tokens,
            num_heads,
            num_kv_heads,
            head_dim,
            storage_dim,
            page_size,
            block_table.stride(0),
            scale,
            sliding_window,
            causal,
            4 if cache_dtype == "bf16" else _format_code(cache_dtype),
            block_d,
            64,
            num_splits,
            num_warps=4,
        )
        if num_splits > 1:
            _merge_attention_splits_kernel[(num_query_tokens * num_heads,)](
                partial, partial_lse, output, head_dim, num_splits, block_d, num_warps=4
            )
        return output

    if query_offsets is None:
        if context_lengths.numel() != 1:
            raise ValueError("query_offsets are required for batched prefill attention")
        query_offsets = torch.tensor([0, num_query_tokens], device=query.device, dtype=torch.int32)
    groups = num_heads // num_kv_heads
    # Reuse each decoded K/V tile across GQA heads, without staging full cache.
    grouped_heads = groups if cache_dtype in ("int8", "bf16") and groups in (1, 2, 3, 4, 6, 8) else 1
    block_q = min(64, triton.next_power_of_2(16 * grouped_heads))
    prefill_block_d = max(16, block_d)
    num_splits = (
        4
        if (
            query.dtype == torch.bfloat16
            and cache_dtype in ("int8", "bf16")
            and context_lengths.numel() <= 4
            and block_table.shape[1] * page_size >= 1024
        )
        else 1
    )
    partial = (
        torch.empty((num_query_tokens, num_heads, num_splits, head_dim), device=query.device, dtype=torch.float32)
        if num_splits > 1
        else output
    )
    partial_lse = (
        torch.empty((num_query_tokens, num_heads, num_splits), device=query.device, dtype=torch.float32)
        if num_splits > 1
        else output
    )
    _quantized_paged_prefill_attention_kernel[
        (
            triton.cdiv(max_query_tokens * grouped_heads, block_q),
            num_heads // grouped_heads,
            context_lengths.numel() * num_splits,
        )
    ](
        query,
        key_cache,
        value_cache,
        key_scale,
        value_scale,
        block_table,
        query_positions,
        query_offsets,
        context_lengths,
        partial,
        partial_lse,
        num_heads,
        num_kv_heads,
        head_dim,
        storage_dim,
        page_size,
        block_table.stride(0),
        scale,
        sliding_window,
        causal,
        4 if cache_dtype == "bf16" else _format_code(cache_dtype),
        block_q,
        prefill_block_d,
        64,
        grouped_heads,
        query.dtype == torch.bfloat16 and cache_dtype in ("int8", "bf16"),
        num_splits,
        num_warps=4,
    )
    if num_splits > 1:
        _merge_attention_splits_kernel[(num_query_tokens * num_heads,)](
            partial, partial_lse, output, head_dim, num_splits, block_d, num_warps=4
        )
    return output
