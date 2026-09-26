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

"""Experimental eager quantized paged attention for CUDA and CPU validation.

Payloads and FP32 scales are owned by the C++ cache allocator. Attention
dequantizes one page at a time and uses online softmax; it never expands the
entire cache pool. This is a correctness baseline, not a fused fast kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata, LayerCache

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention

_FP8_DTYPES = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}
_QUERY_TILE_SIZE = 64


class KVCacheCodec:
    """Per-token/head dynamic symmetric quantization, after RoPE.

    Scales always multiply decoded values. INT4 stores signed [-7, 7]
    two's-complement nibbles, low nibble first. FP8 stores actual float8 bits
    in uint8 tensors, allowing byte copies even without float8 indexing ops.
    """

    def __init__(self, cache_dtype: str, head_dim: int) -> None:
        self.cache_dtype = "fp8_e4m3" if cache_dtype == "fp8" else cache_dtype
        if self.cache_dtype not in ("int8", "int4", *_FP8_DTYPES):
            raise ValueError(f"Unsupported quantized KV cache dtype: {cache_dtype}")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.head_dim = head_dim
        self.storage_dim = (head_dim + 1) // 2 if self.cache_dtype == "int4" else head_dim
        self.storage_dtype = torch.int8 if self.cache_dtype == "int8" else torch.uint8

    def encode(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.shape[-1] != self.head_dim or not tensor.is_floating_point():
            raise ValueError("KV input must be floating point with the configured head_dim")
        values = tensor.float()
        if not torch.isfinite(values).all().item():
            raise ValueError("KV quantization requires finite inputs")
        fp8_dtype = _FP8_DTYPES.get(self.cache_dtype)
        bound = torch.finfo(fp8_dtype).max if fp8_dtype is not None else (7.0 if self.cache_dtype == "int4" else 127.0)
        maximum = values.abs().amax(dim=-1)
        # Keep tiny and zero vectors finite without computing the reciprocal of
        # a potentially subnormal scale. Zero vectors have the canonical scale 1.
        scale = (maximum / bound).clamp_min(torch.finfo(torch.float32).tiny)
        scale = torch.where(maximum == 0, torch.ones_like(scale), scale)
        normalized = (values / scale.unsqueeze(-1)).clamp(-bound, bound)
        if fp8_dtype is not None:
            return normalized.to(fp8_dtype).view(torch.uint8), scale
        quantized = normalized.round().to(torch.int8)
        if self.cache_dtype == "int8":
            return quantized, scale
        nibbles = quantized.to(torch.uint8) & 15
        if self.head_dim % 2:
            nibbles = F.pad(nibbles, (0, 1))
        return nibbles[..., 0::2] | (nibbles[..., 1::2] << 4), scale

    def decode(self, payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if payload.dtype != self.storage_dtype or payload.shape[-1] != self.storage_dim:
            raise ValueError("KV payload does not match its quantization format")
        if scale.dtype != torch.float32 or scale.shape != payload.shape[:-1]:
            raise ValueError("KV scales must be FP32 with one entry per token/head")
        if self.cache_dtype in _FP8_DTYPES:
            values = payload.contiguous().view(_FP8_DTYPES[self.cache_dtype]).float()
        elif self.cache_dtype == "int4":
            nibbles = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2)
            signed = nibbles.to(torch.int16)
            values = torch.where(signed >= 8, signed - 16, signed)[..., : self.head_dim].float()
        else:
            values = payload.float()
        return values * scale.unsqueeze(-1)


def _validate_cache(cache: LayerCache, codec: KVCacheCodec, num_kv_heads: int) -> None:
    tensors = (cache.key, cache.value, cache.key_scale, cache.value_scale)
    if any(tensor is None for tensor in tensors):
        raise ValueError("Quantized attention requires K/V payloads and K/V scales")
    key, value, key_scale, value_scale = tensors
    if key.ndim != 4 or key.shape != value.shape or key.shape[2:] != (num_kv_heads, codec.storage_dim):
        raise ValueError("Quantized cache must use [blocks, page_size, kv_heads, storage_dim]")
    if key.shape[0] <= 0 or key.shape[1] <= 0:
        raise ValueError("Quantized cache requires positive block count and page_size")
    if key.dtype != codec.storage_dtype or value.dtype != codec.storage_dtype:
        raise ValueError("Quantized cache payload dtype mismatch")
    if key_scale.shape != key.shape[:-1] or value_scale.shape != key_scale.shape:
        raise ValueError("Quantized cache scale shape mismatch")
    if key_scale.dtype != torch.float32 or value_scale.dtype != torch.float32:
        raise ValueError("Quantized cache requires FP32 scales")
    if any(tensor.device != key.device or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Quantized cache tensors must be contiguous on the same device")


def write_quantized_kv(
    cache: LayerCache,
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    codec: KVCacheCodec,
) -> None:
    """Scatter payloads and scales together; -1 slots are padding, never writes."""
    if key.ndim != 3 or key.shape != value.shape or key.device != value.device:
        raise ValueError("K/V must have matching [tokens, kv_heads, head_dim] shapes and devices")
    _validate_cache(cache, codec, key.shape[1])
    if key.device != cache.key.device or slots.device != key.device:
        raise ValueError("K/V, slots and cache must be on the same device")
    if slots.ndim != 1 or slots.numel() != key.shape[0] or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot_mapping must contain one integer slot per token")
    capacity = cache.key.shape[0] * cache.key.shape[1]
    if ((slots < -1) | (slots >= capacity)).any().item():
        raise ValueError("KV slot out of range")
    valid = slots >= 0
    indices = slots[valid].long()
    if indices.unique().numel() != indices.numel():
        raise ValueError("Duplicate KV write slots are not supported")
    # Encode both before mutating the persistent cache.
    encoded_key, key_scale = codec.encode(key[valid])
    encoded_value, value_scale = codec.encode(value[valid])
    cache.key.flatten(0, 1).index_copy_(0, indices, encoded_key)
    cache.value.flatten(0, 1).index_copy_(0, indices, encoded_value)
    cache.key_scale.flatten(0, 1).index_copy_(0, indices, key_scale)
    cache.value_scale.flatten(0, 1).index_copy_(0, indices, value_scale)


class QuantizedPagedAttentionBackend(AttentionBackend):
    """Explicit eager backend; supports MHA/GQA prefill, chunking and decode."""

    def __init__(self, cache_dtype: str, head_dim: int, num_kv_heads: int) -> None:
        self._codec = KVCacheCodec(cache_dtype, head_dim)
        self._num_kv_heads = num_kv_heads
        self._kv_caches: list[LayerCache] = []
        self._metadata: AttentionMetadata | None = None
        self._plans: list[tuple[int, int, int, list[int]]] = []

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        if not kv_caches:
            raise ValueError("Quantized attention requires at least one cache")
        for cache in kv_caches:
            _validate_cache(cache, self._codec, self._num_kv_heads)
            if cache.key.shape != kv_caches[0].key.shape or cache.key.device != kv_caches[0].key.device:
                raise ValueError("Quantized cache layout must match across layers")
        self._kv_caches = kv_caches

    @property
    def num_kv_blocks(self) -> int:
        return self._kv_caches[0].key.shape[0]

    @property
    def page_size(self) -> int:
        return self._kv_caches[0].key.shape[1]

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        if graph_mode:
            raise ValueError("Quantized reference attention requires eager execution")
        if not self._kv_caches:
            raise ValueError("Quantized caches are not bound")
        if metadata.block_table is None or metadata.q_seq_lens_host is None:
            raise ValueError("Quantized attention requires block tables and host query lengths")
        query_lengths = metadata.q_seq_lens_host.tolist()
        context_lengths = list(metadata.kv_seq_lens_host_values)
        block_table = metadata.block_table.cpu().tolist()
        if len(query_lengths) != len(context_lengths) or len(block_table) != len(query_lengths):
            raise ValueError("Quantized attention batch metadata does not match")
        plans = []
        expected_slots = []
        offset = 0
        for query_len, context_len, row in zip(query_lengths, context_lengths, block_table):
            if query_len <= 0 or context_len < query_len:
                raise ValueError("Invalid query/context lengths")
            num_pages = (context_len + self.page_size - 1) // self.page_size
            pages = row[:num_pages]
            if len(pages) != num_pages or any(page < 0 or page >= self.num_kv_blocks for page in pages):
                raise ValueError("Invalid quantized KV block table")
            if len(set(pages)) != len(pages):
                raise ValueError("Aliased pages within a sequence are not supported")
            plans.append((offset, query_len, context_len, pages))
            expected_slots.extend(
                pages[position // self.page_size] * self.page_size + position % self.page_size
                for position in range(context_len - query_len, context_len)
            )
            offset += query_len
        if metadata.slot_mapping.cpu().tolist() != expected_slots:
            raise ValueError("KV write slots must match the appended query positions")
        if len(set(expected_slots)) != len(expected_slots):
            raise ValueError("Concurrent queries cannot overwrite shared KV slots")
        self._metadata = metadata
        self._plans = plans

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        if self._metadata is None:
            raise RuntimeError("Quantized attention prepare() was not called")
        if layer.head_dim != self._codec.head_dim or layer.num_kv_heads != self._num_kv_heads:
            raise ValueError("Layer shape does not match quantized cache")
        if layer.num_heads % layer.num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        query = q.reshape(-1, layer.num_heads, layer.head_dim)
        key = k.reshape(-1, layer.num_kv_heads, layer.head_dim)
        value = v.reshape_as(key)
        if query.shape[0] != sum(plan[1] for plan in self._plans) or key.shape[0] != query.shape[0]:
            raise ValueError("Attention tokens do not match the prepared batch")
        if not query.is_floating_point() or query.device != key.device:
            raise ValueError("Queries must be floating point on the KV device")
        if not torch.isfinite(query).all().item():
            raise ValueError("Quantized attention requires finite queries")
        cache = self._kv_caches[layer.layer_id]
        write_quantized_kv(cache, key, value, self._metadata.slot_mapping, self._codec)
        output = torch.empty_like(query)
        for offset, query_len, context_len, pages in self._plans:
            for start in range(0, query_len, _QUERY_TILE_SIZE):
                stop = min(start + _QUERY_TILE_SIZE, query_len)
                tile = query[offset + start : offset + stop]
                positions = torch.arange(start, stop, device=q.device) + context_len - query_len
                output[offset + start : offset + stop] = self._attend_pages(
                    tile, positions, cache, pages, context_len, layer
                )
        return output.flatten(1)

    def _attend_pages(
        self,
        query: torch.Tensor,
        positions: torch.Tensor,
        cache: LayerCache,
        pages: list[int],
        context_len: int,
        layer: Attention,
    ) -> torch.Tensor:
        groups = layer.num_heads // layer.num_kv_heads
        q = query.float().reshape(-1, layer.num_kv_heads, groups, layer.head_dim)
        running_max = torch.full((layer.num_kv_heads, groups, q.shape[0], 1), -torch.inf, device=q.device)
        denominator = torch.zeros_like(running_max)
        accumulator = torch.zeros_like(q)
        for page_index, block_id in enumerate(pages):
            valid = min(self.page_size, context_len - page_index * self.page_size)
            key = self._codec.decode(cache.key[block_id, :valid], cache.key_scale[block_id, :valid])
            value = self._codec.decode(cache.value[block_id, :valid], cache.value_scale[block_id, :valid])
            key_positions = torch.arange(valid, device=q.device) + page_index * self.page_size
            visible = torch.ones((q.shape[0], valid), device=q.device, dtype=torch.bool)
            if layer.causal:
                visible &= key_positions[None, :] <= positions[:, None]
            if layer.sliding_window > 0:
                visible &= key_positions[None, :] > positions[:, None] - layer.sliding_window
            scores = torch.einsum("qhgd,khd->hgqk", q, key) * layer.scale
            scores.masked_fill_(~visible[None, None], -torch.inf)
            next_max = torch.maximum(running_max, scores.amax(-1, keepdim=True))
            # Entire tiles may be masked (causal/SWA); avoid -inf - -inf.
            safe_max = torch.where(torch.isfinite(next_max), next_max, torch.zeros_like(next_max))
            correction = torch.exp(running_max - safe_max)
            weights = torch.exp(scores - safe_max)
            accumulator = accumulator * correction.permute(2, 0, 1, 3) + torch.einsum("hgqk,khd->qhgd", weights, value)
            denominator = denominator * correction + weights.sum(-1, keepdim=True)
            running_max = next_max
        return (accumulator / denominator.permute(2, 0, 1, 3)).reshape_as(query).to(query.dtype)
