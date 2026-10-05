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

"""Native FlashInfer FP8 attention over the allocator's paged byte caches.

This opt-in path uses fixed scalar K/V scales, unlike the experimental
per-token/head dynamic codec. Float8 views alias the persistent byte payloads;
no BF16 cache copy is created. Scale tensors remain allocated by C++ and are
written for consistent page-copy/accounting semantics.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from xllm.python.attention.backend import AttentionMetadata, LayerCache
from xllm.python.attention.flashinfer import FlashInferBackend
from xllm.python.attention.quantized import KVCacheCodec, _validate_cache
from xllm.python.attention.quantized_triton import write_quantized_kv

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


class FlashInferFP8Backend(FlashInferBackend):
    """Eager paged FP8 backend with explicit, immutable scalar scales."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
        cache_dtype: str,
        key_scale: float = 1.0,
        value_scale: float = 1.0,
    ) -> None:
        if cache_dtype not in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            raise ValueError("FlashInfer FP8 requires an FP8 cache dtype")
        if device.type != "cuda" or dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FlashInfer FP8 requires CUDA with BF16/FP16 queries")
        if not all(math.isfinite(value) and value > 0 for value in (key_scale, value_scale)):
            raise ValueError("FP8 K/V scales must be finite and positive")
        self._codec = KVCacheCodec(cache_dtype, head_dim)
        self._cache_dtype = torch.float8_e4m3fn if self._codec.cache_dtype == "fp8_e4m3" else torch.float8_e5m2
        self._fixed_scales = (key_scale, value_scale)
        self._fp8_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        super().__init__(num_heads, num_kv_heads, head_dim, scale, sliding_window, device, dtype)

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        for cache in kv_caches:
            _validate_cache(cache, self._codec, self.num_kv_heads)
            if cache.key.device != self._decode_workspace.device:
                raise ValueError("FP8 cache must be on the backend CUDA device")
            if cache.key.shape != kv_caches[0].key.shape:
                raise ValueError("FP8 page layouts must match across layers")
        super().bind_kv_caches(kv_caches)
        self._fp8_caches = [
            (cache.key.view(self._cache_dtype), cache.value.view(self._cache_dtype)) for cache in kv_caches
        ]

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        if graph_mode:
            raise ValueError("FlashInfer FP8 currently requires eager execution")
        if not self._kv_caches:
            raise RuntimeError("FP8 caches are not bound")
        prefill = metadata.is_prefill or metadata.is_chunked_prefill
        query_indptr = metadata.qo_indptr
        if query_indptr is None:
            query_indptr = metadata.q_cu_seq_lens
        if prefill and query_indptr is None:
            raise ValueError("FP8 paged prefill requires query indptr")
        self._metadata = metadata
        window_left = self.sliding_window - 1 if self.sliding_window > 0 else -1
        options = dict(
            sm_scale=self.scale,
            window_left=window_left,
            q_data_type=self.dtype,
            kv_data_type=self._cache_dtype,
        )
        paging = (metadata.paged_kv_indptr, metadata.paged_kv_indices, metadata.paged_kv_last_page_len)
        shape = (self.num_heads, self.num_kv_heads, self.head_dim, self.page_size)
        if prefill:
            self._prefill_paged_wrapper.plan(query_indptr, *paging, *shape, causal=True, **options)
        else:
            self._decode_wrapper.plan(*paging, *shape, **options)

    def write_cache(self, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor, layer_id: int) -> None:
        cache = self._kv_caches[layer_id]
        if k.shape != v.shape or k.ndim != 3 or k.shape[1:] != (self.num_kv_heads, self.head_dim):
            raise ValueError("FP8 writer requires matching [tokens, kv_heads, head_dim] K/V")
        if slots.ndim != 1 or slots.numel() != k.shape[0] or slots.dtype not in (torch.int32, torch.int64):
            raise ValueError("FP8 writer requires one integer slot per token")
        if any(tensor.device != cache.key.device for tensor in (k, v, slots)):
            raise ValueError("FP8 writer tensors must be on the cache device")
        write_quantized_kv(
            k.contiguous(),
            v.contiguous(),
            slots.contiguous(),
            cache.key,
            cache.value,
            cache.key_scale,
            cache.value_scale,
            self._codec.cache_dtype,
            self.head_dim,
            fixed_scales=self._fixed_scales,
        )

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        if self._metadata is None:
            raise RuntimeError("FlashInferFP8Backend.prepare() was not called")
        self.write_cache(
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            self._metadata.slot_mapping,
            layer.layer_id,
        )
        return self.execute_attention(q, layer)

    def execute_attention(
        self,
        q: torch.Tensor,
        layer: Attention,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._metadata is None:
            raise RuntimeError("FlashInferFP8Backend.prepare() was not called")
        wrapper = (
            self._prefill_paged_wrapper
            if (self._metadata.is_prefill or self._metadata.is_chunked_prefill)
            else self._decode_wrapper
        )
        output = wrapper.run(
            q.view(-1, self.num_heads, self.head_dim),
            self._fp8_caches[layer.layer_id],
            k_scale=self._fixed_scales[0],
            v_scale=self._fixed_scales[1],
        )
        return output.view(-1, self.num_heads * self.head_dim)
