# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Native NVFP4 pages and linear E4M3 block scales consumed directly by FA2.

On SM90 the attention kernel decodes compressed elements in registers/shared
memory. This is compressed resident storage, not Blackwell FP4 Tensor Core
computation. No whole-cache BF16 allocation or decompression is maintained.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import flashinfer.page
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.flashinfer import FlashInferBackend
from xllm.python.attention.native_kv import _full_attention_layers
from xllm.python.attention.native_profile import NativeFlashInferConfig

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


def _validate_nvfp4_cache(cache: LayerCache, heads: int, dim: int, device: torch.device) -> None:
    if cache.key.dtype != torch.uint8 or cache.key.shape[2:] != (heads, dim // 2):
        raise ValueError("NVFP4 K/V requires packed byte pages with head_dim/2 channels")
    for tensor in (cache.key, cache.value, cache.key_scale, cache.value_scale):
        if tensor is None or tensor.device != device or not tensor.is_contiguous():
            raise ValueError("NVFP4 payloads/scales must be contiguous on the backend device")
    shape = (*cache.key.shape[:-1], dim // 16)
    for tensor in (cache.key_scale, cache.value_scale):
        if tensor.shape != shape or tensor.dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise ValueError("NVFP4 requires one linear E4M3 scale byte per 16 channels of each token/head")


class FlashInferNVFP4Backend(FlashInferBackend):
    """SM90 paged prefill/decode with mature FlashInfer quantize/scatter writes."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
        key_scale: float,
        value_scale: float,
    ) -> None:
        if device.type != "cuda" or dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Native NVFP4 requires CUDA with BF16/FP16 queries")
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError(
                "This native NVFP4 FA2 path targets SM90; other architectures require a capability gate"
            )
        if head_dim not in (128, 256) or num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError("SM90 NVFP4 FA2 requires head_dim128/256 and a complete GQA group")
        if not hasattr(flashinfer.page, "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping"):
            raise RuntimeError(
                "FlashInfer lacks the native NVFP4 slot writer; use a verified version providing this API"
            )
        fp32 = torch.finfo(torch.float32)
        if not all(math.isfinite(s) and fp32.tiny <= s <= fp32.max for s in (key_scale, value_scale)):
            raise ValueError("NVFP4 global decode scales must be finite positive normal FP32 values")
        self._fixed_scales = (key_scale, value_scale)
        self._global_scales = tuple(torch.tensor([s], dtype=torch.float32, device=device) for s in self._fixed_scales)
        self._nvfp4_pages: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._nvfp4_scales: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        super().__init__(
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            sliding_window,
            device,
            dtype,
            native_config=NativeFlashInferConfig("fa2", "flashinfer"),
        )

    @property
    def _kv_format(self) -> str:
        return "nvfp4"

    @property
    def _kv_data_type(self) -> torch.dtype:
        return torch.uint8

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        layers = _full_attention_layers(kv_caches)
        for _, cache in layers:
            _validate_nvfp4_cache(cache, self.num_kv_heads, self.head_dim, self._decode_workspace.device)
        super().bind_kv_caches(kv_caches)
        self._nvfp4_pages = {layer_id: (cache.key, cache.value) for layer_id, cache in layers}
        self._nvfp4_scales = {
            layer_id: (cache.key_scale.view(torch.float8_e4m3fn), cache.value_scale.view(torch.float8_e4m3fn))
            for layer_id, cache in layers
        }

    def native_kv_runtime_info(self) -> dict:
        return {
            **super().native_kv_runtime_info(),
            "kv_cache_dtype": "nvfp4",
            "payload_encoding": "E2M1_low_even_high_odd_nibbles",
            "scale_strategy": "linear_E4M3_per_token_head_block16_plus_global_K_and_V",
            "block_scale_layout": "NHD_linear",
            "k_scale": self._fixed_scales[0],
            "v_scale": self._fixed_scales[1],
            "global_scale_resident_bytes": sum(t.numel() * t.element_size() for t in self._global_scales),
            "writer": "flashinfer.page.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping",
            "attention_impl": "flashinfer_fa2_native_compressed_KV",
            "full_BF16_KV_shadow": False,
            "SM90_FP4_tensor_core_compute": False,
            "scale_views_alias_resident_pages": bool(self._nvfp4_scales)
            and all(
                key.data_ptr() == self._kv_caches[layer_id].key_scale.data_ptr()
                and value.data_ptr() == self._kv_caches[layer_id].value_scale.data_ptr()
                for layer_id, (key, value) in self._nvfp4_scales.items()
            ),
        }

    def write_cache(self, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor, layer_id: int) -> None:
        if k.ndim != 3 or k.shape != v.shape or k.shape[1:] != (self.num_kv_heads, self.head_dim):
            raise ValueError("NVFP4 writer requires [tokens, kv_heads, head_dim] K/V")
        if slots.ndim != 1 or slots.numel() != k.shape[0] or slots.dtype not in (torch.int32, torch.int64):
            raise ValueError("NVFP4 writer requires one integer slot per token")
        if k.dtype != self.dtype or v.dtype != self.dtype:
            raise ValueError("NVFP4 writer input dtype must match model activations")
        if any(t.device != self._decode_workspace.device for t in (k, v, slots)):
            raise ValueError("NVFP4 writer inputs must be on the cache device")
        flashinfer.page.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
            k,
            v,
            slots,
            self._nvfp4_pages[layer_id],
            self._nvfp4_scales[layer_id],
            self._global_scales[0],
            self._global_scales[1],
            kv_layout="NHD",
        )

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        if self._metadata is None:
            raise RuntimeError("FlashInferNVFP4Backend.prepare() was not called")
        self.write_cache(
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            self._metadata.slot_mapping,
            layer.layer_id,
        )
        return self.execute_attention(q, layer)

    def execute_attention(
        self, q: torch.Tensor, layer: Attention, k: torch.Tensor | None = None, v: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._metadata is None:
            raise RuntimeError("FlashInferNVFP4Backend.prepare() was not called")
        wrapper = (
            self._prefill_paged_wrapper
            if self._metadata.is_prefill or self._metadata.is_chunked_prefill
            else self._active_decode_wrapper
        )
        output = wrapper.run(
            q.view(-1, self.num_heads, self.head_dim),
            self._nvfp4_pages[layer.layer_id],
            kv_cache_sf=self._nvfp4_scales[layer.layer_id],
            k_scale=self._fixed_scales[0],
            v_scale=self._fixed_scales[1],
        )
        return output.view(-1, self.num_heads * self.head_dim)
