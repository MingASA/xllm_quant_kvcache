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

"""Fixed-scale adapter to the pinned FlashInfer native NVFP4 slot writer."""

from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch

_PINNED_PAGE_SHA256 = "4a20f05410442ff415a44337c7c6246d90777e4c177852b2830fae3cf00eab04"
_KERNEL_NAME = "_nvfp4_quantize_append_paged_kv_cache_with_slot_mapping_kernel"
_KERNEL_PARAMETERS = (
    "append_key", "append_value", "slot_mapping", "paged_k_cache", "paged_v_cache",
    "k_scale_cache", "v_scale_cache", "k_scale", "v_scale", "layout",
)


def _native_slot_writer(page_module: ModuleType) -> Callable[..., None]:
    """Reject private-API drift before binding the original registered op."""
    source = Path(page_module.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != _PINNED_PAGE_SHA256:
        raise RuntimeError("Prevalidated NVFP4 writer requires the qualified FlashInfer page.py SHA256")
    kernel = getattr(page_module, _KERNEL_NAME)
    parameters = inspect.signature(kernel).parameters
    if (tuple(parameters) != _KERNEL_PARAMETERS
            or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                   or p.default is not inspect.Parameter.empty for p in parameters.values())):
        raise RuntimeError("Prevalidated NVFP4 writer requires the qualified native slot-writer signature")
    return kernel


def _validate_fixed_scales(scales: tuple[float, float]) -> None:
    limits = torch.finfo(torch.float32)
    if len(scales) != 2 or not all(math.isfinite(s) and limits.tiny <= s <= limits.max for s in scales):
        raise ValueError("NVFP4 fixed scales must be finite positive normal FP32 values")


class PrevalidatedNVFP4Writer:
    """Own immutable scale values/tensors and bound NHD resident page aliases.

    The backend validates allocator ownership before constructing this adapter.
    Pages and scale tensors stay alive for capture/replay and must not be resized
    or modified outside their native writer/attention operations.
    """

    def __init__(
        self, pages: dict[int, tuple[torch.Tensor, torch.Tensor]],
        block_scales: dict[int, tuple[torch.Tensor, torch.Tensor]],
        scales: tuple[float, float],
    ) -> None:
        _validate_fixed_scales(scales)
        if not pages or pages.keys() != block_scales.keys():
            raise ValueError("NVFP4 writer requires matching nonempty data and scale page bindings")
        key = next(iter(pages.values()))[0]
        self._device = key.device
        if self._device.type != "cuda":
            raise ValueError("NVFP4 writer requires CUDA resident pages")
        if key.ndim != 4:
            raise ValueError("NVFP4 writer requires four-dimensional NHD pages")
        self._input_shape = (key.shape[2], key.shape[3] * 2)
        if self._input_shape[0] <= 0 or self._input_shape[1] not in (128, 256):
            raise ValueError("NVFP4 writer requires positive head count and logical head dimension 128/256")
        self._layers = {}
        for layer_id, pair in pages.items():
            sf_pair = block_scales[layer_id]
            self._validate_page_binding(pair, sf_pair)
            self._layers[layer_id] = (*pair, *sf_pair)

        import flashinfer.page

        self._kernel = _native_slot_writer(flashinfer.page)
        # Construct from already validated Python values. These private tensors
        # never accept caller-owned CUDA scales needing per-call D2H validation.
        self._global_scales = tuple(torch.tensor([s], dtype=torch.float32, device=self._device) for s in scales)

    @property
    def global_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._global_scales

    def _validate_page_binding(
        self, pages: tuple[torch.Tensor, torch.Tensor],
        block_scales: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        key = pages[0]
        if key.ndim != 4 or tuple(key.shape[2:]) != (self._input_shape[0], self._input_shape[1] // 2):
            raise ValueError("NVFP4 data pages must share the bound NHD head geometry")
        expected_sf_shape = (*key.shape[:3], self._input_shape[1] // 16)
        for page in pages:
            if page.dtype != torch.uint8 or page.device != self._device or page.shape != key.shape:
                raise ValueError("NVFP4 data pages must be paired uint8 tensors on the bound CUDA device")
        for sf in block_scales:
            if sf.dtype != torch.float8_e4m3fn or sf.device != self._device or tuple(sf.shape) != expected_sf_shape:
                raise ValueError("NVFP4 block-scale pages must be paired FP8 E4M3 tensors with block size 16")

    def append(
        self, key: torch.Tensor, value: torch.Tensor,
        slot_mapping: torch.Tensor, layer_id: int,
    ) -> None:
        if key.dtype not in (torch.float16, torch.bfloat16) or value.dtype != key.dtype:
            raise ValueError("NVFP4 writer requires paired BF16/FP16 input dtypes")
        if (key.ndim != 3 or value.shape != key.shape or tuple(key.shape[1:]) != self._input_shape
                or key.device != self._device or value.device != self._device):
            raise ValueError("NVFP4 writer inputs must match the bound head shape and CUDA device")
        if (slot_mapping.ndim != 1 or slot_mapping.dtype not in (torch.int32, torch.int64)
                or slot_mapping.device != self._device or slot_mapping.numel() > key.shape[0]):
            raise ValueError("NVFP4 slots must be a one-dimensional int32/int64 tensor on the bound device with enough input rows")
        self._kernel(key, value, slot_mapping, *self._layers[layer_id], *self._global_scales, 0)
