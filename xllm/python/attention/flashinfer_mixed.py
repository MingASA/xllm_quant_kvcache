# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Opt-in real mixed-V FlashInfer FA2 for paged prefill and decode.

All four arms share the stock BF16 planner, launch geometry and MMA pipeline.
Compressed V is converted only inside one shared-memory tile, never a global
BF16 cache. Unsupported models/devices fail explicitly; no backend fallback.
"""

from __future__ import annotations

import ctypes
import json
import math
from typing import TYPE_CHECKING

import flashinfer
import torch

from scripts.logger import logger
from xllm.python.attention.backend import AttentionMetadata, LayerCache
from xllm.python.attention.flashinfer_mixed_build import load_library
from xllm.python.attention.quantized import QuantizedPagedAttentionBackend

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention

_MODES = {"bf16": 0, "int8": 1, "fp8_e4m3": 2, "int4": 3}


class FlashInferMixedVBackend(QuantizedPagedAttentionBackend):
    """Qwen2.5 BF16 head128, runtime GQA, eager SM120/SM90 specialization."""

    def __init__(
        self,
        cache_dtype: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if cache_dtype not in ("auto", "bf16", "bf16_int8", "bf16_fp8_e4m3", "bf16_int4"):
            raise ValueError("Mixed-V FlashInfer requires BF16 K and a supported V format")
        if (
            head_dim != 128
            or num_heads <= 0
            or num_kv_heads <= 0
            or num_heads % num_kv_heads != 0
            or dtype != torch.bfloat16
            or device.type != "cuda"
        ):
            raise ValueError("Mixed-V FlashInfer requires CUDA BF16 head128 and integral positive GQA")
        if sliding_window > 0 or not math.isclose(scale, 128**-0.5, rel_tol=1e-6):
            raise ValueError("Mixed-V FlashInfer supports full causal attention with the standard head128 scale")
        super().__init__("bf16" if cache_dtype == "auto" else cache_dtype, head_dim, num_kv_heads)
        self._device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        self._mode = _MODES[self._codec.cache_dtype]
        self._scale = scale
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._library, self.provenance = load_library(device)
        self._launch = self._library.mixed_prefill
        self._launch.argtypes = (
            [ctypes.c_void_p] * 9
            + [ctypes.c_int32] * 2
            + [ctypes.c_float, ctypes.c_int32]
            + [ctypes.c_void_p] * 4
            + [ctypes.c_int32] * 2
        )
        self._launch.restype = ctypes.c_int32
        # Allocate before cache budgeting, and reuse across every layer and step.
        self._workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._planner = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._workspace, kv_layout="NHD", backend="fa2")
        self._plan_array: ctypes.Array | None = None
        self._paging: tuple[torch.Tensor, ...] | None = None
        logger.info(
            "FLASHINFER_MIXED_BUILD %s",
            json.dumps(
                {
                    name: value
                    for name, value in self.provenance.items()
                    if name not in ("upstream_headers", "cutlass_headers")
                }
            ),
        )

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        super().bind_kv_caches(kv_caches)
        if kv_caches[0].key.device != self._device:
            raise ValueError("Mixed-V cache must match the backend device")

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        if graph_mode:
            raise ValueError("Mixed-V FlashInfer currently requires eager execution")
        if not self._kv_caches:
            raise RuntimeError("Mixed-V FlashInfer caches are not bound")
        host_queries = metadata.q_seq_lens_host
        if host_queries is None or host_queries.device.type != "cpu":
            raise ValueError("Mixed-V FlashInfer requires CPU query lengths")
        tails = metadata.paged_kv_last_page_len
        batch_size = tails.numel()
        lengths = host_queries.tolist()
        if len(lengths) == batch_size + 1 and lengths[0] == 0:
            offsets = lengths
            lengths = [end - start for start, end in zip(offsets, offsets[1:])]
        else:
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
        if len(lengths) != batch_size or any(length <= 0 for length in lengths):
            raise ValueError("Mixed-V query lengths do not match the batch")
        qo_indptr = metadata.qo_indptr
        if qo_indptr is None:
            qo_indptr = metadata.q_cu_seq_lens
        if qo_indptr is None:
            qo_indptr = torch.tensor(offsets, dtype=torch.int32, device=self._device)
        paging = (qo_indptr, metadata.paged_kv_indptr, metadata.paged_kv_indices, tails)
        for tensor in paging:
            if (
                tensor.device != self._device
                or tensor.ndim != 1
                or tensor.dtype != torch.int32
                or not tensor.is_contiguous()
            ):
                raise ValueError("Mixed-V paging metadata must be contiguous CUDA int32")
        if qo_indptr.numel() != batch_size + 1 or metadata.paged_kv_indptr.numel() != batch_size + 1:
            raise ValueError("Mixed-V paging indptr has the wrong size")
        if metadata.slot_mapping.numel() != offsets[-1] or metadata.slot_mapping.device != self._device:
            raise ValueError("Mixed-V write slots must match the appended query tokens")
        self._planner.plan(
            *paging,
            self._num_heads,
            self._num_kv_heads,
            self._head_dim,
            self.page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            sm_scale=self._scale,
        )
        plan = list(self._planner._plan_info)
        if len(plan) != 15 or plan[3] not in (16, 64, 128) or plan[13]:
            raise RuntimeError(f"Unsupported FlashInfer prefill planning ABI: {plan}")
        self._plan_array = (ctypes.c_int64 * 15)(*plan)
        self._paging = paging
        self._query_token_count = offsets[-1]
        self._metadata = metadata

    def execute_attention(self, q: torch.Tensor, layer: Attention) -> torch.Tensor:
        if (
            layer.num_heads != self._num_heads
            or layer.num_kv_heads != self._num_kv_heads
            or layer.head_dim != self._head_dim
            or not layer.causal
            or layer.sliding_window > 0
            or not math.isclose(layer.scale, self._scale)
        ):
            raise ValueError("Layer attention semantics differ from the mixed-V plan")
        query = self._validate_query(q, layer).contiguous()
        if query.dtype != torch.bfloat16 or self._plan_array is None or self._paging is None:
            raise ValueError("Mixed-V FlashInfer requires a prepared BF16 query")
        cache = self._kv_caches[layer.layer_id]
        output = torch.empty_like(query)
        status = self._launch(
            query.data_ptr(),
            cache.key.data_ptr(),
            cache.value.data_ptr(),
            cache.value_scale.data_ptr() if cache.value_scale is not None else 0,
            output.data_ptr(),
            *(tensor.data_ptr() for tensor in self._paging),
            self._paging[-1].numel(),
            self.page_size,
            self._scale,
            self._mode,
            torch.cuda.current_stream(self._device).cuda_stream,
            self._workspace.data_ptr(),
            self._planner._int_workspace_buffer.data_ptr(),
            ctypes.cast(self._plan_array, ctypes.c_void_p),
            self._num_heads,
            self._num_kv_heads,
        )
        if status:
            raise RuntimeError(f"Mixed-V FlashInfer CUDA launch returned error {status}")
        return output.flatten(1)
