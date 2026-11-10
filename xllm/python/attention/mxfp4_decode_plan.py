# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Narrow pinned-FA2 decode planning for a dedicated MXFP4 wrapper.

The first call uses the complete upstream plan. Subsequent calls skip general
backend dispatch and unused FA2 kv-length uploads, but still run the native
planner on every step. No plan/mask, page IDs, or sequence lengths are cached.
"""

from __future__ import annotations

from typing import Any

import torch

MXFP4_DECODE_PLAN_CONTRACT = "pinned_fa2_uint8_graph_mask_v1"


class Mxfp4DecodePlanner:
    """Own planning of a private wrapper; do not mix with wrapper.plan calls."""

    def __init__(
        self, wrapper: Any, *, num_qo_heads: int, num_kv_heads: int,
        head_dim: int, page_size: int, sm_scale: float,
    ) -> None:
        if getattr(wrapper, "_xllm_mxfp4_plan_contract", None) != MXFP4_DECODE_PLAN_CONTRACT:
            raise ValueError("Fast MXFP4 planning requires the private Graph-mask adapter")
        if wrapper._backend != "fa2" or wrapper._jit_module is None:
            raise ValueError("Fast MXFP4 planning requires the private FA2 JIT module")
        if head_dim != 256 or page_size <= 0 or num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
            raise ValueError("Unsupported MXFP4 decode geometry")
        self._wrapper = wrapper
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._initialized = False
        self._query_indptr: dict[int, torch.Tensor] = {}

    def plan(
        self, indptr_host: torch.Tensor, indices: torch.Tensor,
        last_page_len_host: torch.Tensor,
    ) -> None:
        """Consume authoritative physical CSR metadata, never logical KV lengths."""
        wrapper = self._wrapper
        for tensor in (indptr_host, last_page_len_host):
            if tensor.device.type != "cpu" or tensor.dtype != torch.int32 or tensor.ndim != 1:
                raise ValueError("MXFP4 planning requires CPU int32 physical page metadata")
        batch = last_page_len_host.numel()
        if batch == 0 or indptr_host.numel() != batch + 1:
            raise ValueError("MXFP4 physical page metadata has invalid batch dimensions")
        if indices.dtype != torch.int32 or indices.ndim != 1 or indices.device != wrapper.device:
            raise ValueError("MXFP4 page indices must be int32 on the wrapper device")
        if wrapper.is_cuda_graph_enabled:
            if batch != wrapper._fixed_batch_size:
                raise ValueError("MXFP4 Graph batch size changed")
            if indices.numel() > wrapper._paged_kv_indices_buf.numel():
                raise ValueError("MXFP4 Graph page indices exceed capacity")
        query = self._query_indptr.get(batch)
        if query is None:
            query = torch.arange(batch + 1, dtype=torch.int32)
            self._query_indptr[batch] = query
        if not self._initialized:
            wrapper.plan(
                query, indptr_host, indices, last_page_len_host,
                self._num_qo_heads, self._num_kv_heads, self._head_dim, self._page_size,
                causal=True, sm_scale=self._sm_scale, q_data_type=torch.bfloat16,
                kv_data_type=torch.uint8, o_data_type=torch.bfloat16,
            )
            self._initialized = True
            return

        kv_lengths = (
            torch.clamp(indptr_host[1:] - indptr_host[:-1] - 1, min=0) * self._page_size
            + last_page_len_host
        )
        if wrapper.is_cuda_graph_enabled:
            wrapper._paged_kv_indptr_buf.copy_(indptr_host, non_blocking=True)
            wrapper._paged_kv_last_page_len_buf.copy_(last_page_len_host, non_blocking=True)
            wrapper._paged_kv_indices_buf[:indices.numel()].copy_(indices, non_blocking=True)
        else:
            wrapper._qo_indptr_buf = query.to(wrapper.device, non_blocking=True)
            wrapper._paged_kv_indptr_buf = indptr_host.to(wrapper.device, non_blocking=True)
            wrapper._paged_kv_last_page_len_buf = last_page_len_host.to(wrapper.device, non_blocking=True)
            wrapper._paged_kv_indices_buf = indices
        wrapper._batch_size = batch
        wrapper._qo_indptr_last = batch
        wrapper._max_q_len = 1
        wrapper._max_kv_len = int(kv_lengths.max())
        # Keep the uint8 split-KV guard. The native call also refreshes all
        # valid/padded CTA mask entries in the patched scheduler each step.
        wrapper._plan_info = wrapper._cached_module.plan(
            wrapper._float_workspace_buffer, wrapper._int_workspace_buffer,
            wrapper._pin_memory_int_workspace_buffer, query, indptr_host, kv_lengths,
            wrapper._max_total_num_rows or batch, batch,
            self._num_qo_heads, self._num_kv_heads, self._page_size,
            wrapper.is_cuda_graph_enabled, self._head_dim, self._head_dim,
            True, -1, -1, True, 0, 0,
        )
