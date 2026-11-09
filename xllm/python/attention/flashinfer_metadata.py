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

"""Scheduler-owned host metadata for synchronous FlashInfer planning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata


def _prefill_query_indptr(
    metadata: AttentionMetadata, query_indptr: torch.Tensor | None,
) -> torch.Tensor | None:
    """Reuse the CUDA scheduler's leading-zero cumulative host query lengths.

    External metadata without this mirror keeps the original planner path.
    A supplied mirror must satisfy the CUDA contract; per-sequence lengths from
    other builders are not converted or guessed here. The caller retains
    metadata and uses synchronous plan copies while its host view is alive.
    """
    host = getattr(metadata, "q_seq_lens_host", None)
    if host is None:
        return query_indptr
    if (not isinstance(host, torch.Tensor) or host.device.type != "cpu"
            or host.dtype != torch.int32 or host.ndim != 1):
        raise ValueError("FlashInfer host query indptr must be a one-dimensional CPU int32 tensor")
    if (not isinstance(query_indptr, torch.Tensor) or query_indptr.ndim != 1
            or host.numel() != query_indptr.numel() or host.numel() == 0):
        raise ValueError("FlashInfer host query indptr must match the nonempty query indptr shape")
    values = host.tolist()
    if values[0] != 0 or any(right < left for left, right in zip(values, values[1:])):
        raise ValueError("FlashInfer host query indptr must start at zero and be nondecreasing")
    return host
