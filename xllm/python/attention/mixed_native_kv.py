# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Split FP8 attention phases using existing host metadata, once per step.

This does not reorder scheduler requests or change GDN execution. Pure-decode
CUDA Graph buckets are untouched. Mixed-batch XQA remains eager attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from types import SimpleNamespace

import torch

from xllm.python.attention.backend import AttentionMetadata


@dataclass(frozen=True)
class AttentionSubset:
    query_indices: torch.Tensor
    query_indptr: torch.Tensor
    paging: SimpleNamespace


def split_mixed_attention(metadata: AttentionMetadata) -> tuple[AttentionSubset, AttentionSubset] | None:
    """Return decode and prefill subsets without reading device tensors."""
    batch = metadata.paged_kv_last_page_len.numel()
    ends = metadata.q_cu_seq_lens_host_values
    if not ends:
        # The CUDA common builder exposes cumulative lengths through the
        # existing host tensor; its explicit cumulative vector is NPU-only.
        # The newer Slot path instead fills the explicit vector above.
        host = getattr(metadata, "q_seq_lens_host", None)
        if host is None:
            return None
        if host.device.type != "cpu":
            raise ValueError("Mixed attention requires existing host query lengths without device synchronization")
        ends = host.tolist()
        if len(ends) == batch:
            ends = list(accumulate(ends, initial=0))
    if len(ends) == batch:
        ends = [0, *ends]
    if len(ends) != batch + 1 or ends[0] != 0:
        raise ValueError("Mixed attention requires official cumulative host query lengths")
    lengths = [b - a for a, b in zip(ends, ends[1:])]
    decode = [i for i, length in enumerate(lengths) if length == 1]
    prefill = [i for i, length in enumerate(lengths) if length != 1]
    if not decode or not prefill or not any(length > 1 for length in lengths):
        return None
    host_ptr = metadata.paged_kv_indptr_host
    host_last = metadata.paged_kv_last_page_len_host
    if host_ptr is None or host_last is None:
        raise ValueError("Mixed XQA requires existing host paging metadata")
    ptr, last = host_ptr.tolist(), host_last.tolist()
    device = metadata.paged_kv_indices.device
    subsets = []
    for rows in (decode, prefill):
        query_rows, page_rows, qo, indptr = [], [], [0], [0]
        for row in rows:
            query_rows.extend(range(ends[row], ends[row + 1]))
            page_rows.extend(range(ptr[row], ptr[row + 1]))
            qo.append(len(query_rows))
            indptr.append(len(page_rows))
        subset_ptr = torch.tensor(indptr, dtype=torch.int32)
        subset_last = torch.tensor([last[i] for i in rows], dtype=torch.int32)
        page_indices = torch.tensor(page_rows, dtype=torch.int64, device=device)
        paging = SimpleNamespace(
            paged_kv_indptr=subset_ptr.to(device),
            paged_kv_indices=metadata.paged_kv_indices.index_select(0, page_indices),
            paged_kv_last_page_len=subset_last.to(device),
            paged_kv_indptr_host=subset_ptr, paged_kv_last_page_len_host=subset_last,
        )
        subsets.append(AttentionSubset(torch.tensor(query_rows, dtype=torch.int64, device=device),
                                       torch.tensor(qo, dtype=torch.int32), paging))
    return subsets[0], subsets[1]
