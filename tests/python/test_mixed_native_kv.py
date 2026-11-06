# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Mixed request permutation and paging tests, including non-leading decodes."""

from types import SimpleNamespace

import torch

from xllm.python.attention.mixed_native_kv import split_mixed_attention


def test_nonleading_decode_preserves_query_and_page_order() -> None:
    metadata = SimpleNamespace(
        q_cu_seq_lens_host_values=[0, 3, 4, 6, 7],
        paged_kv_indptr_host=torch.tensor([0, 2, 5, 6, 8], dtype=torch.int32),
        paged_kv_last_page_len_host=torch.tensor([3, 7, 2, 9], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 2, 5, 6, 8], dtype=torch.int32),
        paged_kv_indices=torch.tensor([12, 5, 7, 8, 9, 4, 15, 16], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([3, 7, 2, 9], dtype=torch.int32),
    )
    decoded, prefilled = split_mixed_attention(metadata)
    assert decoded.query_indices.tolist() == [3, 6]
    assert decoded.paging.paged_kv_indices.tolist() == [7, 8, 9, 15, 16]
    assert decoded.paging.paged_kv_indptr_host.tolist() == [0, 3, 5]
    assert decoded.paging.paged_kv_last_page_len_host.tolist() == [7, 9]
    assert prefilled.query_indices.tolist() == [0, 1, 2, 4, 5]
    assert prefilled.query_indptr.tolist() == [0, 3, 5]
    assert prefilled.paging.paged_kv_indices.tolist() == [12, 5, 4]
    assert prefilled.paging.paged_kv_indptr_host.tolist() == [0, 2, 3]
    original = torch.arange(7)
    reconstructed = torch.empty_like(original)
    for subset in (decoded, prefilled):
        reconstructed.index_copy_(0, subset.query_indices, original.index_select(0, subset.query_indices))
    assert torch.equal(reconstructed, original)


def test_uniform_phase_does_not_create_subset() -> None:
    for ends in ([0, 3, 6], [0, 1, 2]):
        metadata = SimpleNamespace(q_cu_seq_lens_host_values=ends,
                                   paged_kv_last_page_len=torch.ones(2, dtype=torch.int32))
        assert split_mixed_attention(metadata) is None


def test_cuda_common_builder_uses_existing_host_tensor() -> None:
    metadata = SimpleNamespace(
        q_cu_seq_lens_host_values=[],
        q_seq_lens_host=torch.tensor([0, 3, 4], dtype=torch.int32),
        paged_kv_indptr_host=torch.tensor([0, 1, 3], dtype=torch.int32),
        paged_kv_last_page_len_host=torch.tensor([3, 1], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 1, 3], dtype=torch.int32),
        paged_kv_indices=torch.tensor([9, 5, 7], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([3, 1], dtype=torch.int32),
    )
    decode, prefill = split_mixed_attention(metadata)
    assert decode.query_indices.tolist() == [3]
    assert decode.paging.paged_kv_indices.tolist() == [5, 7]
    assert prefill.query_indices.tolist() == [0, 1, 2]
    assert prefill.paging.paged_kv_indices.tolist() == [9]
