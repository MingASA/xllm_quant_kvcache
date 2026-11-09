# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Real CPU tensor checks of the scheduler host query planning contract."""

from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.flashinfer_metadata import _prefill_query_indptr


def test_host_mirror_is_reused_without_copy_or_reading_device_values() -> None:
    host = torch.tensor([0, 3, 3, 8], dtype=torch.int32)
    query = torch.empty(4, dtype=torch.int32, device="meta")
    metadata = SimpleNamespace(q_seq_lens_host=host)
    assert _prefill_query_indptr(metadata, query) is host


@pytest.mark.parametrize("metadata", [SimpleNamespace(), SimpleNamespace(q_seq_lens_host=None)])
def test_external_metadata_without_host_preserves_original_path(metadata: SimpleNamespace) -> None:
    query = torch.tensor([0, 2, 9], dtype=torch.int32)
    assert _prefill_query_indptr(metadata, query) is query


@pytest.mark.parametrize("host", [
    [0, 3, 8],
    torch.tensor([0, 3, 8], dtype=torch.int64),
    torch.tensor([[0, 3, 8]], dtype=torch.int32),
    torch.empty(3, dtype=torch.int32, device="meta"),
    torch.tensor([3, 5, 8], dtype=torch.int32),
    torch.tensor([0, 8, 3], dtype=torch.int32),
    torch.tensor([0, 8], dtype=torch.int32),
])
def test_invalid_supplied_host_contract_fails_before_planning(host: object) -> None:
    with pytest.raises(ValueError, match="host query indptr"):
        _prefill_query_indptr(SimpleNamespace(q_seq_lens_host=host), torch.empty(3, dtype=torch.int32))


@pytest.mark.parametrize("query", [None, torch.empty((1, 3), dtype=torch.int32), torch.empty(0)])
def test_unknown_or_empty_query_shape_is_not_guessed(query: torch.Tensor | None) -> None:
    host = torch.empty(0, dtype=torch.int32) if query is not None and query.numel() == 0 else torch.zeros(3, dtype=torch.int32)
    with pytest.raises(ValueError, match="shape"):
        _prefill_query_indptr(SimpleNamespace(q_seq_lens_host=host), query)
