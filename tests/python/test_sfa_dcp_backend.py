# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""CPU tests for SFA DCP graph-prepare indexer paging."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("torch_npu", reason="SFA DCP backend tests import the NPU attention backend")

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.sfa_dcp_backend import SfaDcpAttentionBackend
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)


class _FakeDcpGroup:
    def size(self) -> int:
        return 4

    def rank(self) -> int:
        return 0


def _cpu_context(execution_state: AclGraphExecutionState) -> ForwardContext:
    return ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        execution_state=execution_state,
    )


def test_graph_prepare_keeps_valid_indexer_pages_for_padded_lanes() -> None:
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroup(),
        index_topk=2048,
        max_num_reqs=8,
    )
    page_size = 128
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(16, page_size, 1, 512),
                value=torch.empty(16, page_size, 1, 64),
                index=torch.empty(64, page_size, 1, 128),
            )
        ]
    )

    block_table = torch.zeros((8, 2), dtype=torch.int32)
    block_table[:7] = torch.tensor([[1, 2]] * 7, dtype=torch.int32)
    slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1], dtype=torch.int32)
    kv_seq_lens = torch.tensor([1022, 1022, 1022, 1022, 1022, 1022, 1022, 1], dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=slot_mapping,
        block_table=block_table,
        kv_seq_lens=kv_seq_lens,
        kv_seq_lens_host_values=None,
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    with forward_context(_cpu_context(AclGraphExecutionState({}))):
        backend.prepare(metadata, graph_mode=True)

    expanded = backend._expanded_indexer_block_table
    assert expanded is not None
    assert (expanded[-1] >= 0).all()
    assert torch.equal(expanded[-1], torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32))
    assert torch.equal(expanded[0, :4], torch.tensor([4, 5, 6, 7], dtype=torch.int32))


@pytest.mark.parametrize("first_kv_len", [3, 511])
def test_prepare_uses_expanded_rows_for_mtp_verify(first_kv_len: int) -> None:
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroup(),
        index_topk=2048,
        max_num_reqs=8,
    )
    page_size = 128
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(16, page_size, 1, 512),
                value=torch.empty(16, page_size, 1, 64),
                index=torch.empty(64, page_size, 1, 128),
            )
        ]
    )

    captured: dict[str, object] = {}

    class _Builder:
        dcp_local_seq_lens_buf = torch.empty(8)

        @staticmethod
        def build(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(dcp_context=SimpleNamespace())

    backend._builder = _Builder()
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(4, dtype=torch.int32),
        block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        kv_seq_lens=torch.tensor([first_kv_len + 1, first_kv_len + 5], dtype=torch.int32),
        kv_seq_lens_host_values=[first_kv_len + 1, first_kv_len + 5],
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=SimpleNamespace(
            enabled=True,
            kv_seq_lens=torch.tensor([first_kv_len + offset for offset in (0, 1, 4, 5)], dtype=torch.int32),
            block_table=torch.tensor([[10, 11], [10, 11], [20, 21], [20, 21]], dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0, 1, 2, 3, 4] if first_kv_len == 3 else [0, 1, 2, 4, 6], dtype=torch.int32),
            paged_kv_indices=torch.tensor(
                [10, 10, 20, 20] if first_kv_len == 3 else [10, 10, 20, 21, 20, 21], dtype=torch.int32
            ),
            paged_kv_last_page_len=torch.tensor(
                [3, 4, 7, 8] if first_kv_len == 3 else [511, 512, 3, 4], dtype=torch.int32
            ),
            paged_attention_tiling_data=None,
            kv_seq_lens_host=None,
            kv_seq_lens_host_values=[first_kv_len + offset for offset in (0, 1, 4, 5)],
        ),
        is_prefill=False,
        is_chunked_prefill=False,
    )

    with forward_context(_cpu_context(AclGraphExecutionState({}))):
        backend.prepare(metadata, graph_mode=True)

    assert captured["num_reqs"] == 4
    assert captured["num_input_tokens"] == 4
    assert captured["seq_lens"].tolist() == [first_kv_len + offset for offset in (0, 1, 4, 5)]
    assert captured["block_table"].shape == (4, 2)
    assert backend._mla_max_seqlen_k == 1024
