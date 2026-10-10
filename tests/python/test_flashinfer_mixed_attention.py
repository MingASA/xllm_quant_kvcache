# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Dense-reference tests of the complete mixed-V backend and existing writer."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.backend import LayerCache

_ARMS = ("auto", "bf16_int8", "bf16_fp8_e4m3", "bf16_int4")
_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
pytestmark = pytest.mark.skipif(not _SUPPORTED, reason="Pinned mixed-V specialization requires SM120")


@pytest.fixture(scope="module", params=_ARMS)
def backend(request: pytest.FixtureRequest) -> object:
    from xllm.python.attention.flashinfer_mixed import FlashInferMixedVBackend

    return FlashInferMixedVBackend(
        request.param, 12, 2, 128, 128**-0.5, 0, torch.device("cuda", torch.cuda.current_device()), torch.bfloat16
    )


@pytest.mark.parametrize(
    "queries,lengths,page_size",
    [
        ([1], [1], 1),
        ([3, 1, 8], [17, 65, 129], 16),
        ([64, 17], [129, 511], 32),
        ([129], [129], 16),
        ([512], [529], 16),
        ([1, 2], [8193, 4097], 16),
        ([1] * 128, [65 + i % 7 for i in range(128)], 16),
        ([129, 1], [32769, 133], 128),
        ([1] * 128, [65 + i % 7 for i in range(128)], 128),
    ],
)
def test_paged_writer_attention_and_workspace_reuse(
    backend: object,
    queries: list[int],
    lengths: list[int],
    page_size: int,
) -> None:
    from xllm.python.attention.quantized_triton import write_quantized_kv

    torch.manual_seed(20261010)
    counts = [(length + page_size - 1) // page_size for length in lengths]
    pages = sum(counts) + 3
    indices_cpu = torch.randperm(pages, dtype=torch.int32)[: sum(counts)]
    indptr_cpu = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32)
    offsets = torch.tensor([0, *torch.tensor(queries).cumsum(0).tolist()], dtype=torch.int32)
    tails = torch.tensor([(length - 1) % page_size + 1 for length in lengths], dtype=torch.int32)
    key = torch.randn(pages, page_size, 2, 128, dtype=torch.bfloat16, device="cuda")
    values = torch.randn_like(key)
    values[0, 0] = 0
    codec = backend._codec
    payload = torch.empty((*key.shape[:3], codec.storage_dim), dtype=codec.storage_dtype, device="cuda")
    scales = None if codec.cache_dtype == "bf16" else torch.empty(key.shape[:3], dtype=torch.float32, device="cuda")
    cache = LayerCache(key, payload, value_scale=scales)
    write_quantized_kv(
        key.reshape(-1, 2, 128),
        values.reshape(-1, 2, 128),
        torch.arange(pages * page_size, device="cuda"),
        key,
        payload,
        None,
        scales,
        codec.cache_dtype,
        128,
        bf16_key=True,
    )
    query = torch.randn(sum(queries), 12, 128, dtype=torch.bfloat16, device="cuda")
    # Reappend the real query tokens through backend.execute(), preserving tails.
    slots_cpu = torch.cat(
        [
            indices_cpu[indptr_cpu[i] : indptr_cpu[i + 1]][torch.arange(length - query_len, length) // page_size].long()
            * page_size
            + torch.arange(length - query_len, length) % page_size
            for i, (query_len, length) in enumerate(zip(queries, lengths))
        ]
    )
    slots = slots_cpu.cuda()
    appended_key = key.flatten(0, 1)[slots].clone()
    appended_value = values.flatten(0, 1)[slots].clone()
    key_before = key.clone()
    for i, length in enumerate(lengths):
        last = indices_cpu[indptr_cpu[i + 1] - 1].item()
        tail = (length - 1) % page_size + 1
        if scales is None:
            payload[last, tail:] = float("nan")
        else:
            scales[last, tail:] = float("nan")
    metadata = SimpleNamespace(
        q_seq_lens_host=offsets,
        qo_indptr=offsets.cuda(),
        q_cu_seq_lens=None,
        paged_kv_indptr=indptr_cpu.cuda(),
        paged_kv_indices=indices_cpu.cuda(),
        paged_kv_last_page_len=tails.cuda(),
        slot_mapping=slots,
    )
    layer = SimpleNamespace(
        num_heads=12, num_kv_heads=2, head_dim=128, layer_id=0, scale=128**-0.5, causal=True, sliding_window=0
    )
    backend.bind_kv_caches([cache])
    workspace_ptr = backend._workspace.data_ptr()
    backend.prepare(metadata)
    output = backend.execute(query.flatten(1), appended_key, appended_value, layer).reshape_as(query)
    torch.cuda.synchronize()
    assert torch.equal(key, key_before), "BF16 K must remain bitwise unchanged"
    decoded = codec.decode(payload, scales)
    references = []
    for i, (query_len, length) in enumerate(zip(queries, lengths)):
        physical = indices_cpu[indptr_cpu[i] : indptr_cpu[i + 1]].long().cuda()
        k = key[physical].reshape(-1, 2, 128)[:length].float().repeat_interleave(6, 1)
        v = decoded[physical].reshape(-1, 2, 128)[:length].repeat_interleave(6, 1)
        q = query[offsets[i] : offsets[i + 1]].float()
        scores = torch.einsum("qhd,khd->hqk", q, k) * 128**-0.5
        mask = (
            torch.arange(length, device="cuda")[None, :]
            <= (length - query_len + torch.arange(query_len, device="cuda"))[:, None]
        )
        weights = scores.masked_fill(~mask[None, :, :], -torch.inf).softmax(-1)
        references.append(torch.einsum("hqk,khd->qhd", weights, v))
    torch.testing.assert_close(output.float(), torch.cat(references), atol=0.015, rtol=0.02)
    backend.prepare(metadata)
    repeated = backend.execute_attention(query, layer).reshape_as(query)
    torch.testing.assert_close(output, repeated, atol=0, rtol=0)
    assert backend._workspace.data_ptr() == workspace_ptr
    expected = {"bf16": 1024, "int8": 776, "fp8_e4m3": 776, "int4": 648}[codec.cache_dtype]
    actual_bytes = sum(t.numel() * t.element_size() for t in (key, payload, scales) if t is not None)
    assert actual_bytes == pages * page_size * expected
    if codec.cache_dtype == "bf16" and not os.environ.get("XLLM_MIXED_SANITIZER"):
        native = backend._planner.run(query, (key, payload))
        torch.testing.assert_close(output, native, atol=0, rtol=0)
