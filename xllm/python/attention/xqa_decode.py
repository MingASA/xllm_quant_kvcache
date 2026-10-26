# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Explicit SM90 XQA decode over resident pages and stable graph metadata."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from xllm.python.attention.backend import AttentionMetadata


def validate_xqa_lengths(indptr: list[int], last: list[int], page_size: int, maximum: int) -> None:
    """Reject unsupported lengths before changing the device plan."""
    if page_size not in (16, 32, 64, 128) or maximum < page_size or maximum % page_size:
        raise ValueError("XQA requires an explicit page-aligned maximum context")
    if len(indptr) != len(last) + 1 or not last or indptr[0] != 0:
        raise ValueError("XQA host paging shape/origin differs")
    for begin, end, tail in zip(indptr, indptr[1:], last):
        if any(type(value) is not int for value in (begin, end, tail)):
            raise ValueError("XQA host page metadata must be integral")
        if end < begin or not 1 <= tail <= page_size or (end == begin and tail != 1):
            raise ValueError("XQA requires monotone pages and a valid partial tail/padding row")
        if (end - begin - 1) * page_size + tail > maximum:
            raise ValueError("XQA live context exceeds the declared model context")


@triton.jit
def _convert_pages(indptr: tl.tensor, indices: tl.tensor, last: tl.tensor,
                   table: tl.tensor, lengths: tl.tensor, width: tl.constexpr,
                   page_size: tl.constexpr, block: tl.constexpr) -> None:
    row = tl.program_id(0)
    column = tl.arange(0, block)
    begin, end = tl.load(indptr + row), tl.load(indptr + row + 1)
    pages = tl.load(indices + begin + column, column < end - begin, other=0)
    tl.store(table + row * width + column, pages, column < width)
    # Existing graph buckets encode padding with no pages. XQA uses a safe
    # one-token read from initialized reserved page0; padded outputs are discarded.
    length = (end - begin - 1) * page_size + tl.load(last + row)
    tl.store(lengths + row, tl.where(end > begin, length, 1))


class XQADecodePlan:
    """One prepared bucket; no BF16 KV copy, host sync, or planner inside replay."""

    def __init__(self, batch_size: int, page_size: int, maximum: int, device: torch.device) -> None:
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("This explicit production XQA profile is validated only for SM90")
        self._page_size, self._maximum = page_size, maximum
        self._batch_size = batch_size
        self._table = torch.empty((batch_size, maximum // page_size), dtype=torch.int32, device=device)
        self._lengths = torch.empty(batch_size, dtype=torch.uint32, device=device)
        self._metadata_addresses: tuple[int, int, int] | None = None

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool) -> None:
        indptr, last = metadata.paged_kv_indptr_host, metadata.paged_kv_last_page_len_host
        if indptr is None or last is None or indptr.device.type != "cpu" or last.device.type != "cpu":
            raise ValueError("XQA requires the existing CPU planner metadata without a device sync")
        validate_xqa_lengths(indptr.tolist(), last.tolist(), self._page_size, self._maximum)
        if len(last) != self._batch_size:
            raise ValueError("XQA planned batch differs from its stable bucket")
        inputs = (metadata.paged_kv_indptr, metadata.paged_kv_indices, metadata.paged_kv_last_page_len)
        if inputs[0].numel() != len(indptr) or inputs[2].numel() != len(last) or indptr[-1].item() > inputs[1].numel():
            raise ValueError("XQA paging buffers do not cover the host plan")
        if any(tensor.device != self._table.device or tensor.dtype != torch.int32 for tensor in inputs):
            raise ValueError("XQA paged input must be int32 on the planned CUDA device")
        addresses = tuple(tensor.data_ptr() for tensor in inputs)
        if graph_mode and self._metadata_addresses is not None and addresses != self._metadata_addresses:
            raise RuntimeError("XQA graph paging input addresses changed")
        self._metadata_addresses = addresses
        width = self._table.shape[1]
        _convert_pages[(self._batch_size,)](*inputs, self._table, self._lengths,
                                          width, self._page_size, triton.next_power_of_2(width))

    def run(self, query: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor],
            workspace: torch.Tensor, softmax_scale: float, key_scale: float, value_scale: float) -> torch.Tensor:
        from flashinfer.decode import xqa_batch_decode_with_kv_cache

        if self._metadata_addresses is None:
            raise RuntimeError("XQA decode must be prepared before execution")
        if query.ndim != 3 or query.shape[0] != self._batch_size or query.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("XQA requires one BF16/FP16 query per planned sequence")
        key, value = cache
        if (key.dtype not in (query.dtype, torch.float8_e4m3fn) or value.dtype != key.dtype
                or key.shape != value.shape or key.ndim != 4 or key.shape[1] != self._page_size
                or key.shape[0] < 1 or key.shape[2] < 1 or key.shape[-1] != query.shape[-1]
                or query.shape[1] % key.shape[2] or key.stride() != value.stride() or key.stride(-1) != 1
                or any(tensor.device != self._table.device for tensor in (query, key, value))):
            raise ValueError("XQA requires same-device NHD BF16/FP16 or E4M3 cache with compatible heads")
        # The SM90 wrapper forwards Q as a raw packed pointer without Q strides.
        # Fused QKV projections can leave token-stride gaps; materialize inside
        # capture rather than allowing reads into the adjacent K/V projections.
        query = query.contiguous()
        return xqa_batch_decode_with_kv_cache(
            query, cache, workspace, self._table, self._lengths, self._maximum,
            bmm1_scale=softmax_scale * key_scale, bmm2_scale=value_scale,
            kv_layout="NHD", enable_pdl=True,
        )
