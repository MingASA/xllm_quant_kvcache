# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""Bounded H800 FA3 correctness/Graph admission before loading a production model.

No performance claim, model quality claim, or service benchmark is produced.
The enclosing owned container's supervisor enforces resource/deadline limits.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.logger import logger


def _reference(query: object, key: object, value: object, query_count: int) -> object:
    import torch
    heads = query.shape[2]
    keys = key.repeat_interleave(heads // key.shape[2], dim=2).transpose(1, 2).float()
    values = value.repeat_interleave(heads // value.shape[2], dim=2).transpose(1, 2).float()
    scores = query.transpose(1, 2).float() @ keys.transpose(-1, -2) * (query.shape[-1] ** -.5)
    length = key.shape[1]
    allowed = torch.arange(length, device=query.device)[None, :] <= (
        length - query_count + torch.arange(query_count, device=query.device)[:, None])
    scores.masked_fill_(~allowed, -float("inf"))
    return (torch.softmax(scores, dim=-1) @ values).transpose(1, 2).reshape(-1, heads * query.shape[-1])


def check(arm: str) -> dict:
    import torch
    from tools.benchmark_kv_cache import _attention_metadata
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.flashinfer import FlashInferBackend
    from xllm.python.attention.flashinfer_fp8 import FlashInferFP8Backend

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise ValueError("This acceptance is specifically for one isolated H800/SM90 GPU")
    xqa = os.environ.get("XLLM_FLASHINFER_NATIVE_DECODE") == "xqa"
    planner = "fa2" if xqa else "fa3"
    if os.environ.get("XLLM_FLASHINFER_NATIVE_PLANNER") != planner:
        raise ValueError("Explicit matching native prefill planner is required")
    torch.manual_seed(20261026)
    batch, length, page_size, heads, kv_heads, dim = 2, 257, 128, 40, 8, 128
    device = torch.device("cuda")
    shape = (6, page_size, kv_heads, dim)
    is_fp8 = arm == "fp8_e4m3"
    cache = LayerCache(torch.zeros(shape, device=device, dtype=torch.uint8 if is_fp8 else torch.bfloat16),
                       torch.zeros(shape, device=device, dtype=torch.uint8 if is_fp8 else torch.bfloat16),
                       key_scale=torch.ones(shape[:-1], device=device) if is_fp8 else None,
                       value_scale=torch.ones(shape[:-1], device=device) if is_fp8 else None)
    options = {"cache_dtype": "fp8_e4m3", "key_scale": .125, "value_scale": .0625} if is_fp8 else {}
    backend = (FlashInferFP8Backend if is_fp8 else FlashInferBackend)(
        heads, kv_heads, dim, dim**-.5, -1, device, torch.bfloat16, **options)
    backend.bind_kv_caches([cache])
    pages = torch.tensor([[2, 0, 4], [3, 1, 5]], device=device, dtype=torch.int32)
    positions = torch.arange(length, device=device)
    slots = (pages[:, positions // page_size] * page_size + positions % page_size).flatten()
    key = torch.randn(batch, length, kv_heads, dim, device=device, dtype=torch.bfloat16) * .25
    value = torch.randn_like(key) * .5
    if is_fp8:
        backend.write_cache(key.flatten(0, 1), value.flatten(0, 1), slots, 0)
        reference_key = (key.float() / .125).clamp(-448, 448).to(torch.float8_e4m3fn).float() * .125
        reference_value = (value.float() / .0625).clamp(-448, 448).to(torch.float8_e4m3fn).float() * .0625
    else:
        cache.key.view(-1, kv_heads, dim)[slots.long()] = key.flatten(0, 1)
        cache.value.view(-1, kv_heads, dim)[slots.long()] = value.flatten(0, 1)
        reference_key, reference_value = key.float(), value.float()
    for payload in (cache.key, cache.value):
        payload[pages[:, -1].long(), 1:] = 255 if is_fp8 else float("nan")
    layer = SimpleNamespace(layer_id=0, num_heads=heads, num_kv_heads=kv_heads, head_dim=dim)
    q_scale = backend._fp8_query_scale if is_fp8 else 1.0

    def _expected(query: torch.Tensor, context: int) -> torch.Tensor:
        quantized = is_fp8 and (not xqa or query.shape[1] == 1)
        effective = (query.float() / q_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * q_scale if quantized else query.float()
        return _reference(effective, reference_key[:, :context], reference_value[:, :context], query.shape[1])

    comparisons = {}

    def _compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        error = actual.float() - expected
        relative = float((torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected).clamp_min(1e-9)).item())
        comparisons[name] = {"relative_l2": relative, "max_abs": float(error.abs().max().item())}
        torch.testing.assert_close(actual.float(), expected, rtol=.08, atol=.015)
        if relative > .05:
            raise ValueError("Native operator differs from its representable reference: " + name)

    # Nonuniform logits make an incorrect Q/K scale visible to the reference.
    packed_query = torch.randn(batch, 17, heads + 2 * kv_heads, dim, device=device, dtype=torch.bfloat16) * 2
    query = packed_query[:, :, :heads]
    metadata = _attention_metadata(pages, slots.view(batch, length)[:, -17:].flatten(), length, 17, page_size)
    backend.prepare(metadata)
    actual = backend.execute_attention(query.flatten(0, 1), layer)
    _compare("prefill", actual, _expected(query, length))
    if actual.dtype != torch.bfloat16:
        raise ValueError("FA3 attention must return model BF16, not FP8 output")
    # Real capture/replay, followed by a different query and shorter live context.
    packed_query = torch.randn(batch, 1, heads + 2 * kv_heads, dim, device=device, dtype=torch.bfloat16) * 2
    query = packed_query[:, :, :heads]
    metadata = _attention_metadata(pages, slots.view(batch, length)[:, -1:].flatten(), length, 1, page_size)
    backend.prepare(metadata, graph_mode=True)
    for _ in range(3):
        backend.execute_attention(query.flatten(0, 1), layer)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = backend.execute_attention(query.flatten(0, 1), layer)
    graph.replay()
    torch.cuda.synchronize()
    _compare("capture_replay", output, _expected(query, length))
    query.mul_(1.5)
    metadata.paged_kv_indptr_host.copy_(torch.tensor([0, 2, 4], dtype=torch.int32))
    metadata.paged_kv_indptr.copy_(metadata.paged_kv_indptr_host)
    metadata.paged_kv_indices[:4].copy_(pages[:, :2].flatten())
    metadata.paged_kv_last_page_len.fill_(2)
    metadata.paged_kv_last_page_len_host.fill_(2)
    for payload in (cache.key, cache.value):
        payload[pages[:, 1].long(), 2:] = 255 if is_fp8 else float("nan")
    backend.prepare(metadata, graph_mode=True)
    graph.replay()
    torch.cuda.synchronize()
    _compare("changed_context", output, _expected(query, 130))
    if xqa:
        # Native graph buckets retain empty padding rows with indptr repeats.
        # Verify a different bucket with two live rows and two padded rows.
        host_indptr = torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32)
        host_last = torch.tensor([2, 2, 1, 1], dtype=torch.int32)
        padded = SimpleNamespace(is_prefill=False, is_chunked_prefill=False, qo_indptr=None, q_cu_seq_lens=None,
            paged_kv_indptr_host=host_indptr, paged_kv_last_page_len_host=host_last,
            paged_kv_indptr=host_indptr.to(device), paged_kv_last_page_len=host_last.to(device),
            paged_kv_indices=pages[:, :2].flatten().contiguous())
        backend.prepare(padded, graph_mode=True)
        padded_query = torch.cat((query.flatten(0, 1), torch.zeros_like(query.flatten(0, 1))))
        for _ in range(3):
            backend.execute_attention(padded_query, layer)
        torch.cuda.synchronize()
        padded_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(padded_graph):
            padded_output = backend.execute_attention(padded_query, layer)
        padded_graph.replay()
        torch.cuda.synchronize()
        _compare("padded_graph", padded_output[:batch], _expected(query, 130))
        if not torch.isfinite(padded_output).all().item():
            raise ValueError("XQA graph padding produced nonfinite output")
    return {"schema": "kv-native-fa3-operator-acceptance-v1", "classification": "operator_correctness_only",
            "arm": arm, "backend": backend.native_kv_runtime_info(), "device": torch.cuda.get_device_name(0),
            "prefill_planner": planner, "decode_impl": "xqa" if xqa else "flashinfer",
            "padded_graph_bucket_checked": xqa,
            "noncontiguous_q_checked": True,
            "reference_comparisons": comparisons,
            "real_capture_and_replay": True, "partial_page_and_replan": True, "model_loaded": False,
            "service_performance_measured": False, "model_quality_validated": False,
            "reference": "FP32 dense causal attention on the same representable Q/K/V; relative L2 <=.05 plus rtol .08, atol .015"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("bf16", "fp8_e4m3"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.arm)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2); stream.write("\n")
    logger.info("Native production operator/Graph acceptance passed for %s; no model/service/quality claim", args.arm)


if __name__ == "__main__":
    main()
