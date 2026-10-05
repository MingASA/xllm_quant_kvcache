# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Compare BF16 and experimental KV formats under one paged-attention workload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F

from scripts.logger import logger
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.quantized import (
    KVCacheCodec,
    QuantizedPagedAttentionBackend,
    write_quantized_kv,
)


def _command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=_ROOT, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)


def _installed_packages() -> str:
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.append(f"{name}=={distribution.version}")
    return "\n".join(sorted(packages, key=str.casefold))


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _measure(fn: Callable, args: argparse.Namespace, processed_tokens: int) -> dict:
    for _ in range(args.warmup):
        fn()
    _sync(args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        allocated_before_samples = torch.cuda.memory_allocated()
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        fn()
        _sync(args.device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    rates = sorted(processed_tokens / (sample / 1000) for sample in samples)
    result = {
        "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[math.ceil(len(samples) * 0.95) - 1],
        "tokens_per_call": processed_tokens,
        "tokens_per_second_p50": statistics.median(rates),
        "tokens_per_second_p05": rates[math.floor((len(rates) - 1) * 0.05)],
        "samples_ms": samples,
    }
    if args.device == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_increment_bytes"] = result["peak_allocated_bytes"] - allocated_before_samples
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return result


def _dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    length, queries = k.shape[1], q.shape[1]
    mask = (
        torch.arange(length, device=q.device)[None, :]
        <= torch.arange(length - queries, length, device=q.device)[:, None]
    )
    return (
        F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            enable_gqa=True,
        )
        .transpose(1, 2)
        .reshape(q.shape[0] * queries, -1)
    )


def _write_bf16_kv(cache: LayerCache, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor) -> None:
    kernels = sys.modules.get("xllm.python.kernels")
    if key.device.type == "cuda" and kernels is not None:
        kernels.reshape_paged_cache(slots, key.contiguous(), value.contiguous(), cache.key, cache.value)
        return
    if key.device.type == "cuda":
        from xllm.python.attention.quantized_triton import write_bf16_kv

        write_bf16_kv(key.contiguous(), value.contiguous(), slots.contiguous(), cache.key, cache.value)
        return
    valid = slots >= 0
    indices = slots[valid].long()
    cache.key.flatten(0, 1).index_copy_(0, indices, key[valid])
    cache.value.flatten(0, 1).index_copy_(0, indices, value[valid])


def _attention_metadata(
    pages: torch.Tensor, slots: torch.Tensor, context: int, query_tokens: int, page_size: int
) -> SimpleNamespace:
    batch, pages_per_sequence = pages.shape
    device = pages.device
    page_indptr_host = torch.arange(batch + 1, dtype=torch.int32) * pages_per_sequence
    page_last_length = context - (pages_per_sequence - 1) * page_size
    query_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * query_tokens
    return SimpleNamespace(
        block_table=pages,
        slot_mapping=slots,
        q_seq_lens_host=torch.full((batch,), query_tokens, dtype=torch.int32),
        kv_seq_lens_host_values=[context] * batch,
        kv_seq_lens_host=torch.full((batch,), context, dtype=torch.int32),
        paged_kv_indptr=page_indptr_host.to(device=device),
        paged_kv_indptr_host=page_indptr_host,
        paged_kv_indices=pages.reshape(-1).to(dtype=torch.int32),
        paged_kv_last_page_len=torch.full((batch,), page_last_length, dtype=torch.int32, device=device),
        paged_kv_last_page_len_host=torch.full((batch,), page_last_length, dtype=torch.int32),
        qo_indptr=query_indptr,
        q_cu_seq_lens=None,
        kv_cu_seq_lens=None,
        is_prefill=False,
        is_chunked_prefill=query_tokens > 1,
    )


def _flashinfer_backend(
    cache: LayerCache, metadata: SimpleNamespace, args: argparse.Namespace, fp8_format: str | None = None
):
    os.environ.setdefault(
        "FLASHINFER_WORKSPACE_BASE",
        str(Path(tempfile.gettempdir()) / "xllm-flashinfer-workspace"),
    )
    from xllm.python.attention.flashinfer import FlashInferBackend

    backend_type = FlashInferBackend
    options = {}
    if fp8_format is not None:
        from xllm.python.attention.flashinfer_fp8 import FlashInferFP8Backend

        backend_type = FlashInferFP8Backend
        options = dict(cache_dtype=fp8_format, key_scale=args.fp8_k_scale, value_scale=args.fp8_v_scale)
    backend = backend_type(
        args.heads,
        args.kv_heads,
        args.head_dim,
        args.head_dim**-0.5,
        -1,
        torch.device(args.device),
        torch.bfloat16,
        **options,
    )
    backend.bind_kv_caches([cache])
    backend.prepare(metadata)
    return backend


def _prepare_plans(metadata: SimpleNamespace, page_size: int, num_blocks: int) -> list[tuple[int, int, int, list[int]]]:
    """Build the same per-request page plan as the quantized backend."""
    query_lengths = metadata.q_seq_lens_host.tolist()
    context_lengths = list(metadata.kv_seq_lens_host_values)
    block_table = metadata.block_table.cpu().tolist()
    if len(query_lengths) != len(context_lengths) or len(block_table) != len(query_lengths):
        raise ValueError("Paged attention batch metadata does not match")
    plans = []
    expected_slots = []
    offset = 0
    for query_len, context_len, row in zip(query_lengths, context_lengths, block_table):
        if query_len <= 0 or context_len < query_len:
            raise ValueError("Invalid query/context lengths")
        num_pages = (context_len + page_size - 1) // page_size
        pages = row[:num_pages]
        if len(pages) != num_pages or any(page < 0 or page >= num_blocks for page in pages):
            raise ValueError("Invalid paged KV block table")
        if len(set(pages)) != len(pages):
            raise ValueError("Aliased pages within a sequence are not supported")
        plans.append((offset, query_len, context_len, pages))
        expected_slots.extend(
            pages[position // page_size] * page_size + position % page_size
            for position in range(context_len - query_len, context_len)
        )
        offset += query_len
    if metadata.slot_mapping.cpu().tolist() != expected_slots:
        raise ValueError("KV write slots must match the appended query positions")
    if len(set(expected_slots)) != len(expected_slots):
        raise ValueError("Concurrent queries cannot overwrite shared KV slots")
    return plans


def _paged_attention(
    q: torch.Tensor,
    cache: LayerCache,
    codec: KVCacheCodec | None,
    plans: list[tuple[int, int, int, list[int]]],
    layer: SimpleNamespace,
    page_size: int,
) -> torch.Tensor:
    """Run the eager online-softmax page loop for BF16 and quantized caches."""
    query = q.reshape(-1, layer.num_heads, layer.head_dim)
    groups = layer.num_heads // layer.num_kv_heads
    output = torch.empty_like(query)
    for offset, query_len, context_len, pages in plans:
        for start in range(0, query_len, 64):
            stop = min(start + 64, query_len)
            tile = query[offset + start : offset + stop]
            positions = torch.arange(start, stop, device=q.device) + context_len - query_len
            tile_q = tile.float().reshape(-1, layer.num_kv_heads, groups, layer.head_dim)
            running_max = torch.full(
                (layer.num_kv_heads, groups, tile_q.shape[0], 1), -torch.inf, device=q.device
            )
            denominator = torch.zeros_like(running_max)
            accumulator = torch.zeros_like(tile_q)
            for page_index, block_id in enumerate(pages):
                valid = min(page_size, context_len - page_index * page_size)
                key_payload = cache.key[block_id, :valid]
                value_payload = cache.value[block_id, :valid]
                if codec is None:
                    key = key_payload.float()
                    value = value_payload.float()
                else:
                    key = codec.decode(key_payload, cache.key_scale[block_id, :valid])
                    value = codec.decode(value_payload, cache.value_scale[block_id, :valid])
                key_positions = torch.arange(valid, device=q.device) + page_index * page_size
                visible = key_positions[None, :] <= positions[:, None] if layer.causal else torch.ones(
                    (tile_q.shape[0], valid), device=q.device, dtype=torch.bool
                )
                if layer.sliding_window > 0:
                    visible &= key_positions[None, :] > positions[:, None] - layer.sliding_window
                scores = torch.einsum("qhgd,khd->hgqk", tile_q, key) * layer.scale
                scores.masked_fill_(~visible[None, None], -torch.inf)
                next_max = torch.maximum(running_max, scores.amax(-1, keepdim=True))
                safe_max = torch.where(torch.isfinite(next_max), next_max, torch.zeros_like(next_max))
                correction = torch.exp(running_max - safe_max)
                weights = torch.exp(scores - safe_max)
                accumulator = accumulator * correction.permute(2, 0, 1, 3) + torch.einsum(
                    "hgqk,khd->qhgd", weights, value
                )
                denominator = denominator * correction + weights.sum(-1, keepdim=True)
                running_max = next_max
            output[offset + start : offset + stop] = (
                accumulator / denominator.permute(2, 0, 1, 3)
            ).reshape_as(tile).to(tile.dtype)
    return output.flatten(1)


@torch.inference_mode()
def _case(args: argparse.Namespace, length: int, fmt: str, batch: int, queries: int) -> dict:
    dim = args.head_dim
    generator = torch.Generator().manual_seed(args.seed)

    def random(shape: tuple) -> torch.Tensor:
        return torch.randn(shape, generator=generator).to(device=args.device, dtype=torch.bfloat16)

    q = random((batch, queries, args.heads, dim))
    k = random((batch, length, args.kv_heads, dim))
    v = random(tuple(k.shape))
    pages_per_seq = math.ceil(length / args.page_size)
    blocks = batch * pages_per_seq
    # Permuted physical pages catch accidental contiguous-cache assumptions.
    pages = torch.randperm(blocks, generator=generator).reshape(batch, pages_per_seq)
    positions = torch.arange(length)
    slots = (pages[:, positions // args.page_size] * args.page_size + positions % args.page_size).to(args.device)
    row = {
        "format": fmt,
        "context": length,
        "batch": batch,
        "query_tokens": queries,
        "processed_tokens_per_call": batch * queries,
        "sampled_tokens_per_stage": batch * queries * args.iterations,
        "head_dim": dim,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "page_size": args.page_size,
        "device": args.device,
        "bf16_cache_writer": "xllm_reshape_paged_cache"
        if "xllm.python.kernels" in sys.modules
        else "triton_standalone"
        if args.device == "cuda"
        else "torch_index_copy",
        "quantized_cache_writer": "fused_triton" if args.device == "cuda" and fmt != "bf16" else None,
    }
    if fmt == "bf16":
        cache = LayerCache(
            key=torch.zeros(
                (blocks, args.page_size, args.kv_heads, dim), device=args.device, dtype=torch.bfloat16
            ),
            value=torch.zeros(
                (blocks, args.page_size, args.kv_heads, dim), device=args.device, dtype=torch.bfloat16
            ),
        )
        codec = None
    else:
        codec = KVCacheCodec(fmt, dim)
        shape = (blocks, args.page_size, args.kv_heads, codec.storage_dim)
        cache = LayerCache(
            key=torch.zeros(shape, device=args.device, dtype=codec.storage_dtype),
            value=torch.zeros(shape, device=args.device, dtype=codec.storage_dtype),
            key_scale=torch.ones(shape[:-1], device=args.device),
            value_scale=torch.ones(shape[:-1], device=args.device),
        )
    native_fp8 = args.device == "cuda" and fmt.startswith("fp8") and args.fp8_backend == "flashinfer"
    write_kv = (
        (lambda key, value, mapping: _write_bf16_kv(cache, key, value, mapping))
        if codec is None
        else (lambda key, value, mapping: write_quantized_kv(cache, key, value, mapping, codec))
    )
    if native_fp8:
        from xllm.python.attention.quantized_triton import write_quantized_kv as write_fixed_fp8

        def write_kv(key, value, mapping):
            write_fixed_fp8(
                key.contiguous(), value.contiguous(), mapping.contiguous(),
                cache.key, cache.value, cache.key_scale, cache.value_scale, fmt, dim,
                fixed_scales=(args.fp8_k_scale, args.fp8_v_scale),
            )

        row["quantization_scale_policy"] = "fixed_scalar"
        row["fp8_k_scale"] = args.fp8_k_scale
        row["fp8_v_scale"] = args.fp8_v_scale
    elif codec is not None:
        row["quantization_scale_policy"] = "dynamic_per_token_head"
    write_kv(k.flatten(0, 1), v.flatten(0, 1), slots.flatten())
    metadata = SimpleNamespace(
        block_table=pages.to(device=args.device, dtype=torch.int32),
        q_seq_lens_host=torch.full((batch,), queries, dtype=torch.int32),
        kv_seq_lens_host_values=[length] * batch,
        slot_mapping=slots[:, -queries:].reshape(-1),
    )
    plans = _prepare_plans(metadata, args.page_size, blocks)
    layer = SimpleNamespace(
        head_dim=dim,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        layer_id=0,
        scale=dim**-0.5,
        causal=True,
        sliding_window=-1,
    )
    new_k = k[:, -queries:].reshape(-1, args.kv_heads, dim)
    new_v = v[:, -queries:].reshape_as(new_k)
    q_flat = q.reshape(-1, args.heads, dim)
    metadata = _attention_metadata(
        pages.to(device=args.device, dtype=torch.int32),
        slots[:, -queries:].reshape(-1),
        length,
        queries,
        args.page_size,
    )
    if args.device == "cuda":
        backend = _flashinfer_backend(cache, metadata, args) if codec is None else None
        if codec is None and args.bf16_backend == "triton":
            from xllm.python.attention.quantized_triton import quantized_paged_attention

            query_to_sequence = torch.arange(batch, device=args.device, dtype=torch.int32).repeat_interleave(queries)
            query_positions = torch.arange(length - queries, length, device=args.device, dtype=torch.int32).repeat(batch)
            context_lengths = torch.full((batch,), length, device=args.device, dtype=torch.int32)
            query_offsets = torch.arange(batch + 1, device=args.device, dtype=torch.int32) * queries

            def triton_attention(query: torch.Tensor, layer: SimpleNamespace) -> torch.Tensor:
                return quantized_paged_attention(
                    query, cache.key, cache.value, cache.key, cache.value,
                    metadata.block_table, query_to_sequence, query_positions,
                    context_lengths, "bf16", layer.scale, layer.sliding_window,
                    layer.causal, query_offsets, queries,
                ).flatten(1)

            flashinfer_reference = backend.execute_attention(q_flat, layer)
            backend = SimpleNamespace(execute_attention=triton_attention)
        if codec is not None:
            if native_fp8:
                backend = _flashinfer_backend(cache, metadata, args, fp8_format=fmt)
            else:
                backend = QuantizedPagedAttentionBackend(fmt, dim, args.kv_heads)
                backend.bind_kv_caches([cache])
                backend.prepare(metadata)
        write_kv(new_k, new_v, metadata.slot_mapping)
        actual = backend.execute_attention(q_flat, layer)
        if codec is None and args.bf16_backend == "triton":
            torch.testing.assert_close(actual.float(), flashinfer_reference.float(), rtol=0.02, atol=0.005)
        if not torch.isfinite(actual).all().item():
            raise RuntimeError("Nonfinite attention output")
        implementation_max_abs = None
        if codec is not None:
            decoded_key = codec.decode(cache.key, cache.key_scale)[pages.to(args.device)].reshape(
                batch, pages_per_seq * args.page_size, args.kv_heads, dim
            )[:, :length]
            decoded_value = codec.decode(cache.value, cache.value_scale)[pages.to(args.device)].reshape(
                batch, pages_per_seq * args.page_size, args.kv_heads, dim
            )[:, :length]
            quant_reference = _dense(q.float(), decoded_key, decoded_value)
            torch.testing.assert_close(actual.float(), quant_reference, rtol=0.02, atol=0.005)
            implementation_max_abs = (actual.float() - quant_reference).abs().max().item()
            del quant_reference, decoded_key, decoded_value
        bf16_reference = (
            flashinfer_reference if codec is None and args.bf16_backend == "triton"
            else actual if codec is None else None
        )
    else:
        backend = None
        actual = _paged_attention(q_flat, cache, codec, plans, layer, args.page_size)
        if not torch.isfinite(actual).all().item():
            raise RuntimeError("Nonfinite attention output")
        implementation_max_abs = None
        bf16_reference = actual if codec is None else None

    if codec is not None:
        bf16_cache = LayerCache(
            key=torch.zeros(
                (blocks, args.page_size, args.kv_heads, dim), device=args.device, dtype=torch.bfloat16
            ),
            value=torch.zeros(
                (blocks, args.page_size, args.kv_heads, dim), device=args.device, dtype=torch.bfloat16
            ),
        )
        _write_bf16_kv(bf16_cache, k.flatten(0, 1), v.flatten(0, 1), slots.flatten())
        if args.device == "cuda":
            bf16_backend = _flashinfer_backend(bf16_cache, metadata, args)
            bf16_reference = bf16_backend.execute_attention(q_flat, layer)
            del bf16_backend
        else:
            bf16_reference = _paged_attention(q_flat, bf16_cache, None, plans, layer, args.page_size)
        del bf16_cache

    error = actual.float() - bf16_reference.float()
    row["output_relative_l2_vs_bf16"] = (error.norm() / bf16_reference.float().norm().clamp_min(1e-12)).item()
    row["output_max_abs_vs_bf16"] = error.abs().max().item()
    row["implementation_max_abs"] = implementation_max_abs
    row["baseline_kind"] = "flashinfer_paged" if args.device == "cuda" else "eager_cpu_reference"
    row["attention_backend"] = args.bf16_backend if fmt == "bf16" else "flashinfer" if native_fp8 else "triton"
    row["kv_budget_mib"] = args.kv_budget_mib
    cache_tensors = [cache.key, cache.value]
    if cache.key_scale is not None:
        cache_tensors.extend((cache.key_scale, cache.value_scale))
    row["kv_cache_bytes"] = sum(t.numel() * t.element_size() for t in cache_tensors)
    row["kv_cache_mib"] = row["kv_cache_bytes"] / (1024**2)
    row["bf16_cache_bytes"] = blocks * args.page_size * args.kv_heads * dim * 2 * 2
    row["compression_vs_bf16"] = row["bf16_cache_bytes"] / row["kv_cache_bytes"]

    if backend is None:
        def full_call() -> torch.Tensor:
            full_plans = _prepare_plans(metadata, args.page_size, blocks)
            write_kv(new_k, new_v, metadata.slot_mapping)
            return _paged_attention(q_flat, cache, codec, full_plans, layer, args.page_size)

        attention = lambda: _paged_attention(q_flat, cache, codec, plans, layer, args.page_size)
    else:
        def full_call() -> torch.Tensor:
            write_kv(new_k, new_v, metadata.slot_mapping)
            return backend.execute_attention(q_flat, layer)

        attention = lambda: backend.execute_attention(q_flat, layer)

    stages = {
        "cache_write": lambda: write_kv(new_k, new_v, metadata.slot_mapping),
        "attention": attention,
        "full_call": full_call,
    }
    del bf16_reference, actual, error
    if args.device == "cuda" and codec is None and args.bf16_backend == "triton":
        del flashinfer_reference
    _sync(args.device)
    if args.device == "cuda":
        row["allocated_before_timing_bytes"] = torch.cuda.memory_allocated()
    row["timings"] = {
        name: _measure(fn, args, row["processed_tokens_per_call"]) for name, fn in stages.items()
    }
    if args.device == "cuda":
        row["runtime_peak_allocated_bytes"] = max(
            timing["peak_allocated_bytes"] for timing in row["timings"].values()
        )
        row["runtime_peak_increment_bytes"] = max(
            timing["peak_increment_bytes"] for timing in row["timings"].values()
        )
        row["runtime_peak_reserved_bytes"] = max(timing["peak_reserved_bytes"] for timing in row["timings"].values())
    return row


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("bf16", "int8", "fp8_e4m3", "fp8_e5m2", "int4"),
        default=["bf16", "int8", "fp8_e4m3", "fp8_e5m2", "int4"],
    )
    parser.add_argument("--batch", type=int, default=None, help="Single batch size (legacy form)")
    parser.add_argument("--batches", type=int, nargs="+", default=None, help="Batch sizes to sweep")
    parser.add_argument("--query-tokens", type=int, default=None, help="Single query length (legacy form)")
    parser.add_argument("--query-lengths", type=int, nargs="+", default=None, help="Query lengths to sweep")
    for name, default in (
        ("head-dim", 128),
        ("heads", 32),
        ("kv-heads", 8),
        ("page-size", 128),
        ("warmup", 3),
        ("iterations", 20),
        ("threads", 1),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=1, help="Independent benchmark rounds per case")
    parser.add_argument("--output-dir", type=Path, default=Path("kv_benchmark_results"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--bf16-backend", choices=("flashinfer", "triton"), default="flashinfer")
    parser.add_argument("--fp8-backend", choices=("triton", "flashinfer"), default="triton")
    parser.add_argument("--fp8-k-scale", type=float, default=1.0)
    parser.add_argument("--fp8-v-scale", type=float, default=1.0)
    parser.add_argument("--kv-budget-mib", type=float, default=None,
                        help="Single-layer KV-only budget; choose each format's maximum batch at each context")
    args = parser.parse_args()
    if args.kv_budget_mib is not None and (not math.isfinite(args.kv_budget_mib) or args.kv_budget_mib <= 0):
        parser.error("KV budget must be finite and positive")
    if not all(math.isfinite(scale) and scale > 0 for scale in (args.fp8_k_scale, args.fp8_v_scale)):
        parser.error("FP8 scalar scales must be finite and positive")
    if args.fp8_backend == "flashinfer" and args.device != "cuda":
        parser.error("FlashInfer FP8 requires CUDA")
    if args.device != "cuda" and args.bf16_backend != "flashinfer":
        parser.error("Triton BF16 requires CUDA")
    batches = args.batches if args.batches is not None else [args.batch or 1]
    if args.kv_budget_mib is not None:
        if args.batches is not None or args.batch is not None:
            parser.error("KV budget selects batch automatically; do not specify batches")
        batches = [1]
    query_lengths = args.query_lengths if args.query_lengths is not None else [args.query_tokens or 1]
    if any(
        value <= 0
        for value in (
            *args.contexts,
            *batches,
            *query_lengths,
            args.head_dim,
            args.heads,
            args.kv_heads,
            args.page_size,
            args.iterations,
            args.rounds,
            args.threads,
        )
    ):
        parser.error("All shape/count parameters must be positive")
    if args.warmup < 0 or args.heads % args.kv_heads or min(args.contexts) < max(query_lengths):
        parser.error("Require warmup >= 0, heads divisible by kv-heads, and each context >= query length")
    torch.set_num_threads(args.threads)
    destination = args.output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 2,
        "status": "running",
        "arguments": vars(args) | {"output_dir": str(args.output_dir)},
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "git_head": _command(["git", "rev-parse", "HEAD"]),
        "git_branch": _command(["git", "branch", "--show-current"]),
        "git_status": _command(["git", "status", "--porcelain"]),
        "nvidia_smi": _command(["nvidia-smi"]),
        "source_sha256": {
            path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest()
            for path in (
                "tools/benchmark_kv_cache.py",
                "xllm/python/attention/flashinfer.py",
                "xllm/python/attention/flashinfer_fp8.py",
                "xllm/python/attention/quantized.py",
                "xllm/python/attention/quantized_triton.py",
                "xllm/python/attention/backend.py",
            )
        },
    }
    logger.info("Results: %s", destination.resolve())
    metadata["packages"] = _installed_packages()
    (destination / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA unavailable; install a Blackwell-capable PyTorch CUDA wheel and driver. No CPU fallback."
                )
            metadata["gpu"] = {
                "name": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "memory_free_total": torch.cuda.mem_get_info(),
                "compiled_archs": torch.cuda.get_arch_list(),
            }
            x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
            if not torch.isfinite(x @ x).all().item():
                raise RuntimeError("CUDA BF16 matmul preflight failed")
        for fmt in args.formats:
            if fmt != "bf16":
                codec = KVCacheCodec(fmt, args.head_dim)
                payload, scale = codec.encode(torch.ones((2, args.head_dim), device=args.device))
                torch.testing.assert_close(
                    codec.decode(payload, scale), torch.ones_like(scale[:, None]).expand(2, args.head_dim)
                )
        with (
            (destination / "results.jsonl").open("w") as raw,
            (destination / "summary.csv").open("w", newline="") as summary,
        ):
            writer = csv.DictWriter(
                summary,
                fieldnames=[
                    "format",
                    "attention_backend",
                    "kv_budget_mib",
                    "round",
                    "context",
                    "batch",
                    "query_tokens",
                    "processed_tokens_per_call",
                    "stage",
                    "p50_ms",
                    "p95_ms",
                    "tokens_per_second_p50",
                    "kv_cache_bytes",
                    "kv_cache_mib",
                    "runtime_peak_allocated_bytes",
                    "runtime_peak_increment_bytes",
                    "output_relative_l2_vs_bf16",
                    "output_max_abs_vs_bf16",
                    "compression_vs_bf16",
                    "bf16_cache_writer",
                    "quantized_cache_writer",
                ],
            )
            writer.writeheader()
            if not args.preflight_only:
                for round_index in range(args.rounds):
                    for length in args.contexts:
                        for batch in batches:
                            for queries in query_lengths:
                                for fmt in args.formats:
                                    case_batch = batch
                                    if args.kv_budget_mib is not None:
                                        storage_bytes = args.head_dim * 2 if fmt == "bf16" else (
                                            KVCacheCodec(fmt, args.head_dim).storage_dim + 4
                                        )
                                        sequence_bytes = (
                                            math.ceil(length / args.page_size) * args.page_size
                                            * args.kv_heads * 2 * storage_bytes
                                        )
                                        case_batch = int(args.kv_budget_mib * 1024**2) // sequence_bytes
                                        if case_batch < 1:
                                            raise ValueError("KV budget cannot hold one sequence for this context")
                                    metadata["active_case"] = {
                                        "round": round_index + 1,
                                        "format": fmt,
                                        "context": length,
                                        "batch": case_batch,
                                        "query_tokens": queries,
                                    }
                                    row = _case(args, length, fmt, case_batch, queries)
                                    row["round"] = round_index + 1
                                    raw.write(json.dumps(row) + "\n")
                                    raw.flush()
                                    for stage, timing in row["timings"].items():
                                        writer.writerow(
                                            {
                                                key: row.get(key)
                                                for key in (
                                                    "format",
                                                    "attention_backend",
                                                    "kv_budget_mib",
                                                    "round",
                                                    "context",
                                                    "batch",
                                                    "query_tokens",
                                                    "processed_tokens_per_call",
                                                    "kv_cache_bytes",
                                                    "kv_cache_mib",
                                                    "output_relative_l2_vs_bf16",
                                                    "output_max_abs_vs_bf16",
                                                    "compression_vs_bf16",
                                                    "bf16_cache_writer",
                                                    "quantized_cache_writer",
                                                )
                                            }
                                            | {
                                                "stage": stage,
                                                "p50_ms": timing["p50_ms"],
                                                "p95_ms": timing["p95_ms"],
                                                "tokens_per_second_p50": timing["tokens_per_second_p50"],
                                                "runtime_peak_allocated_bytes": timing.get("peak_allocated_bytes"),
                                                "runtime_peak_increment_bytes": timing.get("peak_increment_bytes"),
                                            }
                                        )
                                    summary.flush()
                                    logger.info(
                                        "Completed round=%d %s context=%d batch=%d query_tokens=%d",
                                        round_index + 1,
                                        fmt,
                                        length,
                                        case_batch,
                                        queries,
                                    )
        metadata["status"] = "preflight_passed" if args.preflight_only else "complete"
        metadata.pop("active_case", None)
        return 0
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        logger.exception("Benchmark failed; completed rows are preserved")
        return 1
    finally:
        (destination / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(_main())
