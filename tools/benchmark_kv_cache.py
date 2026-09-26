# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Recorded, single-layer benchmark of the actual experimental KV backend."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
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


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _measure(fn: Callable, args: argparse.Namespace) -> dict:
    for _ in range(args.warmup):
        fn()
    _sync(args.device)
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        fn()
        _sync(args.device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[math.ceil(len(samples) * 0.95) - 1],
        "samples_ms": samples,
    }


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


@torch.inference_mode()
def _case(args: argparse.Namespace, length: int, fmt: str) -> dict:
    batch, queries, dim = args.batch, args.query_tokens, args.head_dim
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
    reference = _dense(q.float(), k.float(), v.float())
    row = {
        "format": fmt,
        "context": length,
        "batch": batch,
        "query_tokens": queries,
        "head_dim": dim,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "page_size": args.page_size,
        "device": args.device,
        "bf16_paged_bytes": 2 * blocks * args.page_size * args.kv_heads * dim * 2,
    }
    if fmt == "bf16":
        execute = lambda: _dense(q, k, v)
        row["baseline_kind"] = "contiguous_torch_sdpa_not_xllm_paged"
        row["cache_tensor_bytes"] = (k.numel() + v.numel()) * k.element_size()
        actual = execute()
        row["implementation_max_abs"] = None
        stages = {"attention_only": execute}
    else:
        codec = KVCacheCodec(fmt, dim)
        shape = (blocks, args.page_size, args.kv_heads, codec.storage_dim)
        cache = LayerCache(
            key=torch.zeros(shape, device=args.device, dtype=codec.storage_dtype),
            value=torch.zeros(shape, device=args.device, dtype=codec.storage_dtype),
            key_scale=torch.ones(shape[:-1], device=args.device),
            value_scale=torch.ones(shape[:-1], device=args.device),
        )
        write_quantized_kv(cache, k.flatten(0, 1), v.flatten(0, 1), slots.flatten(), codec)
        backend = QuantizedPagedAttentionBackend(fmt, dim, args.kv_heads)
        backend.bind_kv_caches([cache])
        metadata = SimpleNamespace(
            block_table=pages.to(device=args.device, dtype=torch.int32),
            q_seq_lens_host=torch.full((batch,), queries, dtype=torch.int32),
            kv_seq_lens_host_values=[length] * batch,
            slot_mapping=slots[:, -queries:].reshape(-1),
        )
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
        backend.prepare(metadata)
        execute = lambda: backend.execute(q.reshape(-1, args.heads, dim), new_k, new_v, layer)
        actual = execute()

        def decoded(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            return codec.decode(payload, scale).flatten(0, 1)[slots].reshape_as(k)

        # Separate algorithm correctness from expected lossy-quantization error.
        quant_reference = _dense(
            q.float(), decoded(cache.key, cache.key_scale), decoded(cache.value, cache.value_scale)
        )
        torch.testing.assert_close(actual.float(), quant_reference, rtol=0.02, atol=0.005)
        row["implementation_max_abs"] = (actual.float() - quant_reference).abs().max().item()
        del quant_reference
        row["cache_tensor_bytes"] = sum(
            t.numel() * t.element_size() for t in (cache.key, cache.value, cache.key_scale, cache.value_scale)
        )
        row["baseline_kind"] = "xllm_quantized_eager_paged"

        def prepare_execute() -> torch.Tensor:
            backend.prepare(metadata)
            return execute()

        stages = {
            "write_only": lambda: write_quantized_kv(cache, new_k, new_v, metadata.slot_mapping, codec),
            "prepare_only": lambda: backend.prepare(metadata),
            "execute_including_write": execute,
            "prepare_execute": prepare_execute,
        }
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("Nonfinite attention output")
    error = actual.float() - reference
    row["quantization_relative_l2"] = (error.norm() / reference.norm().clamp_min(1e-12)).item()
    row["quantization_max_abs"] = error.abs().max().item()
    row["compression_vs_paged_bf16"] = row["bf16_paged_bytes"] / row["cache_tensor_bytes"]
    del reference, actual, error
    _sync(args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        row["allocated_before_timing_bytes"] = torch.cuda.memory_allocated()
    row["timings"] = {name: _measure(fn, args) for name, fn in stages.items()}
    if args.device == "cuda":
        row["peak_allocated_including_fixtures_bytes"] = torch.cuda.max_memory_allocated()
        row["peak_increment_over_setup_bytes"] = (
            torch.cuda.max_memory_allocated() - row["allocated_before_timing_bytes"]
        )
        row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
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
    for name, default in (
        ("batch", 1),
        ("query-tokens", 1),
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
    parser.add_argument("--output-dir", type=Path, default=Path("kv_benchmark_results"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if any(
        getattr(args, name) <= 0
        for name in ("batch", "query_tokens", "head_dim", "heads", "kv_heads", "page_size", "iterations", "threads")
    ):
        parser.error("All shape/count parameters must be positive")
    if args.warmup < 0 or args.heads % args.kv_heads or min(args.contexts) < args.query_tokens:
        parser.error("Require warmup >= 0, heads divisible by kv-heads, and context >= query-tokens")
    torch.set_num_threads(args.threads)
    destination = args.output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
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
                "xllm/python/attention/quantized.py",
                "xllm/python/attention/backend.py",
            )
        },
    }
    logger.info("Results: %s", destination.resolve())
    metadata["packages"] = _command([sys.executable, "-m", "pip", "freeze"])
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
                    "context",
                    "batch",
                    "query_tokens",
                    "stage",
                    "p50_ms",
                    "p95_ms",
                    "cache_tensor_bytes",
                    "quantization_relative_l2",
                ],
            )
            writer.writeheader()
            if not args.preflight_only:
                for length in args.contexts:
                    for fmt in args.formats:
                        metadata["active_case"] = {"format": fmt, "context": length}
                        row = _case(args, length, fmt)
                        raw.write(json.dumps(row) + "\n")
                        raw.flush()
                        for stage, timing in row["timings"].items():
                            writer.writerow(
                                {
                                    key: row[key]
                                    for key in (
                                        "format",
                                        "context",
                                        "batch",
                                        "query_tokens",
                                        "cache_tensor_bytes",
                                        "quantization_relative_l2",
                                    )
                                }
                                | {"stage": stage, "p50_ms": timing["p50_ms"], "p95_ms": timing["p95_ms"]}
                            )
                        summary.flush()
                        logger.info("Completed %s context=%d", fmt, length)
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
