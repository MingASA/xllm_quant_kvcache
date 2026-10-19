# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Calibrate mixed-V FA2 against unchanged FlashInfer BF16; kernel diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts.logger import logger
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.flashinfer_mixed import FlashInferMixedVBackend
from xllm.python.attention.quantized import KVCacheCodec

_ARMS = ("auto", "bf16_int8", "bf16_fp8_e4m3", "bf16_int4")


def _capture(call: Callable[[], torch.Tensor]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(20):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(16):
            call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def _measure(graph: torch.cuda.CUDAGraph, call: Callable[[], torch.Tensor]) -> tuple[float, float]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    gpu_us = start.elapsed_time(end) * 1000 / 160
    wall_start = time.perf_counter()
    for _ in range(50):
        call()
    torch.cuda.synchronize()
    return gpu_us, (time.perf_counter() - wall_start) * 1e6 / 50


def _case(
    batch: int, query_len: int, length: int, repeats: int, gates: dict, num_heads: int = 12, num_kv_heads: int = 2
) -> dict:
    torch.manual_seed(20261010 + batch + length)
    page = 128
    page_count = (length + page - 1) // page
    pages = batch * page_count
    key = torch.randn(pages, page, num_kv_heads, 128, dtype=torch.bfloat16, device="cuda")
    values = torch.randn_like(key)
    query = torch.randn(batch * query_len, num_heads, 128, dtype=torch.bfloat16, device="cuda")
    offsets = torch.arange(batch + 1, dtype=torch.int32) * query_len
    metadata = SimpleNamespace(
        q_seq_lens_host=offsets,
        qo_indptr=offsets.cuda(),
        q_cu_seq_lens=None,
        paged_kv_indptr=(torch.arange(batch + 1, dtype=torch.int32) * page_count).cuda(),
        paged_kv_indices=torch.randperm(pages, dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len=torch.full((batch,), (length - 1) % page + 1, dtype=torch.int32, device="cuda"),
        slot_mapping=torch.zeros(query.shape[0], dtype=torch.int64, device="cuda"),
    )
    layer = SimpleNamespace(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        layer_id=0,
        scale=128**-0.5,
        causal=True,
        sliding_window=0,
    )
    calls: dict[str, Callable[[], torch.Tensor]] = {}
    keepalive = []
    resident = {}
    for arm in _ARMS:
        backend = FlashInferMixedVBackend(arm, num_heads, num_kv_heads, 128, 128**-0.5, 0, key.device, torch.bfloat16)
        if any(backend.provenance[field] != gates["native_build"][field] for field in ("build_key", "artifact_sha256")):
            raise ValueError("Kernel correctness attestation is stale")
        codec = KVCacheCodec("bf16" if arm == "auto" else arm.removeprefix("bf16_"), 128)
        payload, scales = codec.encode(values)
        backend.bind_kv_caches([LayerCache(key, payload, value_scale=scales)])
        backend.prepare(metadata)
        calls[arm] = lambda b=backend: b.execute_attention(query, layer)
        resident[arm] = key.nbytes + payload.nbytes + (scales.nbytes if scales is not None else 0)
        keepalive.append(backend)
        if arm == "auto":
            calls["original_flashinfer_bf16_fa2"] = lambda b=backend: b._planner.run(
                query, (key, values), enable_pdl=False
            )
    torch.testing.assert_close(
        calls["auto"]().reshape_as(query), calls["original_flashinfer_bf16_fa2"](), atol=0, rtol=0
    )
    graphs = {name: _capture(call) for name, call in calls.items()}
    raw = {name: {"gpu_us": [], "eager_us": []} for name in calls}
    rng = random.Random(20261010 + batch + length)
    for _ in range(repeats):
        order = list(calls)
        rng.shuffle(order)
        for name in order:
            gpu, eager = _measure(graphs[name], calls[name])
            raw[name]["gpu_us"].append(gpu)
            raw[name]["eager_us"].append(eager)
    return dict(
        batch=batch,
        query_tokens_per_request=query_len,
        context_tokens=length,
        page_size=page,
        resident_kv_bytes=resident,
        raw=raw,
        median={
            name: {metric: statistics.median(values) for metric, values in data.items()} for name, data in raw.items()
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--num-qo-heads", type=int, default=12)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    args = parser.parse_args()
    if (args.num_qo_heads, args.num_kv_heads) not in ((12, 2), (28, 4), (40, 8)):
        parser.error("Use an attested Qwen2.5 head configuration")
    if args.output.exists() or args.repetitions < 3:
        parser.error("Use a fresh output file and at least three repetitions")
    gates = json.loads(args.gates.read_text())
    if gates.get("attention_backend") != "flashinfer_mixed" or not all(
        gates.get(arm, {}).get("passed") for arm in _ARMS
    ):
        raise ValueError("All four mixed-V correctness gates must pass")
    if [args.num_qo_heads, args.num_kv_heads, 128] not in gates.get("supported_shapes", [[12, 2, 128]]):
        raise ValueError("Correctness gates do not cover the requested model shape")
    for name, expected in gates["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Correctness source attestation is stale: {name}")
    result = dict(
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        scope="kernel-only diagnostic; not server throughput or TPOT",
        repetitions=args.repetitions,
        protocol="Warmed, randomized arm order, graph batch 16 nodes x 10 replays; eager wall 50 calls",
        native_build=gates["native_build"],
        pdl=False,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu,clocks.sm,clocks.mem", "--format=csv"],
            text=True,
        ),
        cases=[],
    )
    cases = [(b, 1, n) for b in (8, 32, 128) for n in (512, 2048, 8192)]
    cases += [(1, 512, 512), (8, 128, 2048), (1, 4096, 8192)]
    for batch, q_len, length in cases:
        result["cases"].append(
            _case(batch, q_len, length, args.repetitions, gates, args.num_qo_heads, args.num_kv_heads)
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        logger.info("Completed kernel calibration: batch=%d q=%d context=%d", batch, q_len, length)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
