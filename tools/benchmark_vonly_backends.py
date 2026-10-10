# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Measure V-only INT8/INT4 attention and the shared writer at batch/concurrency 20."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts.logger import logger
from tools.benchmark_flashinfer_mixed_v import _capture, _measure
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.flashinfer_mixed import FlashInferMixedVBackend
from xllm.python.attention.quantized import KVCacheCodec, QuantizedPagedAttentionBackend
from xllm.python.attention.quantized_triton import write_quantized_kv


def _case(query_len: int, length: int, repeats: int, gates: dict[str, dict]) -> dict:
    torch.manual_seed(20261010 + query_len + length)
    batch, page = 20, 128
    page_count = (length + page - 1) // page
    pages = batch * page_count
    key = torch.randn(pages, page, 2, 128, dtype=torch.bfloat16, device="cuda")
    values = torch.randn_like(key)
    query = torch.randn(batch * query_len, 12, 128, dtype=torch.bfloat16, device="cuda")
    offsets = torch.arange(batch + 1, dtype=torch.int32) * query_len
    indices = torch.randperm(pages, dtype=torch.int32, device="cuda")
    blocks = indices.reshape(batch, page_count)
    positions = torch.arange(length - query_len, length, device="cuda")
    slots = (blocks[:, positions // page].long() * page + positions % page).flatten()
    metadata = SimpleNamespace(
        q_seq_lens_host=offsets,
        qo_indptr=offsets.cuda(),
        q_cu_seq_lens=None,
        kv_seq_lens_host_values=[length] * batch,
        kv_seq_lens_host=None,
        paged_kv_indptr=(torch.arange(batch + 1, dtype=torch.int32) * page_count).cuda(),
        paged_kv_indices=indices,
        paged_kv_last_page_len=torch.full((batch,), (length - 1) % page + 1, dtype=torch.int32, device="cuda"),
        slot_mapping=slots,
        block_table=blocks,
    )
    layer = SimpleNamespace(
        num_heads=12, num_kv_heads=2, head_dim=128, layer_id=0, scale=128**-0.5, causal=True, sliding_window=0
    )
    calls = {}
    keepalive = []
    resident = {}
    numerical = {}
    for arm in ("bf16_int8", "bf16_int4"):
        codec = KVCacheCodec(arm.removeprefix("bf16_"), 128)
        payload, scales = codec.encode(values)
        cache = LayerCache(key, payload, value_scale=scales)
        resident[arm] = key.nbytes + payload.nbytes + scales.nbytes
        outputs = {}
        for name in ("triton", "flashinfer_mixed"):
            backend = (
                QuantizedPagedAttentionBackend(arm, 128, 2)
                if name == "triton"
                else FlashInferMixedVBackend(arm, 12, 2, 128, 128**-0.5, 0, key.device, torch.bfloat16)
            )
            if name == "flashinfer_mixed":
                for field in ("build_key", "artifact_sha256"):
                    if backend.provenance[field] != gates[name]["native_build"][field]:
                        raise ValueError("Native correctness gate is stale")
            backend.bind_kv_caches([cache])
            backend.prepare(metadata)
            label = f"{name}/{arm}"
            calls[label] = lambda b=backend: b.execute_attention(query, layer)
            outputs[name] = calls[label]().clone()
            keepalive.append(backend)
        torch.testing.assert_close(outputs["triton"], outputs["flashinfer_mixed"], atol=0.015, rtol=0.02)
        numerical[arm] = (outputs["triton"].float() - outputs["flashinfer_mixed"].float()).abs().max().item()
        appended_key = key.flatten(0, 1)[slots].clone()
        appended_value = values.flatten(0, 1)[slots].clone()

        def _writer(
            k: torch.Tensor = appended_key,
            v: torch.Tensor = appended_value,
            c: LayerCache = cache,
            fmt: str = codec.cache_dtype,
        ) -> torch.Tensor:
            write_quantized_kv(k, v, slots, c.key, c.value, None, c.value_scale, fmt, 128, bf16_key=True)
            return c.value

        calls[f"shared_writer/{arm}"] = _writer
    graphs = {name: _capture(call) for name, call in calls.items()}
    raw = {name: {"gpu_us": [], "eager_us": []} for name in calls}
    rng = random.Random(20261010 + query_len + length)
    for _ in range(repeats):
        order = list(calls)
        rng.shuffle(order)
        for name in order:
            gpu, eager = _measure(graphs[name], calls[name])
            raw[name]["gpu_us"].append(gpu)
            raw[name]["eager_us"].append(eager)
    return dict(
        batch=20,
        query_tokens_per_request=query_len,
        context_tokens=length,
        page_size=128,
        raw=raw,
        resident_kv_bytes=resident,
        backend_max_abs_difference=numerical,
        median={
            name: {metric: statistics.median(values) for metric, values in data.items()} for name, data in raw.items()
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-gates", type=Path, required=True)
    parser.add_argument("--flashinfer-gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 3:
        parser.error("Use a fresh result file and at least 3 repetitions")
    gates = {
        name: json.loads(path.read_text())
        for name, path in (("triton", args.triton_gates), ("flashinfer_mixed", args.flashinfer_gates))
    }
    for name, gate in gates.items():
        if gate.get("attention_backend", "triton") != name:
            raise ValueError(f"Wrong correctness backend: {name}")
        if not all(gate.get(arm, {}).get("passed") for arm in ("bf16_int8", "bf16_int4")):
            raise ValueError("Both V-only formats must pass correctness")
        for source, expected in gate["source_sha256"].items():
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Stale correctness source: {source}")
    result = dict(
        scope="attention-only and shared writer microbenchmark; not service latency",
        batch=20,
        repetitions=args.repetitions,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        native_build=gates["flashinfer_mixed"]["native_build"],
        protocol="same Q/K/V and shuffled pages; warmed; randomized arm/backend order; GPU event graph 16x10; eager 50 calls",
        cases=[],
    )
    for query_len, length in ((1, 768), (1, 1024), (1, 8192), (768, 768)):
        result["cases"].append(_case(query_len, length, args.repetitions, gates))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        logger.info("Completed V-only comparison: B20 Q%d L%d", query_len, length)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
