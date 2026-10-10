# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Separate attention GPU, eager dispatch, and prepare costs across small batches."""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts.logger import logger
from tools.benchmark_flashinfer_mixed_v import _capture, _measure
from tools.diagnose_vonly_backend_numerics import _metadata
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.flashinfer_mixed import FlashInferMixedVBackend
from xllm.python.attention.quantized import KVCacheCodec, QuantizedPagedAttentionBackend


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh result file")
    result = dict(
        protocol="5 randomized repetitions; graph GPU 16x10 and eager 50 calls; prepare separate",
        context=768,
        page_size=128,
        cases=[],
    )
    for batch in (1, 2, 8, 20):
        for query_len in (1, 128):
            torch.manual_seed(20261010 + batch + query_len)
            metadata = _metadata(query_len, 768, batch)
            key = torch.randn(batch * 6, 128, 2, 128, dtype=torch.bfloat16, device="cuda")
            value = torch.randn_like(key)
            query = torch.randn(batch * query_len, 12, 128, dtype=torch.bfloat16, device="cuda")
            layer = SimpleNamespace(
                num_heads=12, num_kv_heads=2, head_dim=128, layer_id=0, scale=128**-0.5, causal=True, sliding_window=0
            )
            calls, backends, raw = {}, {}, {}
            for arm in ("bf16", "bf16_int8", "bf16_int4"):
                codec = KVCacheCodec(arm.removeprefix("bf16_"), 128)
                payload, scales = codec.encode(value)
                cache = LayerCache(key, payload, value_scale=scales)
                for name in ("triton", "flashinfer_mixed"):
                    backend = (
                        QuantizedPagedAttentionBackend(arm, 128, 2)
                        if name == "triton"
                        else FlashInferMixedVBackend(arm, 12, 2, 128, 128**-0.5, 0, key.device, torch.bfloat16)
                    )
                    backend.bind_kv_caches([cache])
                    backend.prepare(metadata)
                    label = f"{name}/{arm}"
                    backends[label] = backend
                    calls[label] = lambda b=backend, q=query, attention=layer: b.execute_attention(q, attention)
                    raw[label] = dict(gpu_us=[], eager_us=[], prepare_wall_us=[])
            graphs = {name: _capture(call) for name, call in calls.items()}
            rng = random.Random(20261010 + batch + query_len)
            for _ in range(5):
                order = list(calls)
                rng.shuffle(order)
                for name in order:
                    gpu, eager = _measure(graphs[name], calls[name])
                    raw[name]["gpu_us"].append(gpu)
                    raw[name]["eager_us"].append(eager)
            # prepare() replaces Triton's metadata allocations. A captured
            # graph must finish before those allocations can be released.
            del graphs
            for name in order:
                for _ in range(50):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    backends[name].prepare(metadata)
                    raw[name]["prepare_wall_us"].append((time.perf_counter() - start) * 1e6)
            result["cases"].append(
                dict(
                    batch=batch,
                    query_len=query_len,
                    raw=raw,
                    median={name: {k: statistics.median(v) for k, v in values.items()} for name, values in raw.items()},
                )
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            logger.info("Latency decomposition finished: B%d Q%d", batch, query_len)
            del calls, backends
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
