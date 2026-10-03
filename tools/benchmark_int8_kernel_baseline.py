# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture CUDA-event and wall baselines for selected INT8 attention workloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xllm.python.attention.quantized as quantized_attention
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.quantized import KVCacheCodec, QuantizedPagedAttentionBackend, write_quantized_kv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _command(arguments: list[str]) -> str:
    try:
        return subprocess.check_output(arguments, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)


def _benchmark_case(
    name: str,
    query_lengths: list[int],
    context_length: int,
    page_size: int,
    seed: int,
    warmup: int,
    iterations: int,
) -> tuple[dict, torch.Tensor]:
    device = torch.device("cuda")
    batch_size = len(query_lengths)
    num_heads, num_kv_heads, head_dim = 12, 2, 128
    pages_per_sequence = math.ceil(context_length / page_size)
    num_blocks = batch_size * pages_per_sequence
    generator = torch.Generator(device="cpu").manual_seed(seed)

    query_cpu = torch.randn((sum(query_lengths), num_heads, head_dim), generator=generator, dtype=torch.bfloat16)
    key_cpu = torch.randn(
        (batch_size, context_length, num_kv_heads, head_dim), generator=generator, dtype=torch.bfloat16
    )
    value_cpu = torch.randn(
        (batch_size, context_length, num_kv_heads, head_dim), generator=generator, dtype=torch.bfloat16
    )
    query = query_cpu.to(device)
    key = key_cpu.to(device)
    value = value_cpu.to(device)

    page_order = torch.randperm(num_blocks, generator=generator, dtype=torch.int64)
    pages_cpu = page_order.reshape(batch_size, pages_per_sequence).to(torch.int32)
    pages = pages_cpu.to(device)
    positions = torch.arange(context_length, dtype=torch.int64)
    slots_cpu = pages_cpu[:, positions // page_size].to(torch.int64) * page_size + positions % page_size
    cache_slots = slots_cpu.reshape(-1).to(device=device, dtype=torch.int64)
    query_slots = []
    offset = 0
    for sequence, query_length in enumerate(query_lengths):
        query_slots.extend(slots_cpu[sequence, -query_length:].tolist())
        offset += query_length

    codec = KVCacheCodec("int8", head_dim)
    cache_shape = (num_blocks, page_size, num_kv_heads, codec.storage_dim)
    cache = LayerCache(
        key=torch.zeros(cache_shape, device=device, dtype=codec.storage_dtype),
        value=torch.zeros(cache_shape, device=device, dtype=codec.storage_dtype),
        key_scale=torch.ones(cache_shape[:-1], device=device, dtype=torch.float32),
        value_scale=torch.ones(cache_shape[:-1], device=device, dtype=torch.float32),
    )
    write_quantized_kv(
        cache,
        key.reshape(-1, num_kv_heads, head_dim),
        value.reshape(-1, num_kv_heads, head_dim),
        cache_slots,
        codec,
    )

    metadata = SimpleNamespace(
        block_table=pages,
        q_seq_lens_host=torch.tensor(query_lengths, dtype=torch.int32),
        kv_seq_lens_host_values=[context_length] * batch_size,
        kv_seq_lens_host=torch.full((batch_size,), context_length, dtype=torch.int32),
        slot_mapping=torch.tensor(query_slots, device=device, dtype=torch.int64),
    )
    backend = QuantizedPagedAttentionBackend("int8", head_dim, num_kv_heads)
    backend.bind_kv_caches([cache])
    backend.prepare(metadata)
    layer = SimpleNamespace(
        layer_id=0,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=head_dim**-0.5,
        sliding_window=-1,
        causal=True,
    )

    def run() -> torch.Tensor:
        return backend.execute_attention(query, layer)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    wall_samples_ms = []
    event_samples_ms = []
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start_event.record()
        output = run()
        end_event.record()
        end_event.synchronize()
        wall_samples_ms.append((time.perf_counter_ns() - wall_start) / 1e6)
        event_samples_ms.append(start_event.elapsed_time(end_event))
    if not torch.isfinite(output).all().item():
        raise RuntimeError(f"{name}: attention output contains nonfinite values")

    result = {
        "name": name,
        "format": "int8",
        "device": torch.cuda.get_device_name(),
        "batch": batch_size,
        "query_lengths": query_lengths,
        "max_query_tokens": max(query_lengths),
        "total_query_tokens": sum(query_lengths),
        "context_length": context_length,
        "page_size": page_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "cache_blocks": num_blocks,
        "cache_bytes": sum(
            t.numel() * t.element_size() for t in (cache.key, cache.value, cache.key_scale, cache.value_scale)
        ),
        "seed": seed,
        "query_sha256": _tensor_sha256(query_cpu),
        "key_sha256": _tensor_sha256(key_cpu),
        "value_sha256": _tensor_sha256(value_cpu),
        "page_table_sha256": _tensor_sha256(pages_cpu),
        "layout": {
            "query_shape": list(query.shape),
            "query_stride": list(query.stride()),
            "cache_shape": list(cache.key.shape),
            "cache_stride": list(cache.key.stride()),
            "block_table_shape": list(pages.shape),
            "block_table_stride": list(pages.stride()),
            "page_permutation_sha256": _tensor_sha256(page_order),
        },
        "cache_key_sha256": _tensor_sha256(cache.key),
        "cache_value_sha256": _tensor_sha256(cache.value),
        "cache_key_scale_sha256": _tensor_sha256(cache.key_scale),
        "cache_value_scale_sha256": _tensor_sha256(cache.value_scale),
        "wall_ms": {
            "p50": statistics.median(wall_samples_ms),
            "p95": sorted(wall_samples_ms)[math.ceil(len(wall_samples_ms) * 0.95) - 1],
            "samples": wall_samples_ms,
        },
        "cuda_event_ms": {
            "p50": statistics.median(event_samples_ms),
            "p95": sorted(event_samples_ms)[math.ceil(len(event_samples_ms) * 0.95) - 1],
            "samples": event_samples_ms,
        },
        "output_sha256": _tensor_sha256(output),
    }
    return result, output.detach().to(device="cpu").contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/mnt/e/AI/xllm-eval-data/int8-kernel-optimization-20261001")
    )
    parser.add_argument(
        "--kernel-snapshot",
        type=Path,
        default=Path(
            "/mnt/e/AI/xllm-eval-data/int8-kernel-optimization-20261001/"
            "old_kernel_snapshot_20261001T000000Z/quantized_triton.py"
        ),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument(
        "--long-contexts",
        type=int,
        nargs="*",
        default=[],
        help="Also benchmark C16 mixed [4096] + 15x[1] at these contexts",
    )
    parser.add_argument(
        "--prefill-contexts",
        type=int,
        nargs="*",
        default=[],
        help="Also benchmark C1 Q4096 prefill at these contexts",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Run only workloads specified by --long-contexts",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.page_size <= 0:
        parser.error("warmup must be nonnegative; iterations and page-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark does not fall back to CPU")

    destination = args.output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_int8_kernel_baseline")
    destination.mkdir(parents=True, exist_ok=False)
    kernel_path = ROOT / "xllm/python/attention/quantized_triton.py"
    if not args.kernel_snapshot.is_file():
        raise FileNotFoundError(f"Old-kernel snapshot does not exist: {args.kernel_snapshot}")
    snapshot_spec = importlib.util.spec_from_file_location("xllm_int8_kernel_baseline_snapshot", args.kernel_snapshot)
    if snapshot_spec is None or snapshot_spec.loader is None:
        raise RuntimeError(f"Cannot load old-kernel snapshot: {args.kernel_snapshot}")
    snapshot_module = importlib.util.module_from_spec(snapshot_spec)
    snapshot_spec.loader.exec_module(snapshot_module)
    quantized_attention._triton_quantized_paged_attention = snapshot_module.quantized_paged_attention
    environment = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args) | {"output_dir": str(args.output_dir), "kernel_snapshot": str(args.kernel_snapshot)},
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_capability": torch.cuda.get_device_capability(),
        "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
        "nvidia_smi": _command(["/usr/lib/wsl/lib/nvidia-smi"]),
        "git_head": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
        "harness_sha256": _sha256(Path(__file__).resolve()),
        "current_worktree_kernel_path": str(kernel_path),
        "kernel_source_path": str(args.kernel_snapshot),
        "kernel_source_sha256": _sha256(args.kernel_snapshot),
        "current_worktree_kernel_sha256": _sha256(kernel_path),
        "quantized_backend_sha256": _sha256(ROOT / "xllm/python/attention/quantized.py"),
    }
    (destination / "old_quantized_triton.py").write_bytes(args.kernel_snapshot.read_bytes())
    (destination / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    requested_contexts = [*args.long_contexts, *args.prefill_contexts]
    if args.long_only and not requested_contexts:
        parser.error("--long-only requires --long-contexts or --prefill-contexts")
    workloads = (
        []
        if args.long_only
        else [
            ("prefill_c1_q128_ctx4k", [128]),
            ("decode_c16_q1_ctx4k", [1] * 16),
            ("mixed_c16_q1024_plus_15xq1_ctx4k", [1024] + [1] * 15),
        ]
    )
    if any(context < 4096 for context in requested_contexts):
        parser.error("Long-context workloads require context length >= 4096")
    workloads.extend(
        (f"mixed_c16_q4096_plus_15xq1_ctx{context}", [4096] + [1] * 15, context) for context in args.long_contexts
    )
    workloads.extend((f"prefill_c1_q4096_ctx{context}", [4096], context) for context in args.prefill_contexts)
    try:
        with (destination / "results.jsonl").open("w") as stream:
            for index, workload in enumerate(workloads):
                name, query_lengths = workload[:2]
                context_length = workload[2] if len(workload) == 3 else 4096
                result, output = _benchmark_case(
                    name,
                    query_lengths,
                    context_length=context_length,
                    page_size=args.page_size,
                    seed=args.seed + index,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                output_path = destination / f"{name}_output.pt"
                torch.save(output, output_path)
                result["output_file"] = str(output_path)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(
                    f"{name}: event p50={result['cuda_event_ms']['p50']:.3f} ms, "
                    f"wall p50={result['wall_ms']['p50']:.3f} ms",
                    flush=True,
                )
        environment["status"] = "complete"
        return 0
    except Exception as error:
        environment.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (destination / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
