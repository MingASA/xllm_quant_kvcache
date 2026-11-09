# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Finite P1 operator qualification; GPU execution requires --execute explicitly.

Import and --describe use no Torch/FlashInfer imports. Execute only inside an
already admitted, isolated GPU job. This tool neither chooses nor reserves GPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Q_HEADS, KV_HEADS, HEAD_DIM, PAGE_SIZE = 24, 4, 256, 128
FP8_SCALES = (0.125, 0.5)
NV_SCALES = (1.0, 1.0)
WARMUP, REPEATS, INNER = 10, 40, 5
PLAN_WARMUP, PLAN_REPEATS = 3, 20
WORKSPACE_BYTES = 128 * 2**20
ROUTE_ATOL, ROUTE_RTOL = 0.001, 0.001
_GATE_NAMES = (
    "nv_packed_equal", "nv_scale_equal", "nv_changed_input_graph_equal", "invalid_scale_rejected",
    "bf16_host_planner_equal", "fp8_host_planner_equal", "nv_host_planner_equal",
    "public_scale_d2h_observed", "validated_scale_d2h_absent",
)
torch: Any = None
flashinfer: Any = None


def describe(phase: str) -> dict[str, Any]:
    return {
        "schema": "native_kv_p1_qualification_v1", "phase": phase,
        "scope": "operator_only_no_model_or_service_claim",
        "cases": ([{"batch": 1, "context": 128, "query_length": 128},
                   {"batch": 2, "context": 129, "query_length": 1}] if phase == "pilot" else
                  [{"batch": 1, "context": 8192, "query_length": 8192},
                   {"batch": 32, "context": 2048, "query_length": 1},
                   {"batch": 32, "context": 16384, "query_length": 1}]),
        "geometry": {"q_heads": Q_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM, "page_size": PAGE_SIZE},
        "scales": {"fp8_e4m3": FP8_SCALES, "nvfp4": NV_SCALES}, "seed": 20261109,
        "timing": {"warmup": WARMUP, "repeats": REPEATS, "inner": INNER,
                   "plan_warmup": PLAN_WARMUP, "plan_repeats": PLAN_REPEATS},
        "route_tolerance": {"atol": ROUTE_ATOL, "rtol": ROUTE_RTOL},
        "writer_graph_only": True, "attention_graph_claim": False,
        "host_page_csr_reuse": False, "performance_acceptance_threshold": None,
        "planner_arms": {
            "original": {"query_indptr": "CUDA", "non_blocking": True},
            "candidate": {"query_indptr": "existing_CPU_host_view", "non_blocking": False},
        },
        "host_candidate_retention": {"minimum_total_wall_improvement": 0.10,
                                     "maximum_attention_gpu_regression": 0.05,
                                     "decision_owner": "external_frozen_admission"},
        "profiler_invocations_per_writer": 2,
    }


def _summary(samples: list[float]) -> dict[str, Any]:
    return {"samples": samples, "median": statistics.median(samples), "min": min(samples), "max": max(samples)}


def _event_time(operation: Callable[[], Any]) -> dict[str, Any]:
    for _ in range(WARMUP):
        operation()
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(REPEATS)]
    for start, end in pairs:
        start.record()
        for _ in range(INNER):
            operation()
        end.record()
    pairs[-1][1].synchronize()
    return {"unit": "us_per_call", "includes_host_dispatch_gaps": True,
            **_summary([start.elapsed_time(end) * 1000 / INNER for start, end in pairs])}


def _host_time(operation: Callable[[], Any]) -> dict[str, Any]:
    for _ in range(PLAN_WARMUP):
        operation()
    torch.cuda.synchronize()
    api, drained = [], []
    for _ in range(PLAN_REPEATS):
        started = time.perf_counter_ns()
        operation()
        returned = time.perf_counter_ns()
        torch.cuda.synchronize()
        finished = time.perf_counter_ns()
        api.append((returned - started) / 1000)
        drained.append((finished - started) / 1000)
    return {"unit": "us", "cpu_api_return": _summary(api), "including_async_drain": _summary(drained)}


def _nv_pages(num_pages: int) -> tuple[list[Any], tuple[Any, Any], tuple[Any, Any]]:
    from xllm.python.attention.native_kv_layout import nvfp4_page_views

    owners = [torch.zeros((num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM // 2 + HEAD_DIM // 16),
                          dtype=torch.uint8, device="cuda") for _ in range(2)]
    pairs = [nvfp4_page_views(owner, HEAD_DIM) for owner in owners]
    return owners, (pairs[0][0], pairs[1][0]), (pairs[0][1], pairs[1][1])


def _writer_case(case: dict[str, int], measure: bool) -> dict[str, Any]:
    from xllm.python.attention.nvfp4_writer import PrevalidatedNVFP4Writer

    rows = case["batch"] * case["query_length"]
    num_pages = (rows + PAGE_SIZE - 1) // PAGE_SIZE
    owners, pages, sf = _nv_pages(num_pages)
    control_owners, control_pages, control_sf = _nv_pages(num_pages)
    writer = PrevalidatedNVFP4Writer({0: pages}, {0: sf}, NV_SCALES)
    public_scales = tuple(torch.tensor([s], dtype=torch.float32, device="cuda") for s in NV_SCALES)
    key, value = [torch.randn((rows + 2, KV_HEADS, HEAD_DIM * 2), device="cuda", dtype=torch.bfloat16)[..., ::2]
                  for _ in range(2)]
    # Include ignored slots and extra input rows. The negative slot remains in
    # the same place when values change for Graph replay.
    slots = torch.arange(rows, dtype=torch.int64, device="cuda")
    slots[rows // 2] = -1

    def public() -> None:
        flashinfer.page.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
            key.contiguous(), value.contiguous(), slots, control_pages, control_sf, *public_scales, kv_layout="NHD")

    def candidate() -> None:
        writer.append(key.contiguous(), value.contiguous(), slots, 0)

    def equal_bytes() -> tuple[bool, bool]:
        return (all(torch.equal(left, right) for left, right in zip(pages, control_pages)),
                all(torch.equal(left.view(torch.uint8), right.view(torch.uint8))
                    for left, right in zip(sf, control_sf)))

    public()
    candidate()
    eager_packed_equal, eager_scale_equal = equal_bytes()
    # The native operator permits strided outer dimensions, but requires a
    # contiguous head dimension. Production retains contiguous() above; cover
    # the supported strided ABI here without claiming arbitrary strides work.
    strided_key, strided_value = [
        torch.empty((rows + 2, KV_HEADS * 2, HEAD_DIM), device="cuda", dtype=key.dtype)[:, ::2, :]
        for _ in range(2)
    ]
    strided_key.copy_(key)
    strided_value.copy_(value)
    flashinfer.page.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
        strided_key, strided_value, slots, control_pages, control_sf, *public_scales, kv_layout="NHD")
    writer.append(strided_key, strided_value, slots, 0)
    noncontiguous_packed_equal, noncontiguous_scale_equal = equal_bytes()
    for _ in range(3):
        candidate()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    graph_results = []
    for factor in (0.5, -0.25):
        key.mul_(factor)
        value.add_(factor)
        slots.copy_(torch.arange(rows, device="cuda", dtype=torch.int64).flip(0))
        slots[rows // 2] = -1
        public()
        graph.replay()
        torch.cuda.synchronize()
        packed_equal, sf_equal = equal_bytes()
        graph_results.append({"packed_equal": packed_equal, "scale_equal": sf_equal})
    invalid_scales = []
    for invalid in (0.0, -1.0, float("nan"), float("inf"), torch.finfo(torch.float32).tiny / 2):
        for scales in ((invalid, 1.0), (1.0, invalid)):
            try:
                PrevalidatedNVFP4Writer({0: pages}, {0: sf}, scales)
            except ValueError:
                invalid_scales.append(True)
            else:
                invalid_scales.append(False)
    result = {"case": case, "scales": NV_SCALES, "packed_equal": eager_packed_equal, "scale_equal": eager_scale_equal,
              "noncontiguous_packed_equal": noncontiguous_packed_equal,
              "noncontiguous_scale_equal": noncontiguous_scale_equal, "padding_and_extra_rows": True,
              "graph_changed_values_and_slots_bytes_equal": graph_results,
              "invalid_scales_rejected": all(invalid_scales)}
    if measure:
        result["timings"] = {"public": _event_time(public), "prevalidated": _event_time(candidate),
                             "public_host": _host_time(public), "prevalidated_host": _host_time(candidate)}
    result["passed"] = (eager_packed_equal and eager_scale_equal
                        and noncontiguous_packed_equal and noncontiguous_scale_equal
                        and all(r["packed_equal"] and r["scale_equal"] for r in graph_results)
                        and all(invalid_scales))
    return result


def _attention_case(case: dict[str, int], route: str, measure: bool) -> dict[str, Any]:
    from xllm.python.attention.flashinfer_metadata import _prefill_query_indptr

    batch, context, query_length = case["batch"], case["context"], case["query_length"]
    pages_per_seq = (context + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = batch * pages_per_seq
    # Physical cache storage includes unused tail entries; page CSR is built
    # from explicit allocated pages here and is identical in both planner arms.
    key, value = [torch.randn((num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
                  for _ in range(2)]
    query = torch.randn((batch * query_length, Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    query_host = torch.arange(batch + 1, dtype=torch.int32) * query_length
    query_device = query_host.to("cuda")
    indptr = (torch.arange(batch + 1, dtype=torch.int32) * pages_per_seq).to("cuda")
    indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    last = torch.full((batch,), (context - 1) % PAGE_SIZE + 1, dtype=torch.int32, device="cuda")
    scale_options: dict[str, Any] = {}
    kv_dtype, q_dtype, backend = torch.bfloat16, torch.bfloat16, "fa2"
    if route == "fp8_paged":
        pages = tuple((tensor.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                      for tensor, scale in zip((key, value), FP8_SCALES))
        query = query.to(torch.float8_e4m3fn)
        kv_dtype, q_dtype, backend = torch.float8_e4m3fn, torch.float8_e4m3fn, "fa3"
        scale_options = {"q_scale": 1.0, "k_scale": FP8_SCALES[0], "v_scale": FP8_SCALES[1]}
    elif route == "nvfp4_paged":
        owners, pages, sf = _nv_pages(num_pages)
        flashinfer.page.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
            key.flatten(0, 1), value.flatten(0, 1), torch.arange(num_pages * PAGE_SIZE, device="cuda"),
            pages, sf, *NV_SCALES, kv_layout="NHD")
        kv_dtype = torch.uint8
        scale_options = {"kv_cache_sf": sf, "k_scale": NV_SCALES[0], "v_scale": NV_SCALES[1]}
    else:
        pages = (key, value)
    ragged = route == "bf16_ragged"
    if ragged:
        # Remove tail allocation padding from the packed ragged K/V reference.
        key = key.view(batch, -1, KV_HEADS, HEAD_DIM)[:, :context].reshape(-1, KV_HEADS, HEAD_DIM)
        value = value.view(batch, -1, KV_HEADS, HEAD_DIM)[:, :context].reshape(-1, KV_HEADS, HEAD_DIM)
        kv_indptr = (torch.arange(batch + 1, dtype=torch.int32) * context).to("cuda")
        wrappers = [flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda"), "NHD", backend="fa2") for _ in range(2)]
    else:
        wrappers = [flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda"), "NHD", backend=backend) for _ in range(2)]
    metadata = SimpleNamespace(q_seq_lens_host=query_host)

    def plan(index: int) -> None:
        q_ptr = query_device if index == 0 else _prefill_query_indptr(metadata, query_device)
        options = dict(causal=True, sm_scale=HEAD_DIM**-0.5, q_data_type=q_dtype,
                       kv_data_type=kv_dtype, o_data_type=torch.bfloat16, non_blocking=index == 0)
        if ragged:
            wrappers[index].plan(q_ptr, kv_indptr, Q_HEADS, KV_HEADS, HEAD_DIM, **options)
        else:
            wrappers[index].plan(q_ptr, indptr, indices, last, Q_HEADS, KV_HEADS, HEAD_DIM, PAGE_SIZE, **options)

    def attention(index: int) -> Any:
        return (wrappers[index].run(query, key, value) if ragged else
                wrappers[index].run(query, pages, **scale_options))

    def prepare_and_attention(index: int) -> Any:
        plan(index)
        return attention(index)

    outputs = []
    for index in (0, 1):
        outputs.append(prepare_and_attention(index))
    error = (outputs[0].float() - outputs[1].float()).abs()
    equal = bool(torch.isclose(outputs[0], outputs[1], atol=ROUTE_ATOL, rtol=ROUTE_RTOL).all().item())
    finite = bool(torch.isfinite(outputs[0]).all().item() and torch.isfinite(outputs[1]).all().item())
    result = {"case": case, "route": route, "passed": equal and finite,
              "scales": FP8_SCALES if route == "fp8_paged" else NV_SCALES if route == "nvfp4_paged" else None,
              "all_outputs_finite": finite, "max_abs_error": error.max().item() if finite else None,
              "original_non_blocking": True, "candidate_non_blocking": False,
              "query_host_reused": _prefill_query_indptr(metadata, query_device) is query_host,
              "page_metadata_unchanged": True, "attention_graph": False}
    if measure:
        result["planner_timings"] = {"device_query": _host_time(lambda: plan(0)),
                                     "host_query": _host_time(lambda: plan(1))}
        result["prepare_and_attention_timings"] = {
            "original": _host_time(lambda: prepare_and_attention(0)),
            "candidate": _host_time(lambda: prepare_and_attention(1)),
        }
        result["attention_event_timings"] = {"original": _event_time(lambda: attention(0)),
                                             "candidate": _event_time(lambda: attention(1))}
    return result


def _profile_scale_d2h(directory: Path) -> dict[str, Any]:
    from xllm.python.attention.nvfp4_writer import PrevalidatedNVFP4Writer

    directory.mkdir(exist_ok=False)
    owners, pages, sf = _nv_pages(1)
    writer = PrevalidatedNVFP4Writer({0: pages}, {0: sf}, NV_SCALES)
    key, value = [torch.randn((32, KV_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    slots = torch.arange(32, dtype=torch.int64, device="cuda")

    def public() -> None:
        flashinfer.page.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
            key, value, slots, pages, sf, *writer.global_scales, kv_layout="NHD")

    jobs = {"public": public, "prevalidated": lambda: writer.append(key, value, slots, 0)}
    receipts = []
    for label, operation in jobs.items():
        operation()
        torch.cuda.synchronize()
        for invocation in (1, 2):
            path = directory / f"{label}.{invocation}.trace.json"
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with torch.profiler.record_function(f"{label}/call_{invocation}"):
                    operation()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
            kernels = [event for event in events if event.get("cat") == "kernel" and event.get("ph") == "X"]
            copies = [event for event in events if "memcpy" in event.get("cat", "").lower() and event.get("ph") == "X"]
            d2h = [event for event in copies if any(marker in event.get("name", "").lower()
                                                   for marker in ("dtoh", "device -> host", "device to host", "d2h"))]
            receipts.append({"writer": label, "invocation": invocation, "trace": str(path),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "gpu_kernel_events": len(kernels), "d2h_copy_events": len(d2h),
                             "d2h_copies": d2h, "all_memcpy_events": copies})
    passed = all(r["gpu_kernel_events"] > 0 and ((r["d2h_copy_events"] > 0) if r["writer"] == "public"
                                                else (r["d2h_copy_events"] == 0)) for r in receipts)
    return {"passed": passed, "scales": NV_SCALES, "invocations": receipts,
            "scope": "isolated_fixed_scale_eager_writer_calls; not end-to-end attribution"}


def _run(phase: str, output: Path, result: dict[str, Any]) -> None:
    global torch, flashinfer
    import torch as torch_module
    import flashinfer as flashinfer_module
    import flashinfer.page

    torch, flashinfer = torch_module, flashinfer_module
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("P1 qualification requires exactly one admitted visible SM90 GPU")
    torch.manual_seed(20261109)
    torch.cuda.manual_seed_all(20261109)
    result["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                         "flashinfer": getattr(flashinfer, "__version__", None),
                         "page_sha256": hashlib.sha256(Path(flashinfer.page.__file__).read_bytes()).hexdigest()}
    result["writer_cases"], result["attention_cases"] = [], []
    for case in result["description"]["cases"]:
        result["writer_cases"].append(_writer_case(case, phase == "main"))
        for route in ("bf16_ragged", "bf16_paged", "fp8_paged", "nvfp4_paged"):
            result["attention_cases"].append(_attention_case(case, route, phase == "main"))
    result["scale_d2h_profiles"] = _profile_scale_d2h(output.with_suffix(".traces"))
    writers = result["writer_cases"]
    profiles = result["scale_d2h_profiles"]["invocations"]
    gates = result["gates"]
    gates["nv_packed_equal"] = bool(writers) and all(r["packed_equal"] and r["noncontiguous_packed_equal"] for r in writers)
    gates["nv_scale_equal"] = bool(writers) and all(r["scale_equal"] and r["noncontiguous_scale_equal"] for r in writers)
    gates["nv_changed_input_graph_equal"] = bool(writers) and all(
        len(r["graph_changed_values_and_slots_bytes_equal"]) == 2
        and all(g["packed_equal"] and g["scale_equal"] for g in r["graph_changed_values_and_slots_bytes_equal"])
        for r in writers)
    gates["invalid_scale_rejected"] = bool(writers) and all(r["invalid_scales_rejected"] for r in writers)
    for prefix, routes in (("bf16", ("bf16_ragged", "bf16_paged")),
                           ("fp8", ("fp8_paged",)), ("nv", ("nvfp4_paged",))):
        rows = [r for r in result["attention_cases"] if r["route"] in routes]
        gates[prefix + "_host_planner_equal"] = len(rows) == len(writers) * len(routes) > 0 and all(r["passed"] for r in rows)
    for label, gate in (("public", "public_scale_d2h_observed"), ("prevalidated", "validated_scale_d2h_absent")):
        calls = [r for r in profiles if r["writer"] == label]
        gates[gate] = len(calls) == 2 and all(r["gpu_kernel_events"] > 0 and
            (r["d2h_copy_events"] > 0 if label == "public" else r["d2h_copy_events"] == 0) for r in calls)
    result["qualified_formats"] = []
    if gates["bf16_host_planner_equal"]:
        result["qualified_formats"].append("bf16")
    if gates["fp8_host_planner_equal"]:
        result["qualified_formats"].append("fp8_e4m3")
    nv_gates = ("nv_packed_equal", "nv_scale_equal", "nv_changed_input_graph_equal", "invalid_scale_rejected",
                "nv_host_planner_equal", "public_scale_d2h_observed", "validated_scale_d2h_absent")
    if all(gates[name] for name in nv_gates):
        result["qualified_formats"].append("nvfp4")
    result["passed"] = (all(gates.values()) and all(r["passed"] for r in writers + result["attention_cases"])
                        and result["scale_d2h_profiles"]["passed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe(args.phase), sort_keys=True))
        return
    if not args.execute or args.output is None:
        parser.error("GPU work requires --execute and an exclusive --output JSON path")
    result = {"schema": "native_kv_p1_qualification_v1", "phase": args.phase,
              "description": describe(args.phase), "passed": False,
              "gates": {name: False for name in _GATE_NAMES}, "qualified_formats": []}
    # Reserve before importing CUDA libraries; always preserve partial evidence.
    with args.output.open("x", encoding="utf-8") as handle:
        try:
            _run(args.phase, args.output, result)
        except Exception as error:
            result["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
