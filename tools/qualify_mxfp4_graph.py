# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Bounded physical qualification; dispatch only through the external once guard.

No GPU work occurs without --execute. This tool does not admit, reserve, or
select a shared GPU. The Controller owns admission and must supply one device.
A failed receipt is retained and no existing output or trace is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _tensor_sha(tensor: Any) -> str:
    import torch

    return hashlib.sha256(tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _measure(call: Callable[[], Any], *, warmup: int, repeats: int, inner: int) -> dict[str, Any]:
    import torch

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        before = time.perf_counter_ns()
        for _ in range(inner):
            call()
        api_us = (time.perf_counter_ns() - before) / 1000 / inner
        end.record()
        end.synchronize()
        wall_us = (time.perf_counter_ns() - before) / 1000 / inner
        samples.append({"api_return_us": api_us, "wall_drain_us": wall_us,
                        "cuda_event_span_us": start.elapsed_time(end) * 1000 / inner})
    return {"warmup": warmup, "repeats": repeats, "inner": inner, "samples": samples,
            "median": {key: statistics.median(row[key] for row in samples) for key in samples[0]}}


def _profile(plan: Callable[[], None], run: Callable[[], Any], path: Path) -> dict[str, Any]:
    import torch

    if path.exists():
        raise FileExistsError(path)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        with torch.profiler.record_function("mxfp4_planning"):
            plan()
            torch.cuda.synchronize()
        for _ in range(5):
            with torch.profiler.record_function("mxfp4_attention"):
                run()
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    kernels = [event for event in trace["traceEvents"] if event.get("cat") == "kernel"]
    copies = [event for event in trace["traceEvents"] if "memcpy" in event.get("cat", "").lower()]
    if not kernels:
        raise RuntimeError("CUPTI did not provide kernel-active evidence")
    return {"trace": str(path), "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "attention_kernel_active_us_per_call": sum(event["dur"] for event in kernels) / 5,
            "kernels": [{"name": event["name"], "duration_us": event["dur"],
                         "args": event.get("args", {})} for event in kernels],
            "copies": [{"name": event["name"], "duration_us": event.get("dur", 0),
                        "args": event.get("args", {})} for event in copies]}


def _reference(q: Any, key: Any, value: Any, key_sf: Any, value_sf: Any,
               ptr: Any, ids: Any, last: Any, query_ptr: Any) -> tuple[Any, Any]:
    import torch
    from xllm.python.attention.mxfp4_reference import decode_mxfp4

    rows, outputs = [], []
    for request in range(last.numel()):
        query_start, query_end = int(query_ptr[request]), int(query_ptr[request + 1])
        qlen = query_end - query_start
        if qlen == 0:
            continue
        page_ids = ids[int(ptr[request]):int(ptr[request + 1])].long()
        length = (page_ids.numel() - 1) * 128 + int(last[request])
        # Independently decoded resident bytes are used only outside timing.
        k = decode_mxfp4(key[page_ids].cpu().numpy(), key_sf[page_ids].cpu().numpy())
        v = decode_mxfp4(value[page_ids].cpu().numpy(), value_sf[page_ids].cpu().numpy())
        k = torch.from_numpy(k.reshape(-1, 4, 256)[:length]).to(q.device)
        v = torch.from_numpy(v.reshape(-1, 4, 256)[:length]).to(q.device)
        local = torch.linspace(0, qlen - 1, min(16, qlen), device=q.device).long().unique()
        chosen = local + query_start
        scores = torch.einsum("rhd,thd->hrt", q[chosen].float(), k.repeat_interleave(6, dim=1)) / 16
        causal = torch.arange(length, device=q.device)[None, :] <= (length - qlen + local)[:, None]
        scores.masked_fill_(~causal[None], float("-inf"))
        result = torch.einsum("hrt,thd->rhd", scores.softmax(-1), v.repeat_interleave(6, dim=1))
        rows.append(chosen)
        outputs.append(result)
    return torch.cat(rows), torch.cat(outputs)


def _mask(wrapper: Any, query_ptr: Any) -> dict[str, Any]:
    import torch

    info = list(wrapper._plan_info)
    if len(info) != 15 or not info[13] or info[14]:
        raise AssertionError("Expected graph-ready non-split PrefillPlanInfo")
    padded, tile, offset = info[0], info[3], info[12]
    valid = sum(math.ceil(int(length) * 6 / tile) for length in query_ptr.diff())
    actual = wrapper._int_workspace_buffer[offset:offset + padded].cpu()
    expected = (torch.arange(padded) < valid).to(torch.uint8)
    if not torch.equal(actual, expected):
        raise AssertionError(json.dumps({"mask_offset": offset, "valid": valid,
                                         "padded": padded, "actual_mask_bytes": actual.tolist()}))
    return {"padded": padded, "valid": valid, "offset": offset,
            "bytes": actual.tolist(), "plan_info": info, "initialized": True}


def _case(batch: int, kv_len: int, qlen: int, *, fast_plan: bool,
          phase: str, trace_base: Path, geometry_results: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from xllm.python.attention.mxfp4_flashinfer import make_mxfp4_wrapper
    from xllm.python.attention.mxfp4_decode_plan import Mxfp4DecodePlanner

    torch.manual_seed(1701 + batch + kv_len)
    device = torch.device("cuda:0")
    max_pages_per_row = max(math.ceil(kv_len / 128) + 2, 5)
    physical_pages = batch * max_pages_per_row + 8
    cache_shape = (physical_pages, 128, 4, 128)
    scale_shape = (physical_pages, 128, 4, 8)
    key = torch.randint(0, 256, cache_shape, dtype=torch.uint8, device=device)
    value = torch.randint(0, 256, cache_shape, dtype=torch.uint8, device=device)
    key_sf = torch.randint(124, 128, scale_shape, dtype=torch.uint8, device=device)
    value_sf = torch.randint(124, 128, scale_shape, dtype=torch.uint8, device=device)
    q = (torch.randn(batch * qlen, 24, 256, device=device) * .25).to(torch.bfloat16)
    original_q = q.clone()
    inputs = {name: _tensor_sha(tensor) for name, tensor in
              (("q", q), ("key", key), ("value", value), ("key_sf", key_sf), ("value_sf", value_sf))}
    base_query = torch.arange(batch + 1, dtype=torch.int32) * qlen
    baseline_outputs: list[Any] = []
    result: dict[str, Any] = {"batch": batch, "kv_len": kv_len, "query_len": qlen,
                              "head_dim": 256, "query_heads": 24, "kv_heads": 4,
                              "page_size": 128, "inputs_sha256": inputs, "routes": []}
    geometry_results.append(result)
    modes = ["full", "fast"] if fast_plan and qlen == 1 else ["full"]
    for mode in modes:
        q.copy_(original_q)
        buffers = {"qo_indptr_buf": base_query.to(device),
                   "paged_kv_indptr_buf": torch.empty(batch + 1, dtype=torch.int32, device=device),
                   "paged_kv_indices_buf": torch.empty(batch * max_pages_per_row, dtype=torch.int32, device=device),
                   "paged_kv_last_page_len_buf": torch.empty(batch, dtype=torch.int32, device=device)}
        graph_wrapper = make_mxfp4_wrapper(torch.full((64 * 1024 * 1024,), 0xa5, dtype=torch.uint8, device=device),
                                           use_cuda_graph=True, **buffers)
        eager = make_mxfp4_wrapper(torch.full((64 * 1024 * 1024,), 0xa5, dtype=torch.uint8, device=device))
        fast = Mxfp4DecodePlanner(graph_wrapper, num_qo_heads=24, num_kv_heads=4,
                                 head_dim=256, page_size=128, sm_scale=1 / 16) if mode == "fast" else None
        output = torch.empty_like(q)
        graph = None
        records = []
        route: dict[str, Any] = {"mode": mode, "cases": records}
        result["routes"].append(route)
        # Long prefill retains its fixed geometry; decode cases change actual
        # pages and lengths before restoring the geometry used for measurement.
        steps = [0] if qlen > 1 else [0, 1, 2, 3]
        for step in steps:
            result["active_step"] = {"mode": mode, "step": step}
            q.copy_(original_q + step * .01)
            lengths = torch.full((batch,), kv_len, dtype=torch.int32)
            if step == 1:
                lengths = torch.tensor([127 + (row % 3) for row in range(batch)], dtype=torch.int32)
            elif step == 2:
                lengths += torch.arange(batch, dtype=torch.int32) % 2 + 1
            counts = (lengths + 127) // 128
            ptr = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)))
            last = ((lengths - 1) % 128 + 1).clone()
            ids = (torch.arange(int(ptr[-1]), dtype=torch.int32) * 7 + step * 11) % (physical_pages - 8)
            copy_verified = False
            if step in (1, 2):
                ids[ptr[:-1].long()] = 0
            if step == 2:
                target = physical_pages - 2
                for tensor in (key, value, key_sf, value_sf):
                    tensor[target].copy_(tensor[0])
                    assert torch.equal(tensor[target], tensor[0])
                ids[ptr[:-1].long()] = target
                copy_verified = True
            device_ids = ids.to(device)
            query = base_query.clone()

            def full_plan(wrapper: Any) -> None:
                wrapper.plan(query, ptr, device_ids, last, 24, 4, 256, 128,
                             causal=True, sm_scale=1 / 16, q_data_type=torch.bfloat16,
                             kv_data_type=torch.uint8, o_data_type=torch.bfloat16)

            def plan() -> None:
                if fast is None:
                    full_plan(graph_wrapper)
                else:
                    fast.plan(ptr, device_ids, last)

            def run() -> Any:
                return graph_wrapper.run(q, (key, value), key_block_scales=key_sf,
                                         value_block_scales=value_sf, out=output)

            # Poison both actual scheduler workspaces before each native plan.
            torch.cuda.synchronize()
            graph_wrapper._int_workspace_buffer.fill_(0xa5)
            graph_wrapper._pin_memory_int_workspace_buffer.fill_(0xa5)
            plan()
            mask = _mask(graph_wrapper, query)
            result["last_plan_mask"] = mask
            full_plan(eager)
            eager_output = eager.run(q, (key, value), key_block_scales=key_sf, value_block_scales=value_sf)
            run()
            torch.cuda.synchronize()
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                torch.cuda.synchronize()
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            # Same-route graph/direct output equality is exact. Eager tile
            # selection can differ and retains the frozen 0.02 tolerance.
            captured = output.clone()
            run()
            torch.testing.assert_close(captured, output, rtol=0, atol=0)
            torch.testing.assert_close(captured, eager_output, rtol=.02, atol=.02)
            rows, reference = _reference(q, key, value, key_sf, value_sf, ptr, device_ids, last, query)
            torch.testing.assert_close(captured[rows].float(), reference, rtol=.02, atol=.02)
            if mode == "full" and qlen == 1:
                baseline_outputs.append(captured.clone())
            if mode == "fast":
                torch.testing.assert_close(captured, baseline_outputs[step], rtol=0, atol=0)
            records.append({"step": step, "mask": mask, "graph_ready": graph_wrapper.is_cuda_graph_enabled,
                            "non_split": not bool(graph_wrapper._plan_info[14]), "replay": True,
                            "reference_passed": True, "same_route_exact": True,
                            "fast_plan_equal": mode == "fast", "prefix_page_copy": copy_verified,
                            "page_ids": ids.tolist(), "indptr": ptr.tolist(), "last": last.tolist(),
                            "query_sha256": _tensor_sha(q),
                            "reference_rows": rows.tolist(),
                            "maxabs_reference": float((captured[rows].float() - reference).abs().max()),
                            "q_dtype": str(q.dtype), "kv_dtype": str(key.dtype),
                            "scale_dtype": str(key_sf.dtype), "output_dtype": str(captured.dtype)})
        if phase == "main":
            route["planning"] = _measure(plan, warmup=3, repeats=20, inner=1)
            route["attention_direct"] = _measure(run, warmup=10, repeats=40, inner=5)
            route["attention_graph"] = _measure(graph.replay, warmup=10, repeats=40, inner=5)
            route["profile"] = _profile(plan, run, trace_base.with_name(trace_base.name + f"-b{batch}-s{kv_len}-{mode}.trace.json"))
        # Exercise changing validity within one captured padded graph. Fast
        # decode owns a one-query-per-row contract, so this is a separate P0
        # full-plan check after all fast/full comparisons and timing.
        if mode == "full" and batch > 1 and qlen == 1:
            query = base_query.clone()
            query[-1] = query[-2]
            graph_wrapper._int_workspace_buffer.fill_(0xa5)
            graph_wrapper._pin_memory_int_workspace_buffer.fill_(0xa5)
            full_plan(graph_wrapper)
            inactive_mask = _mask(graph_wrapper, query)
            output.fill_(123)
            graph.replay()
            torch.cuda.synchronize()
            rows, reference = _reference(q, key, value, key_sf, value_sf, ptr, device_ids, last, query)
            torch.testing.assert_close(output[rows].float(), reference, rtol=.02, atol=.02)
            assert torch.all(output[batch - 1] == 123)
            route["inactive_rows"] = {"passed": True, "mask": inactive_mask, "tail_untouched": True}
    result.pop("active_step")
    if len(modes) == 2 and phase == "main":
        baseline, candidate = result["routes"]
        host_ratio = candidate["planning"]["median"]["api_return_us"] / baseline["planning"]["median"]["api_return_us"]
        graph_ratio = candidate["attention_graph"]["median"]["cuda_event_span_us"] / baseline["attention_graph"]["median"]["cuda_event_span_us"]
        active_ratio = candidate["profile"]["attention_kernel_active_us_per_call"] / baseline["profile"]["attention_kernel_active_us_per_call"]
        result["performance_gate"] = {"host_ratio": host_ratio, "graph_event_ratio": graph_ratio,
                                      "kernel_active_ratio": active_ratio,
                                      "host_reduction_at_least_10pct": host_ratio <= .90,
                                      "gpu_regression_at_most_5pct": max(graph_ratio, active_ratio) <= 1.05,
                                      "pass": host_ratio <= .90 and max(graph_ratio, active_ratio) <= 1.05}
    return result


def _run(args: argparse.Namespace, receipt: dict[str, Any]) -> None:
    import torch
    from xllm.python.attention.mxfp4_flashinfer import mxfp4_jit_source_manifest, mxfp4_jit_spec

    if torch.__version__ != "2.10.0+cu130" or importlib.metadata.version("flashinfer-python") != "0.6.18.post1":
        raise RuntimeError("Qualification requires frozen Torch 2.10.0+cu130 / FlashInfer 0.6.18.post1")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("Controller must expose exactly one admitted SM90 GPU")
    if "H800" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This protocol is frozen for H800")
    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    receipt["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                          "flashinfer": importlib.metadata.version("flashinfer-python"),
                          "gpu": torch.cuda.get_device_name(0), "gpu_properties": str(properties),
                          "gpu_uuid": str(properties.uuid),
                          "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    receipt["source_manifest"] = mxfp4_jit_source_manifest()
    geometries = [(1, 128, 1), (2, 129, 1), (4, 257, 1)] if args.phase == "pilot" else [(1, 8192, 8192), (32, 2048, 1), (32, 16384, 1)]
    receipt["geometries"] = []
    for batch, length, qlen in geometries:
        _case(batch, length, qlen, fast_plan=args.fast_plan,
              phase=args.phase, trace_base=args.output.with_suffix(""),
              geometry_results=receipt["geometries"])
    spec, _, _ = mxfp4_jit_spec()
    library = spec.get_library_path()
    generated_directory = Path(spec.sources[0]).parent
    receipt["library"] = {"path": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                          "module_name": spec.name,
                          "generated_sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in spec.sources},
                          "generated_inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                               for path in sorted(generated_directory.iterdir())
                                               if path.suffix in (".cu", ".cuh", ".inc")}}
    records = [record for geometry in receipt["geometries"] for route in geometry["routes"] for record in route["cases"]]
    changed = [record for record in records if record["step"] > 0]
    fast_records = [record for geometry in receipt["geometries"] for route in geometry["routes"]
                    if route["mode"] == "fast" for record in route["cases"]]
    inactive = [route["inactive_rows"] for geometry in receipt["geometries"] for route in geometry["routes"] if "inactive_rows" in route]
    gates = {"graph_ready": bool(records) and all(row["graph_ready"] for row in records),
             "non_split": bool(records) and all(row["non_split"] for row in records),
             "padding_mask_initialized": bool(records) and all(row["mask"]["initialized"] for row in records)
                 and any(row["mask"]["padded"] > row["mask"]["valid"] for row in records),
             "dynamic_metadata_reference": bool(changed) and all(row["reference_passed"] for row in changed),
             "changed_input_replay": bool(changed) and all(row["replay"] for row in changed),
             "prefix_page_copy": any(row["prefix_page_copy"] for row in records),
             "inactive_rows": bool(inactive) and all(row["passed"] for row in inactive),
             "reference_passed": bool(records) and all(row["reference_passed"] for row in records)}
    if args.fast_plan:
        gates["fast_plan_equal"] = bool(fast_records) and all(row["fast_plan_equal"] for row in fast_records)
    receipt["gates"] = gates
    performance = [geometry["performance_gate"] for geometry in receipt["geometries"] if "performance_gate" in geometry]
    receipt["service_admitted"] = args.phase == "main" and args.fast_plan and bool(performance) and all(row["pass"] for row in performance) and all(gates.values())
    receipt["completed"] = True
    receipt["pass"] = all(gates.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute only inside an admitted once-guarded job")
    parser.add_argument("--phase", choices=("pilot", "main"), required=True)
    parser.add_argument("--fast-plan", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required; shared-GPU admission belongs to the Controller")
    receipt: dict[str, Any] = {"schema": "kv_p0_p1_operator_acceptance_v1", "phase": args.phase,
                              "mode": "mxfp4_graph_fast_plan" if args.fast_plan else "mxfp4_graph_repair",
                              "completed": False, "pass": False, "gates": {}}
    with args.output.open("x") as output:
        try:
            _run(args, receipt)
        except Exception as error:
            receipt["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            json.dump(receipt, output, indent=2)
            output.write("\n")
    print(json.dumps({"output": str(args.output), "completed": receipt["completed"], "pass": receipt["pass"],
                      "service_admitted": receipt["service_admitted"]}))


if __name__ == "__main__":
    main()
