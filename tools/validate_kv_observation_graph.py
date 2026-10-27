# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Validate changed CUDA Graph range observations before model calibration.

The caller owns shared-GPU admission and watchdogs. This is an untimed
diagnostic, not a model quality or serving performance measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from xllm.python.attention.kv_observation import KVRangeObserver


def _validate() -> dict:
    device = torch.device("cuda:0")
    observer = KVRangeObserver([3, 7], 4, 256, device)
    key = torch.zeros((4, 4, 256), dtype=torch.bfloat16, device=device)
    value = torch.zeros_like(key)
    slots = torch.zeros(4, dtype=torch.int64, device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            observer.observe(3, False, key, value, slots)
            observer.observe(7, False, key, value, slots)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        observer.observe(3, False, key, value, slots)
        observer.observe(7, False, key, value, slots)
    observer.reset()
    for index, (count, k_max, v_max) in enumerate(((2, 3.0, 5.0), (3, 7.0, 11.0))):
        key.fill_(65504)
        value.fill_(65504)
        key[:count].fill_(k_max)
        value[:count].fill_(-v_max)
        slots.copy_(torch.tensor([1, 2, -1, 0] if index == 0 else [1, 2, 3, 0], device=device))
        graph.replay()
    torch.cuda.synchronize()
    snapshot = observer.snapshot()
    for row in snapshot["rows"]:
        expected = 5 * 256 if row["phase"] == "decode" else 0
        maximum = (7.0 if row["kind"] == "key" else 11.0) if expected else 0.0
        if row["finite_values"] != expected or row["nonfinite_values"] or row["absmax"] != maximum:
            raise ValueError("Actual Graph observations disagree with changed valid K/V rows")
    if snapshot["phase_calls_by_layer"] != [[0, 2], [0, 2]]:
        raise ValueError("CUDA Graph range counters did not replay exactly twice")
    observer.reset()
    key.fill_(1)
    value.fill_(2)
    slots.fill_(1)
    observer.observe(3, True, key, value, slots)
    after_reset = observer.snapshot()
    for row in after_reset["rows"]:
        expected = 4 * 256 if row["layer_id"] == 3 and row["phase"] == "prefill" else 0
        if row["finite_values"] != expected:
            raise ValueError("Reset left stale captured range statistics")
    return {"actual_cuda_Graph_capture_replay": True, "changed_KV_and_padding_checked": True,
            "quiescent_reset_verified": True, "snapshot": snapshot, "after_reset": after_reset,
            "GPU_name": torch.cuda.get_device_name(), "SM": torch.cuda.get_device_capability()[0] * 10,
            "model_quality_or_performance_certified": False}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = _validate()
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    _main()
