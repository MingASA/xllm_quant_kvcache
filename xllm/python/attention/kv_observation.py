# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Bounded K/V range observations using ordinary tensor reductions.

The forward path retains only per-layer/phase/head maxima and counters. Host
reads and immutable exports occur at explicitly quiescent request boundaries.
This diagnostic is excluded from timed serving configurations. GPU Graph
compatibility requires a separate execution gate, beyond the CPU tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path

import torch

from scripts.logger import logger

METHOD = "native_BF16_valid_KV_write_ranges_v1"


def _publish(path: Path, value: dict) -> str:
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(data) > 2**20:
        raise ValueError("KV observation exceeds its one-MiB export limit")
    temporary = path.with_name("." + path.name + ".prepared")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    return hashlib.sha256(data).hexdigest()


class KVRangeObserver:
    """Capture-friendly reductions without activation snapshots or host reads."""

    def __init__(self, layer_ids: list[int], heads: int, dimension: int, device: torch.device) -> None:
        if (not layer_ids or len(set(layer_ids)) != len(layer_ids) or min(layer_ids) < 0
                or min(heads, dimension) <= 0):
            raise ValueError("KV observation requires unique original full-attention layer IDs and positive geometry")
        self.layer_ids = tuple(layer_ids)
        self._rows = {layer: index for index, layer in enumerate(layer_ids)}
        self._heads, self._dimension, self._device = heads, dimension, device
        shape = (len(layer_ids), 2, 2, heads)
        self._maxima = torch.zeros(shape, dtype=torch.float32, device=device)
        self._finite = torch.zeros(shape, dtype=torch.int64, device=device)
        self._nonfinite = torch.zeros_like(self._finite)
        self._calls = torch.zeros((len(layer_ids), 2), dtype=torch.int64, device=device)

    @property
    def counter_storage_bytes(self) -> int:
        """Persistent statistics only, excluding temporary reduction workspace."""
        return sum(t.numel() * t.element_size() for t in (self._maxima, self._finite, self._nonfinite, self._calls))

    def observe(self, layer_id: int, prefill: bool, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor) -> None:
        if (key.ndim != 3 or key.shape != value.shape or key.shape[1:] != (self._heads, self._dimension)
                or slots.ndim != 1 or slots.numel() != key.shape[0] or slots.dtype not in (torch.int32, torch.int64)
                or any(t.device != self._device for t in (key, value, slots))):
            raise ValueError("KV observation requires actual same-device K/V write rows and slot mapping")
        if key.shape[0] == 0:
            return
        row, phase = self._rows[layer_id], 0 if prefill else 1
        valid = (slots > 0)[:, None, None]  # Engine padding/reserved slot0 and negative rows are not real writes.
        for kind, tensor in enumerate((key, value)):
            data = tensor.detach().float()
            finite = torch.isfinite(data)
            values = torch.where(valid & finite, data.abs(), 0.0)
            self._maxima[row, phase, kind].copy_(torch.maximum(self._maxima[row, phase, kind], values.amax(dim=(0, 2))))
            self._finite[row, phase, kind].add_((valid & finite).sum(dim=(0, 2)))
            self._nonfinite[row, phase, kind].add_((valid & ~finite).sum(dim=(0, 2)))
        self._calls[row, phase].add_(valid.any().to(torch.int64))

    def reset(self) -> None:
        """Caller must stop requests and drain the model before reset/export."""
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        for tensor in (self._maxima, self._finite, self._nonfinite, self._calls):
            tensor.zero_()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def snapshot(self) -> dict:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        maxima, finite, nonfinite, calls = (tensor.cpu().tolist() for tensor in
                                            (self._maxima, self._finite, self._nonfinite, self._calls))
        rows = []
        for index, layer in enumerate(self.layer_ids):
            for phase, phase_name in enumerate(("prefill", "decode")):
                for kind, kind_name in enumerate(("key", "value")):
                    for head in range(self._heads):
                        rows.append({"layer_id": layer, "phase": phase_name, "kind": kind_name, "head": head,
                                     "finite_values": finite[index][phase][kind][head],
                                     "nonfinite_values": nonfinite[index][phase][kind][head],
                                     "absmax": maxima[index][phase][kind][head]})
        if any(row["nonfinite_values"] for row in rows) or not any(row["finite_values"] for row in rows):
            raise ValueError("Nonfinite or empty K/V observations cannot derive scales")
        observed = {kind: max(row["absmax"] for row in rows if row["kind"] == kind) for kind in ("key", "value")}

        def scales(denominator: float) -> dict[str, float]:
            return {kind: max(torch.finfo(torch.float32).tiny, 2.0**math.ceil(math.log2(maximum * 1.1 / denominator)))
                    if maximum else 1.0 for kind, maximum in observed.items()}

        return {"method": METHOD, "scope": "observed_valid_post_RoPE_K_projected_V_writes_not_unseen_prompt_coverage",
                "layer_ids": list(self.layer_ids), "heads": self._heads, "dimension": self._dimension,
                "counter_storage_bytes": self.counter_storage_bytes, "activation_tensors_retained": False,
                "host_reads_in_forward": False, "phase_calls_by_layer": calls, "observed_absmax": observed,
                "FP8_E4M3_candidate_scales": scales(448.0), "NVFP4_global_candidate_scales": scales(6.0 * 448.0),
                "scale_rule": "power-of-two ceiling of1.1*observed_absmax/format_range, minimum normal FP32",
                "rows": rows, "quality_or_performance_certified": False}

    @classmethod
    def from_config(cls, path: str, layer_ids: list[int], heads: int, dimension: int, device: torch.device) -> KVRangeObserver:
        config_path = Path(path)
        if (device.type != "cuda" or config_path.resolve(strict=True) != config_path
                or config_path.stat().st_uid != os.getuid() or config_path.stat().st_size > 65536):
            raise ValueError("Online KV observation requires an owned bounded config and CUDA")
        config = json.loads(config_path.read_text())
        if (config.get("method") != METHOD or config.get("layer_ids") != layer_ids
                or config.get("heads") != heads or config.get("dimension") != dimension
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", config.get("run_id", "")) is None
                or re.fullmatch(r"[0-9a-f]{64}", config.get("request_sha256", "")) is None):
            raise ValueError("KV observation configuration/geometry does not match actual full-attention caches")
        output = Path(config["output_dir"])
        if output.resolve(strict=True) != output or output.stat().st_uid != os.getuid():
            raise ValueError("KV observation output must be an owned existing isolated directory")
        observer = cls(layer_ids, heads, dimension, device)
        threading.Thread(target=observer._control, args=(output, config), daemon=True,
                         name="kv-observation-boundary-control").start()
        return observer

    def _control(self, output: Path, config: dict) -> None:
        try:
            for operation in ("reset", "export"):
                command = output / ("kv_observer_" + operation + ".json")
                while not command.exists():
                    time.sleep(0.2)
                if (command.resolve(strict=True) != command or command.stat().st_uid != os.getuid()
                        or command.stat().st_size > 65536
                        or json.loads(command.read_text()) != {"action": operation, "request_sha256": config["request_sha256"]}):
                    raise ValueError("Unowned or mismatched quiescent KV observation command")
                digest = None
                if operation == "reset":
                    self.reset()
                else:
                    value = {**self.snapshot(), "run_id": config["run_id"], "request_sha256": config["request_sha256"]}
                    digest = _publish(output / "kv_observer_snapshot.json", value)
                _publish(output / ("kv_observer_" + operation + "_ack.json"),
                         {"action": operation, "request_sha256": config["request_sha256"], "snapshot_sha256": digest})
        except (OSError, ValueError, RuntimeError) as error:
            _publish(output / "kv_observer_error.json", {"error": type(error).__name__ + ": " + str(error)})
            logger.exception("KV observation failed; its calibration is invalid")
