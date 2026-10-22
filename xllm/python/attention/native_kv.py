# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Resident cache evidence and opt-in, bounded native FP8 scale measurement."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from xllm.python.attention.backend import AttentionMetadata, LayerCache


def cache_allocation(caches: list[LayerCache], cache_format: str) -> str:
    layers = []
    for cache in caches:
        if cache.key is None or not cache.key.numel():
            continue
        tensors = {
            name: {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": tensor.numel() * tensor.element_size(),
            }
            for name, tensor in (
                ("key", cache.key),
                ("value", cache.value),
                ("key_scale", cache.key_scale),
                ("value_scale", cache.value_scale),
            )
            if tensor is not None and tensor.numel()
        }
        resident = sum(tensor["bytes"] for tensor in tensors.values())
        layers.append(
            {
                "tensors": tensors,
                "key_format": cache_format,
                "value_format": cache_format,
                "resident_bytes": resident,
                "bytes_per_token": resident // (cache.key.shape[0] * cache.key.shape[1]),
            }
        )
    if not layers:
        raise ValueError("Native FlashInfer requires resident full-attention K/V pages")
    first = next(cache for cache in caches if cache.key is not None and cache.key.numel())
    return json.dumps(
        {
            "layers": layers,
            "resident_bytes": sum(layer["resident_bytes"] for layer in layers),
            "token_capacity": first.key.shape[0] * first.key.shape[1],
        }
    )


class KVScaleCalibration:
    """Measure the first prefill and decode per layer; never used in timed runs.

    Observations contain post-RoPE K and projected V before cache writes. Only
    their absolute maxima/counts reach the CPU; no activation/cache shadow is
    retained. A power-of-two scale covers the observed maximum plus 10% margin,
    not an assertion about unobserved prompts or later decoding positions.
    """

    def __init__(self, output: str) -> None:
        self._output = Path(output)
        if self._output.exists():
            raise FileExistsError(f"Calibration output already exists: {output}")
        self._observations: dict[str, dict] = {}

    def observe(
        self, layer_id: int, key: torch.Tensor, value: torch.Tensor, metadata: AttentionMetadata, num_layers: int
    ) -> None:
        phase = "prefill" if metadata.is_prefill or metadata.is_chunked_prefill else "decode"
        name = f"{phase}/{layer_id}"
        if name in self._observations:
            return
        valid = metadata.slot_mapping >= 0
        tensors = [tensor[valid] for tensor in (key, value)]
        if not tensors[0].numel():
            return
        maxima = [float(tensor.float().abs().amax().item()) for tensor in tensors]
        if not all(math.isfinite(maximum) for maximum in maxima):
            raise ValueError("Nonfinite KV activations in scale calibration")
        self._observations[name] = {"key_absmax": maxima[0], "value_absmax": maxima[1], "elements": tensors[0].numel()}
        if sum(item.startswith(phase + "/") for item in self._observations) != num_layers:
            return
        proposal = self.proposal(num_layers)
        self._output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._output.with_suffix(self._output.suffix + ".tmp")
        with temporary.open("x") as stream:
            json.dump(proposal, stream, indent=2)
            stream.write("\n")
        temporary.replace(self._output)

    def proposal(self, num_layers: int) -> dict:
        maxima = {kind: max(row[f"{kind}_absmax"] for row in self._observations.values()) for kind in ("key", "value")}
        scales = {
            kind: max(torch.finfo(torch.float32).tiny, 2.0 ** math.ceil(math.log2(maximum * 1.1 / 448.0)))
            if maximum
            else 1.0
            for kind, maximum in maxima.items()
        }
        return {
            "scope": "first_prefill_and_first_decode_per_layer_not_quality_certification",
            "strategy": "immutable_global_scalar_per_K_or_V",
            "format": "fp8_e4m3",
            "rule": "scale=max(FP32_min_normal,2**ceil(log2(1.1*observed_absmax/448))); zero-only => 1",
            "num_layers": num_layers,
            "observed_absmax": maxima,
            "fp8_k_scale": scales["key"],
            "fp8_v_scale": scales["value"],
            "covered_absmax": {kind: 448 * scale for kind, scale in scales.items()},
            "observations": self._observations,
        }
