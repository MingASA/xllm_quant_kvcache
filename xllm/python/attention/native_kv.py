# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Resident cache evidence and opt-in, bounded native FP8 scale measurement."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from xllm.python.attention.backend import AttentionMetadata, LayerCache


def _full_attention_layers(caches: list[LayerCache]) -> list[tuple[int, LayerCache]]:
    """Validate paged K/V while preserving model layer IDs in hybrid models."""
    layers: list[tuple[int, LayerCache]] = []
    for layer_id, cache in enumerate(caches):
        if cache.key is None and cache.value is None:
            continue
        if cache.key is None or cache.value is None:
            raise ValueError(f"Full-attention layer {layer_id} requires both K and V")
        if cache.key.ndim != 4 or cache.key.shape != cache.value.shape or min(cache.key.shape) <= 0:
            raise ValueError("Native FlashInfer requires matching positive [pages, page_size, heads, dim] K/V")
        if cache.key.dtype != cache.value.dtype or cache.key.device != cache.value.device:
            raise ValueError("Native K/V must have matching dtypes and devices")
        if not cache.key.is_contiguous() or not cache.value.is_contiguous():
            raise ValueError("Native FlashInfer requires contiguous K/V pages")
        if layers:
            first = layers[0][1].key
            if cache.key.shape != first.shape or cache.key.device != first.device or cache.key.dtype != first.dtype:
                raise ValueError("Native K/V page layouts must match across full-attention layers")
        layers.append((layer_id, cache))
    if not layers:
        raise ValueError("Native FlashInfer requires resident full-attention K/V pages")
    return layers


def cache_allocation(caches: list[LayerCache], cache_format: str) -> str:
    full_layers = _full_attention_layers(caches)
    layers = []
    for layer_id, cache in full_layers:
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
                "layer_id": layer_id,
                "tensors": tensors,
                "key_format": cache_format,
                "value_format": cache_format,
                "resident_bytes": resident,
                "bytes_per_token": resident // (cache.key.shape[0] * cache.key.shape[1]),
            }
        )
    linear_layers = [
        {
            "layer_id": layer_id,
            "tensors": {
                name: {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "bytes": tensor.numel() * tensor.element_size(),
                }
                for name, tensor in (("conv", cache.conv), ("ssm", cache.ssm))
                if tensor is not None
            },
            "resident_bytes": sum(
                tensor.numel() * tensor.element_size() for tensor in (cache.conv, cache.ssm) if tensor is not None
            ),
        }
        for layer_id, cache in enumerate(caches)
        if cache.conv is not None or cache.ssm is not None
    ]
    full_bytes = sum(layer["resident_bytes"] for layer in layers)
    linear_bytes = sum(layer["resident_bytes"] for layer in linear_layers)
    first = full_layers[0][1]
    return json.dumps(
        {
            "layers": layers,
            "resident_bytes": full_bytes,
            "full_attention_resident_bytes": full_bytes,
            "full_attention_layer_ids": [layer_id for layer_id, _ in full_layers],
            "linear_state_layers": linear_layers,
            "linear_state_resident_bytes": linear_bytes,
            "total_resident_bytes": full_bytes + linear_bytes,
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
