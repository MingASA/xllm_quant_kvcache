# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""One-pass static-scale FP8 query conversion for native FA3 attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_query_kernel(source: tl.tensor, destination: tl.tensor, count: tl.constexpr,
                           scale: tl.constexpr, block: tl.constexpr) -> None:
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    values = tl.load(source + offsets, offsets < count, other=0).to(tl.float32) / scale
    values = tl.minimum(tl.maximum(values, -448.0), 448.0)
    tl.store(destination + offsets, values, offsets < count)


def quantize_query_fp8(query: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantize inside graph capture; the immutable scalar never requires a host sync."""
    if query.device.type != "cuda" or query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Native FP8 query conversion requires CUDA BF16/FP16 input")
    source = query.contiguous()
    output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    if source.numel():
        _quantize_query_kernel[(triton.cdiv(source.numel(), 256),)](source, output, source.numel(), scale, 256)
    return output
