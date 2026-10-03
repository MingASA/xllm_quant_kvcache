# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Explicit quality-only INT4 experiments on the BF16 FlashInfer cache.

Only newly projected, post-RoPE K/V are transformed. Persistent cache storage
remains BF16: this is not a compressed-cache or performance implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from scripts.logger import logger
from xllm.python.attention.quantized_triton import _round_to_nearest_even


@triton.jit
def _hadamard(values: tl.tensor, dims: tl.tensor, WIDTH: tl.constexpr, LOG_WIDTH: tl.constexpr) -> tl.tensor:
    for stage in tl.static_range(0, LOG_WIDTH):
        stride = 1 << stage
        peer = tl.gather(values, dims ^ stride, axis=0)
        values = tl.where((dims & stride) == 0, values + peer, peer - values)
    return values * (WIDTH ** -0.5)


@triton.jit
def _int4_qdq_kernel(
    source_ptr: tl.tensor,
    output_ptr: tl.tensor,
    signs_ptr: tl.tensor,
    WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ROTATION: tl.constexpr,
    LOG_WIDTH: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, WIDTH)
    values = tl.load(source_ptr + row * WIDTH + dims).to(tl.float32)
    if ROTATION:
        signs = tl.load(signs_ptr + dims)
        values = _hadamard(values * signs, dims, WIDTH, LOG_WIDTH)
    groups = tl.reshape(values, (WIDTH // GROUP_SIZE, GROUP_SIZE))
    maximum = tl.max(tl.abs(groups), axis=1)
    scales = tl.where(maximum == 0, 1.0, tl.maximum(maximum / 7.0, 1.1754943508222875e-38))
    normalized = tl.clamp(tl.div_rn(groups, scales[:, None]), -7.0, 7.0)
    decoded = _round_to_nearest_even(normalized) * scales[:, None]
    values = tl.reshape(decoded, (WIDTH,))
    if ROTATION:
        values = _hadamard(values, dims, WIDTH, LOG_WIDTH) * signs
    tl.store(output_ptr + row * WIDTH + dims, values)


class KVQualityTransform:
    """Quantize/reconstruct new KV without changing paged-cache ownership."""

    def __init__(
        self,
        mode: str,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int = 17,
    ) -> None:
        if mode not in ("v_only_int4", "int4_rht_g32"):
            raise ValueError(f"Unknown KV quality mode: {mode}")
        if device.type != "cuda" or dtype != torch.bfloat16:
            raise ValueError("KV quality experiments require CUDA and BF16 model/cache tensors")
        if head_dim < 32 or head_dim & (head_dim - 1):
            raise ValueError("KV quality experiments require a power-of-two head_dim >= 32")
        self.mode = mode
        self._head_dim = head_dim
        self._group_size = head_dim if mode == "v_only_int4" else 32
        self._rotation = mode == "int4_rht_g32"
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self._signs = (torch.randint(0, 2, (head_dim,), generator=generator).float() * 2 - 1).to(device)
        self._device = self._signs.device
        logger.warning(
            "KV quality experiment: mode=%s group_size=%d rotation_seed=%d "
            "persistent_cache=BF16 (not compressed)",
            mode,
            self._group_size,
            seed,
        )

    def _transform(self, values: torch.Tensor) -> torch.Tensor:
        if (
            values.ndim != 3
            or values.shape[-1] != self._head_dim
            or values.dtype != torch.bfloat16
            or values.device != self._device
        ):
            raise ValueError("KV quality input must be BF16 [tokens, kv_heads, head_dim] on the configured device")
        values = values.contiguous()
        output = torch.empty_like(values)
        rows = values.shape[0] * values.shape[1]
        if rows:
            _int4_qdq_kernel[(rows,)](
                values,
                output,
                self._signs,
                self._head_dim,
                self._group_size,
                self._rotation,
                self._head_dim.bit_length() - 1,
                num_warps=4,
            )
        return output

    def apply(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if key.shape != value.shape or key.dtype != value.dtype or key.device != value.device:
            raise ValueError("KV quality experiments require matching K/V shapes, dtypes and devices")
        reconstructed_key = self._transform(key) if self._rotation else key
        return reconstructed_key, self._transform(value)
