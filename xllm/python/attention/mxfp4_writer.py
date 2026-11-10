# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Fused OCP block32/E8M0 quantization into paged byte owners."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _write(
    source: tl.tensor, slots: tl.tensor, data: tl.tensor, scales: tl.tensor,
    token_stride: tl.constexpr, head_stride: tl.constexpr,
    channel_stride: tl.constexpr, page_stride: tl.constexpr,
    scale_page_stride: tl.constexpr, tokens: tl.constexpr,
    heads: tl.constexpr, dimension: tl.constexpr, pages: tl.constexpr,
) -> None:
    token, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + token)
    if (slot >= 0) & (slot < pages * tokens):
        group = tl.arange(0, dimension // 32)
        channel = tl.arange(0, 32)
        values = tl.load(source + token * token_stride + head * head_stride
                         + (group[:, None] * 32 + channel[None, :]) * channel_stride).to(tl.float32)
        maximum = tl.max(tl.abs(values), axis=1)
        nonfinite = tl.max(((tl.abs(values) == float("inf")) | (values != values)).to(tl.int32), axis=1) != 0
        # Finite BF16 values have the same exponent field in FP32. Subnormal
        # BF16 blocks all clamp to the minimum E8M0 exponent, so no log2 or
        # approximate reciprocal is needed, including at exact powers of two.
        exponent = (maximum.to(tl.uint32, bitcast=True) >> 23).to(tl.int32) - 129
        exponent = tl.minimum(127, tl.maximum(-127, exponent))
        scale_code = exponent + 127
        inverse_scale = ((127 - exponent) << 23).to(tl.float32, bitcast=True)
        # Explicit non-FTZ multiplication preserves BF16 subnormals when the
        # scale is 2**-127. The reciprocal itself is a normal FP32 value.
        normalized = tl.abs(tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
            args=[values, inverse_scale[:, None]], dtype=tl.float32,
            is_pure=True, pack=1,
        ))
        code = tl.where(normalized <= .25, 0,
               tl.where(normalized < .75, 1,
               tl.where(normalized <= 1.25, 2,
               tl.where(normalized < 1.75, 3,
               tl.where(normalized <= 2.5, 4,
               tl.where(normalized < 3.5, 5,
               tl.where(normalized <= 5, 6, 7)))))))
        sign = (values.to(tl.uint32, bitcast=True) >> 28) & 8
        code = tl.where(nonfinite[:, None], 0, code | sign).to(tl.uint8)
        pair = tl.reshape(code, (dimension // 2, 2))
        packed = tl.sum(pair.to(tl.uint32) << (tl.arange(0, 2)[None, :] * 4), axis=1)
        page, entry = slot // tokens, slot % tokens
        tl.store(data + page * page_stride + (entry * heads + head) * (dimension // 2)
                 + tl.arange(0, dimension // 2), packed.to(tl.uint8))
        tl.store(scales + page * scale_page_stride + (entry * heads + head) * (dimension // 32)
                 + group, tl.where(nonfinite, 255, scale_code).to(tl.uint8))


def mxfp4_page_views(cache: torch.Tensor, dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Alias packed payload and raw E8M0 planes; a page remains one copy unit."""
    pages, tokens, heads = cache.shape[:3]
    payload = tokens * heads * (dimension // 2)
    owner = cache.view(pages, -1)
    return (owner[:, :payload].view(pages, tokens, heads, dimension // 2),
            owner[:, payload:].view(pages, tokens, heads, dimension // 32))


def write_mxfp4_kv(
    key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor,
    pages: tuple[torch.Tensor, torch.Tensor],
    scales: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Write valid slots only, without retaining uncompressed activations."""
    for source, data, scale in zip((key, value), pages, scales):
        _write[(source.shape[0], source.shape[1])](
            source, slots, data, scale, *source.stride(), data.stride(0),
            scale.stride(0), data.shape[1], source.shape[1], source.shape[2],
            data.shape[0], num_warps=4,
        )
