# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused FP8 quantize/scatter; attention stays in FlashInfer."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _write_fp8(
    key: tl.tensor, value: tl.tensor, slots: tl.tensor,
    key_cache: tl.tensor, value_cache: tl.tensor,
    key_token_stride: tl.constexpr, key_head_stride: tl.constexpr, key_channel_stride: tl.constexpr,
    value_token_stride: tl.constexpr, value_head_stride: tl.constexpr, value_channel_stride: tl.constexpr,
    heads: tl.constexpr, dim: tl.constexpr, capacity: tl.constexpr,
    key_scale: tl.constexpr, value_scale: tl.constexpr, block: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slots + token)
    if slot >= 0 and slot < capacity:
        channel = tl.arange(0, block)
        k = tl.load(key + token * key_token_stride + head * key_head_stride + channel * key_channel_stride,
                    mask=channel < dim, other=0).to(tl.float32)
        v = tl.load(value + token * value_token_stride + head * value_head_stride + channel * value_channel_stride,
                    mask=channel < dim, other=0).to(tl.float32)
        offset = (slot * heads + head) * dim + channel
        tl.store(key_cache + offset, tl.clamp(k / key_scale, -448.0, 448.0), mask=channel < dim)
        tl.store(value_cache + offset, tl.clamp(v / value_scale, -448.0, 448.0), mask=channel < dim)


def write_fp8_kv(
    key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor,
    caches: tuple[torch.Tensor, torch.Tensor], scales: tuple[float, float],
) -> None:
    # The model and bound cache establish shapes/dtypes once, before serving.
    # All input strides are explicit, including compiler-selected head layouts.
    _write_fp8[(key.shape[0], key.shape[1])](
        key, value, slots, *caches, key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2), key.shape[1], key.shape[2],
        caches[0].shape[0] * caches[0].shape[1], *scales,
        block=triton.next_power_of_2(key.shape[2]), num_warps=4,
    )
