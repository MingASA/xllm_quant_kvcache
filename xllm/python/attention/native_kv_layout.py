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

"""Packed NVFP4 data and block-scale planes sharing each resident byte page."""

from __future__ import annotations

import torch


def nvfp4_page_views(cache: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Alias an allocator-validated byte page without per-head payload misalignment.

    Only the page stride includes scale bytes. Each data head remains packed
    and 128-byte aligned at D=256; each scale head remains a packed 16-byte row.
    The whole owner page is the unit of prefix copying, VMM mapping and reclaim.
    """
    pages, tokens, heads = cache.shape[:3]
    packed_dim = head_dim // 2
    payload_bytes = tokens * heads * packed_dim
    owner = cache.view(pages, -1)
    data = owner[:, :payload_bytes].view(pages, tokens, heads, packed_dim)
    scales = owner[:, payload_bytes:].view(pages, tokens, heads, head_dim // 16)
    return data, scales.view(torch.float8_e4m3fn)
