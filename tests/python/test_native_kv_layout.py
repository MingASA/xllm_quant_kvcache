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

"""Real CPU tensor checks of compressed page aliases and whole-page copying."""

import torch

from xllm.python.attention.native_kv_layout import nvfp4_page_views


def test_nvfp4_planes_alias_owner_without_payload_head_misalignment() -> None:
    owner = torch.zeros((3, 128, 4, 144), dtype=torch.uint8)
    data, scales = nvfp4_page_views(owner, 256)
    assert data.shape == (3, 128, 4, 128)
    assert scales.shape == (3, 128, 4, 16)
    assert data.data_ptr() == owner.data_ptr()
    assert scales.data_ptr() == owner.data_ptr() + 128 * 4 * 128
    assert data.stride() == (128 * 4 * 144, 4 * 128, 128, 1)
    assert scales.stride() == (128 * 4 * 144, 4 * 16, 16, 1)
    data[0].fill_(17)
    scales[0].view(torch.uint8).fill_(83)
    owner[2].copy_(owner[0])
    assert torch.equal(data[2], data[0])
    assert torch.equal(scales[2].view(torch.uint8), scales[0].view(torch.uint8))
    assert not owner[1].any()


def test_nvfp4_supports_both_admitted_logical_head_sizes() -> None:
    for dimension in (128, 256):
        owner = torch.zeros((2, 16, 4, dimension // 2 + dimension // 16), dtype=torch.uint8)
        data, scales = nvfp4_page_views(owner, dimension)
        assert data.shape[-1] == dimension // 2
        assert scales.shape[-1] == dimension // 16
        assert data.numel() + scales.numel() == owner.numel()
