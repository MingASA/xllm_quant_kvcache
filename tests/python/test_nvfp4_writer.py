# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""CPU boundary checks; native kernel byte identity requires GPU qualification."""

import hashlib
from pathlib import Path
from types import ModuleType

import pytest
import torch

from xllm.python.attention.native_kv_layout import nvfp4_page_views
from xllm.python.attention import nvfp4_writer
from xllm.python.attention.nvfp4_writer import PrevalidatedNVFP4Writer


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"),
                                  torch.finfo(torch.float32).tiny / 2, torch.finfo(torch.float32).max * 2])
def test_invalid_scales_rejected_before_cuda_or_flashinfer_access(scale: float) -> None:
    for scales in ((scale, 1.0), (1.0, scale)):
        with pytest.raises(ValueError, match="normal FP32"):
            PrevalidatedNVFP4Writer({}, {}, scales)


def test_normal_fp32_scale_limits_are_accepted() -> None:
    limits = torch.finfo(torch.float32)
    nvfp4_writer._validate_fixed_scales((limits.tiny, limits.max))


def _cpu_boundary_writer() -> PrevalidatedNVFP4Writer:
    # Exercise input validation/dispatch using real tensors without pretending
    # that CPU storage is accepted by the production CUDA constructor.
    writer = PrevalidatedNVFP4Writer.__new__(PrevalidatedNVFP4Writer)
    writer._device = torch.device("cpu")
    writer._input_shape = (4, 128)
    pair = [nvfp4_page_views(torch.zeros((2, 16, 4, 72), dtype=torch.uint8), 128) for _ in range(2)]
    writer._layers = {2: (pair[0][0], pair[1][0], pair[0][1], pair[1][1])}
    writer._global_scales = (torch.empty(1, device="meta"), torch.empty(1, device="meta"))
    return writer


def test_bound_page_planes_accept_real_native_aliases_and_reject_bad_scales() -> None:
    writer = _cpu_boundary_writer()
    key, value, k_sf, v_sf = writer._layers[2]
    writer._validate_page_binding((key, value), (k_sf, v_sf))
    with pytest.raises(ValueError, match="block-scale"):
        writer._validate_page_binding((key, value), (k_sf.view(torch.uint8), v_sf))


def test_dynamic_dispatch_keeps_padding_noncontiguous_inputs_and_opaque_scales() -> None:
    writer = _cpu_boundary_writer()
    calls = []
    writer._kernel = lambda *args: calls.append(args)
    key = torch.randn((7, 4, 256), dtype=torch.bfloat16)[..., ::2]
    value = torch.randn_like(key)
    slots = torch.tensor([0, -1, 5, 8, -1], dtype=torch.int64)
    writer.append(key, value, slots, 2)
    assert len(calls) == 1
    args = calls[0]
    assert all(left is right for left, right in zip(args[:3], (key, value, slots)))
    assert all(left is right for left, right in zip(args[3:7], writer._layers[2]))
    assert args[7] is writer.global_scales[0] and args[8] is writer.global_scales[1]
    assert args[9] == 0


@pytest.mark.parametrize("case", ["key_dtype", "value_dtype", "shape", "device", "slot_dtype", "slot_shape", "rows"])
def test_invalid_dynamic_inputs_fail_before_native_mutation(case: str) -> None:
    writer = _cpu_boundary_writer()

    def forbidden_kernel(*args: object) -> None:
        raise AssertionError("invalid inputs reached the native kernel")

    writer._kernel = forbidden_kernel
    key, value = [torch.zeros((5, 4, 128), dtype=torch.bfloat16) for _ in range(2)]
    slots = torch.zeros(5, dtype=torch.int32)
    if case == "key_dtype":
        key = key.float()
    elif case == "value_dtype":
        value = value.half()
    elif case == "shape":
        key = key[:, :3]
    elif case == "device":
        slots = torch.empty(5, dtype=torch.int32, device="meta")
    elif case == "slot_dtype":
        slots = slots.float()
    elif case == "slot_shape":
        slots = slots.unsqueeze(0)
    elif case == "rows":
        slots = torch.zeros(6, dtype=torch.int32)
    with pytest.raises(ValueError):
        writer.append(key, value, slots, 2)


def test_private_api_gate_rejects_source_and_signature_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "page.py"
    source.write_text("qualified test fixture\n")
    monkeypatch.setattr(nvfp4_writer, "_PINNED_PAGE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    module = ModuleType("flashinfer.page")
    module.__file__ = str(source)

    def kernel(
        append_key: torch.Tensor, append_value: torch.Tensor, slot_mapping: torch.Tensor,
        paged_k_cache: torch.Tensor, paged_v_cache: torch.Tensor, k_scale_cache: torch.Tensor,
        v_scale_cache: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor, layout: int,
    ) -> None:
        pass

    setattr(module, nvfp4_writer._KERNEL_NAME, kernel)
    assert nvfp4_writer._native_slot_writer(module) is kernel
    source.write_text("drift\n")
    with pytest.raises(RuntimeError, match="SHA256"):
        nvfp4_writer._native_slot_writer(module)
    source.write_text("qualified test fixture\n")
    setattr(module, nvfp4_writer._KERNEL_NAME, lambda *args: None)
    with pytest.raises(RuntimeError, match="signature"):
        nvfp4_writer._native_slot_writer(module)
