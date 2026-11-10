# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Pinned FlashInfer FA2 adaptation: MX scales expand only inside tile smem.

The attention algorithm, planner, packed E2M1 loads and BF16 MMA remain upstream
FlashInfer. One block32 E8M0 byte is duplicated into two block16 *shared-memory*
positions. Its exponent is decoded exactly in registers. No global expanded
scale plane, BF16 cache or history dequantization workspace is allocated.
Only paged SM90 BF16 D256 is exposed. This adapter requires GPU qualification.
"""

from __future__ import annotations

import functools
import hashlib
import re
from pathlib import Path
from typing import Any

_HEADER_SHA256 = "f93cbea96e3a723254c15f5a52d6dd4c13e3892e88eb9b9fed1c0c0048c7d710"


def _adapt_header(source: bytes) -> str:
    if hashlib.sha256(source).hexdigest() != _HEADER_SHA256:
        raise ValueError("MXFP4 adapter requires the qualified FlashInfer 0.6.18.post1 header")
    text = source.decode()
    # Moving the private copy must not change how upstream dependencies resolve.
    text = re.sub(r'#include "\.\./([^\"]+)"', r'#include <flashinfer/\1>', text)
    text = re.sub(r'#include "([^\"/]+\.cuh)"', r'#include <flashinfer/attention/\1>', text)
    helper = """
__device__ __forceinline__ float mxfp4_scale(uint8_t code) {
  const uint32_t bits = code == 0 ? 0x00400000U :
                        code == 255 ? 0x7fc00000U : static_cast<uint32_t>(code) << 23;
  return __uint_as_float(bits);
}
"""
    text = text.replace("namespace flashinfer {", "namespace flashinfer {\n" + helper, 1)
    text = re.sub(r"__nv_fp8_e4m3 (sf[a-z0-9_, ]+);", r"float \1;", text)
    text, count = re.subn(r"(sf\w+)_fp8\.__x = ([^;]+);", r"\1_fp8 = mxfp4_scale(\2);", text)
    if count != 12:
        raise ValueError("FlashInfer scale-fragment conversion contract changed")
    begin = text.index("__device__ __forceinline__ void page_produce_kv_sf(")
    end = text.index("\n/*!", begin)
    page_loader = text[begin:end]
    old = """        cp_async::pred_load_32b<fill_mode>(
            reinterpret_cast<uint32_t*>(sf_smem + flat_byte),
            reinterpret_cast<const uint32_t*>(sf_ptr + sf_gmem_offset), in_bounds);"""
    new = """        // Two block32 E8M0 bytes become four logical block16 SF bytes in
        // this tile only. Data/scales remain compressed in global memory.
        const size_t mx_offset = page_head_base + entry_idx * sf_stride_n + sf_smem_col / 2;
        uint32_t packed = 0;
        if (in_bounds) {
          const uint32_t pair = *reinterpret_cast<const uint16_t*>(sf_ptr + mx_offset);
          packed = (pair & 0xffU) * 0x101U + (pair >> 8) * 0x1010000U;
        }
        if (flat_byte < SF_TOTAL_BYTES) {
          *reinterpret_cast<uint32_t*>(sf_smem + flat_byte) = packed;
        }"""
    if page_loader.count(old) != 1:
        raise ValueError("FlashInfer paged scale-loader contract changed")
    text = text[:begin] + page_loader.replace(old, new) + text[end:]
    return text


def _jit_inputs(
    package: Path | None = None, csrc: Path | None = None,
) -> tuple[str, str, str, dict[str, Any]]:
    import torch

    from xllm.python.attention.mxfp4_graph_sources import read_graph_sources

    if package is None:
        import flashinfer
        from flashinfer.jit import env

        package = Path(flashinfer.__file__).parent
        csrc = env.FLASHINFER_CSRC_DIR
    elif csrc is None:
        csrc = package / "data/csrc"
    original = package / "data/include/flashinfer/attention/prefill.cuh"
    adapted = _adapt_header(original.read_bytes())
    scheduler, binding = read_graph_sources(package, csrc)
    tensors = ["maybe_custom_mask", "maybe_mask_indptr", "maybe_alibi_slopes",
               "maybe_prefix_len_ptr", "maybe_token_pos_in_items_ptr", "maybe_max_item_len_ptr",
               "maybe_k_cache_sf", "maybe_v_cache_sf"]
    tensor_types = ["uint8_t", "int32_t", "float", "uint32_t", "uint16_t", "uint16_t", "uint8_t", "uint8_t"]
    scalars = ["logits_soft_cap", "sm_scale", "rope_rcp_scale", "rope_rcp_theta", "token_pos_in_items_len"]
    parameters = dict(
        backend="fa2", dtype_q=torch.bfloat16, dtype_kv=torch.uint8,
        dtype_o=torch.bfloat16, idtype=torch.int32, head_dim_qk=256, head_dim_vo=256,
        additional_tensor_names=tensors, additional_tensor_dtypes=tensor_types,
        additional_scalar_names=scalars,
        additional_scalar_dtypes=["double", "double", "double", "double", "int64_t"],
        variant_name="DefaultAttention<use_custom_mask, false, false, false>",
        variant_decl="#include<flashinfer/attention/variants.cuh>",
        pos_encoding_mode=0, use_sliding_window=False, use_logits_soft_cap=False,
        use_fp16_qk_reduction=False, fp8_enabled=False,
    )
    return adapted, scheduler, binding, parameters


def mxfp4_jit_source_manifest(
    package: Path | None = None, csrc: Path | None = None,
) -> dict[str, Any]:
    """Identify validated pinned inputs without generating or compiling sources."""
    from xllm.python.attention.mxfp4_graph_sources import graph_source_manifest

    return graph_source_manifest(*_jit_inputs(package, csrc))


def mxfp4_jit_spec() -> tuple[Any, list[str], list[str]]:
    """Create the pinned SM90 spec; CPU-only offline precompilation is allowed."""
    from flashinfer.jit import env
    from flashinfer.jit.attention.modules import gen_customize_batch_prefill_module

    from xllm.python.attention.mxfp4_graph_sources import (
        configure_graph_sources,
        graph_source_manifest,
    )

    adapted, scheduler, binding, parameters = _jit_inputs()
    name = graph_source_manifest(adapted, scheduler, binding, parameters)["module_name"]
    directory = env.FLASHINFER_GEN_SRC_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    header = directory / "mxfp4_prefill.cuh"
    if not header.exists():
        header.write_text(adapted)
    elif header.read_text() != adapted:
        raise ValueError("MXFP4 private JIT header changed")
    spec = gen_customize_batch_prefill_module(uri=name, **parameters)
    configure_graph_sources(spec, directory, scheduler, binding)
    spec.extra_cuda_cflags = [*(spec.extra_cuda_cflags or []), "-include", str(header)]
    return spec, parameters["additional_tensor_names"], parameters["additional_scalar_names"]


@functools.cache
def _load_mxfp4_module() -> tuple[Any, list[str], list[str]]:
    """Compile/register once; eager and Graph wrappers share this module."""
    from flashinfer.prefill import get_batch_prefill_jit_module

    spec, tensors, scalars = mxfp4_jit_spec()
    module = get_batch_prefill_jit_module(spec.name, spec.build_and_load())
    return module, tensors, scalars


def make_mxfp4_wrapper(workspace: Any, **graph_options: Any) -> Any:
    """Construct a private JIT wrapper without changing the installed package."""
    import flashinfer
    import torch

    from xllm.python.attention.mxfp4_decode_plan import MXFP4_DECODE_PLAN_CONTRACT

    plan_source = Path(flashinfer.prefill.__file__).read_bytes()
    if hashlib.sha256(plan_source).hexdigest() != "2ad12a8387b3f6bff192e5945769b68d90cfcab00b9eb2a8a524af8dd71a29de":
        raise ValueError("MXFP4 wrapper requires the pinned FlashInfer Python planning contract")
    if torch.cuda.get_device_capability(workspace.device) != (9, 0):
        raise ValueError("MXFP4 FlashInfer adapter currently targets SM90 only")
    module, tensors, scalars = _load_mxfp4_module()
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2", **graph_options)
    wrapper._jit_module = module
    wrapper._jit_additional_tensor_names = tensors
    wrapper._jit_additional_scalar_names = scalars
    wrapper._xllm_mxfp4_plan_contract = MXFP4_DECODE_PLAN_CONTRACT
    return wrapper
