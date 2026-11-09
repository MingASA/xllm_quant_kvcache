# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Private backport of FlashInfer's non-split Graph padding-mask repair.

Source: flashinfer-ai/flashinfer@4c2b8a7ed4af902df94e83260c8c12e8d3f71872,
include/flashinfer/attention/scheduler.cuh:909-924 and
csrc/batch_prefill_paged.cuh:131-143 (also the equivalent ragged binding).
Only the mask lifetime/binding is backported to the pinned 8bc dependency.
The installed package and the uint8 split-KV correctness guard stay unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


PINNED_FLASHINFER_COMMIT = "8bc3b578027791336c6ae87db5c9d76f82cef8bc"
UPSTREAM_GRAPH_FIX_COMMIT = "4c2b8a7ed4af902df94e83260c8c12e8d3f71872"

_SOURCE_SHA256 = {
    "include/flashinfer/attention/scheduler.cuh": "a11a39aea393cdb380ad9439aac9170cd96849d7426ebbb920c5c8cdadfcc575",
    "csrc/batch_prefill.cu": "6bed87bda87519440d2cc31ba7c349b6068f3458ad8a16d4187829930f83df82",
    "csrc/batch_prefill_jit_binding.cu": "dad0d81c55b92f71a634cb8240dcb0d150aa6ffd391ef3ef5d87b6207eeca4fc",
    "csrc/batch_prefill_customize_config.jinja": "c60626df3c30122721eae44530c18db25f63f934dd695e4803be342855b73ad7",
    "csrc/batch_prefill_paged_kernel_inst.jinja": "0bbe91ae1972c4cbc2f12d1b0e2fddcf1e24050be9ea6b59231ad65a8163c5af",
    "csrc/batch_prefill_ragged_kernel_inst.jinja": "4e177455f914c3665cf38728883d74c076f368c1bea3736c0033675de342f687",
    "csrc/tvm_ffi_utils.h": "4fe41227dc3bd833b492f6b04bb94eb497b78be85847598eb19bd08c016de787",
    "jit/core.py": "454316d67f22ebe3035cf17a13e9d01ffe5eed392d4e96adb952f3955bcb132b",
    "jit/attention/modules.py": "23a8fb1fa8f45db3b527645d15a65c874dba6d3bc7d7df6599d9d7c3a6ddea30",
}


def _replace(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError("Pinned FlashInfer Graph source contract changed")
    return text.replace(old, new)


def _adapt_scheduler(source: bytes) -> str:
    text = source.decode()
    start = text.index("inline cudaError_t PrefillPlanImpl(")
    end = text.index("\ninline cudaError_t PrefillPlan(", start)
    body = text[start:end]
    allocation = """    plan_info.block_valid_mask_offset = int_allocator.aligned_alloc_offset(
        sizeof(bool) * padded_batch_size, 16, "batch_prefill_block_valid_mask");
"""
    pointer = """      bool* block_valid_mask_h =
          GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
"""
    initialize = """      for (uint32_t i = 0; i < padded_batch_size; ++i) {
        block_valid_mask_h[i] = i < new_batch_size;
      }
"""
    body = _replace(body, allocation, "")
    body = _replace(body, pointer, "")
    body = _replace(body, initialize, "")
    graph_mask = """  // Graph launches include padding even when split-KV is disabled.
  // Do not let padded CTAs consume uninitialized request/tile indices.
  if (enable_cuda_graph) {
    plan_info.block_valid_mask_offset = int_allocator.aligned_alloc_offset(
        sizeof(bool) * padded_batch_size, 16, "batch_prefill_block_valid_mask");
    if constexpr (MATERIALIZE) {
      bool* block_valid_mask_h =
          GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
      for (uint32_t i = 0; i < padded_batch_size; ++i) {
        block_valid_mask_h[i] = i < new_batch_size;
      }
    }
  }

"""
    body = _replace(body, "  float_workspace_size_out =", graph_mask + "  float_workspace_size_out =")
    text = text[:start] + body + text[end:]
    # The private header is preincluded from a different directory. Keep its
    # original include guard so subsequent installed scheduler includes skip it.
    text = re.sub(r'#include "\.\./([^\"]+)"', r'#include <flashinfer/\1>', text)
    return _replace(text, '#include "heap.h"', '#include <flashinfer/attention/heap.h>')


def _adapt_batch_prefill(source: bytes) -> str:
    text = source.decode()
    nested = """          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
          }
"""
    text = _replace(text, nested, "", count=2)
    old = """        if (plan_info.enable_cuda_graph) {
          params.total_num_rows ="""
    new = """        if (plan_info.enable_cuda_graph) {
          params.block_valid_mask =
              GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
          params.total_num_rows ="""
    return _replace(text, old, new, count=2)


def read_graph_sources(package: Path, csrc: Path) -> tuple[str, str]:
    """Validate the actual generator inputs before constructing private sources."""
    sources = {}
    for name, expected in _SOURCE_SHA256.items():
        if name.startswith("csrc/"):
            path = csrc / name.removeprefix("csrc/")
        elif name.startswith("include/"):
            path = package / "data" / name
        else:
            path = package / name
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"MXFP4 Graph repair requires pinned FlashInfer source: {name}")
        sources[name] = source
    return (_adapt_scheduler(sources["include/flashinfer/attention/scheduler.cuh"]),
            _adapt_batch_prefill(sources["csrc/batch_prefill.cu"]))


def graph_source_fingerprint(
    prefill: str, scheduler: str, binding: str, parameters: dict[str, Any]
) -> str:
    """Identify the complete pinned source/template and specialization contract."""
    payload = json.dumps(
        {"inputs": _SOURCE_SHA256, "parameters": parameters,
         "prefill": prefill, "scheduler": scheduler, "binding": binding},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_private_source(path: Path, source: str) -> None:
    if path.exists():
        if path.read_text() != source:
            raise ValueError(f"MXFP4 private source changed: {path.name}")
    else:
        path.write_text(source)


def configure_graph_sources(spec: Any, directory: Path, scheduler: str, binding: str) -> None:
    """Bind the patched run translation unit and scheduler to this JIT spec only."""
    original = directory / "batch_prefill.cu"
    if sum(Path(path) == original for path in spec.sources) != 1:
        raise ValueError("FlashInfer custom prefill did not expose its expected run source")
    header = directory / "mxfp4_scheduler.cuh"
    source = directory / "mxfp4_batch_prefill.cu"
    _write_private_source(header, scheduler)
    _write_private_source(source, binding)
    spec.sources = [source if Path(path) == original else path for path in spec.sources]
    spec.extra_cuda_cflags = [*(spec.extra_cuda_cflags or []), "-include", str(header)]
