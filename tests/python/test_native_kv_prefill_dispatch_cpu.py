# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Host dispatch contract only; no Torch import, kernel, or model execution.

Load the real method bodies from the source AST to keep these branch checks
independent of CUDA initialization. Opaque metadata and dtype values are only
forwarded to the external planner. Numerical semantics require GPU qualification.
"""

from __future__ import annotations

import __future__
import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, NoReturn
import unittest
from unittest.mock import Mock


_SOURCE = Path(__file__).resolve().parents[2] / "xllm/python/attention/flashinfer_native_kv.py"


def _load_method(name: str) -> Callable[..., Any]:
    tree = ast.parse(_SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "FlashInferNativeKVBackend")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)

    def _reject_device_import(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError(f"FA2 prefill unexpectedly entered a device dispatch path: {args[0]}")

    scope = {"__builtins__": {**vars(builtins), "__import__": _reject_device_import}}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(_SOURCE), "exec", flags=__future__.annotations.compiler_flag), scope)
    return scope[name]


class NativeKVPrefillDispatchTest(unittest.TestCase):
    def test_fa2_preserves_query_dtype_and_packed_metadata(self) -> None:
        prepare = _load_method("prepare")
        for dtype in ("BF16", "FP16"):
            for phase in ("prefill", "chunked"):
                with self.subTest(dtype=dtype, phase=phase):
                    pointers = [object() for _ in range(4)]
                    metadata = SimpleNamespace(
                        is_prefill=phase == "prefill", is_chunked_prefill=phase == "chunked",
                        qo_indptr=pointers[0], paged_kv_indptr=pointers[1],
                        paged_kv_indices=pointers[2], paged_kv_last_page_len=pointers[3],
                    )
                    backend = SimpleNamespace(
                        _cache_format="fp8_e4m3", _prefill_backend="fa2", _decode_backend="xqa",
                        sliding_window=0, scale=256**-0.5, dtype=dtype,
                        _cache_dtype="FP8", _prefill_query_dtype=dtype,
                        _prefill_paged_wrapper=Mock(),
                        num_heads=24, num_kv_heads=4, head_dim=256, page_size=128,
                    )
                    prepare(backend, metadata)
                    self.assertIsNone(backend._mixed_attention)
                    backend._prefill_paged_wrapper.plan.assert_called_once_with(
                        *pointers, 24, 4, 256, 128, causal=True, sm_scale=256**-0.5,
                        window_left=-1, q_data_type=dtype, kv_data_type="FP8", o_data_type=dtype,
                    )

    def test_runtime_report_limits_mixed_split_to_fa3_xqa(self) -> None:
        report = _load_method("native_kv_runtime_info")
        for prefill in ("fa2", "fa3"):
            for decode in ("flashinfer", "xqa"):
                with self.subTest(prefill=prefill, decode=decode):
                    backend = SimpleNamespace(
                        _cache_format="fp8_e4m3", _fixed_scales=(0.125, 0.0625),
                        _prefill_backend=prefill, _decode_backend=decode,
                        _prefill_query_dtype="FP8" if prefill == "fa3" else "BF16",
                        dtype="BF16", _max_context_tokens=262144,
                        _kv_caches=[], _virtual_memory=False, _global_scales=[],
                    )
                    selected = report(backend)["mixed_FP8_decode"]
                    if (prefill, decode) == ("fa3", "xqa"):
                        self.assertEqual(selected, "XQA_subset_eager_attention_no_mixed_decode_graph")
                    else:
                        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
