# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Compare fast planning against the actual pinned Python plan on CPU tensors."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

_PATH = Path(__file__).resolve().parents[2] / "xllm/python/attention/mxfp4_decode_plan.py"
_SPEC = importlib.util.spec_from_file_location("mxfp4_decode_plan", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
planner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(planner)


def _upstream_plan() -> Any:
    location = os.environ.get("MXFP4_FLASHINFER_PREFILL_SOURCE")
    if location is None:
        raise unittest.SkipTest("Set MXFP4_FLASHINFER_PREFILL_SOURCE to pinned prefill.py")
    source = Path(location).read_bytes()
    assert hashlib.sha256(source).hexdigest() == "2ad12a8387b3f6bff192e5945769b68d90cfcab00b9eb2a8a524af8dd71a29de"
    tree = ast.parse(source)
    wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "BatchPrefillWithPagedKVCacheWrapper")
    method = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == "plan")
    method.decorator_list = []
    # Only the native/CUDA boundary is replaced. All Python planning logic is
    # executed from the hash-checked installed dependency, not copied here.
    namespace = {
        "torch": torch, "canonicalize_torch_dtype": lambda dtype: dtype,
        "_check_workspace_buffer_alignment": lambda *args: None,
        "_nvfp4_kv_requires_disabled_split_kv": lambda dtype: dtype == torch.uint8,
        "get_seq_lens": lambda ptr, last, size: torch.clamp(ptr[1:] - ptr[:-1] - 1, min=0) * size + last,
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(method), str(location), "exec"), namespace)
    return namespace["plan"]


class _NativeRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def plan(self, *args: Any) -> list[int]:
        self.calls.append(tuple(arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args))
        return [len(self.calls)]


def _wrapper(graph: bool, batch: int) -> Any:
    native = _NativeRecorder()
    wrapper = SimpleNamespace(
        _xllm_mxfp4_plan_contract=planner.MXFP4_DECODE_PLAN_CONTRACT,
        _backend="fa2", _jit_module=native, _cached_module=None,
        _float_workspace_buffer=torch.empty(32, dtype=torch.uint8),
        _int_workspace_buffer=torch.full((32,), 0xa5, dtype=torch.uint8),
        _pin_memory_int_workspace_buffer=torch.full((32,), 0xa5, dtype=torch.uint8),
        _kv_lens_buffer=torch.empty(128, dtype=torch.int32),
        device=torch.device("cpu"), is_cuda_graph_enabled=graph,
        _fixed_batch_size=batch, _max_total_num_rows=None,
        _qo_indptr_buf=torch.empty(batch + 1, dtype=torch.int32),
        _paged_kv_indptr_buf=torch.empty(batch + 1, dtype=torch.int32),
        _paged_kv_indices_buf=torch.full((128,), -1, dtype=torch.int32),
        _paged_kv_last_page_len_buf=torch.empty(batch, dtype=torch.int32),
        _custom_mask_buf=None, _mask_indptr_buf=None,
    )
    method = _upstream_plan()
    wrapper.plan = lambda *args, **kwargs: method(wrapper, *args, **kwargs)
    return wrapper


class DecodePlanTest(unittest.TestCase):
    def test_native_arguments_and_dynamic_pages_match_upstream(self) -> None:
        for graph in (False, True):
            for initial_batch in (1, 2, 4):
                fast_wrapper, baseline = _wrapper(graph, initial_batch), _wrapper(graph, initial_batch)
                fast = planner.Mxfp4DecodePlanner(
                    fast_wrapper, num_qo_heads=24, num_kv_heads=4,
                    head_dim=256, page_size=128, sm_scale=1 / 16,
                )
                cached_query = None
                for step in range(4):
                    batch = initial_batch if graph or step != 2 else initial_batch + 1
                    counts = torch.arange(batch, dtype=torch.int32) % 3 + step + 1
                    ptr = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)))
                    last = (torch.arange(batch, dtype=torch.int32) * 19 + step * 17) % 128 + 1
                    ids = ((torch.arange(int(ptr[-1]), dtype=torch.int32) * 7 + step * 3) % 127).clone()
                    fast.plan(ptr.clone(), ids.clone(), last.clone())
                    baseline.plan(torch.arange(batch + 1, dtype=torch.int32), ptr, ids, last,
                                  24, 4, 256, 128, causal=True, sm_scale=1 / 16,
                                  q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
                                  o_data_type=torch.bfloat16)
                    for actual, expected in zip(fast_wrapper._jit_module.calls[-1][3:], baseline._jit_module.calls[-1][3:]):
                        if isinstance(actual, torch.Tensor):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        else:
                            self.assertEqual(actual, expected)
                    for name in ("_qo_indptr_buf", "_paged_kv_indptr_buf", "_paged_kv_last_page_len_buf"):
                        torch.testing.assert_close(getattr(fast_wrapper, name), getattr(baseline, name), rtol=0, atol=0)
                    torch.testing.assert_close(fast_wrapper._paged_kv_indices_buf[:ids.numel()], ids, rtol=0, atol=0)
                    self.assertTrue(fast_wrapper._jit_module.calls[-1][17])  # uint8 split-KV guard
                    self.assertEqual(fast_wrapper._plan_info, [step + 1])
                    self.assertEqual(fast_wrapper._cached_kv_data_type, torch.uint8)
                    if graph:
                        if cached_query is None:
                            cached_query = fast._query_indptr[batch]
                        self.assertIs(cached_query, fast._query_indptr[batch])

    def test_graph_batch_capacity_and_host_dtype_fail_before_native_plan(self) -> None:
        wrapper = _wrapper(True, 2)
        fast = planner.Mxfp4DecodePlanner(wrapper, num_qo_heads=24, num_kv_heads=4,
                                        head_dim=256, page_size=128, sm_scale=1 / 16)
        ptr, ids, last = torch.tensor([0, 1, 2], dtype=torch.int32), torch.tensor([5, 3], dtype=torch.int32), torch.ones(2, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "CPU int32"):
            fast.plan(ptr.to(torch.int64), ids, last)
        with self.assertRaisesRegex(ValueError, "batch size"):
            fast.plan(ptr[:2], ids, last[:1])
        with self.assertRaisesRegex(ValueError, "capacity"):
            fast.plan(ptr, torch.zeros(129, dtype=torch.int32), last)
        self.assertEqual(wrapper._jit_module.calls, [])

    def test_unpatched_wrapper_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "private Graph-mask"):
            planner.Mxfp4DecodePlanner(SimpleNamespace(), num_qo_heads=24,
                                      num_kv_heads=4, head_dim=256, page_size=128, sm_scale=1 / 16)
