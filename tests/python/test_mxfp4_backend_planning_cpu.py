# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Exercise the backend's planner lifecycle without a CUDA import or device."""

from __future__ import annotations

import __future__
import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock


class BackendPlanningTest(unittest.TestCase):
    def setUp(self) -> None:
        source = Path(__file__).resolve().parents[2]/"xllm/python/attention/flashinfer_native_kv.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "FlashInferNativeKVBackend")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_prepare_mxfp4_decode")
        self.factory, self.planner, self.arange = Mock(), Mock(), Mock()
        modules = {
            "xllm.python.attention.mxfp4_flashinfer": SimpleNamespace(make_mxfp4_wrapper=self.factory),
            "xllm.python.attention.mxfp4_decode_plan": SimpleNamespace(Mxfp4DecodePlanner=self.planner),
        }

        def import_external(name: str, *args: Any, **kwargs: Any) -> Any:
            return modules[name]

        namespace = {"torch": SimpleNamespace(arange=self.arange, int32="int32"),
                     "__builtins__": dict(vars(builtins), __import__=import_external)}
        exec(compile(ast.unparse(method), str(source), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        self.prepare = namespace[method.name]
        self.backend = SimpleNamespace(
            _mxfp4_graph_wrappers={}, _mxfp4_decode_planners={},
            _mxfp4_eager_decode_wrapper=Mock(), _decode_workspace=SimpleNamespace(device="cuda:0"),
            num_heads=24, num_kv_heads=4, head_dim=256, page_size=128, scale=1/16,
        )
        self.metadata = SimpleNamespace(
            paged_kv_indptr_host=object(), paged_kv_last_page_len_host=object(),
            paged_kv_indptr=Mock(data_ptr=lambda: 101),
            paged_kv_indices=Mock(data_ptr=lambda: 102),
            paged_kv_last_page_len=Mock(numel=lambda: 2, data_ptr=lambda: 103),
        )
        self.metadata.paged_kv_last_page_len_host = Mock(numel=lambda: 2)

    def test_graph_reuses_planner_but_passes_every_fresh_metadata_view(self) -> None:
        self.prepare(self.backend, self.metadata, graph_mode=True)
        first = self.planner.return_value
        self.metadata.paged_kv_indptr_host = object()
        self.metadata.paged_kv_last_page_len_host = Mock(numel=lambda: 2)
        self.prepare(self.backend, self.metadata, graph_mode=True)
        self.assertEqual(self.factory.call_count, 1)
        self.assertEqual(self.planner.call_count, 1)
        self.assertEqual(self.arange.call_count, 1)
        self.assertEqual(first.plan.call_count, 2)
        first.plan.assert_called_with(self.metadata.paged_kv_indptr_host,
                                      self.metadata.paged_kv_indices,
                                      self.metadata.paged_kv_last_page_len_host)

    def test_graph_address_change_fails_before_replanning(self) -> None:
        self.prepare(self.backend, self.metadata, graph_mode=True)
        self.metadata.paged_kv_indices = Mock(data_ptr=lambda: 999)
        with self.assertRaisesRegex(RuntimeError, "addresses changed"):
            self.prepare(self.backend, self.metadata, graph_mode=True)
        self.assertEqual(self.planner.return_value.plan.call_count, 1)

    def test_eager_planner_owns_query_cache_and_requires_host_csr(self) -> None:
        for _ in range(2):
            self.prepare(self.backend, self.metadata, graph_mode=False)
        self.factory.assert_not_called()
        self.arange.assert_not_called()
        self.assertEqual(self.planner.call_count, 1)
        self.metadata.paged_kv_indptr_host = None
        with self.assertRaisesRegex(ValueError, "official host"):
            self.prepare(self.backend, self.metadata, graph_mode=False)
        self.assertEqual(self.planner.return_value.plan.call_count, 2)


if __name__ == "__main__":
    unittest.main()
