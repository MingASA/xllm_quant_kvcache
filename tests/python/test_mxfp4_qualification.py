# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""CPU checks for the GPU qualifier's independent reference and mask reader."""

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

_PATH = Path(__file__).resolve().parents[2] / "tools/qualify_mxfp4_graph.py"
_SPEC = importlib.util.spec_from_file_location("qualify_mxfp4_graph", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
qualifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qualifier)


class QualificationTest(unittest.TestCase):
    def test_run_keywords_match_the_actual_pinned_flashinfer_signature(self) -> None:
        location = os.environ.get("MXFP4_FLASHINFER_PREFILL_SOURCE")
        if location is None:
            self.skipTest("Set MXFP4_FLASHINFER_PREFILL_SOURCE to the pinned source")
        raw = Path(location).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "2ad12a8387b3f6bff192e5945769b68d90cfcab00b9eb2a8a524af8dd71a29de")
        cls = next(node for node in ast.parse(raw).body if isinstance(node, ast.ClassDef)
                   and node.name == "BatchPrefillWithPagedKVCacheWrapper")
        run = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        permitted = {arg.arg for arg in (*run.args.args, *run.args.kwonlyargs)}
        calls = [node for node in ast.walk(ast.parse(_PATH.read_text())) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "run"]
        self.assertGreaterEqual(len(calls), 2)
        for call in calls:
            self.assertLessEqual({keyword.arg for keyword in call.keywords}, permitted)

    def test_independent_reference_uses_physical_pages_and_causal_prefix(self) -> None:
        from xllm.python.attention.mxfp4_writer import mxfp4_page_views

        q = torch.zeros((2, 24, 256), dtype=torch.bfloat16)
        key_owner = torch.zeros((2, 128, 4, 136), dtype=torch.uint8)
        value_owner = torch.empty_like(key_owner)
        key, key_scale = mxfp4_page_views(key_owner, 256)
        value, value_scale = mxfp4_page_views(value_owner, 256)
        key_scale.fill_(127)
        value_scale.fill_(127)
        value[0].fill_(0x22)  # all one
        value[1].fill_(0x44)  # all two
        ptr = torch.tensor([0, 2], dtype=torch.int32)
        last = torch.tensor([1], dtype=torch.int32)
        query = torch.tensor([0, 2], dtype=torch.int32)
        for ids, expected in (([0, 1], [1.0, 130 / 129]), ([1, 0], [2.0, 257 / 129])):
            rows, output = qualifier._reference(q, key, value, key_scale, value_scale, ptr,
                                               torch.tensor(ids, dtype=torch.int32), last, query)
            self.assertEqual(rows.tolist(), [0, 1])
            reference = torch.tensor(expected).reshape(2, 1, 1).expand_as(output)
            torch.testing.assert_close(output, reference, rtol=1e-6, atol=1e-6)

    def test_mask_reader_rejects_dirty_padding_and_tracks_inactive_rows(self) -> None:
        memory = torch.full((2048,), 0xa5, dtype=torch.uint8)
        info = [66, 2, 900, 16, 0, 0, 0, 0, 0, 0, 0, 0, 1024, 1, 0]
        wrapper = SimpleNamespace(_plan_info=info, _int_workspace_buffer=memory)
        query = torch.tensor([0, 1, 1], dtype=torch.int32)
        with self.assertRaises(AssertionError):
            qualifier._mask(wrapper, query)
        memory[1024:1090].zero_()
        memory[1024] = 1
        result = qualifier._mask(wrapper, query)
        self.assertEqual((result["valid"], result["padded"]), (1, 66))
        info[-1] = 1
        with self.assertRaisesRegex(AssertionError, "non-split"):
            qualifier._mask(wrapper, query)
