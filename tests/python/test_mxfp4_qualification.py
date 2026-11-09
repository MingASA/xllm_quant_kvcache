# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""CPU checks for the GPU qualifier's independent reference and mask reader."""

import importlib.util
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
    def test_independent_reference_uses_physical_pages_and_causal_prefix(self) -> None:
        q = torch.zeros((2, 24, 256), dtype=torch.bfloat16)
        key = torch.zeros((2, 128, 4, 128), dtype=torch.uint8)
        value = torch.empty_like(key)
        value[0].fill_(0x22)  # all one
        value[1].fill_(0x44)  # all two
        scales = torch.full((2, 128, 4, 8), 127, dtype=torch.uint8)
        ptr = torch.tensor([0, 2], dtype=torch.int32)
        last = torch.tensor([1], dtype=torch.int32)
        query = torch.tensor([0, 2], dtype=torch.int32)
        for ids, expected in (([0, 1], [1.0, 130 / 129]), ([1, 0], [2.0, 257 / 129])):
            rows, output = qualifier._reference(q, key, value, scales, scales, ptr,
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
