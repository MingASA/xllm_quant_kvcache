# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Actual diagnostic reductions count clipping and exclude padding."""

from types import SimpleNamespace

import torch

from xllm.python.attention.fp8_numeric_diagnostics import NumericFlashInferBackend, _RECIPES


def test_counts_clipping_and_candidate_recipe_without_changing_inputs() -> None:
    backend = object.__new__(NumericFlashInferBackend)
    backend.head_dim = 256
    backend._metadata = SimpleNamespace(is_prefill=True, is_chunked_prefill=False,
                                        slot_mapping=torch.tensor([1, 2, -1]))
    backend._stats = {
        name:(torch.zeros((1, 2, heads, 4), dtype=torch.int64),
              torch.zeros((1, 2, heads, 2), dtype=torch.float64),
              torch.zeros((1, 2, heads), dtype=torch.float32))
        for name, _, heads in _RECIPES
    }
    backend._calls = torch.zeros((1, 2), dtype=torch.int64)
    k = torch.zeros((3, 4, 256), dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    q = torch.zeros((3, 24, 256), dtype=torch.bfloat16)
    k[0, 0, 0] = 60
    v[0, 0, :2] = torch.tensor([149, -149], dtype=torch.bfloat16)
    q[0, 0, 0] = 500
    for tensor in (q, k, v):
        tensor[2].fill_(1000)
    originals = [t.clone() for t in (q, k, v)]
    backend._observe(q, k, v, 3)
    assert backend._stats['key'][0][0, 0, 0, 1].item() == 1
    assert backend._stats['value_current'][0][0, 0, 0, 1].item() == 2
    assert backend._stats['value_candidate'][0][0, 0, 0, 1].item() == 0
    assert backend._stats['query'][0][0, 0, 0, 1].item() == 1
    assert backend._stats['value_current'][0][0, 0, 0, 0].item() == 512
    assert backend._stats['value_current'][2][0, 0, 0].item() == 149
    assert backend._stats['value_candidate'][1][0, 0, 0, 0] < backend._stats['value_current'][1][0, 0, 0, 0]
    assert all(torch.equal(a, b) for a, b in zip((q, k, v), originals))
