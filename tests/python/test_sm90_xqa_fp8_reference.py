# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU algebra checks only; not GPU qualification or model quality evidence."""

from __future__ import annotations

import pytest
import torch

from tools.sm90_xqa_fp8_reference import reference_decode


@pytest.mark.parametrize("splits", [1, 2, 12])
def test_uniform_logits_merge_all_valid_tokens_and_empty_splits(splits: int) -> None:
    query = torch.zeros(2, 1, 4, 128)
    key = torch.zeros(2, 129, 2, 128)
    value = torch.arange(129).float().view(1, 129, 1, 1).expand(2, 129, 2, 128)
    result = reference_decode(query, key, value, split_count=splits)
    torch.testing.assert_close(result, torch.full((2, 4 * 128), 64.0))


def test_second_GEMM_probability_is_fp8_but_denominator_is_unquantized() -> None:
    query = torch.zeros(1, 1, 1, 128)
    query[..., 0] = 1
    key = torch.zeros(1, 2, 1, 128)
    key[:, 1, :, 0] = 8
    value = torch.zeros_like(key)
    value[:, 0] = 1
    exponential = torch.exp(torch.tensor(-8 / 128**.5))
    quantized = (exponential * 448).to(torch.float8_e4m3fn).float() / 448
    expected = quantized / (exponential + 1)
    result = reference_decode(query, key, value, split_count=1)
    torch.testing.assert_close(result, expected.expand(1, 128))
    assert not torch.isclose(expected, exponential / (exponential + 1))


def test_prefill_or_invalid_split_is_refused() -> None:
    query = torch.zeros(1, 2, 4, 128)
    cache = torch.zeros(1, 129, 2, 128)
    with pytest.raises(ValueError):
        reference_decode(query, cache, cache, split_count=1)
    with pytest.raises(ValueError):
        reference_decode(query[:, :1], cache, cache, split_count=0)
