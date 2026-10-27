# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Independent packed-byte and causal references used by the real GPU gate."""

from __future__ import annotations

import pytest
import torch

from tools.validate_native_kv_pages import _decode_rows, _reference_attention


def test_NVFP4_all_nibbles_block16_scale_order_and_global_factor() -> None:
    codes = torch.arange(16, dtype=torch.uint8)
    packed = codes[::2] | (codes[1::2] << 4)
    payload = packed.repeat(2).view(1, 1, 16)
    # E4M3 bit patterns for 0.5 and 2.0. Each applies to 16 decoded channels.
    scales = torch.tensor([[[0x30, 0x40]]], dtype=torch.uint8)
    actual = _decode_rows(payload, scales, "nvfp4", 3.0)
    positive = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    signed = torch.cat((positive, -positive))
    expected = torch.cat((signed * 1.5, signed * 6)).view(1, 1, 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_FP8_byte_view_uses_single_global_scale_and_BF16_has_none() -> None:
    payload = torch.tensor([0x00, 0x38, 0x40, 0xB8], dtype=torch.uint8)
    torch.testing.assert_close(_decode_rows(payload, None, "fp8_e4m3", 0.25), torch.tensor([0, 0.25, 0.5, -0.25]))
    source = torch.tensor([1, -2], dtype=torch.bfloat16)
    torch.testing.assert_close(_decode_rows(source, None, "bf16", 123.0), source.float())


def test_NVFP4_incomplete_scale_blocks_are_rejected() -> None:
    with pytest.raises(ValueError, match="block16"):
        _decode_rows(torch.zeros(16, dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8), "nvfp4", 1.0)


def test_chunked_prefill_reference_uses_prefix_offset_and_GQA() -> None:
    # Zero logits yield uniform attention over the prefix plus current tokens.
    query = torch.zeros(1, 2, 2, 16)
    key = torch.zeros(1, 4, 1, 16)
    value = torch.arange(1, 5).view(1, 4, 1, 1).expand(1, 4, 1, 16).float()
    result = _reference_attention(query, key, value)
    expected = torch.stack((torch.full((32,), 2.0), torch.full((32,), 2.5)))
    torch.testing.assert_close(result, expected)
