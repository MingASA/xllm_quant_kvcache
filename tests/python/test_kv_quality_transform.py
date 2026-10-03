# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Quality-only KV transforms must match independent codec references."""

from __future__ import annotations

import pytest
import torch

from tools.kv_cache_codec_lab import QuantSpec, encode
from xllm.python.attention.kv_quality_transform import KVQualityTransform
from xllm.python.attention.quantized import KVCacheCodec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("mode", ("v_only_int4", "int4_rht_g32"))
@pytest.mark.parametrize("tokens", (1, 17, 257))
def test_quality_transform_matches_codec(mode: str, tokens: int) -> None:
    torch.manual_seed(701)
    key = torch.randn(tokens, 2, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key[..., 7] *= 128
    # Split projection views are not necessarily contiguous.
    source = torch.stack((key, value), dim=-2)
    key, value = source[..., 0, :], source[..., 1, :]
    transform = KVQualityTransform(mode, 128, torch.device("cuda"), torch.bfloat16)
    actual_key, actual_value = transform.apply(key, value)
    assert actual_value.dtype == torch.bfloat16
    assert actual_value.device == value.device
    if mode == "v_only_int4":
        codec = KVCacheCodec("int4", 128)
        expected_value = codec.decode(*codec.encode(value)).to(value.dtype)
        assert actual_key is key
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    else:
        spec = QuantSpec(format="int4", group_size=32, rotation=True, seed=17)
        expected_key = encode(key.cpu(), spec).decode().to(device=key.device, dtype=key.dtype)
        expected_value = encode(value.cpu(), spec).decode().to(device=value.device, dtype=value.dtype)
        torch.testing.assert_close(actual_key, expected_key, rtol=0.01, atol=0.01)
        torch.testing.assert_close(actual_value, expected_value, rtol=0.01, atol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_v_only_writer_uses_exact_division_at_half_integer_boundary() -> None:
    values = torch.tensor([-318.0, -159.0, 0.0, 1.0] * 32, dtype=torch.bfloat16, device="cuda").reshape(1, 1, 128)
    transform = KVQualityTransform("v_only_int4", 128, torch.device("cuda"), torch.bfloat16)
    key, actual = transform.apply(values, values)
    codec = KVCacheCodec("int4", 128)
    assert key is values
    torch.testing.assert_close(actual, codec.decode(*codec.encode(values)).to(values.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ("unknown", "int4", ""))
def test_unknown_quality_mode_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        KVQualityTransform(mode, 128, torch.device("cpu"), torch.bfloat16)


def test_quality_mode_requires_cuda_bf16() -> None:
    with pytest.raises(ValueError, match="CUDA and BF16"):
        KVQualityTransform("v_only_int4", 128, torch.device("cpu"), torch.bfloat16)
