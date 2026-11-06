# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Bit-level OCP tables and independent edge cases; no GPU dependency."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[2] / "xllm/python/attention/mxfp4_reference.py"
_SPEC = importlib.util.spec_from_file_location("mxfp4_reference", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)


def test_all_nibbles_and_signed_zero() -> None:
    expected = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    codes = np.arange(16, dtype=np.uint8)
    np.testing.assert_array_equal(reference.decode_e2m1(codes), expected)
    np.testing.assert_array_equal(reference.encode_e2m1(expected), codes)
    assert np.signbit(reference.decode_e2m1(codes)[8])


def test_rounding_midpoints_and_overflow() -> None:
    values = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5, np.inf])
    expected = np.array([0, 2, 2, 4, 4, 6, 6, 7], dtype=np.uint8)
    np.testing.assert_array_equal(reference.encode_e2m1(values), expected)
    np.testing.assert_array_equal(reference.encode_e2m1(-values), expected | 8)


def test_known_bytes_scale_and_block_boundary() -> None:
    values = np.tile(np.array([0, .5, 1, 1.5, 2, 3, 4, 6]), 8).reshape(2, 32)
    values[1] *= 8
    packed, scales = reference.encode_mxfp4(values)
    np.testing.assert_array_equal(packed, np.tile([0x10, 0x32, 0x54, 0x76], (2, 4)))
    np.testing.assert_array_equal(scales, [[127], [130]])
    np.testing.assert_array_equal(reference.decode_mxfp4(packed, scales), values)


def test_e8m0_extremes_nan_and_zero() -> None:
    packed = np.zeros((4, 16), dtype=np.uint8)
    packed[:, 0] = 0x21  # 0.5, 1.0
    result = reference.decode_mxfp4(packed, np.array([[0], [127], [254], [255]], dtype=np.uint8))
    assert result[0, 1] == np.float32(2.0 ** -127)
    assert result[1, 1] == 1
    assert result[2, 1] == np.float32(2.0 ** 127)
    assert np.isnan(result[3]).all()
    values = np.zeros((2, 32))
    values[1, 0] = np.nan
    zero, sf = reference.encode_mxfp4(values)
    np.testing.assert_array_equal(sf, [[0], [255]])
    assert not zero.any()


def test_minimum_scale_saturation_and_shape_rejection() -> None:
    values = np.full((1, 32), 2.0 ** -140)
    packed, sf = reference.encode_mxfp4(values)
    assert sf.item() == 0 and not packed.any()
    with pytest.raises(ValueError):
        reference.encode_mxfp4(np.zeros((2, 31)))
    with pytest.raises(ValueError):
        reference.decode_mxfp4(np.zeros((1, 16), dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8))
