# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU oracle for OCP MX v1.0 E2M1 / block32 / E8M0.

This module does not select a serving backend. Nibbles are low-element first;
scale bytes are separate. Conversion follows section 6.3: floor(log2(absmax))-2,
clamped to the E8M0 exponent range. Elements use roundTiesToEven and saturation.
An all-zero block uses scale byte 0 (2**-127), not a nonexistent zero scale.
Nonfinite source blocks produce the NaN scale 255; their element bytes are zero.
FP32 decode overflow produces signed infinity, an implementation-defined case
in section 5.1. Intermediate calculations use FP64 to retain E8M0 byte 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BLOCK_SIZE = 32
_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
_MIDPOINTS = (_E2M1[:-1] + _E2M1[1:]) / 2


def encode_e2m1(values: NDArray) -> NDArray[np.uint8]:
    """Encode scalar E2M1 with signed zero, ties-to-even, and SAT overflow.

    NaN scalar conversion is implementation-defined by OCP; use positive zero.
    Vector conversion instead records NaN on the entire block's scale.
    """
    source = np.asarray(values, dtype=np.float64)
    magnitude = np.abs(source)
    lower = np.searchsorted(_MIDPOINTS, magnitude, side="left")
    ties = magnitude == _MIDPOINTS[np.minimum(lower, 6)]
    code = lower + (ties & ((lower & 1) != 0))
    encoded = code.astype(np.uint8) | (np.signbit(source).astype(np.uint8) << 3)
    return np.where(np.isnan(source), 0, encoded).astype(np.uint8)


def decode_e2m1(codes: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Decode low four bits, preserving the negative-zero encoding."""
    encoded = np.asarray(codes, dtype=np.uint8)
    return np.copysign(_E2M1[encoded & 7], np.where(encoded & 8, -1.0, 1.0))


def encode_mxfp4(values: NDArray) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Return packed bytes and one E8M0 byte per 32 last-dimension values."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim == 0 or source.shape[-1] == 0 or source.shape[-1] % BLOCK_SIZE:
        raise ValueError("MXFP4 requires a nonempty final dimension divisible by 32")
    blocks = source.reshape(*source.shape[:-1], -1, BLOCK_SIZE)
    finite = np.isfinite(blocks).all(axis=-1)
    safe = np.where(finite[..., None], blocks, 0)
    maximum = np.abs(safe).max(axis=-1)
    exponent = np.frexp(maximum)[1] - 3
    exponent = np.where(maximum == 0, -127, np.clip(exponent, -127, 127))
    scale = np.ldexp(np.ones_like(maximum), exponent)
    codes = encode_e2m1(safe / scale[..., None]).reshape(source.shape)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    scales = np.where(finite, exponent + 127, 255).astype(np.uint8)
    return packed, scales


def decode_mxfp4(packed: NDArray[np.uint8], scales: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Decode the physical packed E2M1 and block32 E8M0 representation."""
    if packed.dtype != np.uint8 or scales.dtype != np.uint8:
        raise ValueError("MXFP4 packed data and scale storage must be uint8")
    if packed.ndim == 0 or packed.shape[-1] == 0 or packed.shape[-1] % 16:
        raise ValueError("MXFP4 packed rows must contain complete 16-byte blocks")
    if scales.shape != (*packed.shape[:-1], packed.shape[-1] // 16):
        raise ValueError("MXFP4 requires one scale byte per 16 packed bytes")
    codes = np.stack((packed & 15, packed >> 4), axis=-1)
    elements = decode_e2m1(codes).reshape(*scales.shape, BLOCK_SIZE)
    scale = np.ldexp(np.ones(scales.shape, dtype=np.float64), scales.astype(np.int32) - 127)
    scale = np.where(scales == 255, np.nan, scale)
    with np.errstate(over="ignore", invalid="ignore"):
        return (elements * scale[..., None]).reshape(*packed.shape[:-1], packed.shape[-1] * 2).astype(np.float32)
