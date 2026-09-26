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

"""CPU reference codecs for KV-cache experiments, not serving kernels.

The public layout is [tokens, heads, channels]. Payloads really are packed;
byte accounting includes tensor buffers, including group padding and outliers,
but excludes Python objects, shape metadata and allocator alignment. Decoding
materializes FP32, so storage savings here do not imply faster attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as functional

NF4_VALUES = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FORMATS = {"native", "fp8_e4m3", "fp8_e5m2", "nf4", "fp4_e2m1"} | {f"int{bits}" for bits in range(2, 9)}


@dataclass(frozen=True)
class QuantSpec:
    """Quantization policy; channel groups run over tokens within each page."""

    format: str = "int4"
    group_size: int = 32
    axis: str = "token"
    affine: bool = False
    rotation: bool = False
    outlier_fraction: float = 0.0
    scale_dtype: str = "float32"
    seed: int = 0
    center: bool = False
    residual_rank: int = 0
    error_fraction: float = 0.0
    residual_dtype: str = "float32"


@dataclass
class EncodedTensor:
    """Self-contained encoded tensor with groupwise scale and sparse sidecar."""

    spec: QuantSpec
    shape: tuple[int, int, int]
    padded_dim: int
    payload: torch.Tensor
    scales: torch.Tensor
    offsets: torch.Tensor
    outlier_indices: torch.Tensor
    outlier_values: torch.Tensor
    center: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    residual_left: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    residual_right: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    error_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    error_values: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    def byte_breakdown(self) -> dict[str, int]:
        """Return actual retained tensor bytes, excluding Python metadata."""
        buffers = {
            "payload": self.payload,
            "scales": self.scales,
            "offsets": self.offsets,
            "outlier_indices": self.outlier_indices,
            "outlier_values": self.outlier_values,
            "center": self.center,
            "residual_left": self.residual_left,
            "residual_right": self.residual_right,
            "error_indices": self.error_indices,
            "error_values": self.error_values,
        }
        return {name: value.numel() * value.element_size() for name, value in buffers.items()}

    def nbytes(self) -> int:
        return sum(self.byte_breakdown().values())

    def decode(self) -> torch.Tensor:
        """Reconstruct the original [T,H,D] layout in FP32 on CPU."""
        if self.spec.format == "native":
            return self.payload.float().clone()
        token_count, head_count, _ = self.shape
        width = self.padded_dim if self.spec.axis == "token" else token_count
        rows = token_count * head_count if self.spec.axis == "token" else head_count * self.padded_dim
        groups_per_row = math.ceil(width / self.spec.group_size)
        group_count = rows * groups_per_row
        count = group_count * self.spec.group_size
        if self.spec.format.startswith("fp8"):
            dtype = _fp8_dtype(self.spec.format)
            normalized = self.payload.view(dtype).float()
        else:
            bits = _bits(self.spec.format)
            codes = _unpack(self.payload, bits, count)
            if self.spec.format.startswith("int"):
                normalized = codes.float()
                if not self.spec.affine:
                    normalized -= 1 << (bits - 1)
            else:
                normalized = _codebook(self.spec.format)[codes]
        groups = normalized.reshape(group_count, self.spec.group_size)
        groups = groups * _decode_scales(self.scales, self.spec.scale_dtype)
        if self.spec.affine:
            groups += self.offsets
        groups.flatten()[self.outlier_indices.long()] = self.outlier_values
        values = groups.reshape(rows, -1)[:, :width]
        if self.spec.axis == "token":
            values = values.reshape(token_count, head_count, self.padded_dim)
        else:
            values = values.reshape(head_count, self.padded_dim, token_count)
            values = values.permute(2, 0, 1).contiguous()
        if self.spec.rotation:
            values = _hadamard(values) * _signs(self.padded_dim, self.spec.seed)
        values = values[..., : self.shape[2]].contiguous()
        if self.center.numel():
            values = values + self.center
        if self.residual_left.numel():
            values = values + (self.residual_left.float() @ self.residual_right.float()).permute(1, 0, 2)
        values = values.contiguous()
        values.view(-1)[self.error_indices.long()] += self.error_values
        return values


def _validate(tensor: torch.Tensor, spec: QuantSpec) -> None:
    if tensor.ndim != 3 or any(size == 0 for size in tensor.shape):
        raise ValueError("Expected non-empty [tokens, heads, channels] tensor")
    if not tensor.is_floating_point():
        raise ValueError("KV input must have a floating-point dtype")
    if spec.format not in FORMATS:
        raise ValueError(f"Unknown quantization format: {spec.format}")
    if spec.axis not in {"token", "channel"} or spec.group_size <= 0:
        raise ValueError("axis must be token/channel and group_size must be positive")
    if spec.scale_dtype not in {"float32", "float16", "pow2"}:
        raise ValueError("scale_dtype must be float32, float16 or pow2")
    if not 0.0 <= spec.outlier_fraction < 1.0:
        raise ValueError("outlier_fraction must be in [0, 1)")
    if spec.affine and not spec.format.startswith("int"):
        raise ValueError("Affine quantization is supported only for integers")
    if not isinstance(spec.residual_rank, int) or not 0 <= spec.residual_rank <= min(tensor.shape[0], tensor.shape[2]):
        raise ValueError("residual_rank must be an integer between 0 and min(T,D)")
    if not 0 <= spec.error_fraction < 1:
        raise ValueError("error_fraction must be in [0,1)")
    if spec.residual_dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("residual_dtype must be float32, float16 or bfloat16")
    if spec.format == "native" and (
        spec.rotation or spec.outlier_fraction or spec.center or spec.residual_rank or spec.error_fraction
    ):
        raise ValueError("native stores an exact clone without transforms")


def _bits(format_name: str) -> int:
    return int(format_name[3:]) if format_name.startswith("int") else 4


def _fp8_dtype(format_name: str) -> torch.dtype:
    return torch.float8_e4m3fn if format_name == "fp8_e4m3" else torch.float8_e5m2


def _codebook(format_name: str) -> torch.Tensor:
    values = NF4_VALUES if format_name == "nf4" else FP4_VALUES + tuple(-value for value in FP4_VALUES)
    return torch.tensor(values, dtype=torch.float32)


def _signs(width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 2, (width,), generator=generator).float() * 2 - 1


def _hadamard(values: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal Walsh-Hadamard transform along the final axis."""
    width = values.shape[-1]
    result = values.clone()
    stride = 1
    while stride < width:
        blocks = result.reshape(*result.shape[:-1], -1, 2, stride)
        left, right = blocks[..., 0, :].clone(), blocks[..., 1, :].clone()
        blocks[..., 0, :] = left + right
        blocks[..., 1, :] = left - right
        stride *= 2
    return result / math.sqrt(width)


def _pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned codes LSB-first, including codes crossing byte boundaries."""
    codes = codes.flatten().long()
    offsets = torch.arange(codes.numel(), dtype=torch.int64) * bits
    packed = torch.zeros(math.ceil(codes.numel() * bits / 8), dtype=torch.int64)
    for bit in range(bits):
        positions = offsets + bit
        packed.scatter_add_(0, positions // 8, ((codes >> bit) & 1) << (positions % 8))
    return packed.to(torch.uint8)


def _unpack(payload: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    offsets = torch.arange(count, dtype=torch.int64) * bits
    codes = torch.zeros(count, dtype=torch.int64)
    for bit in range(bits):
        positions = offsets + bit
        codes |= ((payload[positions // 8].long() >> (positions % 8)) & 1) << bit
    return codes


def _encode_scales(scales: torch.Tensor, scale_dtype: str) -> torch.Tensor:
    scales = scales.clamp_min(torch.finfo(torch.float32).tiny)
    if scale_dtype == "pow2":
        exponents = torch.ceil(torch.log2(scales))
        if torch.any(exponents > 127):
            raise ValueError("Scale exceeds pow2 exponent range; choose float32")
        return exponents.clamp_min(-126).to(torch.int8)
    if scale_dtype == "float16":
        if torch.any(scales > torch.finfo(torch.float16).max):
            raise ValueError("Scale exceeds FP16 range; choose float32 or pow2")
        # Preserve subnormal scales, but never store a zero divisor.
        return scales.clamp_min(2.0**-24).to(torch.float16)
    return scales


def _decode_scales(scales: torch.Tensor, scale_dtype: str) -> torch.Tensor:
    return torch.exp2(scales.float()) if scale_dtype == "pow2" else scales.float()


def _extract_outliers(groups: torch.Tensor, valid: torch.Tensor, fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if fraction == 0.0:
        return torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.float32)
    if groups.numel() >= 2**31:
        raise ValueError("Outlier sidecar uses int32 indices; reduce page size")
    counts = torch.ceil(valid.sum(dim=-1).float() * fraction).long()
    order = groups.abs().masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True)
    selected = torch.arange(groups.shape[-1]).unsqueeze(0) < counts.unsqueeze(-1)
    indices = (order + torch.arange(groups.shape[0]).unsqueeze(-1) * groups.shape[-1])[selected]
    values = groups.flatten()[indices].clone()
    groups.flatten()[indices] = 0.0
    return indices.to(torch.int32), values


def encode(tensor: torch.Tensor, spec: QuantSpec) -> EncodedTensor:
    """Encode on CPU; quantize with the *stored* scale to avoid scale mismatch.

    Group padding is masked out of calibration. Channel grouping is suitable
    for sealed pages: appending tokens to an existing group requires re-encoding.
    Outliers are selected in the rotated domain when rotation is enabled.
    """
    _validate(tensor, spec)
    source = tensor.detach().cpu()
    if not torch.isfinite(source.float()).all():
        raise ValueError("KV input must contain only finite FP32-representable values")
    empty = torch.empty(0, dtype=torch.float32)
    if spec.format == "native":
        return EncodedTensor(
            spec,
            tuple(tensor.shape),
            tensor.shape[2],
            source.clone().contiguous(),
            empty,
            empty,
            torch.empty(0, dtype=torch.int32),
            empty,
        )
    values = source.float().clone()
    center = values.double().mean(dim=0, keepdim=True).float() if spec.center else empty
    if spec.center:
        values -= center
    padded_dim = values.shape[-1]
    if spec.rotation:
        padded_dim = 1 << (padded_dim - 1).bit_length()
        values = functional.pad(values, (0, padded_dim - values.shape[-1]))
        values = _hadamard(values * _signs(padded_dim, spec.seed))
    if not torch.isfinite(values).all():
        raise ValueError("Centering or rotation exceeded FP32 range")
    rows = values if spec.axis == "token" else values.permute(1, 2, 0)
    rows = rows.reshape(-1, rows.shape[-1])
    width = rows.shape[-1]
    padded_width = math.ceil(width / spec.group_size) * spec.group_size
    valid = torch.arange(padded_width).unsqueeze(0) < width
    valid = valid.expand(rows.shape[0], -1).reshape(-1, spec.group_size)
    groups = functional.pad(rows, (0, padded_width - width)).reshape(-1, spec.group_size)
    outlier_indices, outlier_values = _extract_outliers(groups, valid, spec.outlier_fraction)
    offsets = empty
    if spec.affine:
        offsets = groups.masked_fill(~valid, torch.inf).amin(dim=-1, keepdim=True)
        maximum = groups.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
        raw_scales = (maximum - offsets) / ((1 << _bits(spec.format)) - 1)
    else:
        peak = groups.abs().masked_fill(~valid, 0).amax(dim=-1, keepdim=True)
        if spec.format.startswith("int"):
            divisor = (1 << (_bits(spec.format) - 1)) - 1
        elif spec.format.startswith("fp8"):
            divisor = torch.finfo(_fp8_dtype(spec.format)).max
        else:
            divisor = 1.0 if spec.format == "nf4" else 6.0
        raw_scales = peak / divisor
    if not torch.isfinite(raw_scales).all():
        raise ValueError("Quantization scale exceeded FP32 range")
    raw_scales = torch.where(raw_scales == 0, torch.ones_like(raw_scales), raw_scales)
    scales = _encode_scales(raw_scales, spec.scale_dtype)
    normalized = (groups - offsets if spec.affine else groups) / _decode_scales(scales, spec.scale_dtype)
    if spec.format.startswith("fp8"):
        dtype = _fp8_dtype(spec.format)
        limit = torch.finfo(dtype).max
        payload = normalized.clamp(-limit, limit).to(dtype).view(torch.uint8).flatten()
    elif spec.format.startswith("int"):
        bits = _bits(spec.format)
        if spec.affine:
            codes = normalized.round().clamp(0, (1 << bits) - 1)
        else:
            shift = 1 << (bits - 1)
            codes = normalized.round().clamp(-shift, shift - 1) + shift
        payload = _pack(codes, bits)
    else:
        codebook = _codebook(spec.format)
        codes = (normalized.unsqueeze(-1) - codebook).abs().argmin(dim=-1)
        payload = _pack(codes, 4)
    encoded = EncodedTensor(
        spec,
        tuple(tensor.shape),
        padded_dim,
        payload,
        scales,
        offsets,
        outlier_indices,
        outlier_values,
        center=center,
    )
    if spec.residual_rank or spec.error_fraction:
        _correct_residual(encoded, source.float())
    return encoded


def _correct_residual(encoded: EncodedTensor, reference: torch.Tensor) -> None:
    """Fit corrections after inverse rotation/centering, in original coordinates.

    SVD is an offline oracle, not a serving implementation. Clone the truncated
    factors: retaining SVD views would keep full-size decompositions alive.
    """
    residual = reference - encoded.decode()
    if not torch.isfinite(residual).all():
        raise ValueError("Residual correction exceeded FP32 range")
    rank = encoded.spec.residual_rank
    if rank:
        left, singular, right = torch.linalg.svd(residual.permute(1, 0, 2), full_matrices=False)
        if encoded.spec.residual_dtype == "float32":
            encoded.residual_left = (left[..., :rank] * singular[:, None, :rank]).clone().contiguous()
            encoded.residual_right = right[:, :rank, :].clone().contiguous()
        else:
            # Balance the singular-value gain across factors before casting,
            # rather than placing the full dynamic range in the left factor.
            gain = singular[:, :rank].sqrt()
            dtype = torch.float16 if encoded.spec.residual_dtype == "float16" else torch.bfloat16
            encoded.residual_left = (left[..., :rank] * gain[:, None, :]).to(dtype).contiguous()
            encoded.residual_right = (right[:, :rank, :] * gain[:, :, None]).to(dtype).contiguous()
            if not torch.isfinite(encoded.residual_left).all() or not torch.isfinite(encoded.residual_right).all():
                raise ValueError("Residual factors overflowed; select a wider residual_dtype")
        residual = reference - encoded.decode()
        if not torch.isfinite(residual).all():
            raise ValueError("Low-rank reconstruction exceeded FP32 range")
    count = math.ceil(reference.numel() * encoded.spec.error_fraction)
    if count:
        if reference.numel() >= 2**31:
            raise ValueError("Sparse residual indices require fewer than 2**31 elements")
        # Stable order resolves equal errors reproducibly; do not select by |x|.
        indices = residual.flatten().abs().argsort(descending=True, stable=True)[:count]
        encoded.error_indices = indices.to(torch.int32)
        encoded.error_values = residual.flatten()[indices].clone()
