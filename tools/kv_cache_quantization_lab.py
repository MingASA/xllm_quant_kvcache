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

"""Offline paged KV quantization lab, deliberately outside the serving path."""

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace

import torch

from tools.kv_cache_codec_lab import EncodedTensor, QuantSpec, encode


@dataclass(frozen=True)
class _HeadwiseEncoding:
    """Independent head payloads; no padded common bitwidth or hidden FP copy."""

    heads: tuple[EncodedTensor, ...]

    def decode(self) -> torch.Tensor:
        return torch.cat([head.decode() for head in self.heads], dim=1)

    def byte_breakdown(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for head in self.heads:
            for name, size in head.byte_breakdown().items():
                result[name] = result.get(name, 0) + size
        return result

    def nbytes(self) -> int:
        return sum(head.nbytes() for head in self.heads)


@dataclass(frozen=True)
class PageQuantPolicy:
    """Offline minimum-byte search with worst-head and worst-vector guards.

    Native must be an explicit candidate. Numerical/codec failures propagate;
    selection never hides them by retrying another format. Budgets describe
    reconstruction, not model quality or end-to-end attention error.
    """

    candidates: tuple[QuantSpec, ...]
    max_relative_l2: float = 0.1
    max_vector_error: float = 0.5
    headwise: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("Candidates must be a nonempty immutable tuple")
        if QuantSpec(format="native") not in self.candidates:
            raise ValueError("Include QuantSpec(format='native') explicitly")
        if any(not math.isfinite(x) or x < 0 for x in (self.max_relative_l2, self.max_vector_error)):
            raise ValueError("Error budgets must be finite and nonnegative")

    def select(self, source: torch.Tensor) -> EncodedTensor | _HeadwiseEncoding:
        if source.ndim != 3 or min(source.shape) <= 0:
            raise ValueError("Expected nonempty [T,H,D] source")
        if self.headwise:
            policy = replace(self, headwise=False)
            return _HeadwiseEncoding(
                tuple(policy.select(source[:, index : index + 1]) for index in range(source.shape[1]))
            )
        reference = source.detach().cpu().double()
        head_norms = reference.square().sum(dim=(0, 2)).sqrt()
        best: EncodedTensor | None = None
        for spec in self.candidates:
            encoded = encode(source, spec)
            if best is not None and encoded.nbytes() >= best.nbytes():
                continue
            error = encoded.decode().double() - reference
            relative = error.square().sum(dim=(0, 2)).sqrt() / head_norms.clamp_min(1e-30)
            vector_error = error.norm(dim=-1).amax()
            if relative.amax() <= self.max_relative_l2 and vector_error <= self.max_vector_error:
                best = encoded
        if best is None:
            raise ValueError("No candidate satisfies budgets; source may exceed FP32 decode precision")
        return best


@dataclass(frozen=True)
class _Page:
    key: EncodedTensor | _HeadwiseEncoding
    value: EncodedTensor | _HeadwiseEncoding
    token_coupled: bool = False


class QuantizedPageCache:
    """Sealed pages plus an unquantized tail; tensors are private and read-only.

    K/V shape is [tokens, kv_heads, head_dim]. Page-local parameters never
    change after publication. A fork shares sealed pages and copies its tail.
    This Python implementation models ownership, not allocator/kernel speed.
    """

    def __init__(
        self,
        key_spec: QuantSpec,
        value_spec: QuantSpec,
        page_size: int = 32,
        residual_tokens: int = 32,
        sink_tokens: int = 0,
        key_policy: PageQuantPolicy | None = None,
        value_policy: PageQuantPolicy | None = None,
    ) -> None:
        if page_size <= 0 or residual_tokens < 0 or sink_tokens < 0:
            raise ValueError("Invalid page size, residual length, or sink length")
        self._key_spec = key_spec
        self._value_spec = value_spec
        self._page_size = page_size
        self._residual_tokens = residual_tokens
        self._sink_tokens = sink_tokens
        self._key_policy = key_policy
        self._value_policy = value_policy
        self._pages: tuple[_Page, ...] = ()
        self._tail_key: torch.Tensor | None = None
        self._tail_value: torch.Tensor | None = None
        self._tokens = 0

    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def sealed_pages(self) -> int:
        return len(self._pages)

    @torch.no_grad()
    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Encode both sides before committing state; never mutate shared pages."""
        if key.ndim != 3 or key.shape != value.shape or min(key.shape) <= 0:
            raise ValueError("K and V must have identical nonempty [T,H,D] shapes")
        if not key.is_floating_point() or not value.is_floating_point():
            raise ValueError("K and V must be floating point")
        if key.device != value.device or key.dtype != value.dtype:
            raise ValueError("K and V must have the same device and dtype")
        if key.device.type != "cpu":
            raise ValueError("This offline cache is CPU-only; device kernels are not implemented")
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("Nonfinite K/V cannot be cached")
        if self._tail_key is not None:
            if (
                key.shape[1:] != self._tail_key.shape[1:]
                or key.dtype != self._tail_key.dtype
                or key.device != self._tail_key.device
            ):
                raise ValueError("Cache shape, dtype, and device cannot change")
            pending_key = torch.cat((self._tail_key, key), dim=0)
            pending_value = torch.cat((self._tail_value, value), dim=0)
        else:
            pending_key, pending_value = key.detach(), value.detach()

        count = max(0, (len(pending_key) - self._residual_tokens) // self._page_size)
        new_pages: list[_Page] = []
        for page_index in range(count):
            start = page_index * self._page_size
            stop = start + self._page_size
            absolute_start = (len(self._pages) + page_index) * self._page_size
            preserve_sink = absolute_start < self._sink_tokens
            key_spec = QuantSpec(format="native") if preserve_sink else self._key_spec
            value_spec = QuantSpec(format="native") if preserve_sink else self._value_spec
            new_pages.append(
                _Page(
                    self._key_policy.select(pending_key[start:stop])
                    if self._key_policy is not None and not preserve_sink
                    else encode(pending_key[start:stop], key_spec),
                    self._value_policy.select(pending_value[start:stop])
                    if self._value_policy is not None and not preserve_sink
                    else encode(pending_value[start:stop], value_spec),
                    not preserve_sink
                    and (
                        self._key_policy is not None
                        or self._value_policy is not None
                        or _couples_tokens(key_spec)
                        or _couples_tokens(value_spec)
                    ),
                )
            )

        # Clone, not just contiguous(): a contiguous slice can retain a complete
        # uncompressed prefill buffer and silently erase the memory benefit.
        consumed = count * self._page_size
        tail_key = pending_key[consumed:].clone()
        tail_value = pending_value[consumed:].clone()
        self._pages = self._pages + tuple(new_pages)
        self._tail_key, self._tail_value = tail_key, tail_value
        self._tokens += len(key)

    def fork(self) -> "QuantizedPageCache":
        """Share the immutable prefix; a branch owns its mutable residual."""
        other = QuantizedPageCache(
            self._key_spec,
            self._value_spec,
            self._page_size,
            self._residual_tokens,
            self._sink_tokens,
            self._key_policy,
            self._value_policy,
        )
        other._pages = self._pages
        other._tokens = self._tokens
        if self._tail_key is not None:
            other._tail_key = self._tail_key.clone()
            other._tail_value = self._tail_value.clone()
        return other

    def byte_breakdown(self) -> dict[str, int]:
        """Resident tensor bytes for one branch, excluding allocator/Python overhead.

        Shared pages are counted once within this branch, but would be double
        counted if reports for several forks were simply added together.
        """
        result: dict[str, int] = {}
        for page in self._pages:
            for prefix, encoded in (("key", page.key), ("value", page.value)):
                for name, size in encoded.byte_breakdown().items():
                    field = f"{prefix}_{name}"
                    result[field] = result.get(field, 0) + size
        result["residual"] = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self._tail_key, self._tail_value)
            if tensor is not None
        )
        return result

    def nbytes(self) -> int:
        return sum(self.byte_breakdown().values())

    def page_report(self) -> list[dict]:
        """Return copied format descriptors, never mutable payload references."""
        return [
            {
                "page": index,
                "key": _describe_encoding(page.key),
                "value": _describe_encoding(page.value),
                "bytes": page.key.nbytes() + page.value.nbytes(),
            }
            for index, page in enumerate(self._pages)
        ]

    def _decoded_pages(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for page in self._pages:
            yield page.key.decode(), page.value.decode()
        if self._tail_key is not None and len(self._tail_key):
            yield self._tail_key.float(), self._tail_value.float()

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """For offline error measurement only; allocates a full FP32 context."""
        if not self._tokens:
            raise ValueError("Cannot materialize an empty cache")
        keys, values = zip(*self._decoded_pages())
        return torch.cat(keys), torch.cat(values)

    @torch.no_grad()
    def attend(self, query: torch.Tensor, query_tile: int = 64, allow_retrospective: bool = False) -> torch.Tensor:
        """Pagewise online softmax with right-aligned causal masking and GQA.

        Query/output: [query_tokens, query_heads, head_dim]; output is FP32.
        Rotations are inverted by each codec before attention. This preserves
        ordinary Q/K semantics at extra cost, not vLLM's fused rotated layout.
        """
        if self._tail_key is None or not self._tokens:
            raise ValueError("Cannot attend to an empty cache")
        if query.ndim != 3 or not 0 < len(query) <= self._tokens or query_tile <= 0:
            raise ValueError("Expected nonempty [Q,H,D], Q <= cached tokens")
        first_position = self._tokens - len(query)
        if not allow_retrospective and any(
            page.token_coupled and (index + 1) * self._page_size - 1 > first_position
            for index, page in enumerate(self._pages)
        ):
            raise ValueError(
                "Page calibration includes tokens later than the first query; "
                "retain the query chunk in the unquantized tail, or explicitly "
                "select allow_retrospective=True for offline reconstruction only"
            )
        heads, dim = self._tail_key.shape[1:]
        if (
            query.shape[1] <= 0
            or query.shape[1] % heads
            or query.shape[2] != dim
            or query.device != self._tail_key.device
            or not query.is_floating_point()
            or not torch.isfinite(query).all()
        ):
            raise ValueError("Query shape/device/type incompatible with the cache")
        outputs: list[torch.Tensor] = []
        for start in range(0, len(query), query_tile):
            q = query[start : start + query_tile].float().transpose(0, 1)
            positions = torch.arange(
                self._tokens - len(query) + start,
                self._tokens - len(query) + start + q.shape[1],
                device=q.device,
            )
            maximum = torch.full(q.shape[:2], -math.inf, device=q.device)
            denominator = torch.zeros_like(maximum)
            accumulator = torch.zeros_like(q)
            offset = 0
            for key, value in self._decoded_pages():
                key = key.repeat_interleave(q.shape[0] // heads, dim=1).transpose(0, 1)
                value = value.repeat_interleave(q.shape[0] // heads, dim=1).transpose(0, 1)
                scores = (q @ key.transpose(-2, -1)) / math.sqrt(dim)
                slots = torch.arange(offset, offset + key.shape[1], device=q.device)
                scores.masked_fill_(slots[None, :] > positions[:, None], -math.inf)
                next_maximum = torch.maximum(maximum, scores.amax(dim=-1))
                correction = (maximum - next_maximum).exp()
                probabilities = (scores - next_maximum[..., None]).exp()
                accumulator = accumulator * correction[..., None] + probabilities @ value
                denominator = denominator * correction + probabilities.sum(dim=-1)
                maximum = next_maximum
                offset += key.shape[1]
            outputs.append((accumulator / denominator[..., None]).transpose(0, 1))
        return torch.cat(outputs)


def _describe_encoding(encoded: EncodedTensor | _HeadwiseEncoding) -> dict:
    if isinstance(encoded, _HeadwiseEncoding):
        return {"granularity": "head", "heads": [asdict(head.spec) for head in encoded.heads]}
    return asdict(encoded.spec)


def _couples_tokens(spec: QuantSpec) -> bool:
    return spec.format != "native" and bool(
        spec.axis == "channel" or spec.center or spec.residual_rank or spec.error_fraction
    )
