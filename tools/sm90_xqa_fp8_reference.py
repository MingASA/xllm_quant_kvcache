# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU model of the additional SM90 XQA FP8 Q/P approximations.

The installed FlashInfer mha_sm90.cu converts BF16 Q into CacheElem(E4M3),
computes each unquantized exponential sum, and stores 448*exp(logit-max) in
E4M3 before the second GEMM. Each split walks interleaved 64-token tiles and
merges online softmax states. This models those approximations in FP32; it is
not a bitwise GMMA emulator and still requires actual GPU correlation.

Source: flashinfer-ai/flashinfer v0.6.18 csrc/xqa/mha_sm90.cu, together with the
pinned installed0.6.18.post1 source snapshot in the research control repository.
"""

from __future__ import annotations

import torch


def reference_decode(
    query: torch.Tensor,
    decoded_key: torch.Tensor,
    decoded_value: torch.Tensor,
    *,
    split_count: int,
    tile_tokens: int = 64,
) -> torch.Tensor:
    """Return [batch, heads*dim] for one query using CPU decoded KV values."""
    if (any(t.device.type != "cpu" for t in (query, decoded_key, decoded_value))
            or query.ndim != 4 or query.shape[1] != 1
            or decoded_key.ndim != 4 or decoded_key.shape != decoded_value.shape
            or query.shape[0] != decoded_key.shape[0] or query.shape[-1] != decoded_key.shape[-1]
            or query.shape[2] % decoded_key.shape[2] or min(split_count, tile_tokens) <= 0):
        raise ValueError("SM90 FP8 reference requires CPU GQA decode rows and a positive explicit split/tile count")
    batch, _, heads, dimension = query.shape
    groups = heads // decoded_key.shape[2]
    q = query.to(torch.float8_e4m3fn).float().transpose(1, 2)
    key = decoded_key.float().repeat_interleave(groups, dim=2).transpose(1, 2)
    value = decoded_value.float().repeat_interleave(groups, dim=2).transpose(1, 2)
    context = key.shape[2]
    states = []
    for split in range(split_count):
        maximum = torch.full((batch, heads, 1, 1), -torch.inf)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros((batch, heads, 1, dimension))
        for start in range(split * tile_tokens, context, split_count * tile_tokens):
            scores = (q @ key[:, :, start:start + tile_tokens].transpose(-1, -2)) * dimension**-0.5
            updated = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
            carry = torch.exp(maximum - updated)
            exponentials = torch.exp(scores - updated)
            probabilities_fp8 = (exponentials * 448.0).to(torch.float8_e4m3fn).float() / 448.0
            numerator = numerator * carry + probabilities_fp8 @ value[:, :, start:start + tile_tokens]
            denominator = denominator * carry + exponentials.sum(dim=-1, keepdim=True)
            maximum = updated
        states.append((maximum, denominator, numerator))
    final_maximum = torch.stack([state[0] for state in states]).amax(dim=0)
    total_denominator = torch.zeros_like(final_maximum)
    total_numerator = torch.zeros((batch, heads, 1, dimension))
    for maximum, denominator, numerator in states:
        carry = torch.exp(maximum - final_maximum)
        total_denominator += denominator * carry
        total_numerator += numerator * carry
    return (total_numerator / total_denominator).transpose(1, 2).reshape(batch, heads * dimension)
