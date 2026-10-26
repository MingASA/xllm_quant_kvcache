# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""Explicit native production profile selection, without runtime fallback."""

from __future__ import annotations

import math
from collections.abc import Mapping


def resolve_native_profile(environment: Mapping[str, str], native: bool, fp8: bool) -> dict[str, object]:
    """Keep the historical FA2 profile while allowing an explicit FP8-Q FA3 path."""
    planner = environment.get("XLLM_FLASHINFER_NATIVE_PLANNER", "fa2" if native else "auto")
    query_flag = environment.get("XLLM_FLASHINFER_FP8_QUERY", "0")
    decode = environment.get("XLLM_FLASHINFER_NATIVE_DECODE", "flashinfer")
    if decode not in ("flashinfer", "xqa"):
        raise ValueError("Invalid explicit native decode selection")
    if query_flag not in ("0", "1") or planner not in ("auto", "fa2", "fa3"):
        raise ValueError("Invalid native planner/query selection")
    explicit = "XLLM_FLASHINFER_NATIVE_PLANNER" in environment or query_flag == "1" or decode == "xqa"
    if explicit and not native:
        raise ValueError("Explicit production profile requires the native paged KV opt-in")
    if native and planner == "auto":
        raise ValueError("Native production plans must name their actual planner, without silent fallback")
    query_fp8 = query_flag == "1"
    if decode == "xqa" and (planner != "fa2" or query_fp8):
        raise ValueError("Production XQA requires explicit FA2 BF16-Q prefill without external Q quantization")
    if query_fp8 and (not fp8 or planner != "fa3"):
        raise ValueError("FP8 query is supported only in explicit native FA3 with resident FP8 KV")
    if fp8 and planner == "fa3" and not query_fp8:
        raise ValueError("Installed FA3 FP8 KV requires FP8 queries; BF16-Q is unsupported")
    query_scale = float(environment.get("XLLM_FLASHINFER_FP8_Q_SCALE", "1.0"))
    if not math.isfinite(query_scale) or not 2**-126 <= query_scale <= (2 - 2**-23) * 2**127:
        raise ValueError("FP8 query scale must be a positive normal FP32 scalar")
    if "XLLM_FLASHINFER_FP8_Q_SCALE" in environment and not query_fp8:
        raise ValueError("Query scale requires explicit FP8 query quantization")
    return {"planner": planner, "query_fp8": query_fp8, "query_scale": query_scale, "decode_impl": decode,
            "scope": "production_xqa" if decode == "xqa" else "production_fa3" if planner == "fa3"
            else "historical_fa2" if native else "default"}
