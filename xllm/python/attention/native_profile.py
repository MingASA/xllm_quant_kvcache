# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""Explicit native production profile selection, without runtime fallback."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class NativeFlashInferConfig:
    """Explicit paged-backend selection independent of research environment flags."""

    planner: str = "fa2"
    decode_impl: str = "flashinfer"
    max_context_tokens: int = 32768

    @classmethod
    def from_runtime_config(
        cls, config: Mapping[str, object], device: torch.device, kv_dtype: torch.dtype
    ) -> NativeFlashInferConfig:
        """Resolve typed service flags before constructing any kernel or cache."""
        planner = config.get("kv_cache_prefill_backend", "auto")
        decode = config.get("kv_cache_decode_backend", "auto")
        if planner not in ("auto", "fa2", "fa3") or decode not in ("auto", "flashinfer", "xqa"):
            raise ValueError("Invalid configured native prefill/decode backend")
        if planner == "auto":
            unquantized = kv_dtype in (torch.bfloat16, torch.float16)
            planner = "fa3" if unquantized and torch.cuda.get_device_capability(device) == (9, 0) else "fa2"
        result = cls(
            planner=str(planner),
            decode_impl="flashinfer" if decode == "auto" else str(decode),
            max_context_tokens=int(config.get("max_position_embeddings", 32768)),
        )
        result.resolve(kv_dtype)
        return result

    def resolve(self, kv_dtype: torch.dtype) -> dict[str, object]:
        if self.planner not in ("fa2", "fa3") or self.decode_impl not in ("flashinfer", "xqa"):
            raise ValueError("Native FlashInfer requires an explicit supported prefill/decode backend")
        if type(self.max_context_tokens) is not int or not 1 <= self.max_context_tokens < 2**31:
            raise ValueError("Native context capacity must be a positive int32 token count")
        if kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8) and self.planner != "fa2":
            raise ValueError("Native low-precision KV with BF16/FP16 queries requires FA2 prefill")
        if kv_dtype == torch.uint8 and self.decode_impl != "flashinfer":
            raise ValueError("SM90 NVFP4 requires the FA2 paged decode backend")
        return {
            "planner": self.planner,
            "decode_impl": self.decode_impl,
            "query_fp8": False,
            "query_scale": 1.0,
            "scope": "native_pages_constructor_config",
        }


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
    if decode == "xqa":
        if query_fp8:
            raise ValueError("XQA requires BF16 queries without external Q quantization")
        if planner != "fa2" and (fp8 or planner != "fa3"):
            raise ValueError("XQA requires FA2 prefill for FP8 KV, or FA2/FA3 prefill for BF16 KV")
    if query_fp8 and (not fp8 or planner != "fa3"):
        raise ValueError("FP8 query is supported only in explicit native FA3 with resident FP8 KV")
    if fp8 and planner == "fa3" and not query_fp8:
        raise ValueError("Installed FA3 FP8 KV requires FP8 queries; BF16-Q is unsupported")
    query_scale = float(environment.get("XLLM_FLASHINFER_FP8_Q_SCALE", "1.0"))
    if not math.isfinite(query_scale) or not 2**-126 <= query_scale <= (2 - 2**-23) * 2**127:
        raise ValueError("FP8 query scale must be a positive normal FP32 scalar")
    if "XLLM_FLASHINFER_FP8_Q_SCALE" in environment and not query_fp8:
        raise ValueError("Query scale requires explicit FP8 query quantization")
    return {
        "planner": planner,
        "query_fp8": query_fp8,
        "query_scale": query_scale,
        "decode_impl": decode,
        "scope": "production_xqa"
        if decode == "xqa"
        else "production_fa3"
        if planner == "fa3"
        else "historical_fa2"
        if native
        else "default",
    }
