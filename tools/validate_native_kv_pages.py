# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Qualify small native paged KV operators on an externally admitted SM90 GPU.

This checks implementation correctness, not model quality or serving speed.
Decompressed references exist on the CPU only and are never timed. The caller
must provide shared-server resource admission and a container watchdog.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from xllm.python.attention.backend import LayerCache


def _decode_rows(payload: torch.Tensor, block_scales: torch.Tensor | None, fmt: str, scale: float) -> torch.Tensor:
    """Independent CPU interpretation of resident bytes, including NVFP4 nibbles."""
    if payload.device.type != "cpu" or (block_scales is not None and block_scales.device.type != "cpu"):
        raise ValueError("Diagnostic decompression is confined to CPU reference tensors")
    if fmt == "bf16":
        return payload.float()
    if fmt == "fp8_e4m3":
        return payload.contiguous().view(torch.float8_e4m3fn).float() * scale
    if fmt != "nvfp4" or block_scales is None:
        raise ValueError("Unknown or incomplete native KV encoding")
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2).long()
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = levels[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    sf = block_scales.contiguous().view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    if sf.shape != values.shape:
        raise ValueError("NVFP4 block16 scales do not cover packed dimensions")
    return values * sf * scale


def _reference_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """CPU FP32 GQA reference with the causal offset for chunked prefill."""
    batch, queries, heads, dim = query.shape
    context = key.shape[1]
    groups = heads // key.shape[2]
    q = query.float().transpose(1, 2)
    k = key.float().repeat_interleave(groups, dim=2).transpose(1, 2)
    v = value.float().repeat_interleave(groups, dim=2).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) * dim**-0.5
    visible = torch.arange(context)[None, :] <= torch.arange(queries)[:, None] + context - queries
    scores.masked_fill_(~visible, float("-inf"))
    return (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch * queries, heads * dim)


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float((actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))


def _allocate_operator_cache(fmt: str, dimension: int, device: torch.device) -> LayerCache:
    from xllm.python.attention.backend import LayerCache

    shape = (7, 64, 4, dimension if fmt != "nvfp4" else dimension // 2)
    dtype = torch.bfloat16 if fmt == "bf16" else torch.uint8
    options = {}
    if fmt != "bf16":
        sf_shape = (*shape[:-1], dimension // 16) if fmt == "nvfp4" else shape[:-1]
        sf_dtype = torch.uint8 if fmt == "nvfp4" else torch.float32
        options = {name: torch.zeros(sf_shape, dtype=sf_dtype, device=device) for name in ("key_scale", "value_scale")}
    # LayerCache is frozen: payload and scale slots must be constructed together.
    return LayerCache(*(torch.zeros(shape, dtype=dtype, device=device) for _ in range(2)), **options)


def _run_case(
    fmt: str, phase: str, dimension: int, decode_impl: str, max_context_tokens: int,
    evidence_directory: Path | None = None,
) -> dict:
    from tools.benchmark_kv_cache import _attention_metadata
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.flashinfer import FlashInferBackend, _get_runtime_kernels
    from xllm.python.attention.flashinfer_fp8 import FlashInferFP8Backend
    from xllm.python.attention.flashinfer_nvfp4 import FlashInferNVFP4Backend
    from xllm.python.attention.native_kv import cache_allocation
    from xllm.python.attention.native_profile import NativeFlashInferConfig

    device = torch.device("cuda", 0)
    batch, heads, kv_heads, page_size, context = 2, 24, 4, 64, 129
    queries = {"prefill": context, "chunked_prefill": 5, "decode": 1}[phase]
    scales = (0.25, 0.125)  # Non-unit, distinct K/V factors catch omitted/doubled global scaling.
    cache = _allocate_operator_cache(fmt, dimension, device)
    linear = [
        LayerCache(None, None, conv=torch.ones(2, 8, 3, device=device), ssm=torch.ones(2, 4, 8, 8, device=device))
        for _ in range(3)
    ]
    args = (heads, kv_heads, dimension, dimension**-0.5, -1, device, torch.bfloat16)
    if fmt == "bf16":
        backend = FlashInferBackend(*args, native_config=NativeFlashInferConfig("fa3", decode_impl, max_context_tokens))
    elif fmt == "fp8_e4m3":
        backend = FlashInferFP8Backend(
            *args, cache_dtype=fmt, key_scale=scales[0], value_scale=scales[1],
            native_config=NativeFlashInferConfig("fa2", decode_impl, max_context_tokens),
        )
    else:
        backend = FlashInferNVFP4Backend(*args, key_scale=scales[0], value_scale=scales[1])
    backend.bind_kv_caches([*linear, cache])
    signature = cache.storage_signature()
    pages = torch.tensor([[3, 1, 5], [4, 2, 6]], device=device, dtype=torch.int32)
    positions = torch.arange(context, device=device)
    slots = (pages[:, positions // page_size] * page_size + positions % page_size).contiguous()
    key = torch.randn(batch, context, kv_heads, dimension, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)

    def write(k: torch.Tensor, v: torch.Tensor, mapping: torch.Tensor) -> None:
        if fmt == "bf16":
            _get_runtime_kernels().reshape_paged_cache(mapping, k, v, cache.key, cache.value)
        else:
            backend.write_cache(k, v, mapping, 3)

    # Include padding in the actual native writer and protect the reserved page.
    mapping = torch.cat((slots.flatten(), torch.tensor([-1, -2], device=device, dtype=torch.int32)))
    write(
        torch.cat((key.flatten(0, 1), torch.randn_like(key[0, :2]))),
        torch.cat((value.flatten(0, 1), torch.randn_like(value[0, :2]))),
        mapping,
    )
    current_slots = slots[:, -queries:].flatten().clone()
    metadata = _attention_metadata(pages, current_slots, context, queries, page_size)
    metadata.is_prefill = phase == "prefill"
    metadata.is_chunked_prefill = phase == "chunked_prefill"
    backend.prepare(metadata, graph_mode=phase == "decode")
    query = torch.randn(batch * queries, heads, dimension, dtype=torch.bfloat16, device=device)
    new_key, new_value = key[:, -queries:].flatten(0, 1).clone(), value[:, -queries:].flatten(0, 1).clone()
    layer = SimpleNamespace(layer_id=3, num_heads=heads, num_kv_heads=kv_heads, head_dim=dimension)
    backend.execute(query, new_key, new_value, layer)
    torch.cuda.synchronize()
    graph = None
    if phase == "decode":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = backend.execute(query, new_key, new_value, layer)
    observations = []
    for replay in range(2):
        query.copy_(torch.randn_like(query))
        new_key.copy_(torch.randn_like(new_key) * (replay + 1))
        new_value.copy_(torch.randn_like(new_value) * (replay + 2))
        if phase == "decode" and replay == 1:
            # Same resident pointers, changed page ownership, slot and context.
            pages.copy_(pages.flip(0))
            context = 130
            positions = torch.arange(context, device=device)
            slots = (pages[:, positions // page_size] * page_size + positions % page_size).contiguous()
            key = torch.randn(batch, context, kv_heads, dimension, dtype=torch.bfloat16, device=device)
            value = torch.randn_like(key)
            write(key.flatten(0, 1), value.flatten(0, 1), slots.flatten())
            metadata.paged_kv_indices.copy_(pages.flatten())
            metadata.paged_kv_last_page_len.fill_(2)
            metadata.paged_kv_last_page_len_host.fill_(2)
            metadata.kv_seq_lens_host.fill_(context)
            metadata.kv_seq_lens_host_values = [context] * batch
            current_slots.copy_(slots[:, -1])
            backend.prepare(metadata, graph_mode=True)
        key[:, -queries:] = new_key.view(batch, queries, kv_heads, dimension)
        value[:, -queries:] = new_value.view(batch, queries, kv_heads, dimension)
        if graph is None:
            actual = backend.execute(query, new_key, new_value, layer)
        else:
            graph.replay()
        torch.cuda.synchronize()
        index = slots.cpu().long().flatten()
        decoded = []
        payload_rows = []
        scale_rows = []
        for payload, sf, scale in zip((cache.key, cache.value), (cache.key_scale, cache.value_scale), scales):
            data = payload.cpu().flatten(0, 1).index_select(0, index)
            sf_data = None if sf is None else sf.cpu().flatten(0, 1).index_select(0, index)
            payload_rows.append(data.contiguous().view(torch.uint8))
            scale_rows.append(sf_data)
            decoded.append(_decode_rows(data, sf_data, fmt, scale).view(batch, context, kv_heads, dimension))
        q_cpu = query.cpu().view(batch, queries, heads, dimension)
        compressed_reference = _reference_attention(q_cpu, *decoded)
        original_reference = _reference_attention(q_cpu, key.cpu(), value.cpu())
        compute_reference = compressed_reference
        reference_name = "FP32_attention_BF16_query_decoded_resident_KV"
        split_count = None
        if fmt == "fp8_e4m3" and phase == "decode" and decode_impl == "xqa":
            from tools.sm90_xqa_fp8_reference import reference_decode

            override = int(os.environ.get("XQA_NB_SUB_SEQ", "0"))
            processors = torch.cuda.get_device_properties(device).multi_processor_count
            split_count = override if override > 0 else min(
                max(1, int(((processors * 3) // (batch * kv_heads)) * .25 + .5)),
                (max_context_tokens + 63) // 64,
            )
            compute_reference = reference_decode(q_cpu, *decoded, split_count=split_count)
            reference_name = "SM90_XQA_FP8_Q_and_tile64_probability_FP8_with_FP32_denominator"
        evidence = None
        if evidence_directory is not None and phase == "decode":
            import numpy as np

            path = evidence_directory / f"{fmt}-{phase}-replay{replay}-reference.npz"
            # Persist CPU arrays before assertions. Failed gates can then be
            # diagnosed without another GPU run; no GPU BF16 shadow is created.
            np.savez_compressed(
                path, query=q_cpu.float().numpy(), decoded_key=decoded[0].numpy(),
                decoded_value=decoded[1].numpy(), original_key=key.cpu().float().numpy(),
                original_value=value.cpu().float().numpy(), actual_output=actual.cpu().float().numpy(),
                resident_key_bytes=payload_rows[0].numpy(), resident_value_bytes=payload_rows[1].numpy(),
                resident_key_scales=(scale_rows[0].numpy() if scale_rows[0] is not None else np.empty(0, dtype=np.uint8)),
                resident_value_scales=(scale_rows[1].numpy() if scale_rows[1] is not None else np.empty(0, dtype=np.uint8)),
                original_BF16_query_reference=compressed_reference.numpy(),
                backend_compute_reference=compute_reference.numpy(),
            )
            evidence = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        error = _relative_error(actual.cpu(), compute_reference)
        if not torch.isfinite(actual).all() or not error <= 0.02:
            failure = {"format": fmt, "phase": phase, "replay": replay, "context": context,
                       "reference": reference_name, "split_count": split_count,
                       "operator_relative_L2": error if torch.isfinite(torch.tensor(error)) else None,
                       "actual_output_finite": bool(torch.isfinite(actual).all()),
                       "CPU_reference_evidence": evidence, "operator_qualified": False}
            if evidence_directory is not None:
                with (evidence_directory / f"{fmt}-{phase}-failure.json").open("x") as stream:
                    json.dump(failure, stream, indent=2, allow_nan=False)
                    stream.write("\n")
            raise ValueError(f"{fmt}/{phase} native attention disagrees with {reference_name}: {error}")
        writer_errors = [_relative_error(decoded[0], key.cpu()), _relative_error(decoded[1], value.cpu())]
        writer_bound = {"bf16": 0.0, "fp8_e4m3": 0.06, "nvfp4": 0.20}[fmt]
        if any(not 0 <= error <= writer_bound for error in writer_errors):
            raise ValueError(f"{fmt}/{phase} native writer exceeds its random-input encoding bound: {writer_errors}")
        observations.append({
            "context": context, "operator_relative_L2": error,
            "operator_reference": reference_name, "XQA_split_count": split_count,
            "KV_only_quantization_relative_L2": _relative_error(compressed_reference, original_reference),
            "additional_backend_Q_P_approximation_relative_L2": _relative_error(compute_reference, compressed_reference),
            "quantization_relative_L2_vs_original_BF16": _relative_error(compute_reference, original_reference),
            "actual_relative_L2_vs_original_BF16_query_decoded_KV": _relative_error(actual.cpu(), compressed_reference),
            "CPU_reference_evidence": evidence,
            "writer_key_relative_L2": writer_errors[0],
            "writer_value_relative_L2": writer_errors[1],
            "random_input_writer_error_bound_not_model_quality_threshold": writer_bound,
        })
    if cache.storage_signature() != signature:
        raise ValueError("Native writer or Graph replaced persistent KV/scale storage")
    if any(not torch.all(t == 1) for state in linear for t in (state.conv, state.ssm)):
        raise ValueError("Attention modified unrelated GDN state")
    if any(torch.count_nonzero(t[0]) for t in (cache.key, cache.value, cache.key_scale, cache.value_scale) if t is not None):
        raise ValueError("Native writer changed the reserved page while skipping negative padding")
    return {
        "format": fmt, "phase": phase, "head_dim": dimension, "heads": heads, "kv_heads": kv_heads,
        "page_size": page_size, "native_runtime": backend.native_kv_runtime_info(),
        "allocation": json.loads(cache_allocation([*linear, cache], fmt)), "observations": observations,
        "writer_in_actual_decode_Graph": graph is not None, "changed_query_KV": True,
        "changed_decode_context_pages_slots": graph is not None, "persistent_storage_unchanged": True,
        "GDN_unchanged": True, "negative_slots_skipped": True,
        "CPU_reference_decompression_only": True, "serving_performance_measured": False,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", choices=("bf16", "fp8_e4m3", "nvfp4"), default=["fp8_e4m3", "nvfp4"],
                        help="BF16 additionally requires the engine's registered native xllm_ops")
    parser.add_argument("--head-dim", type=int, choices=(128, 256), default=256)
    parser.add_argument("--decode-impl", choices=("flashinfer", "xqa"), default="flashinfer",
                        help="BF16/FP8 decode selection; NVFP4 always uses native FA2 decode")
    parser.add_argument("--max-context-tokens", type=int, default=32768,
                        help="XQA metadata capacity; the numerical cases use 129/130 live tokens")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise ValueError("Operator qualification needs exactly one externally admitted SM90 GPU")
    if os.environ.get("XLLM_FP8_CALIBRATION_OUTPUT"):
        raise ValueError("Operator qualification cannot include model calibration")
    if "bf16" in args.formats:
        if not hasattr(torch.ops.xllm_ops, "reshape_paged_cache"):
            raise ValueError("BF16 native writer requires the engine's registered xllm_ops; standalone supports FP8/NVFP4")
        from xllm.python import initialize_runtime

        initialize_runtime()
    torch.set_num_threads(4)
    torch.manual_seed(20261027)
    args.out.mkdir(parents=True, exist_ok=False)
    for fmt in args.formats:
        for phase in ("prefill", "chunked_prefill", "decode"):
            result = _run_case(fmt, phase, args.head_dim, args.decode_impl, args.max_context_tokens, args.out)
            with (args.out / f"{fmt}-{phase}.json").open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
            torch.cuda.empty_cache()
    with (args.out / "acceptance.json").open("x") as stream:
        json.dump({"scope": "operator_qualification_not_model_quality_or_performance", "SM": 90,
                   "formats": args.formats, "head_dim": args.head_dim, "seed": 20261027,
                   "BF16_FP8_decode_impl": args.decode_impl, "XQA_metadata_capacity": args.max_context_tokens,
                   "actual_numerical_context_tokens": [129, 130],
                   "versions": {name: importlib.metadata.version(name) for name in ("torch", "flashinfer-python", "triton")}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    _main()
