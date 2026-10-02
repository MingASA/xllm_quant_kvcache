# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small real-QKV INT4 KV-cache diagnostic; does not alter serving code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as functional
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.logger import logger
from tools.kv_cache_codec_lab import QuantSpec, _hadamard, _signs
from tools.kv_cache_codec_lab import encode as lab_encode
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.quantized import KVCacheCodec, write_quantized_kv
from xllm.python.attention.quantized_triton import quantized_paged_attention

LAYERS = (0, 7, 14, 21, 27)
PAGE_SIZE = 128
CONTEXT_LENGTH = 145
SEED = 17


def _dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """FP32 dense causal GQA attention, rounded back to the model dtype."""
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1).float()
    value = value.repeat_interleave(groups, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", query.float(), key) * scale
    key_positions = torch.arange(key.shape[0], device=query.device)
    visible = key_positions[None, :] <= query_positions[:, None]
    scores.masked_fill_(~visible[None, :, :], -torch.inf)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.einsum("hqk,khd->qhd", weights, value)
    return output.to(query.dtype)


def _error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    actual32 = actual.float()
    reference32 = reference.float()
    delta = actual32 - reference32
    reference_norm = torch.linalg.vector_norm(reference32)
    return {
        "all_finite": bool(torch.isfinite(actual32).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "relative_l2": float((torch.linalg.vector_norm(delta) / reference_norm.clamp_min(1e-20)).item()),
    }


def _make_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    cache_dtype: str,
    poison_tail: bool = False,
    writer: Callable[[LayerCache, torch.Tensor, torch.Tensor, torch.Tensor, KVCacheCodec], None] | None = None,
) -> tuple[LayerCache, torch.Tensor, torch.Tensor]:
    device = key.device
    num_kv_heads, head_dim = key.shape[1:]
    codec = KVCacheCodec(cache_dtype, head_dim)
    num_blocks = 8
    shape = (num_blocks, PAGE_SIZE, num_kv_heads, codec.storage_dim)
    payload_dtype = codec.storage_dtype
    key_cache = torch.zeros(shape, dtype=payload_dtype, device=device)
    value_cache = torch.zeros_like(key_cache)
    key_scale = torch.zeros(shape[:-1], dtype=torch.float32, device=device)
    value_scale = torch.zeros_like(key_scale)
    cache = LayerCache(key=key_cache, value=value_cache, key_scale=key_scale, value_scale=value_scale)

    generator = torch.Generator(device=device).manual_seed(SEED)
    permutation = torch.randperm(num_blocks, generator=generator, device=device, dtype=torch.int32)
    logical_positions = torch.arange(key.shape[0], device=device)
    slots = permutation[logical_positions // PAGE_SIZE].to(torch.int64) * PAGE_SIZE + logical_positions % PAGE_SIZE
    if writer is None:
        write_quantized_kv(cache, key, value, slots, codec)
    else:
        writer(cache, key, value, slots, codec)
    block_table = permutation[None, :].contiguous()
    if poison_tail:
        tail_block = int(permutation[1].item())
        key_scale[tail_block, key.shape[0] - PAGE_SIZE :, :] = float("nan")
        value_scale[tail_block, key.shape[0] - PAGE_SIZE :, :] = float("nan")
    return cache, block_table, permutation


def _triton_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache_dtype: str,
    positions: torch.Tensor,
    poison_tail: bool = False,
    writer: Callable[[LayerCache, torch.Tensor, torch.Tensor, torch.Tensor, KVCacheCodec], None] | None = None,
) -> torch.Tensor:
    cache, block_table, _ = _make_cache(key, value, cache_dtype, poison_tail, writer)
    query_tokens = query.shape[0]
    device = query.device
    result = quantized_paged_attention(
        query.contiguous(),
        cache.key,
        cache.value,
        cache.key_scale,
        cache.value_scale,
        block_table,
        torch.zeros(query_tokens, dtype=torch.int32, device=device),
        positions.to(torch.int32).contiguous(),
        torch.tensor([key.shape[0]], dtype=torch.int32, device=device),
        cache_dtype,
        query.shape[-1] ** -0.5,
        0,
        True,
        torch.tensor([0, query_tokens], dtype=torch.int32, device=device),
        query_tokens,
    )
    return result


def _quantize_dequantize(value: torch.Tensor, cache_dtype: str) -> torch.Tensor:
    codec = KVCacheCodec(cache_dtype, value.shape[-1])
    payload, scales = codec.encode(value)
    return codec.decode(payload, scales)


def _rotation_only(value: torch.Tensor) -> torch.Tensor:
    head_dim = value.shape[-1]
    padded_dim = 1 << (head_dim - 1).bit_length()
    signs = _signs(padded_dim, SEED).to(device=value.device)
    rotated = _hadamard(torch.nn.functional.pad(value.float(), (0, padded_dim - head_dim)) * signs)
    reconstructed = _hadamard(_quantize_dequantize(rotated, "int4")) * signs
    return reconstructed[..., :head_dim]


def _rotation_g32(value: torch.Tensor) -> torch.Tensor:
    head_dim = value.shape[-1]
    padded_dim = 1 << (head_dim - 1).bit_length()
    signs = _signs(padded_dim, SEED).to(device=value.device)
    rotated = _hadamard(torch.nn.functional.pad(value.float(), (0, padded_dim - head_dim)) * signs)
    grouped = rotated.reshape(*rotated.shape[:-1], padded_dim // 32, 32)
    packed_layout = grouped.reshape(rotated.shape[0], -1, 32)
    codec = KVCacheCodec("int4", 32)
    payload, scales = codec.encode(packed_layout)
    restored = codec.decode(payload, scales).reshape_as(grouped).flatten(-2)
    return (_hadamard(restored) * signs)[..., :head_dim]


def _cache_quant_metrics(value: torch.Tensor, cache_dtype: str) -> dict[str, float | bool]:
    reconstructed = _quantize_dequantize(value, cache_dtype)
    return _error_metrics(reconstructed, value)


def _writer_mismatch_examples(
    actual: torch.Tensor,
    expected: torch.Tensor,
    source: torch.Tensor,
    scales: torch.Tensor,
    limit: int = 5,
) -> list[dict[str, float | int]]:
    mismatches = torch.nonzero(actual != expected, as_tuple=False)[:limit]
    examples = []
    for token, head, byte_index in mismatches.tolist():
        for channel in (2 * byte_index, 2 * byte_index + 1):
            if channel >= source.shape[-1]:
                continue
            raw = float(source[token, head, channel].float().item())
            scale = float(scales[token, head].item())
            normalized = raw / scale
            torch_code = int(round(normalized))
            nibble = int((actual[token, head, byte_index].to(torch.int32) >> (4 * (channel % 2))).item()) & 15
            gpu_code = nibble - 16 if nibble >= 8 else nibble
            half_distance = abs(abs(normalized) - (math.floor(abs(normalized)) + 0.5))
            examples.append(
                {
                    "token": token,
                    "head": head,
                    "channel": channel,
                    "value": raw,
                    "scale": scale,
                    "torch_normalized": normalized,
                    "torch_code": torch_code,
                    "gpu_code": gpu_code,
                    "distance_to_half_integer": half_distance,
                }
            )
            if len(examples) >= limit:
                return examples
    return examples


def _layer_report(
    layer_id: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    divrn_writer: Callable[[LayerCache, torch.Tensor, torch.Tensor, torch.Tensor, KVCacheCodec], None],
) -> dict[str, Any]:
    context = min(CONTEXT_LENGTH, key.shape[0])
    if context <= 128:
        raise ValueError(f"Captured context for layer {layer_id} is shorter than the diagnostic requirement")
    query_start = context - 64
    q = query[query_start:context]
    k = key[:context]
    v = value[:context]
    positions = torch.arange(query_start, context, device=q.device)
    scale = q.shape[-1] ** -0.5

    oracle = _dense_attention(q, k, v, positions, scale)
    int4_key = _quantize_dequantize(k, "int4")
    int4_value = _quantize_dequantize(v, "int4")
    int4_dense = _dense_attention(q, int4_key, int4_value, positions, scale)
    int8_dense = _dense_attention(
        q, _quantize_dequantize(k, "int8"), _quantize_dequantize(v, "int8"), positions, scale
    )
    k_only = _dense_attention(q, int4_key, v, positions, scale)
    v_only = _dense_attention(q, k, int4_value, positions, scale)
    rotation_key = _rotation_only(k)
    rotation_value = _rotation_only(v)
    rotation_dense = _dense_attention(q, rotation_key, rotation_value, positions, scale)
    rotation_g32_key = _rotation_g32(k)
    rotation_g32_value = _rotation_g32(v)
    rotation_g32_dense = _dense_attention(q, rotation_g32_key, rotation_g32_value, positions, scale)
    lab_spec = QuantSpec(format="int4", group_size=32, axis="token", rotation=True, seed=SEED)
    lab_key = lab_encode(k.detach().cpu(), lab_spec).decode().to(k.device)
    lab_value = lab_encode(v.detach().cpu(), lab_spec).decode().to(v.device)
    actual_triton = _triton_attention(q, k, v, "int4", positions)
    decode_query = q[-1:]
    decode_position = positions[-1:]
    int4_decode_dense = _dense_attention(decode_query, int4_key, int4_value, decode_position, scale)
    int4_decode_triton = _triton_attention(decode_query, k, v, "int4", decode_position)

    codec = KVCacheCodec("int4", k.shape[-1])
    encoded_key, key_scale = codec.encode(k)
    encoded_value, value_scale = codec.encode(v)
    cache, _, permutation = _make_cache(k, v, "int4")
    positions_all = torch.arange(context, device=k.device)
    physical_blocks = permutation[positions_all // PAGE_SIZE].long()
    physical_offsets = positions_all % PAGE_SIZE
    actual_key_payload = cache.key[physical_blocks, physical_offsets]
    actual_value_payload = cache.value[physical_blocks, physical_offsets]
    actual_key_scale = cache.key_scale[physical_blocks, physical_offsets]
    actual_value_scale = cache.value_scale[physical_blocks, physical_offsets]
    payload_matches = torch.equal(actual_key_payload, encoded_key) and torch.equal(actual_value_payload, encoded_value)
    scales_match = torch.equal(actual_key_scale, key_scale) and torch.equal(actual_value_scale, value_scale)
    gpu_decoded_key = codec.decode(actual_key_payload, actual_key_scale)
    gpu_decoded_value = codec.decode(actual_value_payload, actual_value_scale)
    gpu_written_dense = _dense_attention(q, gpu_decoded_key, gpu_decoded_value, positions, scale)
    gpu_written_triton = _triton_attention(q, k, v, "int4", positions)
    divrn_cache, divrn_table, divrn_permutation = _make_cache(k, v, "int4", writer=divrn_writer)
    divrn_physical_blocks = divrn_permutation[positions_all // PAGE_SIZE].long()
    divrn_physical_offsets = positions_all % PAGE_SIZE
    divrn_key_payload = divrn_cache.key[divrn_physical_blocks, divrn_physical_offsets]
    divrn_value_payload = divrn_cache.value[divrn_physical_blocks, divrn_physical_offsets]
    divrn_key_scale = divrn_cache.key_scale[divrn_physical_blocks, divrn_physical_offsets]
    divrn_value_scale = divrn_cache.value_scale[divrn_physical_blocks, divrn_physical_offsets]
    divrn_key = codec.decode(divrn_key_payload, divrn_key_scale)
    divrn_value = codec.decode(divrn_value_payload, divrn_value_scale)
    divrn_dense = _dense_attention(q, divrn_key, divrn_value, positions, scale)
    divrn_triton = quantized_paged_attention(
        q.contiguous(), divrn_cache.key, divrn_cache.value,
        divrn_cache.key_scale, divrn_cache.value_scale, divrn_table,
        torch.zeros(q.shape[0], dtype=torch.int32, device=q.device), positions.to(torch.int32),
        torch.tensor([context], dtype=torch.int32, device=q.device), "int4", scale, 0, True,
        torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device), q.shape[0],
    )
    poisoned_tail = _triton_attention(q[-1:], k, v, "int4", positions[-1:], poison_tail=True)
    zero_tail = _triton_attention(q[-1:], k, v, "int4", positions[-1:], poison_tail=False)

    return {
        "layer": layer_id,
        "captured_qkv_dtype": str(query.dtype),
        "shape": {"q": list(query.shape), "k": list(key.shape), "v": list(value.shape)},
        "diagnostic_context_tokens": context,
        "diagnostic_query_tokens": int(q.shape[0]),
        "codec": {"axis": "per-token/per-head", "group_size": key.shape[-1], "bound": 7, "scale_dtype": "float32", "rounding": "torch.round ties-to-even"},
        "int4_key_reconstruction": _cache_quant_metrics(k, "int4"),
        "int4_value_reconstruction": _cache_quant_metrics(v, "int4"),
        "int8_key_reconstruction": _cache_quant_metrics(k, "int8"),
        "int8_value_reconstruction": _cache_quant_metrics(v, "int8"),
        "bf16_fp32_dense_oracle": {"all_finite": bool(torch.isfinite(oracle).all().item())},
        "attention_vs_fp32_dense_oracle": {
            "int4_dense": _error_metrics(int4_dense, oracle),
            "int4_triton": _error_metrics(actual_triton, oracle),
            "int8_dense": _error_metrics(int8_dense, oracle),
            "rotation_only_dense": _error_metrics(rotation_dense, oracle),
            "lab_g32_rotation_dense": _error_metrics(rotation_g32_dense, oracle),
            "k_only_int4_dense": _error_metrics(k_only, oracle),
            "v_only_int4_dense": _error_metrics(v_only, oracle),
            "gpu_written_cache_dense": _error_metrics(gpu_written_dense, oracle),
        },
        "triton_vs_dense_int4": _error_metrics(actual_triton, int4_dense),
        "lab_g32_gpu_vs_cpu_codec_key": _error_metrics(rotation_g32_key, lab_key),
        "lab_g32_gpu_vs_cpu_codec_value": _error_metrics(rotation_g32_value, lab_value),
        "triton_vs_gpu_written_cache_dense": _error_metrics(gpu_written_triton, gpu_written_dense),
        "single_token_decode_triton_vs_dense_int4": _error_metrics(int4_decode_triton, int4_decode_dense),
        "gpu_writer_compare": {
            "payload_exact_match": payload_matches,
            "key_payload_mismatched_bytes": int((actual_key_payload != encoded_key).sum().item()),
            "value_payload_mismatched_bytes": int((actual_value_payload != encoded_value).sum().item()),
            "scales_exact_match": scales_match,
            "key_scale_max_abs_delta": float((actual_key_scale - key_scale).abs().max().item()),
            "value_scale_max_abs_delta": float((actual_value_scale - value_scale).abs().max().item()),
            "key_scale_max_relative_delta": float(
                (((actual_key_scale - key_scale).abs()) / key_scale.abs().clamp_min(1e-30)).max().item()
            ),
            "value_scale_max_relative_delta": float(
                (((actual_value_scale - value_scale).abs()) / value_scale.abs().clamp_min(1e-30)).max().item()
            ),
            "key_mismatch_examples": _writer_mismatch_examples(
                actual_key_payload, encoded_key, k, key_scale
            ),
            "value_mismatch_examples": _writer_mismatch_examples(
                actual_value_payload, encoded_value, v, value_scale
            ),
        },
        "div_rn_int4_writer_compare": {
            "payload_exact_match": bool(
                torch.equal(divrn_key_payload, encoded_key) and torch.equal(divrn_value_payload, encoded_value)
            ),
            "key_payload_mismatched_bytes": int((divrn_key_payload != encoded_key).sum().item()),
            "value_payload_mismatched_bytes": int((divrn_value_payload != encoded_value).sum().item()),
            "key_scale_max_abs_delta": float((divrn_key_scale - key_scale).abs().max().item()),
            "value_scale_max_abs_delta": float((divrn_value_scale - value_scale).abs().max().item()),
            "dense_attention_vs_torch_codec": _error_metrics(divrn_dense, int4_dense),
            "triton_attention_vs_dense_from_gpu_written_cache": _error_metrics(divrn_triton, divrn_dense),
            "attention_vs_fp32_dense_oracle": _error_metrics(divrn_dense, oracle),
        },
        "decode_tail_poison": {
            "sequence_length": context,
            "page_size": PAGE_SIZE,
            "scale_tail_nan": True,
            "split_count_expected": 8,
            "zero_tail_all_finite": bool(torch.isfinite(zero_tail).all().item()),
            "nan_tail_all_finite": bool(torch.isfinite(poisoned_tail).all().item()),
        "nan_tail_vs_zero_tail": _error_metrics(poisoned_tail, zero_tail),
        },
    }


def _load_prompt(data_path: Path, shots_path: Path) -> tuple[str, str]:
    row = json.loads(data_path.open(encoding="utf-8").readline())
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    prompt = _gsm_prompt_for_row(row, shots)
    return prompt, hashlib.sha256(prompt.encode()).hexdigest()


def _gsm_prompt_for_row(row: dict[str, Any], shots: list[dict[str, str]]) -> str:
    examples = "\n\n".join(f"Question: {item['question']}\n Answer: {item['answer']}" for item in shots)
    return f"{examples}\n\nQuestion: {row['question']}\n Answer:"


def _capture_real_qkv(model_path: Path, prompt: str) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()
    captures: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    mask_audit: list[dict[str, Any]] = []
    eager = qwen2.eager_attention_forward

    def capture_attention(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape[-2] > 1 and module.layer_idx in LAYERS:
            mask_audit.append(_audit_causal_mask(attention_mask, query, key))
        if module.layer_idx in LAYERS and module.layer_idx not in captures:
            limit = min(512, key.shape[-2])
            captures[module.layer_idx] = tuple(
                tensor[:, :, :limit, :].detach().contiguous() for tensor in (query, key, value)
            )
        return eager(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register("xllm_int4_diagnostic", capture_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register("xllm_int4_diagnostic", ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    model.config._attn_implementation = "xllm_int4_diagnostic"
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    if isinstance(tokens, list):
        tokens = torch.tensor([tokens], dtype=torch.long)
    tokens = tokens.to("cuda")
    with torch.inference_mode():
        model(input_ids=tokens, use_cache=False)
    del model
    torch.cuda.empty_cache()
    expected_layers = set(LAYERS)
    if captures.keys() != expected_layers:
        raise RuntimeError(f"Expected captures for layers {LAYERS}, got {sorted(captures)}")
    if not mask_audit or not all(record["future_masked"] for record in mask_audit):
        raise RuntimeError("The real-QKV capture forward did not receive a verified causal attention mask")
    return {
        layer: tuple(tensor[0].permute(1, 0, 2).contiguous() for tensor in captures[layer])
        for layer in LAYERS
    }


def _hf_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1).float()
    value = value.repeat_interleave(groups, dim=1).float()
    scores = torch.matmul(query.float(), key.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        scores += attention_mask.float()
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.matmul(weights, value).to(query.dtype)
    return output.transpose(1, 2).contiguous()


def _audit_causal_mask(
    attention_mask: torch.Tensor | None,
    query: torch.Tensor,
    key: torch.Tensor,
) -> dict[str, Any]:
    if attention_mask is None:
        raise RuntimeError("Attention callback received no causal mask for a multi-token prefill")
    if attention_mask.ndim != 4:
        raise RuntimeError(f"Expected a 4D causal attention mask, got shape {tuple(attention_mask.shape)}")
    if query.shape[-2] != key.shape[-2]:
        raise RuntimeError("Causal-mask audit currently expects a full-prefill Q/K window")
    if attention_mask.dtype == torch.bool:
        future_masked = not bool(attention_mask[0, 0, 0, -1].item())
    else:
        future_value = float(attention_mask[0, 0, 0, -1].float().item())
        future_masked = not math.isfinite(future_value) or future_value < -1e4
    if not future_masked:
        raise RuntimeError("Causal mask does not block the first query from attending to the final future key")
    return {
        "mask_shape": list(attention_mask.shape),
        "mask_dtype": str(attention_mask.dtype),
        "query_length": int(query.shape[-2]),
        "key_length": int(key.shape[-2]),
        "future_masked": future_masked,
    }


def _apply_attention_mode(
    mode: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    eager: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    module: nn.Module,
    dropout: float,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if mode == "bf16_eager":
        return eager(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    key_thd = key[0].transpose(0, 1).contiguous()
    value_thd = value[0].transpose(0, 1).contiguous()
    if mode == "bf16_fp32_dense":
        pass
    elif mode == "int4_dense":
        key_thd = _quantize_dequantize(key_thd, "int4")
        value_thd = _quantize_dequantize(value_thd, "int4")
    elif mode == "int8_dense":
        key_thd = _quantize_dequantize(key_thd, "int8")
        value_thd = _quantize_dequantize(value_thd, "int8")
    elif mode == "int4_rotation_dense":
        key_thd = _rotation_only(key_thd)
        value_thd = _rotation_only(value_thd)
    elif mode == "int4_lab_g32_rotation_dense":
        key_thd = _rotation_g32(key_thd)
        value_thd = _rotation_g32(value_thd)
    elif mode == "k_only_int4":
        key_thd = _quantize_dequantize(key_thd, "int4")
    elif mode == "v_only_int4":
        value_thd = _quantize_dequantize(value_thd, "int4")
    else:
        raise ValueError(f"Unknown attention diagnostic mode: {mode}")
    key = key_thd.transpose(0, 1).unsqueeze(0).contiguous()
    value = value_thd.transpose(0, 1).unsqueeze(0).contiguous()
    return _hf_dense_attention(query, key, value, attention_mask, scaling), None


def _model_e2e_diagnostic(
    model_path: Path,
    prompt: str,
    answer: str,
    seed: int,
    sample_index: int,
    run_greedy: bool,
) -> dict[str, Any]:
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()
    eager = qwen2.eager_attention_forward
    state: dict[str, Any] = {"mode": "bf16_eager", "mask_audit": []}

    def diagnostic_attention(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.shape[-2] > 1:
            state["mask_audit"].append(_audit_causal_mask(attention_mask, query, key))
        return _apply_attention_mode(
            state["mode"], query, key, value, attention_mask, scaling, eager, module, dropout, kwargs
        )

    model.config._attn_implementation = "eager"
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(prompt_tokens, "input_ids"):
        prompt_tokens = prompt_tokens.input_ids
    if isinstance(prompt_tokens, list):
        prompt_tokens = torch.tensor([prompt_tokens], dtype=torch.long)
    prompt_tokens = prompt_tokens.to("cuda")
    answer_tokens = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids[:, :64].to("cuda")
    if answer_tokens.numel() == 0:
        raise ValueError("The selected GSM8K answer tokenized to an empty sequence")
    forced_input = torch.cat((prompt_tokens, answer_tokens), dim=-1)
    modes = (
        "bf16_eager",
        "bf16_fp32_dense",
        "int4_dense",
        "int8_dense",
        "int4_rotation_dense",
        "int4_lab_g32_rotation_dense",
        "k_only_int4",
        "v_only_int4",
    )
    start = prompt_tokens.shape[-1] - 1
    stop = start + answer_tokens.shape[-1]
    with torch.inference_mode():
        eager_reference = model(input_ids=forced_input, use_cache=False).logits[:, start:stop].float().detach()

    ALL_ATTENTION_FUNCTIONS.register("xllm_int4_e2e_diagnostic", diagnostic_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register("xllm_int4_e2e_diagnostic", ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    model.config._attn_implementation = "xllm_int4_e2e_diagnostic"
    logits_by_mode: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for mode in modes:
            state["mode"] = mode
            output = model(input_ids=forced_input, use_cache=False)
            logits_by_mode[mode] = output.logits[:, start:stop].float().detach()
            del output

    wrapper_delta = (logits_by_mode["bf16_eager"] - eager_reference).abs().max()
    if wrapper_delta.item() != 0.0:
        raise RuntimeError(f"Registered eager wrapper changed BF16 logits (max abs delta={wrapper_delta.item()})")
    split = min(20, answer_tokens.shape[-1] - 1)
    alternate_input = forced_input.clone()
    alternate_input[:, prompt_tokens.shape[-1] + split :] = (
        alternate_input[:, prompt_tokens.shape[-1] + split :] + 1
    ) % model.config.vocab_size
    state["mode"] = "bf16_eager"
    with torch.inference_mode():
        alternate_logits = model(input_ids=alternate_input, use_cache=False).logits[:, start : start + split + 1]
    future_invariance_delta = (alternate_logits.float() - logits_by_mode["bf16_eager"][:, : split + 1]).abs().max()
    if future_invariance_delta.item() != 0.0:
        raise RuntimeError(
            "Changing future teacher-forced tokens changed earlier logits "
            f"(max abs delta={future_invariance_delta.item()})"
        )

    reference = logits_by_mode["bf16_eager"]
    labels = answer_tokens
    reference_prob = torch.softmax(reference, dim=-1)
    reference_log_prob = torch.log_softmax(reference, dim=-1)
    teacher_forced = {}
    for mode, logits in logits_by_mode.items():
        log_prob = torch.log_softmax(logits, dim=-1)
        kl = (reference_prob * (reference_log_prob - log_prob)).sum(dim=-1).mean()
        top1 = (logits.argmax(dim=-1) == reference.argmax(dim=-1)).float().mean()
        loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        teacher_forced[mode] = {
            "mean_kl_reference_to_mode": float(kl.item()),
            "top1_agreement_with_bf16": float(top1.item()),
            "teacher_token_cross_entropy": float(loss.item()),
            "all_finite": bool(torch.isfinite(logits).all().item()),
        }

    greedy_modes = ("bf16_eager", "int4_dense", "int4_rotation_dense") if run_greedy else ()
    generated: dict[str, dict[str, Any]] = {}
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        for mode in greedy_modes:
            state["mode"] = mode
            past = None
            current = prompt_tokens
            generated_ids: list[int] = []
            for _ in range(32):
                result = model(input_ids=current, past_key_values=past, use_cache=True)
                past = result.past_key_values
                token_id = int(result.logits[:, -1].argmax(dim=-1).item())
                generated_ids.append(token_id)
                current = torch.tensor([[token_id]], dtype=torch.long, device="cuda")
                del result
                if token_id == tokenizer.eos_token_id:
                    break
            generated[mode] = {
                "token_ids": generated_ids,
                "decoded_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
                "token_count": len(generated_ids),
            }
    del model
    torch.cuda.empty_cache()
    return {
        "sample_index": sample_index,
        "mode_protocol": "post-RoPE K/V quantize-dequantize, FP32 dense attention, output rounded to query BF16",
        "prompt_tokens": int(prompt_tokens.shape[-1]),
        "teacher_forced_answer_tokens": int(answer_tokens.shape[-1]),
        "causal_mask_audit": {
            "prefill_callbacks_with_verified_future_mask": len(state["mask_audit"]),
            "verified_future_mask_examples": state["mask_audit"][:2],
            "future_token_invariance_max_abs": float(future_invariance_delta.item()),
            "native_eager_vs_registered_wrapper_max_abs": float(wrapper_delta.item()),
        },
        "teacher_forced_modes": teacher_forced,
        "greedy_max_tokens": 32,
        "greedy_modes": generated,
    }


def _make_div_rn_writer() -> Callable[[LayerCache, torch.Tensor, torch.Tensor, torch.Tensor, KVCacheCodec], None]:
    source_path = Path(__file__).resolve().parents[1] / "xllm/python/attention/quantized_triton.py"
    source = source_path.read_text(encoding="utf-8")
    target = "        if FORMAT == 0:\n            # Approximate reciprocal division can flip half-integer ties for"
    replacement = "        if FORMAT == 0 or FORMAT == 3:\n            # Approximate reciprocal division can flip half-integer ties for"
    if source.count(target) != 1:
        raise RuntimeError("Could not identify the writer division branch for the diagnostic-only clone")
    temporary_directory = tempfile.TemporaryDirectory(prefix="xllm-int4-divrn-")
    module_path = Path(temporary_directory.name) / "quantized_triton_diagnostic.py"
    module_path.write_text(source.replace(target, replacement), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("quantized_triton_diagnostic", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the diagnostic-only Triton writer clone")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def writer(
        cache: LayerCache,
        key: torch.Tensor,
        value: torch.Tensor,
        slots: torch.Tensor,
        codec: KVCacheCodec,
    ) -> None:
        module.write_quantized_kv(
            key.contiguous(),
            value.contiguous(),
            slots.contiguous(),
            cache.key,
            cache.value,
            cache.key_scale,
            cache.value_scale,
            codec.cache_dtype,
            codec.head_dim,
        )

    writer._temporary_directory = temporary_directory  # type: ignore[attr-defined]
    return writer


def _poisoned_tail_unit() -> dict[str, Any]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(SEED)
    key = torch.randn(CONTEXT_LENGTH, 2, 128, generator=generator, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query = torch.randn(1, 12, 128, generator=generator, device=device, dtype=torch.bfloat16)
    query_position = torch.tensor([CONTEXT_LENGTH - 1], device=device)
    zero = _triton_attention(query, key, value, "int4", query_position)
    poisoned = _triton_attention(query, key, value, "int4", query_position, poison_tail=True)
    return {
        "context_length": CONTEXT_LENGTH,
        "page_size": PAGE_SIZE,
        "head_dim": 128,
        "query_heads": 12,
        "kv_heads": 2,
        "query_tokens": 1,
        "expected_num_splits": 8,
        "zero_tail_all_finite": bool(torch.isfinite(zero).all().item()),
        "nan_tail_all_finite": bool(torch.isfinite(poisoned).all().item()),
        "nan_tail_vs_zero_tail": _error_metrics(poisoned, zero),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gsm8k-data", type=Path, required=True)
    parser.add_argument("--gsm8k-shots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-e2e-output", type=Path)
    parser.add_argument("--model-e2e-samples", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires the CUDA serving environment")

    prompt, prompt_sha256 = _load_prompt(args.gsm8k_data, args.gsm8k_shots)
    captured = _capture_real_qkv(args.model, prompt)
    divrn_writer = _make_div_rn_writer()
    layers = {}
    for layer, tensors in captured.items():
        query, key, value = tensors
        layers[str(layer)] = _layer_report(layer, query, key, value, divrn_writer)
    result = {
        "diagnostic": "single real GSM8K prompt; no generation; post-RoPE HF Q/K/V capture",
        "prompt_sha256": prompt_sha256,
        "prompt_tokens_captured": max(value[0].shape[0] for value in captured.values()),
        "capture_layers": list(LAYERS),
        "capture_limit_tokens": 512,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "poisoned_tail_repro": _poisoned_tail_unit(),
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "diagnostic output=%s layers=%s poisoned_tail=%s",
        args.output,
        list(layers),
        result["poisoned_tail_repro"],
    )
    if args.model_e2e_output is not None:
        if not 1 <= args.model_e2e_samples <= 4:
            raise ValueError("--model-e2e-samples must be between 1 and 4")
        shots = json.loads(args.gsm8k_shots.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in args.gsm8k_data.read_text(encoding="utf-8").splitlines()[: args.model_e2e_samples]
        ]
        e2e: dict[str, Any] = {"sample_count": len(rows), "samples": []}
        args.model_e2e_output.parent.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows):
            sample_prompt = _gsm_prompt_for_row(row, shots)
            sample_result = _model_e2e_diagnostic(
                args.model,
                sample_prompt,
                row["answer"],
                SEED,
                int(row.get("_source_index", index)),
                index == 0,
            )
            e2e["samples"].append(sample_result)
            args.model_e2e_output.write_text(json.dumps(e2e, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            logger.info("model-e2e completed sample=%d/%d output=%s", index + 1, len(rows), args.model_e2e_output)


if __name__ == "__main__":
    main()
