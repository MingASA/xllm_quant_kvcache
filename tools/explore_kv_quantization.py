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

"""Small reproducible codec/attention exploration, not an LLM accuracy benchmark.

Run from the repository root: python -m tools.explore_kv_quantization
JSON goes to stdout; diagnostics use the shared logger.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, replace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger
from tools.kv_cache_codec_lab import QuantSpec
from tools.kv_cache_quantization_lab import PageQuantPolicy, QuantizedPageCache


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()


def _dense_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    repeats = query.shape[1] // key.shape[1]
    q = query.float().transpose(0, 1)
    k = key.float().repeat_interleave(repeats, dim=1).transpose(0, 1)
    v = value.float().repeat_interleave(repeats, dim=1).transpose(0, 1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    positions = torch.arange(len(key) - len(query), len(key))
    scores.masked_fill_(torch.arange(len(key))[None, :] > positions[:, None], -math.inf)
    return (scores.softmax(dim=-1) @ v).transpose(0, 1)


def _candidates() -> list[tuple[str, QuantSpec, QuantSpec, int, int]]:
    base = QuantSpec(format="int4", group_size=32)
    result = [("native", replace(base, format="native"), replace(base, format="native"), 0, 0)]
    for bits in range(2, 9):
        spec = replace(base, format=f"int{bits}")
        result.append((f"int{bits}_sym", spec, spec, 0, 0))
    for format_name in ("fp8_e4m3", "fp8_e5m2", "nf4", "fp4_e2m1"):
        spec = replace(base, format=format_name)
        result.append((format_name, spec, spec, 0, 0))
    variants = {
        "int4_affine": replace(base, affine=True),
        "int4_group8": replace(base, group_size=8),
        "int4_group64": replace(base, group_size=64),
        "int4_rotated": replace(base, rotation=True),
        "int4_outliers": replace(base, outlier_fraction=1 / 32),
        "int4_fp16_scales": replace(base, scale_dtype="float16"),
        "int4_pow2_scales": replace(base, scale_dtype="pow2"),
    }
    result.extend((name, spec, spec, 0, 0) for name, spec in variants.items())
    result.extend(
        [
            ("k_channel_v_token", replace(base, axis="channel", affine=True), replace(base, affine=True), 32, 0),
            ("k_int8_v_int4", replace(base, format="int8"), base, 32, 0),
            ("int4_residual32", base, base, 32, 0),
            ("int4_residual64_sink16", base, base, 64, 16),
        ]
    )
    return result


def _inputs(distribution: str, tokens: int, seed: int = 2026) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    key = torch.randn(tokens, 2, 64, generator=generator)
    value = torch.randn(tokens, 2, 64, generator=generator)
    query = torch.randn(8, 4, 64, generator=generator)
    if distribution == "channel_outliers":
        key[:, :, ::16] *= 12
        value[::19] *= 8
    elif distribution == "shifted":
        key += 3
        value += 2
    elif distribution == "page_shifts":
        count = math.ceil(tokens / 32)
        key += torch.randn(count, 2, 64, generator=generator).repeat_interleave(32, dim=0)[:tokens] * 5
        value += torch.randn(count, 2, 64, generator=generator).repeat_interleave(32, dim=0)[:tokens] * 5
    elif distribution == "correlated":
        key = 0.1 * key + torch.einsum(
            "thr,hrd->thd", torch.randn(tokens, 2, 2, generator=generator), torch.randn(2, 2, 64, generator=generator)
        )
        value = 0.1 * value + torch.einsum(
            "thr,hrd->thd", torch.randn(tokens, 2, 2, generator=generator), torch.randn(2, 2, 64, generator=generator)
        )
    return query.bfloat16(), key.bfloat16(), value.bfloat16()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--distribution",
        choices=("all", "gaussian", "channel_outliers", "shifted", "page_shifts", "correlated"),
        default="all",
    )
    parser.add_argument(
        "--mechanisms", action="store_true", help="Compare centering, residual repair and expanded page search"
    )
    parser.add_argument("--candidate", default=None, help="Optional exact candidate name")
    parser.add_argument(
        "--adaptive", action="store_true", help="Compare error-budget page selection to fixed INT4/INT8"
    )
    args = parser.parse_args()
    if sum((args.adaptive, args.mechanisms, args.candidate is not None)) > 1:
        parser.error("--adaptive, --mechanisms and --candidate are mutually exclusive")
    if args.tokens < 8:
        parser.error("--tokens must be at least 8")
    torch.set_num_threads(1)
    distributions = ("gaussian", "channel_outliers", "shifted") if args.distribution == "all" else (args.distribution,)
    if args.mechanisms and args.distribution == "all":
        distributions += ("page_shifts", "correlated")
    candidates = [entry for entry in _candidates() if args.candidate is None or entry[0] == args.candidate]
    if not candidates:
        parser.error("Unknown --candidate")
    policy_specs = tuple(QuantSpec(format=f"int{bits}", scale_dtype="float16") for bits in (3, 4, 6, 8)) + (
        QuantSpec(format="int4", rotation=True, scale_dtype="float16"),
        QuantSpec(format="int4", axis="channel", affine=True, scale_dtype="float16"),
        QuantSpec(format="fp8_e4m3", scale_dtype="float16"),
        QuantSpec(format="nf4", scale_dtype="float16"),
        QuantSpec(format="native"),
    )
    key_policy = PageQuantPolicy(policy_specs, max_relative_l2=0.08, max_vector_error=0.5)
    value_policy = PageQuantPolicy(policy_specs, max_relative_l2=0.1, max_vector_error=1.0)
    policies = {"adaptive": (key_policy, value_policy)}
    if args.adaptive:
        candidates = [entry for entry in _candidates() if entry[0] in ("int4_sym", "int8_sym")]
        candidates.append(("adaptive", QuantSpec(), QuantSpec(), 0, 0))
    if args.mechanisms:
        base = QuantSpec()
        additions = {
            "int4_center": replace(base, center=True),
            "int4_rank1": replace(base, residual_rank=1),
            "int4_rank2": replace(base, residual_rank=2),
            "int4_rank1_fp16": replace(base, residual_rank=1, residual_dtype="float16"),
            "int4_rank2_fp16": replace(base, residual_rank=2, residual_dtype="float16"),
            "int4_rank2_bf16": replace(base, residual_rank=2, residual_dtype="bfloat16"),
            "int4_center_rank1_fp16": replace(base, center=True, residual_rank=1, residual_dtype="float16"),
            "int4_error_1_64": replace(base, error_fraction=1 / 64),
            "int4_error_1_32": replace(base, error_fraction=1 / 32),
            "int4_center_rank1": replace(base, center=True, residual_rank=1),
            "int4_rank1_error": replace(base, residual_rank=1, error_fraction=1 / 64),
            "int4_center_rotated": replace(base, center=True, rotation=True),
        }
        candidates = [
            entry
            for entry in _candidates()
            if entry[0] in ("int3_sym", "int4_sym", "int6_sym", "int8_sym", "int4_rotated", "int4_outliers")
        ]
        candidates.extend((name, spec, spec, 0, 0) for name, spec in additions.items())
        candidates.extend(
            [
                ("k_rank2_v_int4", replace(base, residual_rank=2), base, 0, 0),
                ("k_center_v_int4", replace(base, center=True), base, 0, 0),
                ("adaptive", base, base, 0, 0),
                ("adaptive_extended", base, base, 0, 0),
                ("adaptive_headwise", base, base, 0, 0),
            ]
        )
        extra_specs = tuple(replace(spec, scale_dtype="float16") for spec in additions.values())
        policies["adaptive_extended"] = (
            replace(key_policy, candidates=policy_specs + extra_specs),
            replace(value_policy, candidates=policy_specs + extra_specs),
        )
        policies["adaptive_headwise"] = tuple(
            replace(policy, headwise=True) for policy in policies["adaptive_extended"]
        )
        # Queries lie entirely in the uncompressed tail. Page calibration must
        # not use tokens later than any query whose output we evaluate.
        candidates = [(name, key, value, 32, sink) for name, key, value, _, sink in candidates]
    rows: list[dict] = []
    for distribution in distributions:
        logger.info("Exploring %s with %d candidates", distribution, len(candidates))
        query, key, value = _inputs(distribution, args.tokens, args.seed)
        reference = _dense_attention(query, key, value)
        baseline_bytes = (key.numel() + value.numel()) * key.element_size()
        for name, key_spec, value_spec, residual, sink in candidates:
            selected_key_policy, selected_value_policy = policies.get(name, (None, None))
            cache = QuantizedPageCache(
                key_spec,
                value_spec,
                page_size=32,
                residual_tokens=residual,
                sink_tokens=sink,
                key_policy=selected_key_policy,
                value_policy=selected_value_policy,
            )
            # Uneven appends exercise page sealing without adding a test suite.
            for start in range(0, len(key), 37):
                cache.append(key[start : start + 37], value[start : start + 37])
            decoded_key, decoded_value = cache.materialize()
            output = cache.attend(query, allow_retrospective=not args.mechanisms)
            key_error = (decoded_key.double() - key.double()).norm(dim=-1).amax().item()
            value_error = (decoded_value.double() - value.double()).norm(dim=-1).amax().item()
            logit_bound = query.double().norm(dim=-1).amax().item() * key_error / math.sqrt(key.shape[-1])
            output_bound = 2 * min(1.0, logit_bound) * value.double().norm(dim=-1).amax().item() + value_error
            rows.append(
                {
                    "distribution": distribution,
                    "candidate": name,
                    "key_spec": asdict(key_spec),
                    "value_spec": asdict(value_spec),
                    "residual_tokens": residual,
                    "sink_tokens": sink,
                    "resident_bytes": cache.nbytes(),
                    "byte_breakdown": cache.byte_breakdown(),
                    "compression_ratio": baseline_bytes / cache.nbytes(),
                    "effective_bits": cache.nbytes() * 8 / (key.numel() + value.numel()),
                    "key_relative_l2": _relative_error(decoded_key, key),
                    "value_relative_l2": _relative_error(decoded_value, value),
                    "attention_relative_l2": _relative_error(output, reference),
                    "max_key_vector_error": key_error,
                    "max_value_vector_error": value_error,
                    "attention_max_vector_error": (output.double() - reference.double()).norm(dim=-1).amax().item(),
                    "attention_absolute_bound": output_bound,
                    "page_formats": cache.page_report() if selected_key_policy else None,
                    "key_policy": asdict(selected_key_policy) if selected_key_policy else None,
                    "value_policy": asdict(selected_value_policy) if selected_value_policy else None,
                }
            )
    print(
        json.dumps(
            {
                "description": "Synthetic CPU experiment; not model accuracy or serving throughput",
                "torch_version": torch.__version__,
                "seed": args.seed,
                "evaluation_mode": "causal_tail_protected" if args.mechanisms else "retrospective_reconstruction",
                "shape": {"tokens": args.tokens, "kv_heads": 2, "query_heads": 4, "head_dim": 64, "query_tokens": 8},
                "source_dtype": "bfloat16",
                "page_size": 32,
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
