"""Capture post-RoPE Q/K/V stats and FP8 E4M3 QDQ metrics for local Qwen."""

from __future__ import annotations

import json
import pathlib
import statistics

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from tools.diagnose_int4_kv import _audit_causal_mask
from xllm.python.attention.quantized import KVCacheCodec


MODEL = "/mnt/e/AI/models/Qwen2.5-1.5B-Instruct"
OUT = pathlib.Path("results/flashinfer-fp8-smoke-20261005/hf_kv_qdq.json")
PROMPT = "What is 27 times 43? Answer with the number only. Please reason carefully."
ATTENTION_NAME = "xllm_fp8_qdq_stats_20261005"


def distribution(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float().abs().reshape(-1)
    return {
        "min_abs": float(values.min().item()),
        "p50_abs": float(torch.quantile(values, 0.50).item()),
        "p99_abs": float(torch.quantile(values, 0.99).item()),
        "max_abs": float(values.max().item()),
    }


def qdq_metrics(source: torch.Tensor, decoded: torch.Tensor) -> dict[str, float | bool]:
    original = source.detach().float()
    actual = decoded.detach().float()
    nonzero = original.ne(0)
    return {
        "relative_l2": float(
            (torch.linalg.vector_norm(actual - original) / torch.linalg.vector_norm(original).clamp_min(1e-30)).item()
        ),
        "zeroed_nonzero_fraction": float(((actual == 0) & nonzero).sum().item() / nonzero.sum().clamp_min(1).item()),
        "max_abs_error": float((actual - original).abs().max().item()),
        "clipped_gt_448_fraction": float((original.abs() > 448).float().mean().item()),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()
    codec = KVCacheCodec("fp8_e4m3", 128)
    fp8_dtype = torch.float8_e4m3fn
    layers: dict[int, dict] = {}
    mask_audits: list[dict] = []

    def capture(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        layer = int(module.layer_idx)
        if query.shape[-2] > 1:
            mask_audits.append(_audit_causal_mask(attention_mask, query, key))
        entry = {"q_absnorm": distribution(torch.linalg.vector_norm(query.detach().float(), dim=-1))}
        for name, source in (("k", key), ("v", value)):
            source = source.detach().contiguous()
            payload, scale = codec.encode(source)
            dynamic = codec.decode(payload, scale)
            fixed = source.float().clamp(-448, 448).to(fp8_dtype).float()
            entry[name] = {
                "abs": distribution(source),
                "dynamic_scale": distribution(scale),
                "fixed_scale1": qdq_metrics(source, fixed),
                "dynamic_per_token_head": qdq_metrics(source, dynamic),
            }
        layers[layer] = entry
        return qwen2.eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_NAME, capture)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_NAME, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    model.config._attn_implementation = ATTENTION_NAME
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    if isinstance(tokens, list):
        tokens = torch.tensor([tokens], dtype=torch.long)
    with torch.inference_mode():
        model(input_ids=tokens.to("cuda"), use_cache=False)
    if sorted(layers) != list(range(28)):
        raise RuntimeError(f"Expected 28 captured layers; got {sorted(layers)}")
    if len(mask_audits) != 28 or not all(item["future_masked"] for item in mask_audits):
        raise RuntimeError("Causal mask audit failed")

    result = {
        "model": MODEL,
        "prompt": PROMPT,
        "prompt_tokens": int(tokens.shape[-1]),
        "capture": "post-RoPE Q/K/V through Qwen eager attention callback",
        "causal_mask_audit": {"callbacks": len(mask_audits), "all_future_masked": True},
        "layers": layers,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("artifact", OUT, "prompt_tokens", result["prompt_tokens"], "layers", len(layers))
    for name in ("k", "v"):
        for mode in ("fixed_scale1", "dynamic_per_token_head"):
            rel = [layers[index][name][mode]["relative_l2"] for index in layers]
            zero = [layers[index][name][mode]["zeroed_nonzero_fraction"] for index in layers]
            clipped = [layers[index][name][mode]["clipped_gt_448_fraction"] for index in layers]
            print(
                name,
                mode,
                "relative_l2_median/range",
                statistics.median(rel),
                min(rel),
                max(rel),
                "zeroed_nonzero_median",
                statistics.median(zero),
                "max_clipped_fraction",
                max(clipped),
            )
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
