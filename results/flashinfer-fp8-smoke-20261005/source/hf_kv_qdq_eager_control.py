"""Compare BF16 and E4M3 K/V QDQ under identical local Qwen eager attention."""

from __future__ import annotations

import json
import pathlib

import torch
import torch.nn.functional as F
import transformers.models.qwen2.modeling_qwen2 as qwen2
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from tools.diagnose_int4_kv import _audit_causal_mask
from xllm.python.attention.quantized import KVCacheCodec


MODEL = "/mnt/e/AI/models/Qwen2.5-1.5B-Instruct"
OUT = pathlib.Path("results/flashinfer-fp8-smoke-20261005/hf_kv_qdq_eager_control.json")
PROMPT = "What is 27 times 43? Answer with the number only."
ANSWER = "1161"
SEED = 20261005
MAX_NEW_TOKENS = 24
ATTENTION_NAME = "xllm_fp8_eager_control_20261005"
MODES = ("bf16", "fixed_both", "dynamic_both", "fixed_k", "fixed_v")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    if isinstance(prompt_ids, list):
        prompt_ids = torch.tensor([prompt_ids], dtype=torch.long)
    prompt_ids = prompt_ids.to("cuda")
    answer_ids = tokenizer(ANSWER, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    teacher_input = torch.cat((prompt_ids, answer_ids), dim=-1)
    start = prompt_ids.shape[-1] - 1
    stop = start + answer_ids.shape[-1]
    state = {"mode": "bf16"}
    mask_audits: list[dict] = []
    codec = KVCacheCodec("fp8_e4m3", 128)
    fp8_dtype = torch.float8_e4m3fn

    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if query.shape[-2] > 1:
            mask_audits.append(_audit_causal_mask(attention_mask, query, key))
        mode = state["mode"]
        quant_key, quant_value = key, value
        if mode in ("fixed_both", "fixed_k"):
            quant_key = key.detach().float().clamp(-448, 448).to(fp8_dtype).to(key.dtype)
        if mode in ("fixed_both", "fixed_v"):
            quant_value = value.detach().float().clamp(-448, 448).to(fp8_dtype).to(value.dtype)
        if mode == "dynamic_both":
            key_payload, key_scale = codec.encode(key.detach().contiguous())
            value_payload, value_scale = codec.encode(value.detach().contiguous())
            quant_key = codec.decode(key_payload, key_scale).to(key.dtype)
            quant_value = codec.decode(value_payload, value_scale).to(value.dtype)
        return qwen2.eager_attention_forward(
            module, query, quant_key, quant_value, attention_mask, scaling, dropout, **kwargs
        )

    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_NAME, attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_NAME, ALL_MASK_ATTENTION_FUNCTIONS["eager"])

    model.config._attn_implementation = "eager"
    with torch.inference_mode():
        native_logits = model(input_ids=teacher_input, use_cache=False).logits[:, start:stop].float().detach().cpu()
    model.config._attn_implementation = ATTENTION_NAME
    logits: dict[str, torch.Tensor] = {}
    for mode in MODES:
        state["mode"] = mode
        with torch.inference_mode():
            logits[mode] = (
                model(input_ids=teacher_input, use_cache=False).logits[:, start:stop].float().detach().cpu()
            )
    reference = logits["bf16"]

    def compare(actual: torch.Tensor) -> dict[str, float]:
        delta = actual - reference
        return {
            "max_abs_vs_bf16": float(delta.abs().max().item()),
            "relative_l2_vs_bf16": float((delta.norm() / reference.norm().clamp_min(1e-30)).item()),
            "top1_agreement_vs_bf16": float((actual.argmax(-1) == reference.argmax(-1)).float().mean().item()),
        }

    def generate(mode: str) -> dict:
        state["mode"] = mode
        current = prompt_ids
        past = None
        generated: list[int] = []
        with torch.inference_mode():
            for _ in range(MAX_NEW_TOKENS):
                output = model(input_ids=current, past_key_values=past, use_cache=True)
                past = output.past_key_values
                token = int(output.logits[:, -1].argmax(dim=-1).item())
                generated.append(token)
                if token == tokenizer.eos_token_id:
                    break
                current = torch.tensor([[token]], device="cuda", dtype=torch.long)
        return {
            "token_count": len(generated),
            "text": tokenizer.decode(generated, skip_special_tokens=True),
            "ended_with_eos": bool(generated and generated[-1] == tokenizer.eos_token_id),
            "token_ids": generated,
        }

    results = {}
    for mode in MODES:
        logprob = F.log_softmax(logits[mode], dim=-1).gather(
            -1, answer_ids.cpu().unsqueeze(-1)
        ).squeeze(-1)
        results[mode] = {
            "teacher_forced_vs_bf16": compare(logits[mode]),
            "answer_mean_logprob": float(logprob.mean().item()),
        }
        if mode == "bf16":
            results[mode]["native_eager_wrapper_max_abs"] = float((logits[mode] - native_logits).abs().max().item())
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        results[mode]["greedy_24"] = generate(mode)

    if not mask_audits or not all(item["future_masked"] for item in mask_audits):
        raise RuntimeError("Causal mask audit failed")
    record = {
        "model": MODEL,
        "prompt": PROMPT,
        "answer": ANSWER,
        "prompt_tokens": int(prompt_ids.shape[-1]),
        "answer_token_count": int(answer_ids.shape[-1]),
        "max_new_tokens": MAX_NEW_TOKENS,
        "attention_math": "Qwen eager_attention_forward for every arm; only K/V QDQ differs",
        "causal_mask_audit_callbacks": len(mask_audits),
        "all_future_masked": True,
        "results": results,
    }
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print("artifact", OUT, "prompt_tokens", record["prompt_tokens"], "mask_callbacks", len(mask_audits))
    for mode in MODES:
        print(mode, json.dumps(results[mode], ensure_ascii=False))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
