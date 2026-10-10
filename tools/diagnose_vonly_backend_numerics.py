# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Isolate V-cache rounding and propagate fixed-history attention differences.

The checkpoint shell is Transformers Qwen2, with every attention replaced by
the real xLLM backend. This diagnostic does not measure server performance.
BF16 shadow caches are references only, never part of the compressed service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from scripts.logger import logger
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.flashinfer_mixed import FlashInferMixedVBackend
from xllm.python.attention.quantized import KVCacheCodec, QuantizedPagedAttentionBackend


def _error(a: torch.Tensor, b: torch.Tensor) -> dict:
    delta = a.float() - b.float()
    return dict(max_abs=delta.abs().max().item(), rms=delta.square().mean().sqrt().item())


def _metadata(query_len: int, context: int, batch: int = 1) -> SimpleNamespace:
    page_count = (context + 127) // 128
    blocks = torch.arange(batch * page_count, dtype=torch.int32, device="cuda").reshape(batch, page_count)
    positions = torch.arange(context - query_len, context, device="cuda")
    offsets = torch.arange(batch + 1, dtype=torch.int32) * query_len
    return SimpleNamespace(
        q_seq_lens_host=offsets,
        qo_indptr=offsets.cuda(),
        q_cu_seq_lens=None,
        kv_seq_lens_host_values=[context] * batch,
        kv_seq_lens_host=None,
        paged_kv_indptr=(torch.arange(batch + 1, dtype=torch.int32) * page_count).cuda(),
        paged_kv_indices=blocks.flatten(),
        paged_kv_last_page_len=torch.full((batch,), (context - 1) % 128 + 1, dtype=torch.int32, device="cuda"),
        slot_mapping=(blocks[:, positions // 128].long() * 128 + positions % 128).flatten(),
        block_table=blocks,
    )


class Probe:
    def __init__(self, arm: str, layers: int, capacity: int) -> None:
        self.codec = KVCacheCodec(arm.removeprefix("bf16_"), 128)
        self.backends = {
            "triton": QuantizedPagedAttentionBackend(arm, 128, 2),
            "flashinfer_mixed": FlashInferMixedVBackend(
                arm, 12, 2, 128, 128**-0.5, 0, torch.device("cuda", 0), torch.bfloat16
            ),
            "bf16_shadow": FlashInferMixedVBackend(
                "auto", 12, 2, 128, 128**-0.5, 0, torch.device("cuda", 0), torch.bfloat16
            ),
        }
        self.caches, self.shadows = [], []
        for _ in range(layers):
            key = torch.empty(capacity // 128, 128, 2, 128, dtype=torch.bfloat16, device="cuda")
            value = torch.empty(*key.shape[:-1], self.codec.storage_dim, dtype=self.codec.storage_dtype, device="cuda")
            scales = torch.empty(key.shape[:-1], dtype=torch.float32, device="cuda")
            self.caches.append(LayerCache(key, value, value_scale=scales))
            self.shadows.append(LayerCache(key, torch.empty_like(key)))
        for name, backend in self.backends.items():
            backend.bind_kv_caches(self.shadows if name == "bf16_shadow" else self.caches)
        self.records = []

    def prepare(self, query_len: int, context: int, selected: str, step: int) -> None:
        self.metadata = _metadata(query_len, context)
        self.context, self.selected, self.step = context, selected, step
        for backend in self.backends.values():
            backend.prepare(self.metadata)

    def forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        del kwargs
        count = hidden_states.shape[1]
        q = module.q_proj(hidden_states).reshape(1, count, 12, 128).transpose(1, 2)
        k = module.k_proj(hidden_states).reshape(1, count, 2, 128).transpose(1, 2)
        v = module.v_proj(hidden_states).reshape(count, 2, 128)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        q = q.transpose(1, 2).reshape(count, 12, 128).contiguous()
        k = k.transpose(1, 2).reshape(count, 2, 128).contiguous()
        layer = SimpleNamespace(
            num_heads=12,
            num_kv_heads=2,
            head_dim=128,
            layer_id=module.layer_idx,
            scale=128**-0.5,
            causal=True,
            sliding_window=0,
        )
        selected = self.backends[self.selected].execute(q.flatten(1), k, v, layer)
        other_name = "triton" if self.selected == "flashinfer_mixed" else "flashinfer_mixed"
        other = self.backends[other_name].execute_attention(q, layer)
        native = selected if self.selected == "flashinfer_mixed" else other
        triton = selected if self.selected == "triton" else other
        cache, shadow = self.caches[module.layer_idx], self.shadows[module.layer_idx]
        slots = self.metadata.slot_mapping
        decoded_new = self.codec.decode(cache.value.flatten(0, 1)[slots], cache.value_scale.flatten(0, 1)[slots])
        shadow.value.flatten(0, 1)[slots] = decoded_new.bfloat16()
        shadow_output = self.backends["bf16_shadow"].execute_attention(q, layer)
        record = dict(
            step=self.step,
            layer=module.layer_idx,
            native_vs_triton=_error(native, triton),
            native_vs_bf16_shadow=_error(native, shadow_output),
        )
        if module.layer_idx in (0, 13, 27) and self.step in (0, 1, 8, 32):
            keys = cache.key.flatten(0, 1)[: self.context].float().repeat_interleave(6, 1)
            values = self.codec.decode(
                cache.value.flatten(0, 1)[: self.context], cache.value_scale.flatten(0, 1)[: self.context]
            ).repeat_interleave(6, 1)
            weights = torch.einsum("qhd,khd->hqk", q[-1:].float(), keys).mul(128**-0.5).softmax(-1)
            reference = torch.einsum("hqk,khd->qhd", weights, values).flatten(1)
            rounded = torch.einsum("hqk,khd->qhd", weights, values.bfloat16().float()).flatten(1)
            record.update(
                native_vs_fp32=_error(native[-1:], reference),
                triton_vs_fp32=_error(triton[-1:], reference),
                v_rounding_alone=_error(rounded, reference),
            )
        self.records.append(record)
        return module.o_proj(selected.reshape(1, count, 1536)), None


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/mnt/e/AI/models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "results/flashinfer-vs-triton-vonly-c20-gsm100-20261010"
    trace = [json.loads(line) for line in (source / "trace_gsm8k.jsonl").read_text().splitlines()]
    responses = {
        r["index"]: r
        for r in (
            json.loads(line)
            for line in (source / "triton/bf16_int8/gsm8k/c20_r0/responses.jsonl").read_text().splitlines()
        )
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager"
        )
        .cuda()
        .eval()
    )
    results = []
    for arm in ("bf16_int8", "bf16_int4"):
        probe = Probe(arm, 28, 2048)
        for module in model.model.layers:
            attention = module.self_attn

            def _forward(
                instance: torch.nn.Module,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                _probe: Probe = probe,
                **kwargs: object,
            ) -> tuple:
                return _probe.forward(instance, hidden_states, position_embeddings, **kwargs)

            attention.forward = MethodType(_forward, attention)
        for index in range(args.samples):
            prompt = tokenizer.encode(trace[index]["prompt"], add_special_tokens=False)
            forced = tokenizer.encode(responses[index]["prediction"], add_special_tokens=False)[: args.steps]
            branches = {}
            for name in ("triton", "flashinfer_mixed"):
                start = len(probe.records)
                logits = []
                for step in range(len(forced) + 1):
                    ids = prompt if step == 0 else [forced[step - 1]]
                    context = len(prompt) + step
                    probe.prepare(len(ids), context, name, step)
                    output = model(
                        input_ids=torch.tensor([ids], device="cuda"),
                        position_ids=torch.arange(context - len(ids), context, device="cuda")[None],
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    logits.append(output.logits[0, -1].float().cpu())
                branches[name] = torch.stack(logits)
                (args.output / f"{arm}_sample{index}_{name}_layers.json").write_text(
                    json.dumps(probe.records[start:], indent=2) + "\n"
                )
            a, b = branches["triton"], branches["flashinfer_mixed"]
            top = a.topk(2, dim=-1)
            flips = (a.argmax(-1) != b.argmax(-1)).nonzero().flatten().tolist()
            result = dict(
                arm=arm,
                sample=index,
                prompt_tokens=len(prompt),
                forced_tokens=forced,
                logit_error=_error(a, b),
                top1_flips=flips,
                compared_positions=len(forced) + 1,
                triton_top1=a.argmax(-1).tolist(),
                native_top1=b.argmax(-1).tolist(),
                triton_top2_margin=(top.values[:, 0] - top.values[:, 1]).tolist(),
                max_logit_difference_per_step=(a - b).abs().amax(-1).tolist(),
            )
            results.append(result)
            (args.output / "fixed_history.json").write_text(json.dumps(results, indent=2) + "\n")
            logger.info(
                "Fixed history checked: %s sample %d, %d/%d next-token flips", arm, index, len(flips), len(forced) + 1
            )
        del probe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
