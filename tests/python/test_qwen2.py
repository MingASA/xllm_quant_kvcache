# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xllm.python import kernels
from xllm.python.layers import Attention
from xllm.python.models.qwen2 import Qwen2Attention, Qwen2Config, Qwen2ForCausalLM, Qwen2Model


def _config_dict(**overrides: object) -> dict:
    values = {
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "intermediate_size": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 16,
        "vocab_size": 8,
        "tie_word_embeddings": True,
        "tp_size": 1,
        "tp_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def _tiny_lm() -> Qwen2ForCausalLM:
    config = _config_dict()
    model = Qwen2ForCausalLM.__new__(Qwen2ForCausalLM)
    torch.nn.Module.__init__(model)
    model.cfg = Qwen2Config.from_dict(config)
    model.dtype = torch.float32
    model.device = torch.device("cpu")
    model.model = Qwen2Model(model.cfg, torch.float32, torch.device("cpu"))
    from xllm.python.layers import ColumnParallelLinear

    model.lm_head = ColumnParallelLinear(4, 8, 1, gather_output=True, dtype=torch.float32, device=torch.device("cpu"))
    return model


def test_hf_config_defaults_bias_head_dim_and_disables_sliding_window() -> None:
    cfg = Qwen2Config.from_dict(_config_dict(sliding_window=128, max_window_layers=1))

    assert cfg.attention_bias is True
    assert cfg.head_dim == 4
    assert cfg.sliding_window == 0
    assert cfg.n_kv_heads == 1


def test_direct_and_empty_dict_config_defaults_match() -> None:
    direct = Qwen2Config()
    from_empty_dict = Qwen2Config.from_dict({})

    assert direct == from_empty_dict
    assert direct.hidden_size == 1536
    assert direct.n_heads == 12
    assert direct.n_kv_heads == 2
    assert direct.intermediate_size == 8960
    assert direct.max_position_embeddings == 32768
    assert direct.attention_bias is True


def test_config_accepts_unscaled_cpp_rope_sentinels() -> None:
    cfg = Qwen2Config.from_dict(_config_dict(rope_scaling=-1, rope_scaling_rope_type="", rope_scaling_factor=0))

    assert cfg.rope_theta == 10000.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"rope_scaling_rope_type": "yarn"},
        {"rope_scaling_factor": 1.5},
        {"rope_scaling_rope_type": "yarn", "rope_scaling_factor": 1.5},
    ],
)
def test_config_rejects_scaled_rope(overrides: dict[str, object]) -> None:
    with pytest.raises(NotImplementedError, match="scaled RoPE"):
        Qwen2Config.from_dict(_config_dict(rope_scaling=-1, **overrides))


def test_config_rejects_layer_dependent_sliding_window() -> None:
    with pytest.raises(NotImplementedError, match="sliding windows"):
        Qwen2Config.from_dict(_config_dict(use_sliding_window=True))


def test_attention_has_qkv_bias_but_no_qk_norm_or_o_bias() -> None:
    cfg = Qwen2Config.from_dict(_config_dict())
    attention = Qwen2Attention(cfg, 0, torch.float32, torch.device("cpu"))

    assert attention.qkv_proj.bias is not None
    assert attention.o_proj.bias is None
    assert not hasattr(attention, "q_norm")
    assert not hasattr(attention, "k_norm")
    assert attention.attn.sliding_window == 0


def test_model_rope_cache_is_fp32() -> None:
    cfg = Qwen2Config.from_dict(_config_dict())
    model = Qwen2Model(cfg, torch.bfloat16, torch.device("cpu"))

    assert model.rotary.cos_sin_cache.dtype == torch.float32


def test_attention_uses_neox_rope_and_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Qwen2Config.from_dict(_config_dict())
    attention = Qwen2Attention(cfg, 0, torch.float32, torch.device("cpu"))
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with torch.no_grad():
        attention.qkv_proj.weight.copy_(torch.cat((torch.eye(4), 2 * torch.eye(4), 3 * torch.eye(4))))
        attention.qkv_proj.bias.copy_(torch.arange(12, dtype=torch.float32))
        attention.o_proj.weight.copy_(torch.eye(4))

    observed: dict[str, object] = {}

    def standard_rope(
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        head_dim: int,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed["positions"] = positions
        observed["head_dim"] = head_dim
        observed["is_neox"] = True
        half = head_dim // 2
        cos = torch.cat((cos_sin_cache[positions, :half], cos_sin_cache[positions, :half]), dim=-1)
        sin = torch.cat((cos_sin_cache[positions, half:], cos_sin_cache[positions, half:]), dim=-1)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x = x.view(-1, head_dim)
            first, second = x[:, :half], x[:, half:]
            rotated = torch.cat((-second, first), dim=-1)
            return (x * cos + rotated * sin).view_as(q if x.shape == q.shape else k)

        return rotate(q), rotate(k)

    # Keep a direct, readable reference for the standard half-split NeoX rotation.
    def reference(x: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        cos = torch.cat((cos_sin_cache[1, :half], cos_sin_cache[1, :half]))
        sin = torch.cat((cos_sin_cache[1, half:], cos_sin_cache[1, half:]))
        first, second = x[..., :half], x[..., half:]
        rotated = torch.cat((-second, first), dim=-1)
        return x * cos + rotated * sin

    monkeypatch.setattr(kernels, "standard_rope", standard_rope, raising=False)
    captured: dict[str, torch.Tensor] = {}

    def capture_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        captured.update(q=q, k=k, v=v)
        return v

    attention.attn.forward = capture_attention
    cache = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.8, 0.6, 0.9, 0.3]])
    positions = torch.tensor([1])
    output = attention(positions, hidden, cache, None, None)

    qkv = torch.nn.functional.linear(hidden, attention.qkv_proj.weight, attention.qkv_proj.bias)
    q, k, v = qkv.split((4, 4, 4), dim=-1)
    torch.testing.assert_close(captured["q"], reference(q, cache))
    torch.testing.assert_close(captured["k"], reference(k, cache))
    torch.testing.assert_close(captured["v"], v)
    torch.testing.assert_close(output, v)
    assert observed["is_neox"] is True
    assert observed["head_dim"] == 4
    assert observed["positions"] is positions


def test_load_weights_preserves_qkv_bias_and_tied_lm_head(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_lm()
    monkeypatch.setattr(kernels, "prepare_row_parallel_weight", lambda weight: (weight, False), raising=False)
    monkeypatch.setattr(model.model.layers[0].self_attn.o_proj, "process_weights_after_loading", lambda: None)
    monkeypatch.setattr(model.model.layers[0].mlp.down_proj, "process_weights_after_loading", lambda: None)
    tensors = {
        "embed_tokens.weight": torch.arange(32, dtype=torch.float32).view(8, 4),
        "layers.0.input_layernorm.weight": torch.ones(4),
        "layers.0.post_attention_layernorm.weight": torch.ones(4),
        "layers.0.self_attn.q_proj.weight": torch.full((4, 4), 1.0),
        "layers.0.self_attn.k_proj.weight": torch.full((4, 4), 2.0),
        "layers.0.self_attn.v_proj.weight": torch.full((4, 4), 3.0),
        "layers.0.self_attn.q_proj.bias": torch.full((4,), 4.0),
        "layers.0.self_attn.k_proj.bias": torch.full((4,), 5.0),
        "layers.0.self_attn.v_proj.bias": torch.full((4,), 6.0),
        "layers.0.self_attn.o_proj.weight": torch.eye(4),
        "layers.0.mlp.gate_proj.weight": torch.ones((8, 4)),
        "layers.0.mlp.up_proj.weight": torch.ones((8, 4)) * 2,
        "layers.0.mlp.down_proj.weight": torch.ones((4, 8)) * 3,
        "norm.weight": torch.ones(4),
    }

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    attention = model.model.layers[0].self_attn
    torch.testing.assert_close(attention.qkv_proj.bias, torch.tensor([4.0] * 4 + [5.0] * 4 + [6.0] * 4))
    torch.testing.assert_close(attention.qkv_proj.weight[:4], tensors["layers.0.self_attn.q_proj.weight"])
    torch.testing.assert_close(attention.qkv_proj.weight[4:8], tensors["layers.0.self_attn.k_proj.weight"])
    torch.testing.assert_close(attention.qkv_proj.weight[8:], tensors["layers.0.self_attn.v_proj.weight"])
    torch.testing.assert_close(model.lm_head.weight, tensors["embed_tokens.weight"])


@pytest.mark.parametrize(("num_layers", "tie_word_embeddings"), [(1, True), (1, False), (2, True), (2, False)])
def test_tiny_model_logits_match_transformers_with_causal_gqa(
    monkeypatch: pytest.MonkeyPatch,
    num_layers: int,
    tie_word_embeddings: bool,
) -> None:
    transformers = pytest.importorskip("transformers")
    from transformers import Qwen2Config as HfQwen2Config
    from transformers import Qwen2ForCausalLM as HfQwen2ForCausalLM

    from xllm.python.layers import ColumnParallelLinear
    from xllm.python.layers import attention as attention_module
    from xllm.python.models import qwen3

    config = _config_dict(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=12,
        head_dim=4,
        vocab_size=11,
        max_position_embeddings=16,
        num_hidden_layers=num_layers,
        tie_word_embeddings=tie_word_embeddings,
    )
    torch.manual_seed(20261001)
    hf_config = HfQwen2Config(
        vocab_size=11,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        tie_word_embeddings=tie_word_embeddings,
        use_sliding_window=False,
        _attn_implementation="eager",
    )
    hf_model = HfQwen2ForCausalLM(hf_config).eval()

    xllm_model = Qwen2ForCausalLM.__new__(Qwen2ForCausalLM)
    torch.nn.Module.__init__(xllm_model)
    xllm_model.cfg = Qwen2Config.from_dict(config)
    xllm_model.dtype = torch.float32
    xllm_model.device = torch.device("cpu")
    xllm_model.model = Qwen2Model(xllm_model.cfg, torch.float32, torch.device("cpu"))
    xllm_model.lm_head = ColumnParallelLinear(
        8, 11, 1, gather_output=True, dtype=torch.float32, device=torch.device("cpu")
    )
    for layer in xllm_model.model.layers:
        monkeypatch.setattr(layer.self_attn.o_proj, "process_weights_after_loading", lambda: None)
        monkeypatch.setattr(layer.mlp.down_proj, "process_weights_after_loading", lambda: None)
    monkeypatch.setattr(kernels, "prepare_row_parallel_weight", lambda weight: (weight, False), raising=False)
    xllm_model.load_weights([_StateDict(hf_model.state_dict())], tp_rank=0, tp_size=1)

    def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        normalized = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normalized * weight.float()).to(hidden.dtype)

    def fused_add_rms_norm(
        hidden: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden + residual
        return rms_norm(residual, weight, eps), residual

    monkeypatch.setattr(kernels, "rms_norm", rms_norm, raising=False)
    monkeypatch.setattr(kernels, "fused_add_rms_norm", fused_add_rms_norm, raising=False)
    monkeypatch.setattr(
        kernels,
        "silu_and_mul",
        lambda value: torch.nn.functional.silu(value.chunk(2, -1)[0]) * value.chunk(2, -1)[1],
        raising=False,
    )

    def apply_neox_rope(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_dim: int,
        cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half = head_dim // 2
        cos_sin = cache[positions]
        cos = torch.cat((cos_sin[:, :half], cos_sin[:, :half]), dim=-1)
        sin = torch.cat((cos_sin[:, half:], cos_sin[:, half:]), dim=-1)

        def rotate(value: torch.Tensor) -> torch.Tensor:
            value = value.view(value.shape[0], -1, head_dim)
            first, second = value[..., :half], value[..., half:]
            rotated = torch.cat((-second, first), dim=-1)
            return (value * cos[:, None, :] + rotated * sin[:, None, :]).flatten(1)

        return rotate(query), rotate(key)

    monkeypatch.setattr(kernels, "standard_rope", apply_neox_rope, raising=False)

    class _CausalGqaBackend:
        def execute(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            layer: Attention,
        ) -> torch.Tensor:
            q = query.view(-1, layer.num_heads, layer.head_dim).transpose(0, 1)
            k = key.view(-1, layer.num_kv_heads, layer.head_dim).transpose(0, 1)
            v = value.view(-1, layer.num_kv_heads, layer.head_dim).transpose(0, 1)
            repeats = layer.num_heads // layer.num_kv_heads
            k = k.repeat_interleave(repeats, dim=0)
            v = v.repeat_interleave(repeats, dim=0)
            scores = torch.matmul(q, k.transpose(-1, -2)) * layer.scale
            causal_mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
            attended = torch.matmul(scores.softmax(dim=-1), v)
            return attended.transpose(0, 1).reshape(-1, layer.num_heads * layer.head_dim)

    context = SimpleNamespace(cp_context=None, attention_backend=_CausalGqaBackend())
    monkeypatch.setattr(qwen3, "get_forward_context", lambda: context)
    monkeypatch.setattr(attention_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(qwen3, "record_layer_event", lambda _layer_id: None)
    input_ids = torch.tensor([2, 5, 1, 7])
    positions = torch.arange(input_ids.numel(), dtype=torch.int32)

    with torch.no_grad():
        xllm_hidden = xllm_model.model(input_ids, positions)
        actual_logits = xllm_model.compute_logits(xllm_hidden, None)
        expected_logits = hf_model(input_ids=input_ids.unsqueeze(0), use_cache=False).logits.squeeze(0)

    torch.testing.assert_close(actual_logits, expected_logits, rtol=2e-5, atol=2e-5)
