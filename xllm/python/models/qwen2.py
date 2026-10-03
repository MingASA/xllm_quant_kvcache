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

"""Qwen2/2.5 dense CUDA executor with paged backend-owned attention.

Unlike Qwen3, Qwen2 has biased QKV projections and no query/key RMSNorm.
The decoder residual flow and MLP are shared, but attention and checkpoint
loading preserve these architectural differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers import Attention, ColumnParallelLinear, GatedMLP, RMSNorm, RotaryEmbedding, RowParallelLinear
from xllm.python.models.base import PyModelBase
from xllm.python.models.qwen3 import Qwen3Config, Qwen3DecoderLayer, Qwen3Model
from xllm.python.models.weight_utils import WeightLoader, kv_replica_shard

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


@dataclass
class Qwen2Config(Qwen3Config):
    hidden_size: int = 1536
    n_heads: int = 12
    n_kv_heads: int = 2
    intermediate_size: int = 8960
    max_position_embeddings: int = 32768
    attention_bias: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> Qwen2Config:
        values = dict(config)
        values.setdefault("hidden_size", 1536)
        values.setdefault("n_heads", values.get("num_attention_heads", 12))
        values.setdefault("n_kv_heads", values.get("num_key_value_heads", 2))
        values.setdefault("intermediate_size", 8960)
        values.setdefault("attention_bias", True)
        values.setdefault("max_position_embeddings", 32768)
        hidden_size = int(values.get("hidden_size", 1536))
        n_heads = int(values.get("n_heads", values.get("num_attention_heads", 12)))
        if not values.get("head_dim"):
            values["head_dim"] = hidden_size // n_heads
        if values.get("use_sliding_window", False):
            raise NotImplementedError("Qwen2 Python executor does not support layer-dependent sliding windows")
        # C++ reflects ModelArgs.rope_scaling=-1 even for an unscaled model;
        # HF instead omits the field or uses a dictionary. Check both forms.
        if (
            values.get("rope_scaling") not in (None, -1, 0, {})
            or values.get("rope_scaling_rope_type", "") not in ("", "default")
            or values.get("rope_scaling_factor", 0) not in (0, 1)
        ):
            raise NotImplementedError("Qwen2 Python executor does not support scaled RoPE")
        values["sliding_window"] = 0
        return super().from_dict(values)


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        cfg: Qwen2Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads, self.num_kv_heads = cfg.head_split()
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            self.q_size + 2 * self.kv_size,
            cfg.tp_size,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            cfg.tp_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.head_dim**-0.5,
            sliding_window=0,
            layer_id=layer_id,
            causal=causal,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None = None,
    ) -> torch.Tensor:
        del cos, sin
        if mrope_section is not None or positions.ndim != 1:
            raise ValueError("Qwen2 requires one-dimensional standard RoPE positions")
        qkv = self.qkv_proj(hidden)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = kernels.standard_rope(
            positions,
            q,
            k,
            self.head_dim,
            cos_sin_cache,
        )
        return self.o_proj(self.attn(q, k, v))


class Qwen2DecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        cfg: Qwen2Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = Qwen2Attention(cfg, layer_id, dtype, device, causal=causal)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.mlp = GatedMLP(cfg.hidden_size, cfg.intermediate_size, cfg.tp_size, dtype, device)


class Qwen2Model(Qwen3Model):
    def __init__(self, cfg: Qwen2Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(cfg, dtype, device, decoder_layer_type=Qwen2DecoderLayer)
        # FlashInfer standard RoPE requires an FP32 cache. Build from FP32
        # frequencies, not by promoting the already-rounded model-dtype table.
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=torch.float32,
            device=device,
        )


class Qwen2ForCausalLM(PyModelBase):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Qwen2Config.from_dict(config)
        self.dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        self.device = torch.device(config.get("device", "cuda"))
        if self.device.type != "cuda":
            raise NotImplementedError("Qwen2 Python executor currently supports CUDA only")
        cfg = self.cfg
        if cfg.tp_size * cfg.dp_size != int(config.get("world_size", cfg.tp_size * cfg.dp_size)):
            raise ValueError("world_size must equal tp_size * dp_size")
        if not 0 <= cfg.dp_rank < cfg.dp_size:
            raise ValueError("dp_rank must be in [0, dp_size)")
        if cfg.vocab_size % cfg.tp_size:
            raise ValueError("vocab_size must be divisible by tp_size")
        self.model = Qwen2Model(cfg, self.dtype, self.device)
        self.lm_head = ColumnParallelLinear(
            cfg.hidden_size,
            cfg.vocab_size // cfg.tp_size,
            cfg.tp_size,
            gather_output=True,
            dtype=self.dtype,
            device=self.device,
        )

    def load_weights(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> None:
        cfg = self.cfg
        kv_world, kv_rank = kv_replica_shard(cfg.n_kv_heads, tp_rank, tp_size)
        loader = WeightLoader(self, state_dicts, tp_size, tp_rank, src_prefixes=("model.", ""))
        loader.copy_shard("model.embed_tokens.weight", dim=1)
        for i, layer in enumerate(self.model.layers):
            src = f"layers.{i}."
            dst = f"model.layers.{i}."
            for norm in ("input_layernorm", "post_attention_layernorm"):
                loader.copy_in(dst + norm + ".weight", loader.load_tensor(src + norm + ".weight"))
            for suffix in ("weight", "bias") if cfg.attention_bias else ("weight",):
                q = loader.load_shard(src + "self_attn.q_proj." + suffix, 0)
                k = loader.load_shard(src + "self_attn.k_proj." + suffix, 0, world=kv_world, rank=kv_rank)
                v = loader.load_shard(src + "self_attn.v_proj." + suffix, 0, world=kv_world, rank=kv_rank)
                loader.copy_in(dst + "self_attn.qkv_proj." + suffix, torch.cat([q, k, v], dim=0))
            loader.copy_in(dst + "self_attn.o_proj.weight", loader.load_shard(src + "self_attn.o_proj.weight", 1))
            loader.load_gated_mlp(dst + "mlp.", src + "mlp.")
            layer.self_attn.o_proj.process_weights_after_loading()
            layer.mlp.down_proj.process_weights_after_loading()
        loader.copy_replicated("model.norm.weight")
        lm_name = "embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
        loader.copy_in("lm_head.weight", loader.load_shard(lm_name, dim=0))
