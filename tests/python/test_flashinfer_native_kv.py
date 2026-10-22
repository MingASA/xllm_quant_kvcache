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

"""Small native FlashInfer GQA=5 checks for BF16 queries and FP8 pages."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("FlashInfer tests require CUDA", allow_module_level=True)
pytest.importorskip("flashinfer", reason="FlashInfer is not installed")

from tools.benchmark_kv_cache import _attention_metadata, _dense
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.flashinfer_fp8 import FlashInferFP8Backend
from xllm.python.attention.quantized import KVCacheCodec


@pytest.fixture
def gqa5_backend(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setenv("XLLM_FLASHINFER_NATIVE_KV", "1")
    torch.manual_seed(20261022)
    device = torch.device("cuda")
    heads, kv_heads, dim, page_size = 40, 8, 128, 128
    shape = (2, page_size, kv_heads, dim)
    cache = LayerCache(
        torch.zeros(shape, dtype=torch.uint8, device=device),
        torch.zeros(shape, dtype=torch.uint8, device=device),
        key_scale=torch.ones(shape[:-1], dtype=torch.float32, device=device),
        value_scale=torch.ones(shape[:-1], dtype=torch.float32, device=device),
    )
    backend = FlashInferFP8Backend(
        heads,
        kv_heads,
        dim,
        dim**-0.5,
        -1,
        device,
        torch.bfloat16,
        "fp8_e4m3",
        key_scale=0.5,
        value_scale=2.0,
    )
    backend.bind_kv_caches([cache])
    assert backend._decode_use_tensor_cores
    return SimpleNamespace(
        backend=backend,
        cache=cache,
        pages=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        heads=heads,
        kv_heads=kv_heads,
        dim=dim,
        page_size=page_size,
        device=device,
    )


def _decoded_prefix(
    state: SimpleNamespace, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    codec = KVCacheCodec("fp8_e4m3", state.dim)
    pages = state.pages.long()
    decoded_key = codec.decode(state.cache.key, state.cache.key_scale)[pages].reshape(1, -1, state.kv_heads, state.dim)
    decoded_value = codec.decode(state.cache.value, state.cache.value_scale)[pages].reshape(
        1, -1, state.kv_heads, state.dim
    )
    length = key.shape[1]
    return decoded_key[:, :length], decoded_value[:, :length]


def test_gqa5_native_fp8_paged_prefill_scale_and_partial_page(gqa5_backend: SimpleNamespace) -> None:
    state = gqa5_backend
    length = state.page_size + 1
    key = torch.randn(1, length, state.kv_heads, state.dim, dtype=torch.bfloat16, device=state.device) * 0.2
    value = torch.randn_like(key) * 0.2
    query = torch.randn(1, length, state.heads, state.dim, dtype=torch.bfloat16, device=state.device)
    slots = torch.arange(length, dtype=torch.int64, device=state.device)
    metadata = _attention_metadata(state.pages, slots, length, length, state.page_size)
    metadata.is_prefill = True
    metadata.is_chunked_prefill = False
    state.backend.prepare(metadata)
    layer = SimpleNamespace(layer_id=0, num_heads=state.heads, num_kv_heads=state.kv_heads, head_dim=state.dim)

    actual = state.backend.execute(query.flatten(0, 1), key.flatten(0, 1), value.flatten(0, 1), layer)

    decoded_key, decoded_value = _decoded_prefix(state, key, value)
    expected = _dense(query.float(), decoded_key, decoded_value)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, rtol=0.03, atol=0.008)
    assert torch.all(state.cache.key_scale[0, :].flatten()[: state.kv_heads * state.page_size] == 0.5)
    assert torch.all(state.cache.value_scale.flatten()[: length * state.kv_heads] == 2.0)
    before = tuple(
        tensor.clone()
        for tensor in (state.cache.key, state.cache.value, state.cache.key_scale, state.cache.value_scale)
    )
    state.backend.write_cache(
        key[0, :1],
        value[0, :1],
        torch.tensor([-1], dtype=torch.int64, device=state.device),
        0,
    )
    for previous, current in zip(
        before, (state.cache.key, state.cache.value, state.cache.key_scale, state.cache.value_scale)
    ):
        assert torch.equal(previous, current)


def test_gqa5_native_fp8_tensor_core_decode_scale(gqa5_backend: SimpleNamespace) -> None:
    state = gqa5_backend
    prefix_length = state.page_size + 1
    key = torch.randn(1, prefix_length + 1, state.kv_heads, state.dim, dtype=torch.bfloat16, device=state.device) * 0.2
    value = torch.randn_like(key) * 0.2
    prefix_slots = torch.arange(prefix_length, dtype=torch.int64, device=state.device)
    state.backend.write_cache(key[0, :prefix_length], value[0, :prefix_length], prefix_slots, 0)

    metadata = _attention_metadata(
        state.pages,
        torch.tensor([prefix_length], dtype=torch.int64, device=state.device),
        prefix_length + 1,
        1,
        state.page_size,
    )
    state.backend.prepare(metadata)
    query = torch.randn(1, state.heads, state.dim, dtype=torch.bfloat16, device=state.device)
    layer = SimpleNamespace(layer_id=0, num_heads=state.heads, num_kv_heads=state.kv_heads, head_dim=state.dim)

    actual = state.backend.execute(query, key[0, prefix_length], value[0, prefix_length], layer)

    decoded_key, decoded_value = _decoded_prefix(state, key, value)
    expected = _dense(query[:, None].float(), decoded_key, decoded_value)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, rtol=0.03, atol=0.008)
    assert state.backend._decode_wrapper._backend == "fa2"
    assert torch.all(state.cache.key_scale[1, 1, :] == 0.5)
    assert torch.all(state.cache.value_scale[1, 1, :] == 2.0)


def test_gqa5_native_fp8_decode_cuda_graph_replans_and_replays(gqa5_backend: SimpleNamespace) -> None:
    state = gqa5_backend
    prefix_length = state.page_size + 1
    key = torch.randn(1, prefix_length + 2, state.kv_heads, state.dim, dtype=torch.bfloat16, device=state.device)
    value = torch.randn_like(key) * 0.2
    prefix_slots = torch.arange(prefix_length, dtype=torch.int64, device=state.device)
    state.backend.write_cache(key[0, :prefix_length], value[0, :prefix_length], prefix_slots, 0)
    metadata = _attention_metadata(
        state.pages,
        torch.tensor([prefix_length], dtype=torch.int64, device=state.device),
        prefix_length + 1,
        1,
        state.page_size,
    )
    metadata.paged_kv_indices = metadata.paged_kv_indices.contiguous()
    metadata.paged_kv_last_page_len_host.fill_(2)
    metadata.paged_kv_last_page_len.fill_(2)
    state.backend.prepare(metadata, graph_mode=True)
    assert state.backend._active_decode_wrapper is state.backend._graph_decode_wrappers[1]
    cache_views = state.backend._fp8_caches[0]
    assert cache_views[0].data_ptr() == state.cache.key.data_ptr()
    assert cache_views[1].data_ptr() == state.cache.value.data_ptr()

    query = torch.randn(1, state.heads, state.dim, dtype=torch.bfloat16, device=state.device)
    current_key = key[:, prefix_length].contiguous()
    current_value = value[:, prefix_length].contiguous()
    layer = SimpleNamespace(layer_id=0, num_heads=state.heads, num_kv_heads=state.kv_heads, head_dim=state.dim)

    # Resolve kernels and graph planner before capture.
    state.backend.execute(query, current_key, current_value, layer)
    torch.cuda.synchronize()
    static_output = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = state.backend.execute(query, current_key, current_value, layer)

    # Reuse every metadata address but advance the decode by one slot/page token.
    metadata.slot_mapping.fill_(prefix_length + 1)
    metadata.paged_kv_last_page_len_host.fill_(3)
    metadata.paged_kv_last_page_len.fill_(3)
    query.copy_(torch.randn_like(query))
    current_key.copy_(key[:, prefix_length + 1])
    current_value.copy_(value[:, prefix_length + 1])
    state.backend.prepare(metadata, graph_mode=True)
    assert state.backend._active_decode_wrapper is state.backend._graph_decode_wrappers[1]
    assert state.backend._graph_decode_buffer_ptrs[1] == state.backend._buffer_ptrs(metadata)

    eager_cache = LayerCache(
        state.cache.key.clone(),
        state.cache.value.clone(),
        key_scale=state.cache.key_scale.clone(),
        value_scale=state.cache.value_scale.clone(),
    )
    eager_backend = FlashInferFP8Backend(
        state.heads,
        state.kv_heads,
        state.dim,
        state.dim**-0.5,
        -1,
        state.device,
        torch.bfloat16,
        "fp8_e4m3",
        key_scale=0.5,
        value_scale=2.0,
    )
    eager_backend.bind_kv_caches([eager_cache])
    eager_metadata = _attention_metadata(
        state.pages.clone(),
        torch.tensor([prefix_length + 1], dtype=torch.int64, device=state.device),
        prefix_length + 2,
        1,
        state.page_size,
    )
    eager_backend.prepare(eager_metadata)
    eager_output = eager_backend.execute(query, current_key, current_value, layer)

    graph.replay()
    torch.cuda.synchronize()
    graph_output = static_output.clone()
    assert torch.isfinite(graph_output).all()
    torch.testing.assert_close(graph_output.float(), eager_output.float(), rtol=0.03, atol=0.008)

    decoded_key, decoded_value = _decoded_prefix(state, key, value)
    expected = _dense(query[:, None].float(), decoded_key, decoded_value)
    torch.testing.assert_close(graph_output.float(), expected, rtol=0.03, atol=0.008)


def test_native_fp8_empty_request_copy_reuse_and_negative_slot(gqa5_backend: SimpleNamespace) -> None:
    state = gqa5_backend
    prefix_key = torch.randn(1, 2, state.kv_heads, state.dim, dtype=torch.bfloat16, device=state.device) * 0.2
    prefix_value = torch.randn_like(prefix_key) * 0.2
    state.backend.write_cache(
        prefix_key[0],
        prefix_value[0],
        torch.tensor([0, 1], dtype=torch.int64, device=state.device),
        0,
    )

    source_page = tuple(tensor[0].clone() for tensor in (state.cache.key, state.cache.value))
    source_scales = tuple(tensor[0].clone() for tensor in (state.cache.key_scale, state.cache.value_scale))
    state.cache.key[1].copy_(state.cache.key[0])
    state.cache.value[1].copy_(state.cache.value[0])
    state.cache.key_scale[1].copy_(state.cache.key_scale[0])
    state.cache.value_scale[1].copy_(state.cache.value_scale[0])
    assert torch.equal(state.cache.key[1], source_page[0])
    assert torch.equal(state.cache.value[1], source_page[1])
    assert torch.equal(state.cache.key_scale[1], source_scales[0])
    assert torch.equal(state.cache.value_scale[1], source_scales[1])
    assert state.backend._fp8_caches[0][0].data_ptr() == state.cache.key.data_ptr()
    assert state.backend._fp8_caches[0][1].data_ptr() == state.cache.value.data_ptr()

    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([-1, state.page_size + 2], dtype=torch.int64, device=state.device),
        paged_kv_indptr=torch.tensor([0, 0, 1], dtype=torch.int32, device=state.device),
        paged_kv_indptr_host=torch.tensor([0, 0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([1], dtype=torch.int32, device=state.device),
        paged_kv_last_page_len=torch.tensor([0, 3], dtype=torch.int32, device=state.device),
        paged_kv_last_page_len_host=torch.tensor([0, 3], dtype=torch.int32),
        qo_indptr=None,
        q_cu_seq_lens=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    state.backend.prepare(metadata)
    query = torch.randn(2, state.heads, state.dim, dtype=torch.bfloat16, device=state.device)
    current_key = torch.randn(2, state.kv_heads, state.dim, dtype=torch.bfloat16, device=state.device) * 0.2
    current_value = torch.randn_like(current_key) * 0.2
    current_key[1] = torch.randn_like(current_key[1]) * 0.2
    current_value[1] = torch.randn_like(current_value[1]) * 0.2
    layer = SimpleNamespace(layer_id=0, num_heads=state.heads, num_kv_heads=state.kv_heads, head_dim=state.dim)
    page_zero_before = tuple(
        tensor[0].clone()
        for tensor in (state.cache.key, state.cache.value, state.cache.key_scale, state.cache.value_scale)
    )

    actual = state.backend.execute(query.flatten(0, 1), current_key, current_value, layer).view(
        2, state.heads, state.dim
    )

    for previous, current in zip(
        page_zero_before,
        (state.cache.key[0], state.cache.value[0], state.cache.key_scale[0], state.cache.value_scale[0]),
    ):
        assert torch.equal(previous, current)
    codec = KVCacheCodec("fp8_e4m3", state.dim)
    decoded_key = codec.decode(state.cache.key, state.cache.key_scale)[1:2, :3]
    decoded_value = codec.decode(state.cache.value, state.cache.value_scale)[1:2, :3]
    expected = _dense(query[1:2, None].float(), decoded_key, decoded_value)
    assert torch.isfinite(actual[1]).all()
    torch.testing.assert_close(actual[1:2].flatten(1).float(), expected, rtol=0.03, atol=0.008)


def test_gqa5_native_bf16_profile_runs_paged_prefill_and_graph_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.attention.flashinfer import FlashInferBackend, _get_runtime_kernels
    from xllm.python.attention.quantized_triton import write_bf16_kv

    monkeypatch.setenv("XLLM_FLASHINFER_NATIVE_KV", "1")
    torch.manual_seed(20261023)
    device = torch.device("cuda")
    heads, kv_heads, dim, page_size = 40, 8, 128, 128
    shape = (2, page_size, kv_heads, dim)
    cache = LayerCache(
        torch.zeros(shape, dtype=torch.bfloat16, device=device),
        torch.zeros(shape, dtype=torch.bfloat16, device=device),
    )

    def write_pages(
        slots: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor
    ) -> None:
        write_bf16_kv(key, value, slots, key_cache, value_cache)

    monkeypatch.setattr(_get_runtime_kernels(), "reshape_paged_cache", write_pages, raising=False)
    backend = FlashInferBackend(heads, kv_heads, dim, dim**-0.5, -1, device, torch.bfloat16)
    backend.bind_kv_caches([cache])
    assert backend.native_kv_runtime_info()["planner_backend"] == "fa2"

    prefix_length = page_size + 1
    key = torch.randn(1, prefix_length + 1, kv_heads, dim, dtype=torch.bfloat16, device=device) * 0.2
    value = torch.randn_like(key) * 0.2
    prefill_query = torch.randn(1, prefix_length, heads, dim, dtype=torch.bfloat16, device=device)
    pages = torch.tensor([[0]], dtype=torch.int32, device=device)
    prefill_metadata = _attention_metadata(
        pages,
        torch.arange(prefix_length, dtype=torch.int64, device=device),
        prefix_length,
        prefix_length,
        page_size,
    )
    prefill_metadata.is_prefill = True
    prefill_metadata.is_chunked_prefill = False
    backend.prepare(prefill_metadata)
    layer = SimpleNamespace(layer_id=0, num_heads=heads, num_kv_heads=kv_heads, head_dim=dim)
    prefill_output = backend.execute(
        prefill_query.flatten(0, 1), key[:, :prefix_length].flatten(0, 1), value[:, :prefix_length].flatten(0, 1), layer
    )
    expected_prefill = _dense(prefill_query.float(), key[:, :prefix_length].float(), value[:, :prefix_length].float())
    torch.testing.assert_close(prefill_output.float(), expected_prefill, rtol=0.02, atol=0.005)

    decode_metadata = _attention_metadata(
        pages,
        torch.tensor([prefix_length], dtype=torch.int64, device=device),
        prefix_length + 1,
        1,
        page_size,
    )
    backend.prepare(decode_metadata, graph_mode=True)
    decode_query = torch.randn(1, heads, dim, dtype=torch.bfloat16, device=device)
    decode_key = key[:, prefix_length].contiguous()
    decode_value = value[:, prefix_length].contiguous()
    backend.execute(decode_query, decode_key, decode_value, layer)
    torch.cuda.synchronize()
    static_output = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = backend.execute(decode_query, decode_key, decode_value, layer)
    backend.prepare(decode_metadata, graph_mode=True)

    eager_cache = LayerCache(cache.key.clone(), cache.value.clone())
    eager_backend = FlashInferBackend(heads, kv_heads, dim, dim**-0.5, -1, device, torch.bfloat16)
    eager_backend.bind_kv_caches([eager_cache])
    eager_metadata = _attention_metadata(
        pages.clone(),
        torch.tensor([prefix_length], dtype=torch.int64, device=device),
        prefix_length + 1,
        1,
        page_size,
    )
    eager_backend.prepare(eager_metadata)
    eager_output = eager_backend.execute(decode_query, decode_key, decode_value, layer)

    graph.replay()
    torch.cuda.synchronize()
    graph_output = static_output.clone()
    torch.testing.assert_close(graph_output.float(), eager_output.float(), rtol=0.02, atol=0.005)
    full_key = key.float()
    full_value = value.float()
    expected_decode = _dense(decode_query[:, None].float(), full_key, full_value)
    torch.testing.assert_close(graph_output.float(), expected_decode, rtol=0.02, atol=0.005)
    assert backend._active_decode_wrapper is backend._graph_decode_wrappers[1]


def test_model_executor_fp8_gqa5_binds_warms_prefills_and_replays_graph(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Exercise the native FP8 graph route through the real model executor."""
    import logging

    from xllm.python import kernels
    from xllm.python.layers.attention import Attention
    from xllm.python.model_executor.executor import ModelExecutor

    monkeypatch.setenv("XLLM_QUANTIZED_BACKEND", "flashinfer")
    monkeypatch.setenv("XLLM_FLASHINFER_NATIVE_KV", "1")
    monkeypatch.setenv("XLLM_FP8_K_SCALE", "0.5")
    monkeypatch.setenv("XLLM_FP8_V_SCALE", "2.0")

    def update_graph_metadata(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indptr: torch.Tensor,
        page_indices: torch.Tensor,
        last_page_len: torch.Tensor,
        dst_tokens: torch.Tensor,
        dst_positions: torch.Tensor,
        dst_slots: torch.Tensor,
        dst_kv_lens: torch.Tensor,
        dst_kv_delta: torch.Tensor,
        dst_page_indptr: torch.Tensor,
        dst_page_indices: torch.Tensor,
        dst_last_page_len: torch.Tensor,
        padded_batch: int,
    ) -> torch.Tensor:
        batch = tokens.numel()
        dst_tokens.zero_()
        dst_positions.zero_()
        dst_slots.zero_()
        dst_kv_lens.zero_()
        dst_kv_delta.zero_()
        dst_page_indptr.zero_()
        dst_last_page_len.fill_(1)
        dst_page_indices.zero_()
        dst_tokens[:batch].copy_(tokens)
        dst_positions[:batch].copy_(positions)
        dst_slots[:batch].copy_(slots)
        dst_kv_lens[: batch + 1].copy_(kv_lens[: batch + 1])
        dst_kv_delta[:batch].copy_(kv_lens[1 : batch + 1] - kv_lens[:batch])
        dst_page_indptr[: batch + 1].copy_(page_indptr[: batch + 1])
        dst_page_indices[: page_indices.numel()].copy_(page_indices)
        dst_last_page_len[:batch].copy_(last_page_len[:batch])
        return dst_tokens

    monkeypatch.setattr(kernels, "update_decode_graph_metadata", update_graph_metadata, raising=False)

    device = torch.device("cuda")
    heads, kv_heads, dim, page_size, max_len = 40, 8, 128, 128, 256

    class TinyForward(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = Attention(heads, kv_heads, dim, dim**-0.5, -1, 0)
            self.q = torch.nn.Parameter(torch.randn(max_len, heads, dim, device=device, dtype=torch.bfloat16))
            self.k = torch.nn.Parameter(torch.randn(max_len, kv_heads, dim, device=device, dtype=torch.bfloat16) * 0.1)
            self.v = torch.nn.Parameter(torch.randn(max_len, kv_heads, dim, device=device, dtype=torch.bfloat16) * 0.1)

        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            del input_ids
            pos = positions.to(torch.long)
            return self.attention(self.q[pos].flatten(1), self.k[pos].flatten(1), self.v[pos].flatten(1))

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TinyForward()

    model = TinyModel()
    executor = ModelExecutor(
        model,
        {
            "kv_cache_dtype": "fp8_e4m3",
            "python_graph_backend": "cudagraphs",
            "max_position_embeddings": max_len,
        },
        max_seqs_per_batch=1,
    )
    cache_shape = (2, page_size, kv_heads, dim)
    cache = LayerCache(
        torch.zeros(cache_shape, dtype=torch.uint8, device=device),
        torch.zeros(cache_shape, dtype=torch.uint8, device=device),
        key_scale=torch.ones(cache_shape[:-1], dtype=torch.float32, device=device),
        value_scale=torch.ones(cache_shape[:-1], dtype=torch.float32, device=device),
    )
    executor.bind_kv_caches([cache])
    assert executor.decode_graph_runner.captured_batch_sizes == (1,)
    assert executor.attention_backend.native_kv_runtime_info()["fp8_views_alias_resident_pages"]

    pages = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    prefix_len = 3
    prefill_metadata = _attention_metadata(
        pages,
        torch.arange(prefix_len, dtype=torch.int32, device=device),
        prefix_len,
        prefix_len,
        page_size,
    )
    prefill_metadata.q_seq_lens_host = torch.tensor([prefix_len], dtype=torch.int32)
    prefill_metadata.kv_seq_lens_host = torch.tensor([0, prefix_len], dtype=torch.int32)
    prefill_metadata.kv_cu_seq_lens = torch.tensor([0, prefix_len], dtype=torch.int32, device=device)
    prefill_metadata.kv_seq_lens_host_values = [prefix_len]
    prefill_metadata.is_prefill = True
    prefill_metadata.is_chunked_prefill = False
    executor.execute(
        torch.arange(prefix_len, dtype=torch.int32, device=device),
        torch.arange(prefix_len, dtype=torch.int32, device=device),
        prefill_metadata,
    )

    decode_metadata = _attention_metadata(
        pages,
        torch.tensor([prefix_len], dtype=torch.int32, device=device),
        prefix_len + 1,
        1,
        page_size,
    )
    decode_metadata.q_seq_lens_host = torch.tensor([1], dtype=torch.int32)
    decode_metadata.kv_seq_lens_host = torch.tensor([0, prefix_len + 1], dtype=torch.int32)
    decode_metadata.kv_cu_seq_lens = torch.tensor([0, prefix_len + 1], dtype=torch.int32, device=device)
    decode_metadata.kv_seq_lens_host_values = [prefix_len + 1]
    decode_metadata.is_prefill = False
    decode_metadata.is_chunked_prefill = False

    def eager_must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("decode unexpectedly used EagerRunner")

    monkeypatch.setattr(executor.eager_runner, "execute", eager_must_not_run)
    with caplog.at_level(logging.INFO):
        actual = executor.execute(
            torch.tensor([prefix_len], dtype=torch.int32, device=device),
            torch.tensor([prefix_len], dtype=torch.int32, device=device),
            decode_metadata,
        )
    assert "NATIVE_KV_GRAPH_REPLAY" in caplog.text
    assert torch.isfinite(actual).all()

    codec = KVCacheCodec("fp8_e4m3", dim)
    decoded_key = codec.decode(cache.key, cache.key_scale)[pages.long()].reshape(1, -1, kv_heads, dim)[
        :, : prefix_len + 1
    ]
    decoded_value = codec.decode(cache.value, cache.value_scale)[pages.long()].reshape(1, -1, kv_heads, dim)[
        :, : prefix_len + 1
    ]
    query = model.model.q[prefix_len : prefix_len + 1].float().unsqueeze(0)
    expected = _dense(query, decoded_key, decoded_value)
    torch.testing.assert_close(actual.float(), expected, rtol=0.03, atol=0.008)
