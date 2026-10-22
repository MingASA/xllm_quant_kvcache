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

"""CPU checks for native FlashInfer cache evidence and dispatch policy."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.request import Request

import pytest
import torch
import torch.nn as nn

from tools import check_native_flashinfer_service as service_check
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.native_kv import KVScaleCalibration, cache_allocation
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.executor import ModelExecutor, _create_attention_backend
from xllm.python.model_executor.runners.decode_cuda_graph import DecodeCudaGraphRunner


def _cache(cache_format: str) -> LayerCache:
    shape = (2, 4, 1, 8)
    if cache_format == "bf16":
        return LayerCache(
            torch.zeros(shape, dtype=torch.bfloat16),
            torch.zeros(shape, dtype=torch.bfloat16),
        )
    return LayerCache(
        torch.zeros(shape, dtype=torch.uint8),
        torch.zeros(shape, dtype=torch.uint8),
        key_scale=torch.ones(shape[:-1], dtype=torch.float32),
        value_scale=torch.ones(shape[:-1], dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("cache_format", "expected_layer_bytes", "expected_total_bytes", "expected_bytes_per_token"),
    [("bf16", 256, 12288, 32), ("fp8_e4m3", 192, 9216, 24)],
)
def test_cache_allocation_reports_48_layers_of_resident_storage(
    cache_format: str, expected_layer_bytes: int, expected_total_bytes: int, expected_bytes_per_token: int
) -> None:
    report = json.loads(cache_allocation([_cache(cache_format) for _ in range(48)], cache_format))

    assert len(report["layers"]) == 48
    assert report["layers"][0]["resident_bytes"] == expected_layer_bytes
    assert report["layers"][0]["bytes_per_token"] == expected_bytes_per_token
    assert report["resident_bytes"] == expected_total_bytes
    assert report["token_capacity"] == 8
    assert all(layer["key_format"] == layer["value_format"] == cache_format for layer in report["layers"])


def test_scale_calibration_uses_valid_slot_maxima_and_power_of_two_scales(tmp_path: Path) -> None:
    output = tmp_path / "calibration.json"
    calibration = KVScaleCalibration(str(output))
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        slot_mapping=torch.tensor([0, -1, 1]),
    )
    key = torch.tensor([[[1.0, 10.0]], [[10000.0, -10000.0]], [[2.0, 100.0]]])
    value = torch.tensor([[[2.0, 3.0]], [[20000.0, -20000.0]], [[10.0, 20.0]]])

    calibration.observe(0, key, value, metadata, num_layers=1)

    proposal = json.loads(output.read_text())
    assert proposal["observed_absmax"] == {"key": 100.0, "value": 20.0}
    assert proposal["fp8_k_scale"] == 0.25
    assert proposal["fp8_v_scale"] == 0.0625
    assert proposal["observations"]["prefill/0"]["elements"] == 4


def _attention() -> Attention:
    return Attention(num_heads=8, num_kv_heads=2, head_dim=64, scale=0.125, sliding_window=0, layer_id=0)


def test_executor_emits_native_backend_marker_once_with_graph_runner_state() -> None:
    backend = MagicMock()
    backend.native_kv_runtime_info.return_value = {
        "class": "FlashInferFP8Backend",
        "kv_cache_dtype": "fp8_e4m3",
        "attention_impl": "flashinfer",
    }
    executor = object.__new__(ModelExecutor)
    executor.model = nn.Sequential(_attention())
    executor.attention_backend = backend
    executor.eager_runner = MagicMock()
    executor.decode_graph_runner = SimpleNamespace(bind_layer_caches=MagicMock())
    executor.inductor_runner = None
    executor._kv_bound = False
    cache = (torch.zeros(1), torch.zeros(1))
    logger_mock = MagicMock()

    with patch("xllm.python.model_executor.executor.logger", logger_mock):
        executor.bind_kv_caches([cache])
        executor.bind_kv_caches([cache])

    backend.bind_kv_caches.assert_called_once()
    logger_mock.info.assert_called_once()
    args = logger_mock.info.call_args.args
    assert args[0] == "NATIVE_KV_BACKEND %s"
    evidence = json.loads(args[1])
    assert evidence["graph_enabled"] is False
    assert evidence["graph_captured_batches"] == []
    assert evidence["graph_runner"] == "SimpleNamespace"


def test_factory_requires_explicit_native_fp8_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLLM_QUANTIZED_BACKEND", "flashinfer")
    monkeypatch.setenv("XLLM_FLASHINFER_NATIVE_KV", "1")
    monkeypatch.delenv("XLLM_FP8_K_SCALE", raising=False)
    monkeypatch.delenv("XLLM_FP8_V_SCALE", raising=False)
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_cuda", return_value=True),
        pytest.raises(ValueError, match="explicit measured K/V scales"),
    ):
        _create_attention_backend(
            _attention(),
            torch.device("cuda"),
            torch.bfloat16,
            {"kv_cache_dtype": "fp8_e4m3"},
        )


def test_factory_allows_cuda_graphs_for_native_flashinfer_fp8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLLM_QUANTIZED_BACKEND", "flashinfer")
    monkeypatch.setenv("XLLM_FLASHINFER_NATIVE_KV", "1")
    monkeypatch.setenv("XLLM_FP8_K_SCALE", "0.25")
    monkeypatch.setenv("XLLM_FP8_V_SCALE", "0.5")
    backend_type = MagicMock(name="FlashInferFP8Backend")
    module = types.ModuleType("xllm.python.attention.flashinfer_fp8")
    module.FlashInferFP8Backend = backend_type

    with (
        patch("xllm.python.model_executor.executor.current_platform.is_cuda", return_value=True),
        patch.dict(sys.modules, {module.__name__: module}),
    ):
        result = _create_attention_backend(
            _attention(),
            torch.device("cuda"),
            torch.bfloat16,
            {"kv_cache_dtype": "fp8_e4m3", "python_graph_backend": "cudagraphs"},
        )

    assert result is backend_type.return_value
    assert backend_type.call_args.kwargs["key_scale"] == 0.25
    assert backend_type.call_args.kwargs["value_scale"] == 0.5


def test_factory_keeps_native_bf16_flashinfer_graph_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XLLM_QUANTIZED_BACKEND", raising=False)
    monkeypatch.delenv("XLLM_FLASHINFER_NATIVE_KV", raising=False)
    backend_type = MagicMock(name="FlashInferBackend")
    module = types.ModuleType("xllm.python.attention.flashinfer")
    module.FlashInferBackend = backend_type

    with (
        patch("xllm.python.model_executor.executor.current_platform.is_cuda", return_value=True),
        patch.dict(sys.modules, {module.__name__: module}),
    ):
        result = _create_attention_backend(
            _attention(),
            torch.device("cuda"),
            torch.bfloat16,
            {"kv_cache_dtype": "auto", "python_graph_backend": "cudagraphs"},
        )

    assert result is backend_type.return_value


@pytest.mark.parametrize(
    ("quantized_backend", "cache_dtype"),
    [("triton", "int8"), ("triton", "fp8_e4m3"), ("flashinfer_mixed", "fp8_e4m3")],
)
def test_factory_keeps_cuda_graphs_closed_for_other_backends(
    monkeypatch: pytest.MonkeyPatch, quantized_backend: str, cache_dtype: str
) -> None:
    monkeypatch.setenv("XLLM_QUANTIZED_BACKEND", quantized_backend)
    monkeypatch.delenv("XLLM_FLASHINFER_NATIVE_KV", raising=False)
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_cuda", return_value=True),
        pytest.raises(ValueError, match="requires Python eager execution"),
    ):
        _create_attention_backend(
            _attention(),
            torch.device("cuda"),
            torch.bfloat16,
            {"kv_cache_dtype": cache_dtype, "python_graph_backend": "cudagraphs"},
        )


class _FakeSSEResponse:
    headers = {"Content-Type": "text/event-stream; charset=utf-8"}

    def __init__(self, events: list[dict | str]) -> None:
        self._lines = [
            ("data: " + (event if isinstance(event, str) else json.dumps(event)) + "\n\n").encode() for event in events
        ]

    def __enter__(self) -> _FakeSSEResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._lines)


def _sse_chunks() -> list[dict | str]:
    return [
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Answer "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "42"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        "[DONE]",
    ]


def test_service_stream_collects_role_content_finish_usage_and_done(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_urlopen(request: Request, timeout: int) -> _FakeSSEResponse:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _FakeSSEResponse(_sse_chunks())

    monkeypatch.setattr(service_check.urllib.request, "urlopen", fake_urlopen)

    result = service_check._stream("http://localhost:8000/v1", "offline prompt", 17)

    assert result["prediction"] == "Answer 42"
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    assert result["sse_done"] is True
    assert len(result["chunks"]) == 4
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["body"]["stream"] is True
    assert captured["timeout"] == 17


def test_service_stream_rejects_missing_done(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = _sse_chunks()[:-1]
    monkeypatch.setattr(
        service_check.urllib.request,
        "urlopen",
        lambda _request, timeout: _FakeSSEResponse(chunks),
    )

    with pytest.raises(ValueError, match="Incomplete SSE/usage"):
        service_check._stream("http://localhost:8000/v1", "offline prompt", 17)


@pytest.mark.parametrize("capture_fails", [False, True])
def test_executor_captures_cuda_graphs_before_binding_completes(capture_fails: bool) -> None:
    backend = MagicMock()
    backend.native_kv_runtime_info.return_value = {"class": "FlashInferFP8Backend"}
    executor = object.__new__(ModelExecutor)
    executor.model = nn.Sequential(_attention(), nn.Linear(1, 1))
    executor.attention_backend = backend
    executor.eager_runner = MagicMock()
    executor.inductor_runner = None
    executor._kv_bound = False
    runner = DecodeCudaGraphRunner(executor.model, backend, torch.device("cpu"), 1, 32)
    executor.decode_graph_runner = runner

    def capture(_device: torch.device, _dtype: torch.dtype) -> None:
        assert not executor._kv_bound
        assert len(runner.layer_caches) == 1
        if capture_fails:
            raise RuntimeError("capture failed")
        runner._graphs[(1, (1,))] = object()
        runner._warmed_up = True

    logger_mock = MagicMock()
    with (
        patch.object(runner, "warmup", side_effect=capture) as warmup,
        patch("xllm.python.model_executor.executor.logger", logger_mock),
    ):
        if capture_fails:
            with pytest.raises(RuntimeError, match="capture failed"):
                executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
            assert not executor._kv_bound
            logger_mock.info.assert_not_called()
        else:
            executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
            executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
            warmup.assert_called_once()
            assert executor._kv_bound
            evidence = json.loads(logger_mock.info.call_args.args[1])
            assert evidence["graph_enabled"] is True
            assert evidence["graph_captured_batches"] == [1]
