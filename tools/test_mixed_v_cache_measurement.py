# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Exercise timing and bounded replay using an actual local HTTP/SSE server."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from tools import run_mixed_v_cache_experiment as runner


def _row(index: int = 0) -> dict[str, Any]:
    return dict(
        task="gsm8k",
        id=f"gsm8k/{index}",
        prompt=str(index),
        max_tokens=8,
        chat=True,
        prompt_tokens=2,
        prompt_sha256=f"hash{index}",
    )


async def _events(
    response: web.StreamResponse, tokens: int = 4, text: str = "one two", total: int | None = None
) -> None:
    events = [
        {"choices": [{"index": 0, "delta": {"content": text}}]},
        {"choices": [{"index": 0, "finish_reason": "stop"}]},
        {"usage": dict(prompt_tokens=2, completion_tokens=tokens, total_tokens=2 + tokens if total is None else total)},
    ]
    for event in events:
        data = ("data: " + json.dumps(event) + "\n\n").encode()
        # Split network writes inside an SSE line; events must still be parsed once.
        await response.write(data[:7])
        await response.write(data[7:])
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()


async def _start(handler: Any) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    server = web.AppRunner(app)
    await server.setup()
    await web.TCPSite(server, "127.0.0.1", 0).start()
    return server, f"http://127.0.0.1:{server.addresses[0][1]}"


def test_tpot_includes_finish_tail() -> None:
    async def _check() -> None:
        async def _handler(request: web.Request) -> web.StreamResponse:
            await request.json()
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n')
            await asyncio.sleep(0.08)
            await _events(response, text="")
            return response

        server, url = await _start(_handler)
        try:
            async with aiohttp.ClientSession() as session:
                result = await runner._request(session, url, _row(), 0)
            assert result["stream_events"] == 1
            assert result["text_span_tpot_s"] == 0
            assert result["tpot_s"] >= 0.08 / 3
            assert result["tpot_s"] == pytest.approx((result["e2e_s"] - result["ttft_s"]) / 3)
            assert result["request_end_monotonic_s"] - result["request_start_monotonic_s"] == result["e2e_s"]
        finally:
            await server.cleanup()

    asyncio.run(_check())


@pytest.mark.parametrize(
    "tokens,text,total,invalid",
    [
        (1, "", 3, False),
        (1, "x", 3, False),
        (0, "", 2, False),
        (9, "x", 11, True),
        (-1, "x", 1, True),
        (4, "x", 99, True),
        (0, "x", 2, True),
    ],
)
def test_usage_boundaries(tokens: int, text: str, total: int, invalid: bool) -> None:
    async def _check() -> None:
        async def _handler(request: web.Request) -> web.StreamResponse:
            await request.json()
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await _events(response, tokens, text, total)
            return response

        server, url = await _start(_handler)
        try:
            async with aiohttp.ClientSession() as session:
                if invalid:
                    with pytest.raises(ValueError):
                        await runner._request(session, url, _row(), 0)
                else:
                    result = await runner._request(session, url, _row(), 0)
                    assert result["tpot_s"] is None
                    assert (result["ttft_s"] is None) == (text == "")
        finally:
            await server.cleanup()

    asyncio.run(_check())


def test_replay_boundaries_and_concurrency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _check() -> None:
        active = peak = calls = 0
        submitted: list[int] = []

        async def _handler(request: web.Request) -> web.StreamResponse:
            nonlocal active, peak, calls
            body = await request.json()
            calls += 1
            call = calls
            submitted.append(int(body["messages"][0]["content"]))
            active += 1
            peak = max(peak, active)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await asyncio.sleep(0.12 if call <= 4 else 0.04)
            await _events(response)
            active -= 1
            return response

        async def _sample(session: Any, url: str, output: Path, stop: asyncio.Event) -> None:
            output.write_text("")
            await stop.wait()
            # Sampler cleanup must not inflate measured makespan.
            await asyncio.sleep(0.15)

        monkeypatch.setattr(runner, "_sample", _sample)
        server, url = await _start(_handler)
        try:
            start = time.perf_counter()
            await runner._replay(url, [_row(i) for i in range(12)], 4, tmp_path, 2)
            wall = time.perf_counter() - start
            summary = json.loads((tmp_path / "summary.json").read_text())
            responses = [json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()]
            assert calls == 16 and peak == summary["peak_client_inflight"] == 4
            assert sorted(submitted[:4]) == list(range(4))
            assert sorted(submitted[4:]) == list(range(12))
            assert [r["index"] for r in sorted(responses, key=lambda r: r["request_start_monotonic_s"])] == list(
                range(12)
            )
            assert sorted(r["index"] for r in responses) == list(range(12))
            assert summary["requests"] == 12 and summary["generated_tokens"] == 48
            assert summary["request_throughput"] == 12 / summary["elapsed_s"]
            assert summary["token_throughput"] == 48 / summary["elapsed_s"]
            assert wall - summary["elapsed_s"] >= 0.12 + 0.15
            assert summary["elapsed_s"] >= max(r["e2e_s"] for r in responses)
            assert all(
                summary["measurement_start_monotonic_s"]
                <= r["request_start_monotonic_s"]
                <= r["request_end_monotonic_s"]
                <= summary["measurement_end_monotonic_s"]
                for r in responses
            )
            assert summary["metric_samples"] == dict(ttft_s=12, tpot_s=12, e2e_s=12)
        finally:
            await server.cleanup()

    asyncio.run(_check())


@pytest.mark.parametrize("arm", runner.ARMS)
def test_resident_allocation_accounting(arm: str) -> None:
    value_format, dtype, width = {
        "auto": ("bf16", "torch.bfloat16", 128),
        "bf16_int8": ("int8", "torch.int8", 128),
        "bf16_fp8_e4m3": ("fp8_e4m3", "torch.uint8", 128),
        "bf16_int4": ("int4", "torch.uint8", 64),
    }[arm]
    tensors = {
        "key": dict(dtype="torch.bfloat16", shape=[2, 128, 2, 128], bytes=131072),
        "value": dict(dtype=dtype, shape=[2, 128, 2, width], bytes=512 * width * (2 if arm == "auto" else 1)),
    }
    if arm != "auto":
        tensors["value_scale"] = dict(dtype="torch.float32", shape=[2, 128, 2], bytes=2048)
    resident = sum(t["bytes"] for t in tensors.values())
    allocation = dict(
        layers=[
            dict(
                tensors=tensors,
                key_format="bf16",
                value_format=value_format,
                resident_bytes=resident,
                bytes_per_token=resident // 256,
            )
        ],
        resident_bytes=resident,
        token_capacity=256,
    )
    runner._validate_allocation(arm, allocation)
    invalid = copy.deepcopy(allocation)
    invalid["layers"][0]["tensors"]["key"]["dtype"] = "torch.int8"
    with pytest.raises(ValueError):
        runner._validate_allocation(arm, invalid)
    invalid = copy.deepcopy(allocation)
    invalid["resident_bytes"] += 1
    with pytest.raises(ValueError):
        runner._validate_allocation(arm, invalid)
