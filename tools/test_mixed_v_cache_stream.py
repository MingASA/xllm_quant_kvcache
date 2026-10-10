# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Check the trace runner's streaming boundary and authoritative token counts."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from tools.run_mixed_v_cache_experiment import _request


@pytest.mark.parametrize("failure", (None, "no_usage", "no_done", "wrong_prompt_tokens"))
def test_stream_usage_and_protocol(failure: str | None) -> None:
    async def _check() -> None:
        captured: list[dict[str, Any]] = []

        async def _handler(request: web.Request) -> web.StreamResponse:
            captured.append(await request.json())
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            events = [
                {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"delta": {"content": "one"}}]},
                {"choices": [{"delta": {"content": " two"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            if failure != "no_usage":
                events.append(
                    {
                        "usage": {
                            "prompt_tokens": 3 if failure == "wrong_prompt_tokens" else 2,
                            "completion_tokens": 4,
                            "total_tokens": 6,
                        }
                    }
                )
            for event in events:
                await response.write(("data: " + json.dumps(event) + "\n\n").encode())
            if failure != "no_done":
                await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/chat/completions", _handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        row = {
            "task": "gsm8k",
            "id": "gsm8k/0",
            "prompt": "query",
            "max_tokens": 512,
            "chat": True,
            "prompt_tokens": 2,
            "prompt_sha256": "hash",
        }
        try:
            async with aiohttp.ClientSession() as session:
                if failure:
                    with pytest.raises(ValueError):
                        await _request(session, f"http://127.0.0.1:{port}", row, 0)
                else:
                    result = await _request(session, f"http://127.0.0.1:{port}", row, 0)
                    assert result["prediction"] == "one two"
                    assert result["stream_events"] == 2
                    assert result["usage"]["completion_tokens"] == 4
                    assert result["tpot_s"] == (result["e2e_s"] - result["ttft_s"]) / 3
                    assert result["e2e_s"] >= result["ttft_s"] >= 0
            assert captured[0]["max_tokens"] == 512
            assert captured[0]["temperature"] == 0
            assert captured[0]["stop_token_ids"] == [151645, 151643]
            assert captured[0]["stream_options"] == {"include_usage": True}
        finally:
            await runner.cleanup()

    asyncio.run(_check())
