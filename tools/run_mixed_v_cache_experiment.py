# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Replay immutable benchmark traces against eager mixed-dtype xLLM servers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger
from tools import eval_kv_cache_quality as evaluator
from tools import run_kv_fp8_quality_ablation as assets

ARMS = ("auto", "bf16_int8", "bf16_fp8_e4m3", "bf16_int4")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(output: Path, performance_samples: int) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(assets.MODEL, local_files_only=True)
    templates, lengths, _, revision = evaluator._longbench_assets(assets.LONG_BENCH_REPO)
    shots = json.loads(assets.GSM_SHOTS.read_text())
    sources = {"gsm8k": assets.GSM_DATA, **{task: assets.PREPARED_LONG / f"{task}.jsonl" for task in assets.TASKS}}
    quality: list[dict[str, Any]] = []
    for task, path in sources.items():
        for index, row in enumerate(evaluator._jsonl(path)):
            prompt = (
                evaluator._gsm_prompt(row["question"], shots)
                if task == "gsm8k"
                else templates[task].format(context=row["context"], input=row["input"])
            )
            chat = task not in ("lcc", "repobench-p")
            max_tokens = 512 if task == "gsm8k" else int(lengths[task])
            prompt_tokens = evaluator._token_count(tokenizer, prompt, chat)
            if prompt_tokens + max_tokens > evaluator.MAX_CONTEXT_TOKENS:
                raise ValueError(f"{task}/{index}: prompt+max_tokens exceeds context limit; no truncation")
            quality.append(
                {
                    "id": f"{task}/{index}",
                    "task": task,
                    "row": row,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": max_tokens,
                    "chat": chat,
                }
            )
    traces = {"quality": quality}
    # Fixed benchmark rows, with original prompts and generation caps. Equal per-task representation.
    limit = performance_samples or None
    traces["gsm8k"] = [row for row in quality if row["task"] == "gsm8k"][:limit]
    long_rows = {task: [row for row in quality if row["task"] == task] for task in assets.TASKS}
    traces["longbench"] = [
        long_rows[task][i] for i in range(min(len(rows) for rows in long_rows.values())) for task in assets.TASKS
    ][:limit]
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in traces.items():
        path = output / f"trace_{name}.jsonl"
        if path.exists():
            raise FileExistsError(f"Immutable trace already exists: {path}")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    _write_json(
        output / "trace_manifest.json",
        {
            "checkpoint": str(assets.MODEL),
            "model_provenance": json.loads(assets.PROVENANCE.read_text()),
            "longbench_revision": revision,
            "sources": {str(path): _hash_file(path) for path in sources.values()},
            "shots_sha256": _hash_file(assets.GSM_SHOTS),
            "traces": {
                name: {"requests": len(rows), "sha256": _hash_file(output / f"trace_{name}.jsonl")}
                for name, rows in traces.items()
            },
            "protocol": "temperature=0, top_p=1, seed=17, normal EOS, original benchmark max_tokens/stop strings",
            "performance_scope": "full" if not performance_samples else "prefix",
            "performance_order": "benchmark order; LongBench round robin in existing eight-task order",
        },
    )


async def _request(session: Any, url: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": assets.MODEL_ID,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "seed": 17,
        "max_tokens": row["max_tokens"],
        "stop_token_ids": [151645, 151643],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if row["task"] == "gsm8k":
        body["stop"] = ["Question:", "</s>", "<|im_end|>"]
    if row["chat"]:
        body["messages"] = [{"role": "user", "content": row["prompt"]}]
        endpoint = "/v1/chat/completions"
    else:
        body["prompt"] = row["prompt"]
        endpoint = "/v1/completions"
    start_unix = time.time()
    start = time.perf_counter()
    chunks: list[str] = []
    times: list[float] = []
    usage: dict[str, Any] | None = None
    finish_reason = None
    done = False
    async with session.post(url + endpoint, json=body) as response:
        response.raise_for_status()
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            item = json.loads(payload)
            if "error" in item:
                raise RuntimeError(item["error"])
            if item.get("usage"):
                usage = item["usage"]
            choices = item.get("choices", [])
            if len(choices) > 1 or any(choice.get("index", 0) != 0 for choice in choices):
                raise ValueError("Expected exactly one output sequence (n=1)")
            for choice in choices:
                text = choice.get("delta", {}).get("content", "") if row["chat"] else choice.get("text", "")
                if text:
                    chunks.append(text)
                    times.append(time.perf_counter())
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    end = time.perf_counter()
    if not done or usage is None or finish_reason is None:
        raise ValueError(
            f"Incomplete streamed response for {row['id']}: done={done}, usage={usage}, finish={finish_reason}"
        )
    if any(type(usage.get(key)) is not int for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
        raise ValueError(f"Invalid API token counts: {usage}")
    tokens = usage["completion_tokens"]
    if not 0 <= tokens <= row["max_tokens"] or usage["total_tokens"] != usage["prompt_tokens"] + tokens:
        raise ValueError(f"Inconsistent API token counts: {usage}")
    if finish_reason not in ("stop", "length") or (finish_reason == "length" and tokens != row["max_tokens"]):
        raise ValueError(f"Unexpected finish reason/count: {finish_reason}, {tokens}")
    if chunks and tokens == 0:
        raise ValueError("Nonempty output with zero generated tokens")
    if usage["prompt_tokens"] != row["prompt_tokens"]:
        raise ValueError(f"API/local prompt token count mismatch: {row['id']}")
    return {
        "index": index,
        "id": row["id"],
        "task": row["task"],
        "prompt_sha256": row["prompt_sha256"],
        "prediction": "".join(chunks),
        "usage": usage,
        "finish_reason": finish_reason,
        "ttft_s": times[0] - start if times else None,
        "tpot_s": (end - times[0]) / (tokens - 1) if times and tokens > 1 else None,
        "text_span_tpot_s": (times[-1] - times[0]) / (tokens - 1) if times and tokens > 1 else None,
        "request_start_unix_s": start_unix,
        "request_start_monotonic_s": start,
        "request_end_monotonic_s": end,
        "e2e_s": end - start,
        "stream_events": len(times),
        "stream_event_times_s": [t - start for t in times],
    }


async def _sample(session: Any, url: str, output: Path, stop: asyncio.Event) -> None:
    import aiohttp

    with output.open("w") as stream:
        while not stop.is_set():
            record: dict[str, Any] = {"unix_time": time.time()}
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                *(["-i", os.environ["XLLM_TELEMETRY_GPU"]] if os.environ.get("XLLM_TELEMETRY_GPU") else []),
                "--query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            record["gpu_raw"] = stdout.decode().strip()
            if len(record["gpu_raw"].splitlines()) > 1:
                raise ValueError("Select one telemetry GPU with XLLM_TELEMETRY_GPU (prefer its UUID)")
            if process.returncode:
                record["gpu_error"] = stderr.decode().strip()
            # Telemetry is best-effort at its external API boundary; failures remain visible in raw data.
            try:
                async with session.get(url + "/brpc_metrics", timeout=2) as response:
                    record["metrics_status"] = response.status
                    record["metrics"] = await response.text() if response.status == 200 else None
            except (OSError, asyncio.TimeoutError, aiohttp.ClientError) as error:
                record["metrics_error"] = str(error)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass


async def _bootstrap(url: str, row: dict[str, Any]) -> dict[str, Any]:
    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        return await _request(session, url, row, 0)


def _percentiles(values: list[float]) -> dict[str, float] | None:
    import numpy as np

    if not values:
        return None
    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)}


def _attention_compiles_between(start_unix: float, end_unix: float) -> list[dict[str, Any]]:
    # Read metadata after sampling stops, outside timed request processing.
    cache = Path(os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton/cache")))
    if not cache.is_dir():
        raise ValueError(f"Triton compilation cache is unavailable for measurement audit: {cache}")
    records = []
    artifacts = set(cache.glob("*/_quantized_paged*kernel.json"))
    artifacts.update(cache.glob("*/_quantize_and_scatter_kv_kernel.json"))
    artifacts.update(cache.glob("*/_scatter_bf16_kv_kernel.json"))
    for path in sorted(artifacts):
        modified = path.stat().st_mtime
        if start_unix <= modified <= end_unix:
            records.append({"path": str(path), "modified_unix_s": modified})
    return records


async def _watch_event_loop(stop: asyncio.Event, lags: list[float], stalls: list[dict[str, float]]) -> None:
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        expected = loop.time() + 0.1
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            lag = max(0.0, loop.time() - expected)
            lags.append(lag)
            if lag >= 0.02:
                stalls.append({"unix_time": time.time(), "lag_s": lag})


async def _replay(url: str, rows: list[dict[str, Any]], concurrency: int, output: Path, warmup: int) -> None:
    import aiohttp

    output.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=3600), connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        warmup_semaphore = asyncio.Semaphore(concurrency)

        async def _warm(index: int, row: dict[str, Any]) -> None:
            async with warmup_semaphore:
                await _request(session, url, row, index)

        async with asyncio.TaskGroup() as group:
            for index, row in enumerate(rows[: max(warmup, concurrency)]):
                group.create_task(_warm(index, row))
        semaphore = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = []
        stop = asyncio.Event()
        sampler = asyncio.create_task(_sample(session, url, output / "telemetry.jsonl", stop))
        lags: list[float] = []
        stalls: list[dict[str, float]] = []
        watcher = asyncio.create_task(_watch_event_loop(stop, lags, stalls))
        active = 0
        peak_active = 0
        with (output / "responses.jsonl").open("x") as journal:

            async def _execute(index: int, row: dict[str, Any]) -> None:
                nonlocal active, peak_active
                async with semaphore:
                    active += 1
                    peak_active = max(peak_active, active)
                    try:
                        result = await _request(session, url, row, index)
                    finally:
                        active -= 1
                    results.append(result)
                    journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                    journal.flush()

            start_unix = time.time()
            cpu_start = time.process_time()
            start = time.perf_counter()
            try:
                async with asyncio.TaskGroup() as group:
                    for index, row in enumerate(rows):
                        group.create_task(_execute(index, row))
            finally:
                measured_end = time.perf_counter()
                elapsed = measured_end - start
                cpu_elapsed = time.process_time() - cpu_start
                stop.set()
                await asyncio.gather(sampler, watcher)
        compiles = _attention_compiles_between(start_unix, start_unix + elapsed)
        if compiles:
            _write_json(
                output / "invalid_measurement.json",
                {"reason": "attention/cache-writer JIT during timed requests", "compiles": compiles},
            )
            raise ValueError(f"Attention/cache-writer JIT contaminated timed replay; preserved raw data: {output}")
        generated = sum(row["usage"]["completion_tokens"] for row in results)
        _write_json(
            output / "summary.json",
            {
                "measurement_version": 2,
                "attention_compiles_during_measurement": compiles,
                "requests": len(results),
                "measurement_start_unix_s": start_unix,
                "measurement_start_monotonic_s": start,
                "measurement_end_monotonic_s": measured_end,
                "peak_client_inflight": peak_active,
                "client_cpu_s": cpu_elapsed,
                "client_cpu_fraction_one_core": cpu_elapsed / elapsed,
                "event_loop_lag_s": _percentiles(lags),
                "event_loop_lag_max_s": max(lags, default=0.0),
                "event_loop_lag_samples": len(lags),
                "event_loop_stalls_over_20ms": stalls,
                "metric_samples": {
                    metric: sum(r[metric] is not None for r in results) for metric in ("ttft_s", "tpot_s", "e2e_s")
                },
                "elapsed_s": elapsed,
                "request_throughput": len(results) / elapsed,
                "generated_tokens": generated,
                "token_throughput": generated / elapsed,
                "concurrency": concurrency,
                "length_exhaustion": sum(r["finish_reason"] == "length" for r in results),
                **{
                    metric: _percentiles([r[metric] for r in results if r[metric] is not None])
                    for metric in ("ttft_s", "tpot_s", "e2e_s")
                },
                "tpot_definition": "(stream completion - first nonempty text)/(API generated tokens - 1); client metric",
                "latency_boundary": "from HTTP submission after client semaphore acquisition; excludes client slot waiting",
                "ttft_definition": "time to first nonempty text event; includes detokenization buffering",
            },
        )


def _validate_allocation(arm: str, allocation: dict[str, Any]) -> None:
    value_format, value_dtype = {
        "auto": ("bf16", "torch.bfloat16"),
        "bf16_int8": ("int8", "torch.int8"),
        "bf16_fp8_e4m3": ("fp8_e4m3", "torch.uint8"),
        "bf16_int4": ("int4", "torch.uint8"),
    }[arm]
    layers = allocation["layers"]
    if not layers:
        raise ValueError("Missing resident cache layers")
    total = 0
    capacity = allocation["token_capacity"]
    for layer in layers:
        tensors = layer["tensors"]
        expected_names = {"key", "value"} | ({"value_scale"} if arm != "auto" else set())
        if set(tensors) != expected_names or layer["key_format"] != "bf16" or layer["value_format"] != value_format:
            raise ValueError(f"Wrong resident cache format for {arm}")
        key, value = tensors["key"], tensors["value"]
        shape = key["shape"]
        if len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("Invalid key cache shape")
        value_shape = shape[:-1] + [(shape[-1] + 1) // 2 if arm == "bf16_int4" else shape[-1]]
        if key["dtype"] != "torch.bfloat16" or value["dtype"] != value_dtype or value["shape"] != value_shape:
            raise ValueError("Mixed cache dtype/shape mismatch")
        if arm != "auto":
            scale = tensors["value_scale"]
            if scale["dtype"] != "torch.float32" or scale["shape"] != shape[:-1]:
                raise ValueError("Wrong resident value scale allocation")
        for tensor in tensors.values():
            element_bytes = {"torch.bfloat16": 2, "torch.int8": 1, "torch.uint8": 1, "torch.float32": 4}[
                tensor["dtype"]
            ]
            if tensor["bytes"] != math.prod(tensor["shape"]) * element_bytes:
                raise ValueError("Resident tensor byte count mismatch")
        resident = sum(tensor["bytes"] for tensor in tensors.values())
        if (
            shape[0] * shape[1] != capacity
            or resident != layer["resident_bytes"]
            or resident != capacity * layer["bytes_per_token"]
        ):
            raise ValueError("Cache capacity/byte accounting mismatch")
        total += resident
    if allocation["resident_bytes"] != total:
        raise ValueError("Total resident cache byte count mismatch")


def _run(args: argparse.Namespace) -> None:
    if "XLLM_KV_QUALITY_MODE" in os.environ and os.environ["XLLM_KV_QUALITY_MODE"] != "none":
        raise ValueError("QDQ quality modes must be disabled for real compressed-cache experiments")
    manifest = json.loads((args.output / "trace_manifest.json").read_text())
    if args.joint_quality:
        if min(args.concurrency) < 8 or max(args.concurrency) > 256:
            raise ValueError("Joint full-trace protocol requires concurrency between 8 and 256")
        if "quality" in args.workloads:
            raise ValueError("Joint quality scores measured workload responses; no separate quality replay")
        reference_path = args.output / "trace_quality.jsonl"
        if _hash_file(reference_path) != manifest["traces"]["quality"]["sha256"]:
            raise ValueError(f"Trace was modified: {reference_path}")
        reference = evaluator._jsonl(reference_path)
        for workload in args.workloads:
            expected = {
                r["id"]: r["prompt_sha256"] for r in reference if (r["task"] == "gsm8k") == (workload == "gsm8k")
            }
            actual_rows = evaluator._jsonl(args.output / f"trace_{workload}.jsonl")
            actual = {r["id"]: r["prompt_sha256"] for r in actual_rows}
            if actual != expected or len(actual_rows) != len(expected):
                raise ValueError(f"Joint quality requires the full {workload} benchmark trace")
    gates = json.loads(args.gates.read_text())
    if gates.get("attention_backend", "triton") != args.attention_backend:
        raise ValueError("Correctness gate does not attest the selected attention backend")
    for arm in args.arms:
        if not gates.get(arm, {}).get("passed"):
            raise ValueError(f"Correctness gate not passed for {arm}")
    for relative_path, expected_hash in gates.get("source_sha256", {}).items():
        if _hash_file(ROOT / relative_path) != expected_hash:
            raise ValueError(f"Correctness gate is stale: {relative_path}")
    if gates.get("binary_sha256") != _hash_file(assets.BINARY):
        raise ValueError("Correctness gate does not attest the current server binary")
    source_paths = [
        ROOT / "tools/run_mixed_v_cache_experiment.py",
        ROOT / "tools/report_mixed_v_cache_experiment.py",
        ROOT / "tools/eval_kv_cache_quality.py",
        ROOT / "tools/run_kv_fp8_quality_ablation.py",
        *[
            ROOT / path
            for path in subprocess.check_output(
                ["git", "ls-files", "xllm/python", "xllm/core/framework/kv_cache"], cwd=ROOT, text=True
            ).splitlines()
        ],
    ]
    if args.attention_backend == "flashinfer_mixed":
        source_paths.extend(
            [
                ROOT / "xllm/python/attention/flashinfer_mixed.py",
                ROOT / "xllm/python/attention/flashinfer_mixed_build.py",
                *[
                    path
                    for path in (ROOT / "xllm/python/kernels_cuda/flashinfer_mixed").rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                ],
            ]
        )
    environment = {
        "attention_backend": args.attention_backend,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_diff": subprocess.check_output(["git", "diff"], cwd=ROOT, text=True),
        "source_sha256": {str(path.relative_to(ROOT)): _hash_file(path) for path in source_paths},
        "binary_sha256": _hash_file(assets.BINARY),
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "triton", "aiohttp", "numpy")},
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "PATH",
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES",
                "TRITON_CACHE_DIR",
                "XLLM_TELEMETRY_GPU",
                "XLLM_FLASHINFER_MIXED_KERNEL_VARIANT",
                "XLLM_MODEL_PATH",
            )
        },
        "gpu_static": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"], text=True
        ),
    }
    if args.attention_backend == "flashinfer_mixed":
        import flashinfer

        environment["packages"]["flashinfer"] = flashinfer.__version__
    env_path = args.output / "environment.json"
    if env_path.exists() and json.loads(env_path.read_text()) != environment:
        raise ValueError("Source/environment changed; use a new output directory")
    _write_json(env_path, environment)
    _write_json(
        args.output / f"invocation_{time.time_ns()}.json",
        {
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "argv": sys.argv,
            "gates_sha256": _hash_file(args.gates),
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        },
    )
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for relative in (
        "tools/run_mixed_v_cache_experiment.py",
        "tools/report_mixed_v_cache_experiment.py",
        "tools/eval_kv_cache_quality.py",
        "tools/run_kv_fp8_quality_ablation.py",
    ):
        (snapshot / Path(relative).name).write_bytes((ROOT / relative).read_bytes())
    for arm in args.arms:
        arm_root = args.output / arm
        arm_root.mkdir(exist_ok=True)
        command = [
            str(assets.BINARY),
            f"--model={assets.MODEL}",
            "--model_impl=python",
            f"--python_model_path={ROOT}",
            "--host=127.0.0.1",
            f"--port={args.port}",
            "--max_cache_size=0",
            "--enable_prefix_cache=false",
            f"--max_memory_utilization={args.max_memory_utilization}",
            *([] if args.use_default_token_budget else ["--max_tokens_per_batch=4096"]),
            f"--max_seqs_per_batch={args.max_seqs_per_batch}",
            f"--max_concurrent_requests={args.max_concurrent_requests}",
            f"--kv_cache_dtype={arm}",
            "--enable_graph=false",
            "--enable_prefill_piecewise_graph=false",
            "--python_graph_backend=off",
        ]
        env = dict(os.environ, XLLM_QUANTIZED_BACKEND=args.attention_backend, XLLM_KV_QUALITY_MODE="none")
        _write_json(
            arm_root / "server_command.json",
            {
                "command": command,
                "overrides": {name: env[name] for name in ("XLLM_QUANTIZED_BACKEND", "XLLM_KV_QUALITY_MODE")},
            },
        )
        url = f"http://127.0.0.1:{args.port}"
        log_path = arm_root / f"server_{time.time_ns()}.log"
        _write_json(arm_root / "latest_server.json", {"log": str(log_path)})
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited before API readiness: {log_path}")
                    try:
                        with urllib.request.urlopen(url + "/v1/models", timeout=2) as response:
                            models = json.loads(response.read())
                        if assets.MODEL_ID in {item.get("id") for item in models.get("data", [])}:
                            break
                    except (OSError, json.JSONDecodeError):
                        time.sleep(1)
                else:
                    raise TimeoutError(f"API readiness timed out: {log_path}")
                # Python caches bind lazily on the first forward, after API readiness.
                probe_trace = args.output / f"trace_{args.workloads[0]}.jsonl"
                if _hash_file(probe_trace) != manifest["traces"][args.workloads[0]]["sha256"]:
                    raise ValueError(f"Trace was modified: {probe_trace}")
                probe_row = evaluator._jsonl(probe_trace)[0]
                probe_result = asyncio.run(_bootstrap(url, probe_row))
                _write_json(
                    arm_root / "startup_probe.json",
                    {"trace_id": probe_row["id"], "measured": False, "response": probe_result},
                )
                deadline = time.monotonic() + 30
                while "KV_ALLOCATION " not in log_path.read_text(errors="replace"):
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"Server allocation inspection failed after first forward: {log_path}")
                    time.sleep(1)
                text = log_path.read_text(errors="replace")
                allocation = json.loads(text.split("KV_ALLOCATION ", 1)[1].splitlines()[0])
                _validate_allocation(arm, allocation)
                if args.attention_backend == "flashinfer_mixed":
                    if "FLASHINFER_MIXED_BUILD " not in text:
                        raise ValueError("Server did not load the mixed-V FlashInfer backend")
                    native = json.loads(text.split("FLASHINFER_MIXED_BUILD ", 1)[1].splitlines()[0])
                    for field in ("build_key", "artifact_sha256"):
                        if native[field] != gates.get("native_build", {}).get(field):
                            raise ValueError(f"Correctness gate is stale for the native kernel: {field}")
                    _write_json(arm_root / "native_build.json", native)
                options = [line for line in text.splitlines() if "Master init options" in line]
                if not options:
                    raise ValueError("Missing effective scheduler options in server log")
                effective = {}
                for name in ("max_tokens_per_batch", "max_tokens_per_chunk_for_prefill", "max_seqs_per_batch"):
                    match = re.search(r"\b" + name + r"\s*[:=]\s*(\d+)", options[-1])
                    if match is None:
                        raise ValueError(f"Missing effective scheduler option: {name}")
                    effective[name] = int(match.group(1))
                if effective["max_seqs_per_batch"] != args.max_seqs_per_batch:
                    raise ValueError("Effective batch sequence limit differs from requested configuration")
                expected_budget = 10240 if args.use_default_token_budget else 4096
                if effective["max_tokens_per_batch"] != expected_budget:
                    raise ValueError("Effective token budget differs from the pinned protocol")
                _write_json(arm_root / "scheduler_effective.json", effective)
                _write_json(arm_root / "allocation.json", allocation)
                for workload in args.workloads:
                    trace = args.output / f"trace_{workload}.jsonl"
                    if _hash_file(trace) != manifest["traces"][workload]["sha256"]:
                        raise ValueError(f"Trace was modified: {trace}")
                    rows = evaluator._jsonl(trace)
                    if workload != "quality":
                        rows = rows * args.replay_cycles
                    levels = [args.quality_concurrency] if workload == "quality" else args.concurrency
                    for concurrency in levels:
                        for repetition in range(args.repetitions):
                            run_path = arm_root / workload / f"c{concurrency}_r{repetition}"
                            run_config = {
                                "measurement_version": 2,
                                "attention_backend": args.attention_backend,
                                "arm": arm,
                                "workload": workload,
                                "concurrency": concurrency,
                                "repetition": repetition,
                                "replay_cycles": args.replay_cycles if workload != "quality" else 1,
                                "quality_protocol": "joint_full_trace" if args.joint_quality else "separate",
                                "trace_sha256": _hash_file(trace),
                                "warmup": args.warmup,
                                "warmup_requests": min(len(rows), max(args.warmup, concurrency)),
                                "warmup_concurrency": concurrency,
                                "server_command": command,
                            }
                            config_path = run_path / "run_config.json"
                            if (run_path / "summary.json").exists():
                                if json.loads(config_path.read_text()) != run_config:
                                    raise ValueError(f"Completed run has different settings: {run_path}")
                                continue
                            if len(rows) < concurrency:
                                raise ValueError(
                                    f"Trace has fewer requests than concurrency: {len(rows)} < {concurrency}"
                                )
                            _write_json(config_path, run_config)
                            _write_json(run_path / "allocation.json", allocation)
                            logger.info(
                                "Replay %s %s c=%d repetition=%d (%d real queries)",
                                arm,
                                workload,
                                concurrency,
                                repetition,
                                len(rows),
                            )
                            asyncio.run(_replay(url, rows, concurrency, run_path, args.warmup))
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gates", type=Path)
    parser.add_argument("--attention-backend", choices=("triton", "flashinfer_mixed"), default="triton")
    parser.add_argument("--performance-samples", type=int, default=0, help="0 selects the full benchmark traces")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--workloads", nargs="+", choices=("gsm8k", "longbench", "quality"), default=["gsm8k", "longbench"]
    )
    parser.add_argument("--joint-quality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--replay-cycles", type=int, default=2)
    parser.add_argument("--quality-concurrency", type=int, default=16)
    parser.add_argument("--max-memory-utilization", type=float, default=0.93)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--port", type=int, default=18995)
    parser.add_argument("--use-default-token-budget", action="store_true")
    parser.add_argument("--max-seqs-per-batch", type=int, default=128)
    parser.add_argument("--max-concurrent-requests", type=int, default=200)
    args = parser.parse_args()
    if (
        not 0 < args.max_memory_utilization < 1
        or args.performance_samples < 0
        or 0 < args.performance_samples < 8
        or min(args.concurrency) <= 0
        or min(args.repetitions, args.replay_cycles, args.quality_concurrency, args.warmup) <= 0
    ):
        parser.error(
            "Counts must be positive, memory utilization in (0, 1), performance samples 0 (full) or at least 8"
        )
    if args.max_seqs_per_batch <= 0 or args.max_concurrent_requests < max(args.concurrency):
        parser.error("Positive batch limit and admission limit >= client concurrency required")
    if args.action == "prepare":
        _prepare(args.output, args.performance_samples)
    else:
        if args.gates is None:
            parser.error("--gates is required for run")
        _run(args)


if __name__ == "__main__":
    main()
