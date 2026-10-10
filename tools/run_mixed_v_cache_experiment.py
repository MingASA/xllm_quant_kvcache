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
import os
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
    traces["gsm8k"] = [row for row in quality if row["task"] == "gsm8k"][:performance_samples]
    long_rows = {task: [row for row in quality if row["task"] == task] for task in assets.TASKS}
    traces["longbench"] = [
        long_rows[task][i] for i in range(min(len(rows) for rows in long_rows.values())) for task in assets.TASKS
    ][:performance_samples]
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
            "performance_order": "fixed prefixes; LongBench round robin in existing eight-task order",
        },
    )


async def _request(session: Any, url: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": assets.MODEL_ID,
        "temperature": 0,
        "top_p": 1,
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
            for choice in item.get("choices", []):
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
    tokens = int(usage["completion_tokens"])
    if int(usage["prompt_tokens"]) != row["prompt_tokens"]:
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
        "tpot_s": (times[-1] - times[0]) / (tokens - 1) if times and tokens > 1 else None,
        "e2e_s": end - start,
        "stream_events": len(times),
        "stream_event_times_s": [t - start for t in times],
    }


async def _sample(session: Any, url: str, output: Path, stop: asyncio.Event) -> None:
    with output.open("w") as stream:
        while not stop.is_set():
            record: dict[str, Any] = {"unix_time": time.time()}
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            record["gpu_raw"] = stdout.decode().strip()
            if process.returncode:
                record["gpu_error"] = stderr.decode().strip()
            # Telemetry is best-effort at its external API boundary; failures remain visible in raw data.
            try:
                async with session.get(url + "/brpc_metrics", timeout=2) as response:
                    record["metrics_status"] = response.status
                    record["metrics"] = await response.text() if response.status == 200 else None
            except (OSError, asyncio.TimeoutError) as error:
                record["metrics_error"] = str(error)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass


def _percentiles(values: list[float]) -> dict[str, float] | None:
    import numpy as np

    if not values:
        return None
    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)}


async def _replay(url: str, rows: list[dict[str, Any]], concurrency: int, output: Path, warmup: int) -> None:
    import aiohttp

    output.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=3600), connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        for index, row in enumerate(rows[:warmup]):
            await _request(session, url, row, index)
        semaphore = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = []
        stop = asyncio.Event()
        sampler = asyncio.create_task(_sample(session, url, output / "telemetry.jsonl", stop))
        with (output / "responses.jsonl").open("x") as journal:

            async def execute(index: int, row: dict[str, Any]) -> None:
                async with semaphore:
                    result = await _request(session, url, row, index)
                    results.append(result)
                    journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                    journal.flush()

            start = time.perf_counter()
            try:
                async with asyncio.TaskGroup() as group:
                    for index, row in enumerate(rows):
                        group.create_task(execute(index, row))
            finally:
                elapsed = time.perf_counter() - start
                stop.set()
                await sampler
        generated = sum(row["usage"]["completion_tokens"] for row in results)
        _write_json(
            output / "summary.json",
            {
                "requests": len(results),
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
                "tpot_definition": "(last nonempty stream event - first event)/(API generated tokens - 1)",
                "ttft_definition": "time to first nonempty text event; includes detokenization buffering",
            },
        )


def _run(args: argparse.Namespace) -> None:
    if "XLLM_KV_QUALITY_MODE" in os.environ and os.environ["XLLM_KV_QUALITY_MODE"] != "none":
        raise ValueError("QDQ quality modes must be disabled for real compressed-cache experiments")
    manifest = json.loads((args.output / "trace_manifest.json").read_text())
    gates = json.loads(args.gates.read_text())
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
        *[
            ROOT / path
            for path in subprocess.check_output(
                ["git", "ls-files", "xllm/python", "xllm/core/framework/kv_cache"], cwd=ROOT, text=True
            ).splitlines()
        ],
    ]
    environment = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_diff": subprocess.check_output(["git", "diff"], cwd=ROOT, text=True),
        "source_sha256": {str(path.relative_to(ROOT)): _hash_file(path) for path in source_paths},
        "binary_sha256": _hash_file(assets.BINARY),
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "triton", "aiohttp", "numpy")},
        "runtime_environment": {
            name: os.environ.get(name) for name in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")
        },
        "gpu_static": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"], text=True
        ),
    }
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
            "--max_memory_utilization=0.87",
            "--max_tokens_per_batch=4096",
            "--max_seqs_per_batch=128",
            f"--kv_cache_dtype={arm}",
            "--enable_graph=false",
            "--enable_prefill_piecewise_graph=false",
            "--python_graph_backend=off",
        ]
        env = dict(os.environ, XLLM_QUANTIZED_BACKEND="triton", XLLM_KV_QUALITY_MODE="none")
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
                while "KV_ALLOCATION " not in log_path.read_text(errors="replace"):
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"Server allocation/readiness failed: {log_path}")
                    time.sleep(1)
                text = log_path.read_text(errors="replace")
                allocation = json.loads(text.split("KV_ALLOCATION ", 1)[1].splitlines()[0])
                _write_json(arm_root / "allocation.json", allocation)
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
                                "arm": arm,
                                "workload": workload,
                                "concurrency": concurrency,
                                "repetition": repetition,
                                "replay_cycles": args.replay_cycles if workload != "quality" else 1,
                                "trace_sha256": _hash_file(trace),
                                "warmup": args.warmup,
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
    parser.add_argument("--performance-samples", type=int, default=64)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--workloads", nargs="+", choices=("gsm8k", "longbench", "quality"), default=["gsm8k", "longbench", "quality"]
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--replay-cycles", type=int, default=4)
    parser.add_argument("--quality-concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--port", type=int, default=18995)
    args = parser.parse_args()
    if (
        args.performance_samples < 8
        or min(args.concurrency) <= 0
        or min(args.repetitions, args.replay_cycles, args.quality_concurrency, args.warmup) <= 0
    ):
        parser.error("Counts must be positive; performance trace needs at least eight samples")
    if args.action == "prepare":
        _prepare(args.output, args.performance_samples)
    else:
        if args.gates is None:
            parser.error("--gates is required for run")
        _run(args)


if __name__ == "__main__":
    main()
