# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""One paired Qwen14B semantic/SSE and resident-page check, not an AIPerf runner."""

from __future__ import annotations

import argparse
import difflib
import importlib.metadata
import json
import math
import os
import re
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger
from tools.eval_kv_cache_quality import _token_count
from tools.qwen25_experiment_model import checkpoint_provenance, sha256_file
from tools.run_qwen25_joint_kv_quality import _stop, _validate_allocation, _wait_ready

PROMPTS = [
    {
        "prompt": "A customer buys 3 boxes containing 4 pencils each. Each pencil costs 2 dollars. "
        "They pay 30 dollars. How much change do they receive? Explain briefly, then write #### followed by the answer.",
        "expected_regex": r"####\s*6\b",
    },
    {
        "prompt": "Track the object carefully. At first the key is in the red box. Mira moves it into the blue box. "
        "Leo moves the empty red box to the shelf. Mira gives the blue box to Nora. "
        "Which color box currently contains the key? Reply with only the color.",
        "expected_regex": r"(?i)\bblue\b",
    },
    {
        "prompt": "Read the code without executing it: xs = [2, 3, 5]; ys = [x*x for x in xs if x % 2]; "
        "print(sum(ys)). What exact integer is printed? Reply with only the integer.",
        "expected_regex": r"\b34\b",
    },
]


def _stream(url: str, prompt: str, timeout: int) -> dict:
    body = {
        "model": "Qwen2.5-14B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "seed": 17,
        "max_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True},
        "stop_token_ids": [151645, 151643],
    }
    request = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks, parts, usage, finish, done = [], [], None, None, False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise ValueError("Expected OpenAI-compatible SSE Content-Type")
        for line in response:
            line = line.decode().strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                raise ValueError(f"Unexpected SSE line: {line}")
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise ValueError(f"Stream error: {chunk['error']}")
            chunks.append(chunk)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    parts.append(choice["delta"]["content"])
                finish = choice.get("finish_reason") or finish
    if (
        not done
        or finish not in ("stop", "length")
        or not parts
        or not usage
        or type(usage.get("completion_tokens")) is not int
        or usage["completion_tokens"] <= 0
    ):
        raise ValueError("Incomplete SSE/usage")
    return {"prediction": "".join(parts), "usage": usage, "finish_reason": finish, "sse_done": done, "chunks": chunks}


def _evidence(log: Path, expected: str, graph: bool, block_size: int, profile: dict) -> dict:
    text = log.read_text(errors="replace")

    def records(marker: str) -> list[dict]:
        return [json.loads(line.split(marker, 1)[1]) for line in text.splitlines() if marker in line]

    backends, allocations = records("NATIVE_KV_BACKEND "), records("KV_ALLOCATION ")
    if len(backends) != 1 or len(allocations) != 1:
        raise ValueError("Require exactly one runtime backend marker and allocation per single-GPU worker")
    backend = backends[0]
    wanted_class = "FlashInferBackend" if expected == "auto" else "FlashInferFP8Backend"
    if any(
        backend.get(key) != value
        for key, value in {
            "class": wanted_class,
            "kv_cache_dtype": expected,
            "graph_enabled": graph,
            "attention_impl": "flashinfer",
            "planner_backend": "fa2",
            "paged_prefill": True,
            "quality_transform": False,
        }.items()
    ):
        raise ValueError(f"Unexpected actual backend: {backend}")
    if graph:
        replays = records("NATIVE_KV_GRAPH_REPLAY ")
        if not backend.get("graph_captured_batches") or len(replays) != 1 or replays[0].get("class") != wanted_class:
            raise ValueError("Graph check requires captured buckets and actual decode replay evidence")
    _validate_allocation(expected, allocations[0], profile, block_size)
    return {"backend": backend, "allocation": allocations[0]}


def _run(args: argparse.Namespace, arm: str, scales: tuple[float, float] | None, profile: dict) -> list[dict]:
    root = args.output / arm
    root.mkdir()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpu,
        XLLM_KV_QUALITY_MODE="none",
        XLLM_FLASHINFER_NATIVE_KV="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
    )
    for key in ("XLLM_QUANTIZED_BACKEND", "XLLM_FP8_CALIBRATION_OUTPUT"):
        env.pop(key, None)
    if arm == "fp8_e4m3":
        env.update(
            XLLM_QUANTIZED_BACKEND="flashinfer", XLLM_FP8_K_SCALE=str(scales[0]), XLLM_FP8_V_SCALE=str(scales[1])
        )
    elif scales is None:
        env["XLLM_FP8_CALIBRATION_OUTPUT"] = str(args.output / "calibration.json")
    command = [
        str(args.server_binary.resolve()),
        f"--model={args.model.resolve()}",
        "--model_id=Qwen2.5-14B-Instruct",
        "--model_impl=python",
        f"--python_model_path={ROOT}",
        f"--port={args.port}",
        "--host=127.0.0.1",
        f"--kv_cache_dtype={arm}",
        f"--python_graph_backend={args.graph}",
        f"--enable_graph={'true' if args.graph != 'off' else 'false'}",
        "--enable_prefill_piecewise_graph=false",
        "--enable_prefix_cache=false",
        "--max_cache_size=0",
        f"--max_memory_utilization={args.max_memory_utilization}",
        "--block_size=128",
        f"--max_tokens_per_batch={args.max_tokens_per_batch}",
        f"--max_seqs_per_batch={args.max_seqs_per_batch}",
    ]
    (root / "launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "environment": {
                    key: value for key, value in env.items() if key.startswith("XLLM_") or key == "CUDA_VISIBLE_DEVICES"
                },
            },
            indent=2,
        )
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    log_path = root / "server.log"
    logger.info("Checking %s actual service, graph=%s", arm, args.graph)
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            _wait_ready(process, args)
            outputs = []
            for row in PROMPTS:
                result = _stream(f"http://127.0.0.1:{args.port}/v1", row["prompt"], args.timeout)
                result["semantic_match"] = bool(re.search(row["expected_regex"], result["prediction"]))
                outputs.append(result)
            if max(row["usage"]["completion_tokens"] for row in outputs) <= 1:
                raise ValueError("Need at least one multi-token answer to exercise autoregressive decode")
            (root / "responses.json").write_text(json.dumps(outputs, indent=2))
            evidence = _evidence(log_path, arm, args.graph != "off", 128, profile)
            if arm == "fp8_e4m3" and (evidence["backend"]["k_scale"], evidence["backend"]["v_scale"]) != scales:
                raise ValueError("Actual FP8 scales differ from the measured/selected values")
            (root / "evidence.json").write_text(json.dumps(evidence, indent=2))
            return outputs
        finally:
            _stop(process)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=18997)
    parser.add_argument("--graph", choices=("off", "cudagraphs"), default="off")
    parser.add_argument("--fp8-k-scale", type=float)
    parser.add_argument("--fp8-v-scale", type=float)
    parser.add_argument("--scales-file", type=Path, help="Prior calibration.json, from this same checkpoint")
    parser.add_argument("--max-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-tokens-per-batch", type=int, default=10240)
    parser.add_argument("--max-seqs-per-batch", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args()
    if (
        any(char in args.gpu for char in ", \t\n")
        or not args.gpu
        or not 1 <= args.port <= 65535
        or not 0 < args.max_memory_utilization < 1
        or min(args.max_tokens_per_batch, args.max_seqs_per_batch, args.timeout, args.startup_timeout) <= 0
    ):
        parser.error("Use exactly one GPU, a valid port and positive resource limits")
    if args.scales_file and (args.fp8_k_scale is not None or args.fp8_v_scale is not None):
        parser.error("Use either --scales-file or both explicit scales")
    if args.scales_file:
        recorded = json.loads(args.scales_file.read_text())
        args.fp8_k_scale, args.fp8_v_scale = recorded["fp8_k_scale"], recorded["fp8_v_scale"]
    scales = None
    if args.fp8_k_scale is not None or args.fp8_v_scale is not None:
        if not all(
            value is not None and math.isfinite(value) and value > 0 for value in (args.fp8_k_scale, args.fp8_v_scale)
        ):
            parser.error("Both finite positive scales are required")
        scales = (args.fp8_k_scale, args.fp8_v_scale)
    if args.graph != "off" and scales is None:
        parser.error("Measure scales once in eager mode before a graph comparison")
    model = checkpoint_provenance(args.model)
    if model["profile"]["size"] != "14b" or not os.access(args.server_binary, os.X_OK):
        parser.error("Need local Qwen2.5-14B and its runnable target-device server binary")
    if args.scales_file:
        source_manifest = args.scales_file.parent / "environment.json"
        if not source_manifest.is_file() or json.loads(source_manifest.read_text())["model"] != model:
            raise ValueError("Scales must come from a service check with matching checkpoint provenance")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {
        "model": model,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "server_sha256": sha256_file(args.server_binary),
        "prompts": PROMPTS,
        "source_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "xllm/python/attention/flashinfer.py",
                "xllm/python/attention/flashinfer_fp8.py",
                "xllm/python/attention/native_kv.py",
                "xllm/python/model_executor/executor.py",
                "xllm/python/model_executor/runners/decode_cuda_graph.py",
                "tools/check_native_flashinfer_service.py",
            )
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "flashinfer-python",
                "triton",
                "transformers",
            )
        },
        "scope": "semantic/SSE correctness only; no performance or benchmark quality claims",
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))
    baseline = _run(args, "auto", scales, model["profile"])
    if scales is None:
        proposal = json.loads((args.output / "calibration.json").read_text())
        if len(proposal["observations"]) != 2 * model["profile"]["num_layers"]:
            raise ValueError("Scale evidence must include prefill and decode for every layer")
        scales = (proposal["fp8_k_scale"], proposal["fp8_v_scale"])
    candidate = _run(args, "fp8_e4m3", scales, model["profile"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    differences = []
    for prompt, bf16, fp8 in zip(PROMPTS, baseline, candidate):
        local_tokens = _token_count(tokenizer, prompt["prompt"], True)
        if any(row["usage"].get("prompt_tokens") != local_tokens for row in (bf16, fp8)):
            raise ValueError("Local/API prompt tokens differ; paired preprocessing was not verified")
        left, right = (tokenizer.encode(row["prediction"], add_special_tokens=False) for row in (bf16, fp8))
        differences.append(
            {
                "bf16_semantic_match": bf16["semantic_match"],
                "fp8_semantic_match": fp8["semantic_match"],
                "identical_text": bf16["prediction"] == fp8["prediction"],
                "retokenized_output_similarity": difflib.SequenceMatcher(a=left, b=right).ratio(),
                "first_different_retokenized_output_token": next(
                    (i for i, (x, y) in enumerate(zip(left, right)) if x != y),
                    min(len(left), len(right)) if left != right else None,
                ),
                "length_exhaustion": [row["finish_reason"] == "length" for row in (bf16, fp8)],
            }
        )
    (args.output / "report.json").write_text(
        json.dumps({"fp8_k_scale": scales[0], "fp8_v_scale": scales[1], "differences": differences}, indent=2)
    )
    if not all(row["semantic_match"] for row in baseline + candidate):
        raise ValueError("Semantic check failed; inspect responses/report before any performance experiment")
    logger.info("Native paired service evidence saved: %s", args.output)


if __name__ == "__main__":
    main()
