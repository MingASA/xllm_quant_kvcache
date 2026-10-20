# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""Audit and run single-GPU Qwen2.5-14B joint-KV quality experiments."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger
from tools import eval_kv_cache_quality as evaluator
from tools.qwen25_experiment_model import checkpoint_provenance, read_profile, sha256_file

ARMS = ("auto", "int8", "fp8_e4m3", "int4")
EVALUATOR = ROOT / "tools/eval_kv_cache_quality.py"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path)
    parser.add_argument("--gsm8k-shots", type=Path)
    parser.add_argument("--longbench-data", type=Path)
    parser.add_argument("--longbench-repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-binary", type=Path, default=ROOT / "build/cmake.cuda-x86/xllm/xllm")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("gsm8k", "longbench", *evaluator.LONG_BENCH_TASKS),
        default=["gsm8k", "longbench"],
    )
    parser.add_argument("--gpu", default="0", help="Exactly one visible GPU index or UUID")
    parser.add_argument("--port", type=int, default=18996)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-seqs-per-batch", type=int, default=200)
    parser.add_argument("--max-tokens-per-batch", type=int, default=10240)
    parser.add_argument("--max-concurrent-requests", type=int, default=200)
    parser.add_argument("--max-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-cache-size", type=int, default=0, help="Native xLLM cache budget, bytes; 0=automatic")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--limit", type=int, help="Smoke only: fixed prefix per task, never full quality evidence")
    parser.add_argument(
        "--approved-data-manifest",
        type=Path,
        help="Reviewed dry-run JSON: data hashes and unique counts must match before a full run",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="JSON plan/audit only; no GPU, services or output files")
    return parser


def _task_paths(args: argparse.Namespace) -> dict[str, Path]:
    tasks = dict.fromkeys(
        t for choice in args.tasks for t in (evaluator.LONG_BENCH_TASKS if choice == "longbench" else (choice,))
    )
    paths = {}
    for task in tasks:
        if task == "gsm8k":
            if args.gsm8k is None or args.gsm8k_shots is None:
                raise ValueError("GSM8K requires explicit --gsm8k and --gsm8k-shots")
            paths[task] = args.gsm8k.resolve()
        else:
            if args.longbench_data is None or args.longbench_repo is None:
                raise ValueError("LongBench requires explicit --longbench-data and --longbench-repo")
            paths[task] = args.longbench_data.resolve() / f"{task}.jsonl"
    return paths


def _validate_args(args: argparse.Namespace) -> dict[str, Path]:
    if not args.gpu.strip() or any(c in args.gpu for c in ", \t\n") or args.gpu == "-1":
        raise ValueError("--gpu must identify exactly one GPU")
    if not 1 <= args.port <= 65535 or args.concurrency < 1 or args.concurrency > args.max_seqs_per_batch:
        raise ValueError("Invalid port/concurrency or concurrency exceeds max_seqs_per_batch")
    if args.max_concurrent_requests and args.max_concurrent_requests < args.concurrency:
        raise ValueError("Admission budget must cover concurrency")
    if not 0 < args.max_memory_utilization < 1:
        raise ValueError("Memory utilization must be between 0 and 1")
    if min(args.max_tokens_per_batch, args.block_size, args.timeout, args.startup_timeout) < 1:
        raise ValueError("Token budget, block size and timeouts must be positive")
    if args.max_cache_size < 0 or args.max_concurrent_requests < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("Invalid cache/admission/limit")
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("Duplicate arms")
    return _task_paths(args)


def _data_audit(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    audit: dict[str, Any] = {"datasets": {}, "protocol_files": {}}
    for task, path in paths.items():
        rows = evaluator._jsonl(path)
        if not rows:
            raise ValueError(f"Empty dataset: {task}")
        required = ("question", "answer") if task == "gsm8k" else ("context", "input", "answers")
        if any(any(key not in row for key in required) for row in rows):
            raise ValueError(f"Missing dataset fields in {task}")
        content = [row["question"] if task == "gsm8k" else (row["context"], row["input"]) for row in rows]
        unique = len({_digest(item) for item in content})
        ids = [row.get("_id", row.get("id")) for row in rows]
        ids = [str(item) for item in ids if item is not None]
        if unique != len(rows) or len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate questions/IDs in {task}; do not concatenate repeated QDQ rounds")
        audit["datasets"][task] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "unique_questions": unique,
            "selected_rows": len(rows[: args.limit]),
        }
    if "gsm8k" in paths:
        shots = json.loads(args.gsm8k_shots.read_text())
        if (
            not isinstance(shots, list)
            or len(shots) != 5
            or any(set(("question", "answer")) - row.keys() for row in shots)
        ):
            raise ValueError("Exactly five question/answer few-shot examples are required")
        audit["protocol_files"]["gsm8k_shots"] = sha256_file(args.gsm8k_shots)
    if any(task != "gsm8k" for task in paths):
        for name in ("config/dataset2prompt.json", "config/dataset2maxlen.json", "metrics.py"):
            audit["protocol_files"][f"LongBench/{name}"] = sha256_file(args.longbench_repo / "LongBench" / name)
    return audit


def _server_command(args: argparse.Namespace, arm: str) -> list[str]:
    return [
        str(args.server_binary.resolve()),
        f"--model={args.model.resolve()}",
        "--model_id=Qwen2.5-14B-Instruct",
        "--model_impl=python",
        f"--python_model_path={ROOT}",
        "--host=127.0.0.1",
        f"--port={args.port}",
        f"--kv_cache_dtype={arm}",
        "--enable_graph=false",
        "--enable_prefill_piecewise_graph=false",
        "--python_graph_backend=off",
        "--enable_prefix_cache=false",
        f"--max_cache_size={args.max_cache_size}",
        f"--block_size={args.block_size}",
        f"--max_memory_utilization={args.max_memory_utilization}",
        f"--max_tokens_per_batch={args.max_tokens_per_batch}",
        f"--max_seqs_per_batch={args.max_seqs_per_batch}",
        f"--max_concurrent_requests={args.max_concurrent_requests}",
    ]


def _environment(args: argparse.Namespace) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "XLLM_QUANTIZED_BACKEND": "triton",
        "XLLM_KV_QUALITY_MODE": "none",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _evaluator_command(
    args: argparse.Namespace, task: str, path: Path, arm: str, manifest: Path, resume: bool
) -> list[str]:
    label = arm.replace("_", "-")
    command = [
        sys.executable,
        "-B",
        str(EVALUATOR),
        "--task",
        task,
        "--data",
        str(path),
        "--tokenizer",
        str(args.model.resolve()),
        "--model",
        "Qwen2.5-14B-Instruct",
        "--arm",
        label,
        "--url",
        f"http://127.0.0.1:{args.port}/v1",
        "--server-manifest",
        str(manifest),
        "--output",
        str(args.output.resolve() / "runs" / task),
        "--seed",
        str(args.seed),
        "--concurrency",
        str(args.concurrency),
        "--timeout",
        str(args.timeout),
        "--stop-token-ids",
        "151645",
        "151643",
        "--strict-token-count",
    ]
    command.extend(
        ("--gsm8k-shots", str(args.gsm8k_shots.resolve()))
        if task == "gsm8k"
        else ("--longbench-repo", str(args.longbench_repo.resolve()))
    )
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if resume:
        command.append("--resume")
    return command


def _plan(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "scope": "smoke_only" if args.limit else "one_unique_dataset_round",
        "tasks": list(paths),
        "arms": args.arms,
        "environment": _environment(args),
        "resource_configuration": {
            key: getattr(args, key)
            for key in (
                "concurrency",
                "max_seqs_per_batch",
                "max_tokens_per_batch",
                "max_concurrent_requests",
                "max_memory_utilization",
                "max_cache_size",
                "block_size",
                "port",
                "gpu",
            )
        },
        "seed": args.seed,
        "limit": args.limit,
        "servers": {arm: _server_command(args, arm) for arm in args.arms},
        "evaluation": {
            arm: {
                task: _evaluator_command(
                    args,
                    task,
                    path,
                    arm,
                    args.output.resolve() / "arms" / arm / "ATTEMPT" / "server_manifest.json",
                    False,
                )
                for task, path in paths.items()
            }
            for arm in args.arms
        },
    }


def _validate_allocation(arm: str, allocation: dict[str, Any], profile: dict[str, Any], block_size: int) -> None:
    """Attest actual resident payloads and scales, including packed INT4 and FP8 bytes."""
    fmt, dtype, size = {
        "auto": ("bf16", "torch.bfloat16", 2),
        "int8": ("int8", "torch.int8", 1),
        "fp8_e4m3": ("fp8_e4m3", "torch.uint8", 1),
        "int4": ("int4", "torch.uint8", 1),
    }[arm]
    layers = allocation.get("layers", [])
    capacity = allocation.get("token_capacity")
    if len(layers) != profile["num_layers"] or type(capacity) is not int or capacity <= 0:
        raise ValueError("Invalid cache layer count/capacity")
    total = 0
    for layer in layers:
        if layer.get("key_format") != fmt or layer.get("value_format") != fmt:
            raise ValueError("Actual K/V formats do not match joint quantization arm")
        tensors = layer["tensors"]
        names = {"key", "value"} | ({"key_scale", "value_scale"} if arm != "auto" else set())
        if set(tensors) != names:
            raise ValueError("Wrong resident K/V scale set (BF16 QDQ is not compressed cache)")
        shape = tensors["key"]["shape"]
        dim = (profile["head_dim"] + 1) // 2 if arm == "int4" else profile["head_dim"]
        if (
            len(shape) != 4
            or any(type(n) is not int or n <= 0 for n in shape)
            or shape[1:] != [block_size, profile["num_kv_heads"], dim]
            or shape[0] * block_size != capacity
        ):
            raise ValueError("Resident K/V shape or token capacity mismatch")
        for name in names:
            tensor = tensors[name]
            expected_shape = shape[:-1] if name.endswith("scale") else shape
            expected_dtype = "torch.float32" if name.endswith("scale") else dtype
            element_bytes = 4 if name.endswith("scale") else size
            if (
                tensor["shape"] != expected_shape
                or tensor["dtype"] != expected_dtype
                or tensor["bytes"] != math.prod(expected_shape) * element_bytes
            ):
                raise ValueError(f"Invalid resident tensor: {name}")
        resident = sum(t["bytes"] for t in tensors.values())
        if resident != layer["resident_bytes"] or resident != capacity * layer["bytes_per_token"]:
            raise ValueError("Resident cache byte accounting mismatch")
        total += resident
    if total != allocation["resident_bytes"]:
        raise ValueError("Total cache byte accounting mismatch")


def _prepared(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    assets = evaluator._longbench_assets(args.longbench_repo) if any(t != "gsm8k" for t in paths) else None
    shots = json.loads(args.gsm8k_shots.read_text()) if "gsm8k" in paths else None
    result = {}
    for task, path in paths.items():
        rows = evaluator._jsonl(path)[: args.limit]
        chat = task not in ("lcc", "repobench-p")
        prompts = [
            evaluator._gsm_prompt(row["question"], shots)
            if task == "gsm8k"
            else assets[0][task].format(context=row["context"], input=row["input"])
            for row in rows
        ]
        cap = 512 if task == "gsm8k" else int(assets[1][task])
        counts = [evaluator._token_count(tokenizer, prompt, chat) for prompt in prompts]
        if any(n + cap > evaluator.MAX_CONTEXT_TOKENS for n in counts):
            raise ValueError(f"{task}: context exceeded; no truncation or resampling")
        result[task] = {"rows": rows, "prompts": prompts, "counts": counts, "chat": chat, "max_tokens": cap}
    return result


def _verify_journal(args: argparse.Namespace, task: str, arm: str, prepared: dict[str, Any]) -> set[int] | None:
    label = arm.replace("_", "-")
    result = args.output / "runs" / task / f"{label}.jsonl"
    config = result.with_name(f"{label}.config.json")
    if not result.exists() and not config.exists():
        return None
    if not args.resume:
        raise FileExistsError(f"Existing results require --resume: {result}")
    if not result.exists() or not config.exists():
        raise ValueError("Resume requires both config and durable journal; existing files preserved")
    saved = json.loads(config.read_text())
    if saved["arm"] != label or saved["examples"] != len(prepared["rows"]):
        raise ValueError("Resume arm/count mismatch")
    data = _task_paths(args)[task]
    expected = {
        "task": task,
        "data": str(data),
        "input_sha256": sha256_file(data),
        "model": "Qwen2.5-14B-Instruct",
        "tokenizer": str(args.model.resolve()),
        "model_impl": "python",
        "kv_cache_dtype": arm,
        "quality_transform_mode": "none",
        "endpoint": f"http://127.0.0.1:{args.port}/v1",
        "sampling": {"temperature": 0, "top_p": 1, "seed": args.seed},
        "generation": {
            "temperature": 0,
            "top_p": 1,
            "seed": args.seed,
            "max_tokens": prepared["max_tokens"],
            "stop_token_ids": [151645, 151643],
            "stop_strings": ["Question:", "</s>", "<|im_end|>"] if task == "gsm8k" else [],
        },
        "max_tokens": prepared["max_tokens"],
        "max_prompt_plus_generation_tokens": evaluator.MAX_CONTEXT_TOKENS,
        "api_mode": "chat-completions" if prepared["chat"] else "completions",
        "execution_protocol": {"request_concurrency": args.concurrency, "max_seqs_per_batch": args.max_seqs_per_batch},
        "runtime_versions": evaluator._runtime_versions(task),
    }
    if task == "gsm8k":
        expected["gsm8k_shots_sha256"] = sha256_file(args.gsm8k_shots)
    else:
        expected["longbench_revision"] = evaluator._longbench_assets(args.longbench_repo)[3]
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Resume config differs: {key}; files preserved")
    provenance = json.loads((args.output / "experiment_manifest.json").read_text())["model_provenance"]
    manifest = saved["server_manifest"]
    evidence = manifest.get("server_attestation", {})
    if manifest != _server_manifest(args, arm, provenance, evidence):
        raise ValueError("Saved server manifest differs from the current experiment")
    attempt = Path(evidence["attempt"]).resolve()
    if not attempt.is_relative_to((args.output / "arms" / arm).resolve()):
        raise ValueError("Allocation evidence is outside this arm's output directory")
    if (
        sha256_file(attempt / "allocation.json") != evidence["allocation_sha256"]
        or json.loads((attempt / "server_manifest.json").read_text()) != manifest
        or sha256_file(attempt / "server_manifest.json") != saved["server_manifest_sha256"]
    ):
        raise ValueError("Saved allocation/server manifest evidence changed")
    _validate_allocation(
        arm, json.loads((attempt / "allocation.json").read_text()), provenance["profile"], args.block_size
    )
    completed = evaluator._validate_resume_prefix(
        result, prepared["rows"], prepared["prompts"], prepared["counts"], prepared["chat"]
    )
    for row in evaluator._jsonl(result):
        if any(row.get(key) != value for key, value in prepared["rows"][row["_eval_index"]].items()):
            raise ValueError("Journal source record fields were changed")
        choice = row["api_response"]["choices"][0]
        usage = row["usage"]
        if choice.get("finish_reason") not in ("stop", "length"):
            raise ValueError("Journal contains incomplete/failed generation")
        completion = usage.get("completion_tokens")
        if (
            type(completion) is not int
            or not 0 <= completion <= prepared["max_tokens"]
            or usage.get("total_tokens") != usage["prompt_tokens"] + completion
            or any(
                usage.get(key) != row["api_response"].get("usage", {}).get(key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
        ):
            raise ValueError("Journal completion token usage is invalid")
    return completed


def _server_manifest(
    args: argparse.Namespace, arm: str, provenance: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    template = json.loads((args.model / "tokenizer_config.json").read_text()).get("chat_template")
    return {
        "server_started": True,
        "model_impl": "python",
        "graph_mode": "off",
        "kv_cache_mode": arm,
        "quality_transform_mode": "none",
        "backend": "triton",
        "model": "Qwen2.5-14B-Instruct",
        "checkpoint_path": str(args.model.resolve()),
        "checkpoint_sha256": provenance["weights"]["sha256"],
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "chat_template_sha256": _digest(template),
        "runtime_configuration": {
            key: getattr(args, key)
            for key in (
                "max_seqs_per_batch",
                "max_tokens_per_batch",
                "max_memory_utilization",
                "max_cache_size",
                "max_concurrent_requests",
                "block_size",
            )
        },
        "server_attestation": evidence,
    }


def _wait_ready(process: subprocess.Popen, args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited before readiness; see this attempt's server.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2) as response:
                if "Qwen2.5-14B-Instruct" in {item.get("id") for item in json.load(response).get("data", [])}:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise TimeoutError("Server readiness timed out")


def _stop(process: subprocess.Popen) -> None:
    # Only the session created by this runner; also clean up workers if its leader exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=15)


def _attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while (path := root / f"attempt-{number:04d}").exists():
        number += 1
    path.mkdir()
    return path


def _run_arm(
    args: argparse.Namespace,
    paths: dict[str, Path],
    arm: str,
    prepared: dict[str, Any],
    provenance: dict[str, Any],
    env: dict[str, str],
) -> None:
    pending = []
    for task in paths:
        indices = _verify_journal(args, task, arm, prepared[task])
        if indices is None or len(indices) < len(prepared[task]["rows"]):
            pending.append(task)
    if not pending:
        logger.info("%s: all task journals verified; no server started", arm)
        return
    attempt = _attempt(args.output / "arms" / arm)
    logger.info("Starting %s: %d pending tasks; evidence in %s", arm, len(pending), attempt)
    command = _server_command(args, arm)
    _new_json(attempt / "launch.json", {"command": command, "environment": _environment(args)})
    # Check only this loopback port; never stop another process or container.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    with (attempt / "server.log").open("xb") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            _wait_ready(process, args)
            response = evaluator._api_call(
                f"http://127.0.0.1:{args.port}/v1",
                "Qwen2.5-14B-Instruct",
                "Reply with one word.",
                1,
                args.seed,
                args.timeout,
                True,
                [151645, 151643],
                [],
            )
            if not response.get("choices"):
                raise ValueError("Cache bootstrap failed")
            _new_json(attempt / "bootstrap.json", response)
            deadline = time.monotonic() + 30
            while "KV_ALLOCATION " not in (text := (attempt / "server.log").read_text(errors="replace")):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise ValueError("No actual cache allocation evidence after first forward")
                time.sleep(1)
            allocations = [
                json.loads(line.split("KV_ALLOCATION ", 1)[1]) for line in text.splitlines() if "KV_ALLOCATION " in line
            ]
            if len(allocations) != 1:
                raise ValueError("Expected exactly one single-GPU cache allocation attestation")
            _validate_allocation(arm, allocations[0], provenance["profile"], args.block_size)
            _new_json(attempt / "allocation.json", allocations[0])
            manifest = attempt / "server_manifest.json"
            _new_json(
                manifest,
                _server_manifest(
                    args,
                    arm,
                    provenance,
                    {
                        "attempt": str(attempt),
                        "allocation_sha256": sha256_file(attempt / "allocation.json"),
                        "allocation_verified": True,
                    },
                ),
            )
            # Exercise the evaluator's manifest contract before launching its subprocess.
            evaluator._load_server_manifest(manifest, arm.replace("_", "-"), "Qwen2.5-14B-Instruct", args.model)
            for task in pending:
                logger.info("Evaluating %s/%s; detailed progress is in eval-%s.log", arm, task, task)
                result = args.output / "runs" / task / f"{arm.replace('_', '-')}.jsonl"
                command = _evaluator_command(args, task, paths[task], arm, manifest, result.exists())
                with (attempt / f"eval-{task}.log").open("xb") as task_log:
                    subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=task_log, stderr=subprocess.STDOUT)
                # Validate complete durable responses even on the initial run.
                original_resume = args.resume
                args.resume = True
                try:
                    indices = _verify_journal(args, task, arm, prepared[task])
                    if indices != set(range(len(prepared[task]["rows"]))):
                        raise ValueError("Evaluator returned without a complete journal")
                finally:
                    args.resume = original_resume
        finally:
            _stop(process)


def _save_or_compare(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing evidence differs; preserved: {path}")
    else:
        _new_json(path, value)


def _attest_arm(args: argparse.Namespace, paths: dict[str, Path], arm: str, prepared: dict[str, Any]) -> None:
    original_resume = args.resume
    args.resume = True
    try:
        evidence = {}
        for task in paths:
            indices = _verify_journal(args, task, arm, prepared[task])
            if indices != set(range(len(prepared[task]["rows"]))):
                raise ValueError(f"Incomplete {arm}/{task} journal")
            result = args.output / "runs" / task / f"{arm.replace('_', '-')}.jsonl"
            evidence[task] = {
                "expected_rows": len(indices),
                "unique_completed_indices": len(indices),
                "result_sha256": sha256_file(result),
                "config_sha256": sha256_file(result.with_suffix(".config.json")),
            }
        _save_or_compare(args.output / "arms" / arm / "integrity.json", evidence)
    finally:
        args.resume = original_resume


def _report(args: argparse.Namespace, paths: dict[str, Path], prepared: dict[str, Any]) -> None:
    metrics = evaluator._longbench_assets(args.longbench_repo)[2] if any(t != "gsm8k" for t in paths) else None
    table = []
    for task in paths:
        for arm in args.arms:
            label = arm.replace("_", "-")
            path = args.output / "runs" / task / f"{label}.jsonl"
            rows = sorted(evaluator._jsonl(path), key=lambda row: row["_eval_index"])
            scores, items = evaluator._score_pair(
                task, prepared[task]["rows"], {label: [row["prediction"] for row in rows]}, metrics
            )
            evidence = {
                "task": task,
                "arm": arm,
                "n": len(rows),
                "scores": scores,
                "items": items,
                "result_sha256": sha256_file(path),
                "length_exhaustion": sum(
                    row["api_response"]["choices"][0]["finish_reason"] == "length" for row in rows
                ),
            }
            _save_or_compare(args.output / "scores" / task / f"{arm}.json", evidence)
            table.append({key: evidence[key] for key in ("task", "arm", "n", "scores", "length_exhaustion")})
        if "auto" not in args.arms:
            continue
        for arm in args.arms:
            if arm == "auto":
                continue
            label = arm.replace("_", "-")
            directory = args.output / "paired" / task / arm
            results = args.output / "runs" / task
            identity = {
                "auto_sha256": sha256_file(results / "auto.jsonl"),
                "candidate_sha256": sha256_file(results / f"{label}.jsonl"),
            }
            if directory.exists():
                saved_identity = json.loads((directory / "inputs.json").read_text())
                if (
                    any(saved_identity.get(key) != value for key, value in identity.items())
                    or not (directory / "paired_scores.json").is_file()
                    or sha256_file(directory / "paired_scores.json") != saved_identity.get("score_sha256")
                ):
                    raise ValueError("Existing paired scores incomplete/changed; preserved")
                continue
            staging = _attempt(args.output / "score_attempts" / task / arm)
            command = [
                sys.executable,
                "-B",
                str(EVALUATOR),
                "--task",
                task,
                "--auto-results",
                str(results / "auto.jsonl"),
                f"--{label}-results",
                str(results / f"{label}.jsonl"),
                "--output",
                str(staging),
                "--strict-token-count",
            ]
            if task != "gsm8k":
                command.extend(("--longbench-repo", str(args.longbench_repo.resolve())))
            with (staging / "scoring.log").open("xb") as log:
                subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            _new_json(
                staging / "inputs.json", {**identity, "score_sha256": sha256_file(staging / "paired_scores.json")}
            )
            directory.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(directory)
    macros = (
        {
            arm: sum(
                next(iter(row["scores"].values()))
                for row in table
                if row["arm"] == arm and row["task"] in evaluator.LONG_BENCH_TASKS
            )
            / 8
            for arm in args.arms
        }
        if set(evaluator.LONG_BENCH_TASKS).issubset(paths)
        else {}
    )
    summary = {
        "scope": "smoke_only" if args.limit else "one_unique_dataset_round",
        "tasks": table,
        "longbench_8_task_macro": macros,
        "paired": "paired/<task>/<arm>/paired_scores.json"
        if "auto" in args.arms
        else "unavailable: baseline not selected",
    }
    _save_or_compare(args.output / "summary.json", summary)
    csv_path = args.output / "summary.csv"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("task", "arm", "n", "metric", "score", "length_exhaustion"))
    for row in table:
        for metric, value in row["scores"].items():
            writer.writerow((row["task"], row["arm"], row["n"], metric, value, row["length_exhaustion"]))
    content = buffer.getvalue()
    if csv_path.exists():
        if csv_path.read_bytes() != content.encode():
            raise ValueError("Existing summary.csv differs or is incomplete; preserved")
    else:
        with csv_path.open("x", newline="") as stream:
            stream.write(content)


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        paths = _validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    plan = _plan(args, paths)
    if args.dry_run:
        try:
            plan["data_audit"] = _data_audit(args, paths)
        except FileNotFoundError as error:
            plan["data_audit"] = None
            plan["pending_input"] = str(error)
        sys.stdout.write(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
        return
    if args.limit is None and args.approved_data_manifest is None:
        parser.error("Full runs require --approved-data-manifest from reviewed --dry-run JSON")
    audit = _data_audit(args, paths)
    if args.approved_data_manifest:
        approved = json.loads(args.approved_data_manifest.read_text())
        if approved.get("data_audit", approved) != audit:
            raise ValueError("Approved data hashes/counts differ; no server started")
    logger.info("Hashing local checkpoint shards and validating model/configuration: %s", args.model)
    provenance = checkpoint_provenance(args.model)
    if {path.name for path in args.model.glob("*.safetensors")} != set(provenance["weights"]["files"]):
        raise ValueError("Unindexed or missing safetensors shards; checkpoint identity is ambiguous")
    if provenance["profile"]["size"] != "14b":
        raise ValueError("This runner requires the Qwen2.5-14B-Instruct config")
    if not os.access(args.server_binary, os.X_OK):
        raise ValueError("--server-binary must be executable")
    os.environ.update(_environment(args))
    prepared = _prepared(args, paths)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible CUDA GPU is required")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "triton", "transformers")}
    sources = subprocess.check_output(
        ["git", "ls-files", "xllm/python", "xllm/core/framework/kv_cache"], cwd=ROOT, text=True
    ).splitlines()
    sources += [
        str(Path(__file__).relative_to(ROOT)),
        str(EVALUATOR.relative_to(ROOT)),
        "tools/qwen25_experiment_model.py",
    ]
    manifest = {
        "schema_version": 1,
        "plan": plan,
        "data_audit": audit,
        "model_provenance": provenance,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: sha256_file(ROOT / name) for name in sources},
        "binary_sha256": sha256_file(args.server_binary),
        "versions": versions,
        "gpu": {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))},
        "prepared": {
            task: {
                "prompt_sha256": _digest(data["prompts"]),
                "prompt_tokens": data["counts"],
                "max_tokens": data["max_tokens"],
            }
            for task, data in prepared.items()
        },
    }
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _save_or_compare(args.output / "experiment_manifest.json", manifest)
        # Validate all existing journals before launching any arm.
        for arm in args.arms:
            for task in paths:
                _verify_journal(args, task, arm, prepared[task])
        for arm in args.arms:
            _run_arm(args, paths, arm, prepared, provenance, dict(os.environ))
            _attest_arm(args, paths, arm, prepared)
        _report(args, paths, prepared)
        _save_or_compare(
            args.output / "complete.json",
            {
                "summary_sha256": sha256_file(args.output / "summary.json"),
                "csv_sha256": sha256_file(args.output / "summary.csv"),
                "paired_sha256": {
                    str(path.relative_to(args.output)): sha256_file(path)
                    for path in sorted((args.output / "paired").rglob("paired_scores.json"))
                },
            },
        )
    logger.info("Joint-KV quality run complete: %s", args.output)


if __name__ == "__main__":
    main()
