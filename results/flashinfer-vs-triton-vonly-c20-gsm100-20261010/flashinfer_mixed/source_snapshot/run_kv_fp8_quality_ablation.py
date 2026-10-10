# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Run resumable GSM8K and LongBench K/V FP8-QDQ quality ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.logger import logger

DATA_ROOT = Path("/mnt/e/AI/xllm-eval-data")
MODEL = Path("/mnt/e/AI/models/Qwen2.5-1.5B-Instruct")
MODEL_ID = "Qwen2.5-1.5B-Instruct"
LONG_BENCH = DATA_ROOT / "longbench-5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
PREPARED_LONG = LONG_BENCH / "prepared/seed17_n50_contextcap32704"
GSM_DATA = DATA_ROOT / "int4-model-eval-20261002/gsm8k/data.jsonl"
GSM_SHOTS = DATA_ROOT / "gsm8k-3101c7d5072418e28b9008a6636bde82a006892c/prepared/five_train_examples.json"
LONG_BENCH_REPO = DATA_ROOT / "../xllm-eval-tools"
PROVENANCE = DATA_ROOT / "model_provenance.json"
BINARY = ROOT / "build/cmake.cuda-x86/xllm/xllm"
EVALUATOR = ROOT / "tools/eval_kv_cache_quality.py"
RUNNER = Path(__file__).resolve()
BASE_URL = "http://127.0.0.1:18994/v1"
ARMS = (("auto", "none"), ("k-only-fp8", "k_only_fp8"), ("v-only-fp8", "v_only_fp8"))
TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "2wikimqa", "hotpotqa", "musique", "lcc", "repobench-p")
FLASHINFER_OPS = "/mnt/e/AI/xllm-build-tools/flashinfer-cache/.cache/flashinfer/0.6.18.post1/120f/cached_ops"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_manifest(task: str, output_root: Path, limit: int | None) -> dict[str, Any]:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    common: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "FP8 E4M3 scale=1 QDQ precision ablation; persistent cache remains BF16",
        "task_scope": task,
        "limit": limit,
        "arms": [{"arm": arm, "quality_mode": mode} for arm, mode in ARMS],
        "execution_protocol": {
            "request_concurrency_by_dataset": {"gsm8k": 64, "longbench": 8},
            "max_seqs_per_batch_by_dataset": {"gsm8k": 64, "longbench": 8},
            "model_impl": "python",
            "kv_cache_dtype": "auto",
            "backend": "flashinfer",
            "enable_graph": False,
            "max_cache_size": 0,
            "max_memory_utilization": 0.87,
            "max_tokens_per_batch": 4096,
            "fp8_format": "E4M3",
            "key_scale": 1.0,
            "value_scale": 1.0,
            "persistent_cache_dtype": "bfloat16",
        },
        "model": {
            "id": MODEL_ID,
            "path": str(MODEL),
            "checkpoint_sha256": provenance["weights"]["sha256"],
            "config_sha256": _sha256_file(MODEL / "config.json"),
            "chat_template_sha256": hashlib.sha256(
                json.dumps(
                    json.loads((MODEL / "tokenizer_config.json").read_text(encoding="utf-8")).get("chat_template"),
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        },
        "source_files": {
            "runner_sha256": _sha256_file(RUNNER),
            "evaluator_sha256": _sha256_file(EVALUATOR),
            "model_provenance_sha256": _sha256_file(PROVENANCE),
        },
        "output_root": str(output_root.resolve()),
    }
    if task in ("gsm8k", "all"):
        common["gsm8k"] = {
            "data": str(GSM_DATA),
            "data_sha256": _sha256_file(GSM_DATA),
            "shots": str(GSM_SHOTS),
            "shots_sha256": _sha256_file(GSM_SHOTS),
            "examples": len([line for line in GSM_DATA.read_text(encoding="utf-8").splitlines() if line.strip()]),
        }
    if task in ("longbench", "all"):
        manifest_path = PREPARED_LONG / "manifest.json"
        common["longbench"] = {
            "prepared_root": str(PREPARED_LONG),
            "prepared_manifest_sha256": _sha256_file(manifest_path),
            "prepared_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            "official_repo": str(LONG_BENCH_REPO.resolve()),
            "tasks": {
                task_name: {
                    "path": str(PREPARED_LONG / f"{task_name}.jsonl"),
                    "sha256": _sha256_file(PREPARED_LONG / f"{task_name}.jsonl"),
                    "examples": len(
                        [
                            line
                            for line in (PREPARED_LONG / f"{task_name}.jsonl").read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        ]
                    ),
                }
                for task_name in TASKS
            },
        }
    return common


def _assert_manifest_compatible(path: Path, current: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        current_without_timestamp = dict(current)
        existing_without_timestamp = dict(existing)
        current_without_timestamp.pop("created_at_utc", None)
        existing_without_timestamp.pop("created_at_utc", None)
        if existing_without_timestamp != current_without_timestamp:
            raise ValueError(f"experiment manifest differs from current inputs/configuration: {path}")
    else:
        _write_json(path, current)


def _create_server_manifest(arm: str, quality_mode: str, concurrency: int) -> dict[str, Any]:
    tokenizer_config = json.loads((MODEL / "tokenizer_config.json").read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "server_started": False,
        "server_mode": "python",
        "model_impl": "python",
        "graph_mode": "off",
        "kv_cache_mode": "auto",
        "quality_transform_mode": quality_mode,
        "fp8_backend": "flashinfer",
        "fp8_format": "E4M3",
        "fp8_key_scale": 1.0,
        "fp8_value_scale": 1.0,
        "persistent_cache_dtype": "bfloat16",
        "checkpoint_path": str(MODEL),
        "checkpoint_sha256": provenance["weights"]["sha256"],
        "model_config_sha256": _sha256_file(MODEL / "config.json"),
        "chat_template_sha256": hashlib.sha256(
            json.dumps(tokenizer_config.get("chat_template"), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "model": MODEL_ID,
        "runtime_configuration": {
            "max_cache_size": 0,
            "max_memory_utilization": 0.87,
            "max_tokens_per_batch": 4096,
            "max_seqs_per_batch": concurrency,
            "enable_graph": False,
            "enable_prefill_piecewise_graph": False,
            "python_graph_backend": "off",
        },
        "arm": arm,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return manifest


def _server_is_ready(process: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.monotonic() + 240
    url = BASE_URL + "/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            raise RuntimeError(f"xLLM exited before readiness (status {process.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.loads(response.read())
            if MODEL_ID in {item.get("id") for item in payload.get("data", [])}:
                return
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1)
    raise TimeoutError(f"xLLM API did not become ready; inspect {log_path}")


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)


def _post_completion(prompt: str) -> dict[str, Any]:
    request_body = json.dumps(
        {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 32,
            "seed": 20261005,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=request_body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def _arithmetic_smoke(output_dir: Path) -> None:
    prompts = [("What is 27 times 43? Answer with the number only.", "1161")]
    prompts.extend(
        (f"What is {left} + {right}? Answer with the number only.", str(left + right))
        for left, right in zip(range(11, 19), range(33, 57, 3))
    )
    rows = []
    for prompt, expected in prompts:
        response = _post_completion(prompt)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("arithmetic smoke received a response without choices")
        rows.append(
            {
                "prompt": prompt,
                "expected": expected,
                "response": choices[0].get("message", {}).get("content", ""),
                "finish_reason": choices[0].get("finish_reason"),
                "usage": response.get("usage", {}),
            }
        )
    _write_json(output_dir / "arithmetic_smoke.json", {"results": rows})


def _evaluator_command(
    task: str,
    data: Path,
    arm: str,
    manifest_path: Path,
    output_dir: Path,
    concurrency: int,
    limit: int | None,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--task",
        task,
        "--data",
        str(data),
        "--tokenizer",
        str(MODEL),
        "--model",
        MODEL_ID,
        "--arm",
        arm,
        "--url",
        BASE_URL,
        "--server-manifest",
        str(manifest_path),
        "--output",
        str(output_dir),
        "--seed",
        "17",
        "--concurrency",
        str(concurrency),
        "--stop-token-ids",
        "151645",
        "151643",
        "--strict-token-count",
    ]
    if task == "gsm8k":
        command.extend(("--gsm8k-shots", str(GSM_SHOTS)))
    else:
        command.extend(("--longbench-repo", str(LONG_BENCH_REPO)))
    if limit is not None:
        command.extend(("--limit", str(limit)))
    if resume:
        command.append("--resume")
    return command


def _run_checked_with_progress(command: list[str], log_path: Path, result_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(30)
            if process.poll() is None:
                completed = (
                    sum(1 for line in result_path.open(encoding="utf-8") if line.strip()) if result_path.exists() else 0
                )
                logger.info("Progress %s: %d durable response rows", result_path.parent.name, completed)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def _run_arm(
    arm: str,
    quality_mode: str,
    tasks: list[tuple[str, Path, int]],
    output_root: Path,
    experiment: dict[str, Any],
    limit: int | None,
    resume: bool,
    do_arithmetic_smoke: bool,
) -> None:
    arm_root = output_root / "arms" / arm
    manifest_path = arm_root / "server_manifest.json"
    _write_json(
        manifest_path, _create_server_manifest(arm, quality_mode, max(concurrency for _, _, concurrency in tasks))
    )
    env = os.environ.copy()
    env["FLASHINFER_OPS_PATH"] = env.get("FLASHINFER_OPS_PATH", FLASHINFER_OPS)
    env["FLASHINFER_CUDA_ARCH_LIST"] = env.get("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
    env["XLLM_QUANTIZED_BACKEND"] = "flashinfer"
    env["XLLM_KV_QUALITY_MODE"] = quality_mode
    env["XLLM_FP8_K_SCALE"] = "1.0"
    env["XLLM_FP8_V_SCALE"] = "1.0"
    server_log_path = arm_root / "server.log"
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[bytes] | None = None
    with server_log_path.open("ab") as server_log:
        try:
            process = subprocess.Popen(
                [
                    str(BINARY),
                    f"--model={MODEL}",
                    "--model_impl=python",
                    f"--python_model_path={ROOT}",
                    "--host=127.0.0.1",
                    "--port=18994",
                    "--max_cache_size=0",
                    "--max_memory_utilization=0.87",
                    "--max_tokens_per_batch=4096",
                    f"--max_seqs_per_batch={max(concurrency for _, _, concurrency in tasks)}",
                    "--kv_cache_dtype=auto",
                    "--enable_graph=false",
                    "--enable_prefill_piecewise_graph=false",
                    "--python_graph_backend=off",
                ],
                cwd=ROOT,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _server_is_ready(process, server_log_path)
            manifest = _create_server_manifest(arm, quality_mode, max(concurrency for _, _, concurrency in tasks))
            manifest["server_started"] = True
            manifest["server_attestation"] = f"Readiness verified for {arm} on {BASE_URL}."
            manifest["verified_api_port"] = 18994
            manifest["smoke_response_status"] = 200
            _write_json(manifest_path, manifest)
            if do_arithmetic_smoke:
                _arithmetic_smoke(arm_root)
            for task, data, concurrency in tasks:
                output_dir = output_root / "runs" / task
                output_dir.mkdir(parents=True, exist_ok=True)
                command = _evaluator_command(task, data, arm, manifest_path, output_dir, concurrency, limit, resume)
                log_path = output_dir / f"{arm}.run.log"
                if not resume and log_path.exists():
                    raise FileExistsError(f"refusing to overwrite existing evaluator log: {log_path}")
                _run_checked_with_progress(command, log_path, output_dir / f"{arm}.jsonl")
        finally:
            _stop_process(process)


def _run_scoring(task: str, output_root: Path) -> None:
    names = (*TASKS, "gsm8k") if task == "all" else TASKS if task == "longbench" else ("gsm8k",)
    for task_name in names:
        task_dir = output_root / "runs" / task_name
        for candidate in ("k-only-fp8", "v-only-fp8"):
            command = [
                sys.executable,
                str(EVALUATOR),
                "--task",
                task_name,
                "--auto-results",
                str(task_dir / "auto.jsonl"),
                f"--{candidate}-results",
                str(task_dir / f"{candidate}.jsonl"),
                "--output",
                str(task_dir / f"paired-{candidate}"),
                "--strict-token-count",
            ]
            if task_name != "gsm8k":
                command.extend(("--longbench-repo", str(LONG_BENCH_REPO)))
            subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("gsm8k", "longbench", "all"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("/mnt/e/AI/xllm-eval-data/fp8-kv-quality-ablation-20261005")
    )
    parser.add_argument(
        "--limit", type=int, help="Run a fixed prefix subset; use only for isolated smoke output roots."
    )
    parser.add_argument("--resume", action="store_true", help="Continue verified partial arm JSONL journals.")
    parser.add_argument(
        "--smoke-only", action="store_true", help="Run three-arm arithmetic and GSM8K four-row smoke only."
    )
    args = parser.parse_args()
    if not BINARY.is_file():
        parser.error(f"xLLM binary not found: {BINARY}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.smoke_only and args.task != "gsm8k":
        parser.error("--smoke-only requires task=gsm8k")
    output_root = args.output_root.resolve()
    tasks: list[tuple[str, Path, int]] = []
    scope = args.task
    task_groups: list[list[tuple[str, Path, int]]] = []
    if args.task in ("gsm8k", "all"):
        task_groups.append([("gsm8k", GSM_DATA, 64)])
    if args.task in ("longbench", "all"):
        task_groups.append([(task, PREPARED_LONG / f"{task}.jsonl", 8) for task in TASKS])
    limit = 4 if args.smoke_only else args.limit
    current_manifest = _source_manifest(scope, output_root, limit)
    _assert_manifest_compatible(output_root / "experiment_manifest.json", current_manifest)
    for path in (BINARY, EVALUATOR, MODEL / "config.json", MODEL / "tokenizer_config.json", GSM_SHOTS):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.smoke_only:
        task_groups = [[("gsm8k", GSM_DATA, 64)]]
    for arm, quality_mode in ARMS:
        for tasks in task_groups:
            _run_arm(
                arm,
                quality_mode,
                tasks,
                output_root,
                current_manifest,
                limit,
                args.resume,
                args.smoke_only,
            )
    if not args.smoke_only:
        _run_scoring(args.task, output_root)
    logger.info("Completed %s FP8 QDQ ablation phase under %s", args.task, output_root)


if __name__ == "__main__":
    main()
