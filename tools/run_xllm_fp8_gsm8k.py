# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Run paired xLLM GSM8K BF16, FP8 E4M3, and FP8 E5M2 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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

MODEL = Path("/mnt/e/AI/models/Qwen2.5-1.5B-Instruct")
MODEL_ID = "Qwen2.5-1.5B-Instruct"
DATA_ROOT = Path("/mnt/e/AI/xllm-eval-data")
DATA = DATA_ROOT / "int4-model-eval-20261002/gsm8k/data.jsonl"
SHOTS = DATA_ROOT / "gsm8k-3101c7d5072418e28b9008a6636bde82a006892c/prepared/five_train_examples.json"
PROVENANCE = DATA_ROOT / "model_provenance.json"
OUTPUT_ROOT = DATA_ROOT / "xllm-fp8-gsm8k-20261005"
BINARY = ROOT / "build/cmake.cuda-x86/xllm/xllm"
EVALUATOR = ROOT / "tools/eval_kv_cache_quality.py"
BASE_URL = "http://127.0.0.1:18994/v1"
FLASHINFER_OPS = "/mnt/e/AI/xllm-build-tools/flashinfer-cache/.cache/flashinfer/0.6.18.post1/120f/cached_ops"
SMOKE_ARM_LIMIT = 4
ARMS = (
    {"arm": "auto", "kv_cache_dtype": "auto", "backend": "flashinfer", "concurrency": 32},
    {"arm": "fp8-e4m3", "kv_cache_dtype": "fp8_e4m3", "backend": "triton", "concurrency": 50},
    {"arm": "fp8-e5m2", "kv_cache_dtype": "fp8_e5m2", "backend": "triton", "concurrency": 50},
)
SOURCE_FILES = (
    Path("tools/run_xllm_fp8_gsm8k.py"),
    Path("tools/eval_kv_cache_quality.py"),
    Path("tools/test_eval_kv_cache_quality_resume.py"),
    Path("xllm/python/attention/quantized_triton.py"),
    Path("xllm/python/attention/quantized.py"),
    Path("xllm/python/attention/flashinfer.py"),
    Path("xllm/python/attention/backend.py"),
    Path("xllm/python/model_executor/executor.py"),
    Path("xllm/python/layers/attention.py"),
)


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


def _snapshot_sources(output_root: Path) -> dict[str, str]:
    snapshot_root = output_root / "source_snapshot"
    hashes: dict[str, str] = {}
    for relative_path in SOURCE_FILES:
        source = ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Required source snapshot file is missing: {source}")
        destination = snapshot_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = _sha256_file(source)
        copy_hash = _sha256_file(destination)
        if source_hash != copy_hash:
            raise OSError(f"Source snapshot hash mismatch for {relative_path}")
        hashes[str(relative_path)] = source_hash
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    _write_json(
        output_root / "source_snapshot_manifest.json",
        {"created_at_utc": datetime.now(timezone.utc).isoformat(), "git_head": head, "sha256": hashes},
    )
    return hashes


def _experiment_manifest(output_root: Path, source_hashes: dict[str, str], limit: int | None) -> dict[str, Any]:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    tokenizer_data = json.loads((MODEL / "tokenizer_config.json").read_text(encoding="utf-8"))
    chat_template_hash = hashlib.sha256(
        json.dumps(tokenizer_data.get("chat_template"), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "GSM8K test split",
        "input": {"path": str(DATA), "sha256": _sha256_file(DATA), "rows": 1319},
        "few_shot": {"path": str(SHOTS), "sha256": _sha256_file(SHOTS), "examples": 5},
        "model": {
            "id": MODEL_ID,
            "path": str(MODEL),
            "checkpoint_sha256": provenance["weights"]["sha256"],
            "config_sha256": _sha256_file(MODEL / "config.json"),
            "chat_template_sha256": chat_template_hash,
        },
        "generation": {
            "temperature": 0,
            "top_p": 1,
            "seed": 17,
            "max_tokens": 512,
            "stop_strings": ["Question:", "</s>", "<|im_end|>"],
            "stop_token_ids": [151645, 151643],
        },
        "runtime": {
            "model_impl": "python",
            "graph_mode": "off",
            "max_cache_size": 0,
            "max_memory_utilization": 0.87,
            "max_tokens_per_batch": 4096,
            "backend_by_arm": {arm["arm"]: arm["backend"] for arm in ARMS},
            "kv_cache_dtype_by_arm": {arm["arm"]: arm["kv_cache_dtype"] for arm in ARMS},
            "request_concurrency_by_arm": {arm["arm"]: arm["concurrency"] for arm in ARMS},
            "max_seqs_per_batch_by_arm": {arm["arm"]: arm["concurrency"] for arm in ARMS},
            "quality_transform_mode": "none",
        },
        "limit": limit,
        "source_snapshot_sha256": source_hashes,
        "output_root": str(output_root.resolve()),
    }


def _assert_experiment_compatible(path: Path, current: dict[str, Any]) -> None:
    if not path.exists():
        _write_json(path, current)
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    previous.pop("created_at_utc", None)
    current.pop("created_at_utc", None)
    if previous != current:
        raise ValueError(f"existing experiment manifest differs from current inputs or sources: {path}")


def _server_manifest(arm: dict[str, Any]) -> dict[str, Any]:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    tokenizer_data = json.loads((MODEL / "tokenizer_config.json").read_text(encoding="utf-8"))
    return {
        "server_started": False,
        "server_mode": "python",
        "model_impl": "python",
        "graph_mode": "off",
        "kv_cache_mode": arm["kv_cache_dtype"],
        "quality_transform_mode": "none",
        "quantized_backend": arm["backend"],
        "checkpoint_path": str(MODEL),
        "checkpoint_sha256": provenance["weights"]["sha256"],
        "model_config_sha256": _sha256_file(MODEL / "config.json"),
        "chat_template_sha256": hashlib.sha256(
            json.dumps(tokenizer_data.get("chat_template"), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "model": MODEL_ID,
        "runtime_configuration": {
            "max_cache_size": 0,
            "max_memory_utilization": 0.87,
            "max_tokens_per_batch": 4096,
            "max_seqs_per_batch": arm["concurrency"],
            "enable_graph": False,
            "enable_prefill_piecewise_graph": False,
            "python_graph_backend": "off",
        },
        "verified_at_utc": None,
        "arm": arm["arm"],
        "request_concurrency": arm["concurrency"],
    }


def _wait_ready(process: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            raise RuntimeError(f"xLLM exited before API readiness (status {process.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(BASE_URL + "/models", timeout=2) as response:
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


def _run_evaluator(
    arm: dict[str, Any], manifest_path: Path, output_root: Path, limit: int | None, resume: bool
) -> None:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--task",
        "gsm8k",
        "--data",
        str(DATA),
        "--gsm8k-shots",
        str(SHOTS),
        "--tokenizer",
        str(MODEL),
        "--server-manifest",
        str(manifest_path),
        "--model",
        MODEL_ID,
        "--arm",
        arm["arm"],
        "--url",
        BASE_URL,
        "--output",
        str(output_root / "runs"),
        "--concurrency",
        str(arm["concurrency"]),
        "--seed",
        "17",
        "--stop-token-ids",
        "151645",
        "151643",
        "--strict-token-count",
    ]
    if limit is not None:
        command.extend(("--limit", str(limit)))
    if resume:
        command.append("--resume")
    log_path = output_root / "runs" / f"{arm['arm']}.run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and log_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluator log: {log_path}")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result_path = output_root / "runs" / f"{arm['arm']}.jsonl"
        while process.poll() is None:
            time.sleep(30)
            if process.poll() is None:
                completed = 0
                if result_path.exists():
                    with result_path.open(encoding="utf-8") as result_stream:
                        completed = sum(1 for line in result_stream if line.strip())
                logger.info("%s durable GSM8K rows: %d", arm["arm"], completed)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def _run_arm(arm: dict[str, Any], output_root: Path, limit: int | None, resume: bool) -> None:
    arm_root = output_root / "arms" / arm["arm"]
    arm_root.mkdir(parents=True, exist_ok=True)
    manifest_path = arm_root / "server_manifest.json"
    manifest = _server_manifest(arm)
    _write_json(manifest_path, manifest)
    environment = os.environ.copy()
    environment["XLLM_QUANTIZED_BACKEND"] = arm["backend"]
    environment["XLLM_KV_QUALITY_MODE"] = "none"
    if arm["backend"] == "flashinfer":
        environment["FLASHINFER_OPS_PATH"] = environment.get("FLASHINFER_OPS_PATH", FLASHINFER_OPS)
        environment["FLASHINFER_CUDA_ARCH_LIST"] = environment.get("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
    server_log_path = arm_root / "server.log"
    with server_log_path.open("ab") as server_log:
        server: subprocess.Popen[bytes] | None = None
        try:
            server = subprocess.Popen(
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
                    f"--max_seqs_per_batch={arm['concurrency']}",
                    f"--kv_cache_dtype={arm['kv_cache_dtype']}",
                    "--enable_graph=false",
                    "--enable_prefill_piecewise_graph=false",
                    "--python_graph_backend=off",
                ],
                cwd=ROOT,
                env=environment,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_ready(server, server_log_path)
            manifest["server_started"] = True
            manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["verified_api_port"] = 18994
            manifest["server_attestation"] = f"Fresh HTTP model readiness verified for {arm['arm']} arm."
            _write_json(manifest_path, manifest)
            _run_evaluator(arm, manifest_path, output_root, limit, resume)
        finally:
            _stop_process(server)


def _score_candidates(output_root: Path) -> None:
    run_root = output_root / "runs"
    for arm, flag in (("fp8-e4m3", "--fp8-e4m3-results"), ("fp8-e5m2", "--fp8-e5m2-results")):
        command = [
            sys.executable,
            str(EVALUATOR),
            "--task",
            "gsm8k",
            "--auto-results",
            str(run_root / "auto.jsonl"),
            flag,
            str(run_root / f"{arm}.jsonl"),
            "--output",
            str(run_root / f"paired-{arm}"),
            "--strict-token-count",
        ]
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume only matching validated arm journals.")
    args = parser.parse_args()
    if not BINARY.is_file():
        parser.error(f"xLLM binary not found: {BINARY}")
    output_root = (args.output_root or (OUTPUT_ROOT / "smoke" if args.phase == "smoke" else OUTPUT_ROOT)).resolve()
    limit = SMOKE_ARM_LIMIT if args.phase == "smoke" else None
    output_root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        source_manifest_path = output_root / "source_snapshot_manifest.json"
        if not source_manifest_path.exists():
            parser.error("--resume requires a saved source snapshot manifest")
        source_hashes = json.loads(source_manifest_path.read_text(encoding="utf-8"))["sha256"]
    else:
        source_hashes = _snapshot_sources(output_root)
    manifest = _experiment_manifest(output_root, source_hashes, limit)
    _assert_experiment_compatible(output_root / "experiment_manifest.json", manifest)
    for required in (DATA, SHOTS, MODEL / "config.json", MODEL / "tokenizer_config.json", PROVENANCE):
        if not required.is_file():
            raise FileNotFoundError(required)
    for arm in ARMS:
        _run_arm(arm, output_root, limit, args.resume)
    if args.phase == "full":
        _score_candidates(output_root)
    logger.info("Completed xLLM FP8 GSM8K %s phase under %s", args.phase, output_root)


if __name__ == "__main__":
    main()
