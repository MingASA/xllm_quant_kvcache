# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Run paired GSM8K shards, one local xLLM server at a time."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_kv_cache_quality import _jsonl

_MODEL = "/mnt/e/AI/models/Qwen2.5-1.5B-Instruct"
_MODEL_ID = "Qwen2.5-1.5B-Instruct"
_ROOT = Path("/home/mingasa/projects/xllm_quant_kvcache")
_BINARY = _ROOT / "build/cmake.cuda-x86/xllm/xllm"
_SHOTS = Path(
    "/mnt/e/AI/xllm-eval-data/gsm8k-3101c7d5072418e28b9008a6636bde82a006892c/"
    "prepared/five_train_examples.json"
)
_TOKENIZER = Path(_MODEL)
_URL = "http://127.0.0.1:18994/v1"
_GPU_SAMPLER = "/usr/lib/wsl/lib/nvidia-smi"
_FLASHINFER_OPS = "/mnt/e/AI/xllm-build-tools/flashinfer-cache/.cache/flashinfer/0.6.18.post1/120f/cached_ops"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_eval(
    arm: str,
    data: Path,
    manifest: Path,
    output: Path,
    concurrency: int,
    limit: int | None,
    log_path: Path,
) -> float:
    result_path = output / f"{arm}.jsonl"
    config_path = output / f"{arm}.config.json"
    args = [
        "python",
        str(_ROOT / "tools/eval_kv_cache_quality.py"),
        "--task=gsm8k",
        f"--data={data}",
        f"--gsm8k-shots={_SHOTS}",
        f"--tokenizer={_TOKENIZER}",
        f"--server-manifest={manifest}",
        f"--model={_MODEL_ID}",
        f"--arm={arm}",
        f"--url={_URL}",
        f"--output={output}",
        f"--concurrency={concurrency}",
        "--strict-token-count",
    ]
    if limit is not None:
        args.append(f"--limit={limit}")
    if result_path.exists() or config_path.exists():
        args.append("--resume")
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(args, cwd=_ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    return time.monotonic() - started


def _stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)


def _wait_ready(process: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"xLLM exited before API readiness:\n{tail}")
        try:
            with urllib.request.urlopen(_URL + "/models", timeout=2) as response:
                payload = json.loads(response.read())
            ids = {item.get("id") for item in payload.get("data", [])}
            if _MODEL_ID in ids:
                return
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1)
    raise TimeoutError(f"xLLM API did not become ready; see {log_path}")


def _metrics(path: Path, sample_path: Path) -> dict[str, Any]:
    rows = _jsonl(path)
    if not rows:
        return {"examples": 0, "completion_tokens": 0, "http_span_seconds": None, "tokens_per_second": None}
    starts = [datetime.fromisoformat(row["request_started_utc"]) for row in rows]
    finishes = [datetime.fromisoformat(row["request_finished_utc"]) for row in rows]
    span = (max(finishes) - min(starts)).total_seconds()
    latencies = sorted(row["client_latency_seconds"] for row in rows)

    def percentile(fraction: float) -> float:
        offset = (len(latencies) - 1) * fraction
        lower = int(offset)
        upper = min(lower + 1, len(latencies) - 1)
        return latencies[lower] + (latencies[upper] - latencies[lower]) * (offset - lower)

    tokens = sum(row.get("usage", {}).get("completion_tokens", 0) for row in rows)
    first_start = min(starts).astimezone(ZoneInfo("America/Los_Angeles")).replace(tzinfo=None)
    last_finish = max(finishes).astimezone(ZoneInfo("America/Los_Angeles")).replace(tzinfo=None)
    gpu_samples = []
    if sample_path.exists():
        with sample_path.open(encoding="utf-8", newline="") as sample_file:
            for sample in csv.reader(sample_file):
                if len(sample) < 2:
                    continue
                sample_time = datetime.strptime(sample[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                if first_start <= sample_time <= last_finish:
                    gpu_samples.append(int(sample[1].strip().split()[0]))
    return {
        "examples": len(rows),
        "completion_tokens": tokens,
        "http_span_seconds": span,
        "tokens_per_second": tokens / span if span else None,
        "requests_per_second": len(rows) / span if span else None,
        "client_latency_p50_seconds": percentile(0.50),
        "client_latency_p95_seconds": percentile(0.95),
        "gpu_peak_mib_during_requests": max(gpu_samples) if gpu_samples else None,
    }


def _start_arm(root: Path, batch: dict[str, Any], arm: str, concurrency: int) -> float:
    batch_dir = root / batch["name"]
    data = batch_dir / "data.jsonl"
    manifest_path = root / f"{arm}_server_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["server_attestation"] = f"Fresh readiness verified for {arm} GSM8K batch {batch['name']}."
    manifest["runtime_configuration"]["max_seqs_per_batch"] = concurrency
    _write_json(manifest_path, manifest)

    kv_type = "auto" if arm == "auto" else "int8"
    server_log_path = batch_dir / f"{arm}_server.log"
    server_log = server_log_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    environment["FLASHINFER_OPS_PATH"] = _FLASHINFER_OPS
    environment["FLASHINFER_CUDA_ARCH_LIST"] = "12.0f"
    server_args = [
        str(_BINARY),
        f"--model={_MODEL}",
        "--model_impl=python",
        f"--python_model_path={_ROOT}",
        "--host=127.0.0.1",
        "--port=18994",
        "--max_cache_size=0",
        "--max_memory_utilization=0.87",
        "--max_tokens_per_batch=4096",
        f"--max_seqs_per_batch={concurrency}",
        f"--kv_cache_dtype={kv_type}",
        "--enable_graph=false",
        "--enable_prefill_piecewise_graph=false",
        "--python_graph_backend=off",
    ]
    server: subprocess.Popen[bytes] | None = None
    sampler: subprocess.Popen[bytes] | None = None
    try:
        server = subprocess.Popen(
            server_args,
            cwd=_ROOT,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_ready(server, server_log_path)
        manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
        with (batch_dir / f"{arm}_gpu_samples.csv").open("w", encoding="utf-8") as sample_file:
            sampler = subprocess.Popen(
                [_GPU_SAMPLER, "--query-gpu=timestamp,memory.used,utilization.gpu", "--format=csv,noheader", "-l", "1"],
                stdout=sample_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            measured = batch_dir / arm
            elapsed = _run_eval(
                arm,
                data,
                manifest_path,
                measured,
                concurrency,
                None,
                batch_dir / f"{arm}_eval.log",
            )
            return elapsed
    finally:
        _stop(sampler)
        _stop(server)
        server_log.close()


def _score_pair(
    root: Path,
    batch: dict[str, Any],
    auto_elapsed: float,
    int8_elapsed: float,
    auto_concurrency: int,
    int8_concurrency: int,
) -> dict[str, Any]:
    batch_dir = root / batch["name"]
    auto_results = batch_dir / "auto/auto.jsonl"
    int8_results = batch_dir / "int8/int8.jsonl"
    score_dir = batch_dir / "paired_score"
    command = [
        "python",
        str(_ROOT / "tools/eval_kv_cache_quality.py"),
        "--task=gsm8k",
        f"--auto-results={auto_results}",
        f"--int8-results={int8_results}",
        f"--output={score_dir}",
        "--strict-token-count",
    ]
    with (batch_dir / "scoring.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=_ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    scores = json.loads((score_dir / "paired_scores.json").read_text(encoding="utf-8"))
    rows = scores["items"]
    strict_flips = {key: sum(row["flip"] == key for row in rows) for key in ("up", "down", "tie")}
    flex_flips = {
        key: sum(row["flexible_flip"] == key for row in rows) for key in ("up", "down", "tie")
    }
    stage = {
        "batch": batch["name"],
        "source_start": batch["start"],
        "source_end_exclusive": batch["end_exclusive"],
        "n": len(rows),
        "execution_protocol": {
            "auto": {
                "request_concurrency": auto_concurrency,
                "max_seqs_per_batch": auto_concurrency,
            },
            "int8": {
                "request_concurrency": int8_concurrency,
                "max_seqs_per_batch": int8_concurrency,
            },
        },
        "auto": {
            **_metrics(auto_results, batch_dir / "auto_gpu_samples.csv"),
            "wall_seconds": auto_elapsed,
        },
        "int8": {
            **_metrics(int8_results, batch_dir / "int8_gpu_samples.csv"),
            "wall_seconds": int8_elapsed,
        },
        "strict_correct": {
            "auto": round(scores["scores"]["auto_strict_em"] * len(rows)),
            "int8": round(scores["scores"]["int8_strict_em"] * len(rows)),
            "up_down_tie": strict_flips,
        },
        "flexible_correct": {
            "auto": round(scores["scores"]["auto_flexible_em"] * len(rows)),
            "int8": round(scores["scores"]["int8_flexible_em"] * len(rows)),
            "up_down_tie": flex_flips,
        },
    }
    strict_total = {"auto": 0, "int8": 0, "up_down_tie": {key: 0 for key in ("up", "down", "tie")}}
    flex_total = {"auto": 0, "int8": 0, "up_down_tie": {key: 0 for key in ("up", "down", "tie")}}
    legacy_scores = root.parent / "gsm8k-batch50-full-20261001/batch_0000/paired_scores.json"
    if legacy_scores.exists():
        legacy = json.loads(legacy_scores.read_text(encoding="utf-8"))
        legacy_n = legacy["n"]
        strict_total["auto"] += round(legacy["scores"]["auto_strict_em"] * legacy_n)
        strict_total["int8"] += round(legacy["scores"]["int8_strict_em"] * legacy_n)
        flex_total["auto"] += round(legacy["scores"]["auto_flexible_em"] * legacy_n)
        flex_total["int8"] += round(legacy["scores"]["int8_flexible_em"] * legacy_n)
        for row in legacy["items"]:
            strict_total["up_down_tie"][row["flip"]] += 1
            flex_total["up_down_tie"][row["flexible_flip"]] += 1
    for previous_path in sorted(root.glob("batch_*/stage.json")):
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous["source_start"] >= batch["start"]:
            continue
        for total, key in ((strict_total, "strict_correct"), (flex_total, "flexible_correct")):
            total["auto"] += previous[key]["auto"]
            total["int8"] += previous[key]["int8"]
            for outcome, count in previous[key]["up_down_tie"].items():
                total["up_down_tie"][outcome] += count
    for total, key in ((strict_total, "strict_correct"), (flex_total, "flexible_correct")):
        total["auto"] += stage[key]["auto"]
        total["int8"] += stage[key]["int8"]
        for outcome, count in stage[key]["up_down_tie"].items():
            total["up_down_tie"][outcome] += count
    stage["cumulative_unique"] = {
        "n": 50 + sum(
            json.loads(path.read_text(encoding="utf-8"))["n"]
            for path in root.glob("batch_*/stage.json")
            if json.loads(path.read_text(encoding="utf-8"))["source_start"] < batch["start"]
        ) + len(rows),
        "strict_correct": strict_total,
        "flexible_correct": flex_total,
    }
    _write_json(batch_dir / "stage.json", stage)
    return stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=50)
    parser.add_argument("--auto-concurrency", type=int, default=32)
    parser.add_argument("--int8-concurrency", type=int, default=50)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    for batch in manifest["batches"]:
        if batch["start"] < args.start_index:
            continue
        auto_elapsed = _start_arm(args.root, batch, "auto", args.auto_concurrency)
        int8_elapsed = _start_arm(args.root, batch, "int8", args.int8_concurrency)
        stage = _score_pair(
            args.root,
            batch,
            auto_elapsed,
            int8_elapsed,
            args.auto_concurrency,
            args.int8_concurrency,
        )
        print(json.dumps(stage, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
