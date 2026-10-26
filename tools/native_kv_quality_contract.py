# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU-testable attestation contract for native FlashInfer joint-KV quality."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROFILE = "native-flashinfer-joint-v1"


def native_environment(arm: str, key_scale: float, value_scale: float) -> dict[str, str]:
    if arm not in ("auto", "fp8_e4m3"):
        raise ValueError("Native quality supports only auto/fp8_e4m3")
    if not all(math.isfinite(value) and value > 0 for value in (key_scale, value_scale)):
        raise ValueError("Native FP8 requires positive finite fixed scales")
    env = {"XLLM_FLASHINFER_NATIVE_KV": "1", "XLLM_KV_QUALITY_MODE": "none"}
    if arm == "fp8_e4m3":
        env.update(XLLM_QUANTIZED_BACKEND="flashinfer", XLLM_FP8_K_SCALE=str(key_scale),
                   XLLM_FP8_V_SCALE=str(value_scale))
    return env


def clean_native_environment(inherited: dict[str, str], selected: dict[str, str]) -> dict[str, str]:
    env = {key: value for key, value in inherited.items() if not key.startswith("XLLM_")}
    env.update(selected)
    return env


def native_evidence(text: str, arm: str, graph_mode: str, scales: tuple[float, float]) -> dict[str, Any]:
    backend_lines = [line for line in text.splitlines() if "NATIVE_KV_BACKEND " in line]
    replay_lines = [line for line in text.splitlines() if "NATIVE_KV_GRAPH_REPLAY " in line]
    if len(backend_lines) != 1:
        raise ValueError("Require one actual native backend log record")
    backend = json.loads(backend_lines[0].split("NATIVE_KV_BACKEND ", 1)[1])
    expected = "FlashInferBackend" if arm == "auto" else "FlashInferFP8Backend"
    required = {"class": expected, "kv_cache_dtype": arm, "attention_impl": "flashinfer",
                "planner_backend": "fa2", "paged_prefill": True, "quality_transform": False,
                "graph_enabled": graph_mode == "cudagraphs", "decode_tensor_cores": True,
                "flashinfer_version": "0.6.18.post1"}
    if arm not in ("auto", "fp8_e4m3") or graph_mode not in ("off", "cudagraphs"):
        raise ValueError("Unsupported native quality arm/Graph mode")
    if any(backend.get(key) != value for key, value in required.items()):
        raise ValueError("Native FA2/backend/Graph evidence differs from requested configuration")
    replays = [json.loads(line.split("NATIVE_KV_GRAPH_REPLAY ", 1)[1]) for line in replay_lines]
    captured = backend.get("graph_captured_batches", [])
    if graph_mode == "cudagraphs":
        if not captured or not replays:
            raise ValueError("Native Graph quality requires actual capture and replay")
        if any(row.get("class") != expected or row.get("captured_batch_size") not in captured for row in replays):
            raise ValueError("Native replay does not match its captured backend/bucket")
    elif captured or replays:
        raise ValueError("Unexpected Graph evidence in eager quality mode")
    if arm == "fp8_e4m3" and (
        backend.get("k_scale") != scales[0] or backend.get("v_scale") != scales[1]
        or backend.get("fp8_views_alias_resident_pages") is not True
        or backend.get("kv_data_type") != "torch.float8_e4m3fn"
    ):
        raise ValueError("Native FP8 fixed scales/resident alias differ")
    return {"backend": backend, "graph_replays": replays, "raw_backend": backend_lines[0],
            "raw_replays": replay_lines, "server_log_prefix_sha256": hashlib.sha256(text.encode()).hexdigest()}


def validate_native_manifest(path: Path, manifest: dict[str, Any], arm: str) -> None:
    if manifest.get("execution_profile") != PROFILE or manifest.get("backend") != "flashinfer":
        raise ValueError("Native quality requires the explicit FlashInfer execution profile")
    if arm not in ("auto", "fp8-e4m3"):
        raise ValueError("Native quality manifest cannot attest legacy/mixed arms")
    evidence = manifest.get("server_attestation", {})
    snapshot_path = path.parent / "native_evidence.json"
    allocation_path = path.parent / "allocation.json"
    if snapshot_path.is_symlink() or allocation_path.is_symlink():
        raise ValueError("Native attestation files must not be symlinks")
    snapshot_bytes = snapshot_path.read_bytes()
    if hashlib.sha256(snapshot_bytes).hexdigest() != evidence.get("native_evidence_sha256"):
        raise ValueError("Native evidence file hash mismatch")
    if (evidence.get("allocation_verified") is not True
            or hashlib.sha256(allocation_path.read_bytes()).hexdigest() != evidence.get("allocation_sha256")):
        raise ValueError("Native resident allocation attestation mismatch")
    snapshot = json.loads(snapshot_bytes)
    scales = manifest.get("paired_fp8_scales", {})
    selected = (scales.get("key"), scales.get("value"))
    if not all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in selected):
        raise ValueError("Native manifest must pin positive finite FP8 scales")
    text = "\n".join([snapshot["raw_backend"], *snapshot["raw_replays"]])
    observed = native_evidence(text, arm.replace("-", "_"), manifest["graph_mode"], selected)
    if observed["backend"] != snapshot["backend"] or observed["graph_replays"] != snapshot["graph_replays"]:
        raise ValueError("Native raw marker snapshot and structured evidence disagree")
    if evidence.get("native_backend") != observed["backend"]:
        raise ValueError("Native manifest backend differs from captured raw evidence")
