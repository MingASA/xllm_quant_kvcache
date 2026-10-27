# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU-testable attestation contract for native FlashInfer joint-KV quality."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROFILE = "native-flashinfer-joint-v1"
FA3_PROFILE = "native-flashinfer-fa3-joint-v1"
XQA_PROFILE = "native-flashinfer-xqa-joint-v1"
BF16_XQA_PROFILE = "native-flashinfer-bf16-xqa-reference-v1"


def native_environment(arm: str, key_scale: float, value_scale: float, *,
                       planner: str = "fa2", query_scale: float | None = None,
                       decode: str = "flashinfer") -> dict[str, str]:
    if arm not in ("auto", "fp8_e4m3"):
        raise ValueError("Native quality supports only auto/fp8_e4m3")
    if not all(math.isfinite(value) and value > 0 for value in (key_scale, value_scale)):
        raise ValueError("Native FP8 requires positive finite fixed scales")
    if planner not in ("fa2", "fa3"):
        raise ValueError("Unsupported native planner")
    if decode not in ("flashinfer", "xqa") or (decode == "xqa" and arm == "fp8_e4m3" and planner != "fa2"):
        raise ValueError("FP8 XQA quality requires FA2 BF16-Q prefill")
    if decode == "xqa" and query_scale is not None:
        raise ValueError("XQA quality must not request external Q quantization")
    if planner == "fa3" and decode != "xqa" and (query_scale is None or not math.isfinite(query_scale) or query_scale <= 0):
        raise ValueError("FA3 quality requires an explicit positive finite FP8 query scale")
    if planner == "fa2" and query_scale is not None:
        raise ValueError("FP8 query scale requires the explicit FA3 profile")
    env = {"XLLM_FLASHINFER_NATIVE_KV": "1", "XLLM_KV_QUALITY_MODE": "none"}
    if decode == "xqa":
        env.update(XLLM_FLASHINFER_NATIVE_PLANNER=planner, XLLM_FLASHINFER_NATIVE_DECODE="xqa")
    if planner == "fa3":
        env["XLLM_FLASHINFER_NATIVE_PLANNER"] = "fa3"
    if arm == "fp8_e4m3":
        env.update(XLLM_QUANTIZED_BACKEND="flashinfer", XLLM_FP8_K_SCALE=str(key_scale),
                   XLLM_FP8_V_SCALE=str(value_scale))
        if planner == "fa3":
            env.update(XLLM_FLASHINFER_FP8_QUERY="1", XLLM_FLASHINFER_FP8_Q_SCALE=str(query_scale))
    return env


def clean_native_environment(inherited: dict[str, str], selected: dict[str, str]) -> dict[str, str]:
    env = {key: value for key, value in inherited.items() if not key.startswith("XLLM_")}
    env.update(selected)
    return env


def native_evidence(text: str, arm: str, graph_mode: str, scales: tuple[float, float], *,
                    planner: str = "fa2", query_scale: float | None = None,
                    decode: str = "flashinfer") -> dict[str, Any]:
    native_environment(arm, *scales, planner=planner, query_scale=query_scale, decode=decode)
    backend_lines = [line for line in text.splitlines() if "NATIVE_KV_BACKEND " in line]
    replay_lines = [line for line in text.splitlines() if "NATIVE_KV_GRAPH_REPLAY " in line]
    if len(backend_lines) != 1:
        raise ValueError("Require one actual native backend log record")
    backend = json.loads(backend_lines[0].split("NATIVE_KV_BACKEND ", 1)[1])
    expected = "FlashInferBackend" if arm == "auto" else "FlashInferFP8Backend"
    required = {"class": expected, "kv_cache_dtype": arm, "attention_impl": "flashinfer",
                "planner_backend": planner, "paged_prefill": True, "quality_transform": False,
                "graph_enabled": graph_mode == "cudagraphs", "decode_tensor_cores": True,
                "flashinfer_version": "0.6.18.post1"}
    if arm not in ("auto", "fp8_e4m3") or graph_mode not in ("off", "cudagraphs"):
        raise ValueError("Unsupported native quality arm/Graph mode")
    if any(backend.get(key) != value for key, value in required.items()):
        raise ValueError("Native planner/backend/Graph evidence differs from requested configuration")
    if planner == "fa3" and decode != "xqa":
        dtype = "torch.bfloat16" if arm == "auto" else "torch.float8_e4m3fn"
        if (backend.get("profile_scope") != "production_fa3" or backend.get("q_data_type") != dtype
                or backend.get("kv_data_type") != dtype
                or backend.get("q_scale") != (None if arm == "auto" else query_scale)):
            raise ValueError("FA3 actual Q/KV dtype or query scale differs")
    if decode == "xqa":
        expected_internal = "torch.bfloat16" if arm == "auto" else "torch.float8_e4m3fn"
        if any(backend.get(key) != value for key, value in {
            "profile_scope": "production_xqa", "decode_impl": "xqa", "q_data_type": "torch.bfloat16",
            "kv_data_type": expected_internal,
            "q_scale": None, "decode_internal_q_data_type": expected_internal, "xqa_max_context": 32768,
            "xqa_enable_pdl": True,
            "xqa_unused_kv_policy": "finite_full_initialization",
            "xqa_graph_padding": "reserved_page0_one_token_discard",
        }.items()):
            raise ValueError("Actual XQA prefill/decode/padding precision differs")
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
    profile = manifest.get("execution_profile")
    if profile not in (PROFILE, FA3_PROFILE, XQA_PROFILE, BF16_XQA_PROFILE) or manifest.get("backend") != "flashinfer":
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
    planner = "fa3" if profile in (FA3_PROFILE, BF16_XQA_PROFILE) else "fa2"
    if profile == FA3_PROFILE and manifest.get("native_planner") != planner:
        raise ValueError("FA3 quality manifest must explicitly pin its planner")
    if profile == XQA_PROFILE and (manifest.get("native_planner") != "fa2"
                                  or manifest.get("native_decode") != "xqa"
                                  or manifest.get("fp8_query_scale") is not None):
        raise ValueError("XQA manifest must explicitly pin prefill/decode/query precision")
    if profile == BF16_XQA_PROFILE and (arm != "auto" or manifest.get("native_planner") != "fa3"
                                       or manifest.get("native_decode") != "xqa"
                                       or manifest.get("fp8_query_scale") is not None):
        raise ValueError("BF16 XQA reference requires explicit BF16 FA3 prefill, XQA decode and no FP8 Q scale")
    observed = native_evidence(text, arm.replace("-", "_"), manifest["graph_mode"], selected,
                               planner=planner, query_scale=manifest.get("fp8_query_scale"),
                               decode="xqa" if profile in (XQA_PROFILE, BF16_XQA_PROFILE) else "flashinfer")
    if observed["backend"] != snapshot["backend"] or observed["graph_replays"] != snapshot["graph_replays"]:
        raise ValueError("Native raw marker snapshot and structured evidence disagree")
    if evidence.get("native_backend") != observed["backend"]:
        raise ValueError("Native manifest backend differs from captured raw evidence")
