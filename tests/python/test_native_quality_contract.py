# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU-only native quality contracts, raw attestation and legacy compatibility."""

import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import eval_kv_cache_quality as evaluator
from tools import run_qwen25_joint_kv_quality as runner
from tools.native_kv_quality_contract import clean_native_environment, native_evidence


def _args(tmp_path: Path, *extra: str) -> object:
    return runner._parser().parse_args([
        "--model", str(tmp_path / "model"), "--output", str(tmp_path / "out"),
        "--tasks", "narrativeqa", "--longbench-data", str(tmp_path / "data"),
        "--longbench-repo", str(tmp_path / "scorer"), "--arms", "auto", "fp8_e4m3",
        "--attention-backend", "native-flashinfer", "--graph-mode", "cudagraphs",
        "--fp8-k-scale", "0.125", "--fp8-v-scale", "0.0625", "--max-seqs-per-batch", "8",
        "--cohort-scope", "pilot", "--binary-source-commit", "c" * 40, *extra])


def _backend(arm: str) -> dict:
    value = {"class": "FlashInferBackend" if arm == "auto" else "FlashInferFP8Backend",
             "kv_cache_dtype": arm, "attention_impl": "flashinfer", "planner_backend": "fa2",
             "paged_prefill": True, "quality_transform": False, "graph_enabled": True,
             "graph_captured_batches": [1, 2, 4, 8], "decode_tensor_cores": True,
             "flashinfer_version": "0.6.18.post1"}
    if arm == "fp8_e4m3":
        value.update(k_scale=0.125, v_scale=0.0625, fp8_views_alias_resident_pages=True,
                     kv_data_type="torch.float8_e4m3fn")
    return value


def _log(backend: dict) -> str:
    replay = {"class": backend["class"], "batch_size": 1, "captured_batch_size": 1}
    return "NATIVE_KV_BACKEND " + json.dumps(backend) + "\nNATIVE_KV_GRAPH_REPLAY " + json.dumps(replay)


def _manifest(tmp_path: Path, arm: str = "fp8_e4m3", planner: str = "fa2") -> tuple[Path, dict]:
    args = _args(tmp_path, *(["--native-planner", "fa3", "--fp8-q-scale", "1.0"] if planner == "fa3" else []))
    args.model.mkdir()
    (args.model / "config.json").write_text('{"model_type":"qwen2"}')
    (args.model / "tokenizer_config.json").write_text('{"chat_template":"frozen"}')
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "allocation.json").write_text('{"fixture":"allocation already verified by runner"}')
    backend = _backend(arm)
    if planner == "fa3":
        backend.update(planner_backend="fa3", profile_scope="production_fa3",
                       q_data_type="torch.bfloat16" if arm == "auto" else "torch.float8_e4m3fn",
                       kv_data_type="torch.bfloat16" if arm == "auto" else "torch.float8_e4m3fn",
                       q_scale=None if arm == "auto" else 1.0)
    native = native_evidence(_log(backend), arm, "cudagraphs", (0.125, 0.0625), planner=planner,
                             query_scale=1.0 if planner == "fa3" else None)
    (attempt / "native_evidence.json").write_text(json.dumps(native))
    evidence = {"attempt": str(attempt), "allocation_verified": True,
                "allocation_sha256": hashlib.sha256((attempt / "allocation.json").read_bytes()).hexdigest(),
                "native_backend": native["backend"],
                "native_evidence_sha256": hashlib.sha256((attempt / "native_evidence.json").read_bytes()).hexdigest()}
    manifest = runner._server_manifest(args, arm, {"weights": {"sha256": "a" * 64}}, evidence)
    path = attempt / "server_manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_native_arguments_and_per_arm_environment_are_explicit(tmp_path: Path) -> None:
    args = _args(tmp_path)
    runner._validate_args(args)
    baseline, fp8 = runner._environment(args, "auto"), runner._environment(args, "fp8_e4m3")
    assert "XLLM_QUANTIZED_BACKEND" not in baseline
    assert "XLLM_FP8_K_SCALE" not in baseline
    assert fp8["XLLM_QUANTIZED_BACKEND"] == "flashinfer"
    cleaned = clean_native_environment({"XLLM_FP8_CALIBRATION_OUTPUT": "/bad", "PATH": "/fixture"}, baseline)
    assert "XLLM_FP8_CALIBRATION_OUTPUT" not in cleaned
    assert cleaned["PATH"] == "/fixture"
    for arm in args.arms:
        command = runner._server_command(args, arm)
        assert "--python_graph_backend=cudagraphs" in command
        assert "--enable_graph=true" in command
        assert "--enable_prefix_cache=false" in command


@pytest.mark.parametrize("extra", [["--arms", "int8"], ["--fp8-k-scale", "nan"],
                                   ["--max-seqs-per-batch", "200"], ["--binary-source-commit", "wrong"]])
def test_native_invalid_scope_fails_before_model_start(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(ValueError):
        runner._validate_args(_args(tmp_path, *extra))


@pytest.mark.parametrize("arm", ["auto", "fp8_e4m3"])
def test_evaluator_accepts_only_actual_native_graph_attestation(tmp_path: Path, arm: str) -> None:
    path, manifest = _manifest(tmp_path, arm)
    observed = evaluator._load_server_manifest(path, arm.replace("_", "-"), manifest["model"], tmp_path / "model")
    assert observed["backend"] == "flashinfer"
    assert observed["binary_source_commit"] == "c" * 40
    (path.parent / "native_evidence.json").write_text('{}')
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluator._load_server_manifest(path, arm.replace("_", "-"), manifest["model"], tmp_path / "model")


def test_graph_flag_or_backend_json_alone_cannot_pass(tmp_path: Path) -> None:
    backend = _backend("fp8_e4m3")
    with pytest.raises(ValueError, match="capture and replay"):
        native_evidence("NATIVE_KV_BACKEND " + json.dumps(backend), "fp8_e4m3", "cudagraphs", (0.125, 0.0625))
    backend["k_scale"] = 1.0
    with pytest.raises(ValueError, match="fixed scales"):
        native_evidence(_log(backend), "fp8_e4m3", "cudagraphs", (0.125, 0.0625))


def test_native_snapshot_does_not_relax_legacy_graph_gate(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    manifest.pop("execution_profile")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="graph_mode off"):
        evaluator._load_server_manifest(path, "fp8-e4m3", manifest["model"], tmp_path / "model")


def test_narrativeqa_duplicate_prompt_preserves_distinct_records(tmp_path: Path) -> None:
    args = _args(tmp_path)
    data = tmp_path / "data/narrativeqa.jsonl"
    data.parent.mkdir()
    data.write_text('\n'.join(json.dumps(row) for row in [
        {"_id": "one", "context": "same", "input": "question", "answers": ["a"]},
        {"_id": "two", "context": "same", "input": "question", "answers": ["b"]}]))
    for name in ("metrics.py", "config/dataset2prompt.json", "config/dataset2maxlen.json"):
        path = args.longbench_repo / "LongBench" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    audit = runner._data_audit(args, {"narrativeqa": data})["datasets"]["narrativeqa"]
    assert audit["rows"] == audit["unique_records"] == 2
    assert audit["unique_questions"] == 1
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    data.write_text('\n'.join(map(json.dumps, [rows[0], rows[0]])))
    with pytest.raises(ValueError, match="Duplicate"):
        runner._data_audit(args, {"narrativeqa": data})


def test_bootstrap_forces_decode_but_quality_protocol_does_not(tmp_path: Path) -> None:
    args = _args(tmp_path)
    response = {"usage": {"completion_tokens": 16}, "choices": [{"finish_reason": "length"}]}
    captured = {}

    def urlopen(request: object, timeout: int) -> io.BytesIO:
        captured.update(json.loads(request.data))
        return io.BytesIO(json.dumps(response).encode())

    with patch.object(runner.urllib.request, "urlopen", urlopen):
        runner._native_bootstrap(args)
    assert captured["ignore_eos"] is True
    command = runner._evaluator_command(args, "narrativeqa", tmp_path / "data", "auto", tmp_path / "manifest", False)
    assert "--ignore-eos" not in command
    response["choices"][0]["finish_reason"] = "stop"
    with patch.object(runner.urllib.request, "urlopen", urlopen), pytest.raises(ValueError, match="fixed length"):
        runner._native_bootstrap(args)


@pytest.mark.parametrize("arm", ["auto", "fp8_e4m3"])
def test_fa3_quality_attests_actual_query_dtype_and_separate_profile(tmp_path: Path, arm: str) -> None:
    path, manifest = _manifest(tmp_path, arm, "fa3")
    observed = evaluator._load_server_manifest(path, arm.replace("_", "-"), manifest["model"], tmp_path / "model")
    assert observed["execution_profile"] == "native-flashinfer-fa3-joint-v1"
    snapshot = json.loads((path.parent / "native_evidence.json").read_text())
    with pytest.raises(ValueError, match="planner/backend/Graph"):
        native_evidence(_log(snapshot["backend"]), arm, "cudagraphs", (0.125, 0.0625))
    for field, wrong in [("q_data_type", "torch.float32"), ("q_scale", 2.0), ("planner_backend", "fa2")]:
        backend = copy.deepcopy(snapshot["backend"])
        backend[field] = wrong
        with pytest.raises(ValueError):
            native_evidence(_log(backend), arm, "cudagraphs", (0.125, 0.0625), planner="fa3", query_scale=1.0)


def test_fa3_quality_environment_and_production_prefix_are_explicit(tmp_path: Path) -> None:
    args = _args(tmp_path, "--native-planner", "fa3", "--fp8-q-scale", "1.0", "--max-seqs-per-batch", "128")
    runner._validate_args(args)
    bf16, fp8 = runner._environment(args, "auto"), runner._environment(args, "fp8_e4m3")
    assert bf16["XLLM_FLASHINFER_NATIVE_PLANNER"] == fp8["XLLM_FLASHINFER_NATIVE_PLANNER"] == "fa3"
    assert "XLLM_FLASHINFER_FP8_QUERY" not in bf16
    assert fp8["XLLM_FLASHINFER_FP8_QUERY"] == "1"
    assert float(fp8["XLLM_FLASHINFER_FP8_Q_SCALE"]) == 1.0
    assert "--enable_prefix_cache=true" in runner._server_command(args, "auto")


@pytest.mark.parametrize("extra", [["--native-planner", "fa3"], ["--fp8-q-scale", "1.0"],
                                  ["--native-planner", "fa3", "--fp8-q-scale", "nan"],
                                  ["--native-planner", "fa3", "--fp8-q-scale", "0"],
                                  ["--runtime-source-commit", "bad"]])
def test_quality_rejects_implicit_or_invalid_query_profile(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(ValueError):
        runner._validate_args(_args(tmp_path, *extra))
