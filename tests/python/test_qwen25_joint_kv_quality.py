# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Offline contracts for the joint-KV quality runner."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_kv_cache_quality as evaluator
from tools import run_qwen25_joint_kv_quality as runner
from tools.qwen25_experiment_model import sha256_file


def _write_gsm_inputs(root: Path, duplicate: bool = False, count: int = 2) -> tuple[Path, Path]:
    rows = [
        {"id": f"gsm8k/{index}", "question": f"What is {index} + 1?", "answer": f"#### {index + 1}"}
        for index in range(count)
    ]
    if duplicate:
        rows[1]["question"] = rows[0]["question"]
    dataset = root / "gsm8k.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    shots = root / "shots.json"
    shots.write_text(json.dumps([{"question": f"shot {index}", "answer": "answer"} for index in range(5)]))
    return dataset, shots


def _gsm_args(root: Path, dataset: Path, shots: Path) -> tuple[list[str], Path]:
    output = root / "output"
    return (
        [
            "--model",
            str(root / "model"),
            "--gsm8k",
            str(dataset),
            "--gsm8k-shots",
            str(shots),
            "--output",
            str(output),
            "--tasks",
            "gsm8k",
            "--dry-run",
        ],
        output,
    )


def _allocation(arm: str) -> dict:
    block_size, pages, kv_heads, head_dim = 16, 2, 8, 128
    capacity = pages * block_size
    fmt, dtype, width = {
        "auto": ("bf16", "torch.bfloat16", 2),
        "int8": ("int8", "torch.int8", 1),
        "fp8_e4m3": ("fp8_e4m3", "torch.uint8", 1),
        "int4": ("int4", "torch.uint8", 1),
    }[arm]
    dim = (head_dim + 1) // 2 if arm == "int4" else head_dim
    shape = [pages, block_size, kv_heads, dim]
    tensors = {name: {"shape": shape, "dtype": dtype, "bytes": math.prod(shape) * width} for name in ("key", "value")}
    bytes_per_token = kv_heads * (dim * 2 * width + (8 if arm != "auto" else 0))
    if arm != "auto":
        scale_shape = shape[:-1]
        for name in ("key_scale", "value_scale"):
            tensors[name] = {
                "shape": scale_shape,
                "dtype": "torch.float32",
                "bytes": math.prod(scale_shape) * 4,
            }
    resident = sum(tensor["bytes"] for tensor in tensors.values())
    return {
        "token_capacity": capacity,
        "resident_bytes": resident,
        "layers": [
            {
                "key_format": fmt,
                "value_format": fmt,
                "tensors": tensors,
                "resident_bytes": resident,
                "bytes_per_token": bytes_per_token,
            }
        ],
    }


def _resume_fixture(root: Path, journal_indices: list[int] | None = None) -> tuple:
    dataset, shots = _write_gsm_inputs(root, count=3)
    argv, output = _gsm_args(root, dataset, shots)
    argv.remove("--dry-run")
    argv.extend(("--arms", "int8", "--resume", "--block-size", "16"))
    args = runner._parser().parse_args(argv)
    model = args.model
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "template"}))
    rows = evaluator._jsonl(dataset)
    prepared = {
        "rows": rows,
        "prompts": [f"prompt-{index}" for index in range(len(rows))],
        "counts": [7] * len(rows),
        "chat": True,
        "max_tokens": 512,
    }
    profile = {"size": "14b", "num_heads": 40, "num_kv_heads": 8, "head_dim": 128, "num_layers": 48}
    provenance = {"profile": profile, "weights": {"sha256": "a" * 64}}
    output.mkdir()
    (output / "experiment_manifest.json").write_text(json.dumps({"model_provenance": provenance}))
    attempt = output / "arms" / "int8" / "attempt-0001"
    attempt.mkdir(parents=True)
    allocation = _allocation("int8")
    layer = allocation["layers"][0]
    allocation["layers"] = [copy.deepcopy(layer) for _ in range(profile["num_layers"])]
    allocation["resident_bytes"] = layer["resident_bytes"] * profile["num_layers"]
    allocation_path = attempt / "allocation.json"
    allocation_path.write_text(json.dumps(allocation))
    evidence = {
        "attempt": str(attempt),
        "allocation_sha256": sha256_file(allocation_path),
        "allocation_verified": True,
    }
    server_manifest = runner._server_manifest(args, "int8", provenance, evidence)
    server_manifest_path = attempt / "server_manifest.json"
    server_manifest_path.write_text(json.dumps(server_manifest, indent=2) + "\n")
    run_dir = output / "runs" / "gsm8k"
    run_dir.mkdir(parents=True)
    config_path = run_dir / "int8.config.json"
    config = {
        "task": "gsm8k",
        "data": str(dataset.resolve()),
        "input_sha256": sha256_file(dataset),
        "model": "Qwen2.5-14B-Instruct",
        "arm": "int8",
        "endpoint": f"http://127.0.0.1:{args.port}/v1",
        "model_impl": "python",
        "quality_transform_mode": "none",
        "kv_cache_dtype": "int8",
        "tokenizer": str(model.resolve()),
        "sampling": {"temperature": 0, "top_p": 1, "seed": args.seed},
        "generation": {
            "temperature": 0,
            "top_p": 1,
            "seed": args.seed,
            "max_tokens": prepared["max_tokens"],
            "stop_token_ids": [151645, 151643],
            "stop_strings": ["Question:", "</s>", "<|im_end|>"],
        },
        "max_tokens": prepared["max_tokens"],
        "max_prompt_plus_generation_tokens": evaluator.MAX_CONTEXT_TOKENS,
        "examples": len(rows),
        "api_mode": "chat-completions",
        "execution_protocol": {
            "request_concurrency": args.concurrency,
            "max_seqs_per_batch": args.max_seqs_per_batch,
        },
        "runtime_versions": evaluator._runtime_versions("gsm8k"),
        "server_manifest": server_manifest,
        "server_manifest_sha256": sha256_file(server_manifest_path),
        "gsm8k_shots": str(shots.resolve()),
        "gsm8k_shots_sha256": sha256_file(shots),
    }
    config_path.write_text(json.dumps(config))
    if journal_indices is not None:
        journal_path = run_dir / "int8.jsonl"
        with journal_path.open("w") as stream:
            for index in journal_indices:
                row = rows[index]
                prediction = f"#### {index + 1}"
                response = {
                    "choices": [{"message": {"content": prediction}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                }
                usage = {
                    "local_prompt_tokens": 7,
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                }
                record = evaluator._result_record(index, row, prediction, usage, response, prepared["prompts"][index])
                stream.write(json.dumps(record) + "\n")
    return args, prepared, config_path, output / "runs" / "gsm8k" / "int8.jsonl"


def test_dry_run_audits_without_gpu_service_or_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, shots = _write_gsm_inputs(tmp_path)
    argv, output = _gsm_args(tmp_path, dataset, shots)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not start a service or inspect a model")

    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    monkeypatch.setattr(runner, "_run_arm", forbidden)
    monkeypatch.setattr(runner, "checkpoint_provenance", forbidden)
    runner.main(argv)

    plan = json.loads(capsys.readouterr().out)
    assert plan["data_audit"]["datasets"]["gsm8k"]["unique_questions"] == 2
    assert plan["environment"]["XLLM_QUANTIZED_BACKEND"] == "triton"
    assert not output.exists()


@pytest.mark.parametrize("arm", runner.ARMS)
def test_allocation_contract_accepts_real_compressed_cache_shapes(arm: str) -> None:
    runner._validate_allocation(arm, _allocation(arm), {"num_layers": 1, "num_kv_heads": 8, "head_dim": 128}, 16)


def test_allocation_contract_rejects_bf16_qdq_for_int8() -> None:
    allocation = _allocation("int8")
    allocation["layers"][0]["tensors"].pop("key_scale")
    allocation["layers"][0]["tensors"].pop("value_scale")
    with pytest.raises(ValueError, match="scale set"):
        runner._validate_allocation("int8", allocation, {"num_layers": 1, "num_kv_heads": 8, "head_dim": 128}, 16)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("key_dtype", "Invalid resident tensor: key"),
        ("value_shape", "Resident K/V shape or token capacity mismatch"),
        ("scale_dtype", "Invalid resident tensor: key_scale"),
        ("bytes", "Invalid resident tensor: value"),
        ("layer_count", "Invalid cache layer count/capacity"),
    ],
)
def test_allocation_contract_rejects_malformed_resident_cache(mutation: str, message: str) -> None:
    allocation = _allocation("int8")
    profile = {"num_layers": 1, "num_kv_heads": 8, "head_dim": 128}
    if mutation == "key_dtype":
        allocation["layers"][0]["tensors"]["key"]["dtype"] = "torch.bfloat16"
    elif mutation == "value_shape":
        allocation["layers"][0]["tensors"]["value"]["shape"][-1] -= 1
    elif mutation == "scale_dtype":
        allocation["layers"][0]["tensors"]["key_scale"]["dtype"] = "torch.bfloat16"
    elif mutation == "bytes":
        allocation["layers"][0]["tensors"]["value"]["bytes"] -= 1
    else:
        profile["num_layers"] = 48

    with pytest.raises(ValueError, match=message):
        runner._validate_allocation("int8", allocation, profile, 16)


def test_fp8_server_manifest_matches_evaluator_arm_label(tmp_path: Path) -> None:
    model = tmp_path / "checkpoint"
    model.mkdir()
    config = model / "config.json"
    tokenizer = model / "tokenizer_config.json"
    config.write_text(json.dumps({"model_type": "qwen2"}))
    tokenizer.write_text(json.dumps({"chat_template": "template"}))
    args = SimpleNamespace(
        model=model,
        max_seqs_per_batch=8,
        max_tokens_per_batch=1024,
        max_memory_utilization=0.8,
        max_cache_size=0,
        max_concurrent_requests=8,
        block_size=16,
    )
    manifest = runner._server_manifest(
        args,
        "fp8_e4m3",
        {"weights": {"sha256": "a" * 64}},
        {"allocation_verified": True, "allocation_sha256": "b" * 64},
    )
    manifest_path = tmp_path / "server_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    loaded = evaluator._load_server_manifest(manifest_path, "fp8-e4m3", "Qwen2.5-14B-Instruct", model)

    assert loaded["kv_cache_mode"] == "fp8_e4m3"
    assert loaded["backend"] == "triton"


def test_data_audit_rejects_duplicate_questions(tmp_path: Path) -> None:
    dataset, shots = _write_gsm_inputs(tmp_path, duplicate=True)
    argv, _ = _gsm_args(tmp_path, dataset, shots)
    args = runner._parser().parse_args(argv)
    paths = runner._validate_args(args)

    with pytest.raises(ValueError, match="Duplicate questions/IDs"):
        runner._data_audit(args, paths)


def test_gsm8k_pair_scoring_uses_same_rows_for_all_arms() -> None:
    rows = [
        {"id": "one", "question": "one plus one?", "answer": "#### 2"},
        {"id": "two", "question": "two plus two?", "answer": "#### 4"},
    ]
    scores, items = evaluator._score_pair(
        "gsm8k",
        rows,
        {"auto": ["#### 2", "#### 4"], "int8": ["#### 3", "#### 4"]},
        None,
    )

    assert scores == {
        "auto_strict_em": 1.0,
        "auto_flexible_em": 1.0,
        "int8_strict_em": 0.5,
        "int8_flexible_em": 0.5,
    }
    assert [item["flip"] for item in items] == ["down", "tie"]
    assert [item["flexible_flip"] for item in items] == ["down", "tie"]


def test_resume_accepts_valid_noncontiguous_indices(tmp_path: Path) -> None:
    args, prepared, _, result = _resume_fixture(tmp_path, [0, 2])

    assert runner._verify_journal(args, "gsm8k", "int8", prepared) == {0, 2}


@pytest.mark.parametrize(
    ("journal", "message"),
    [
        (b'{"truncated":', "incomplete final line"),
        ("duplicate", "duplicate evaluation index"),
    ],
)
def test_resume_rejects_corrupt_or_duplicate_journal(tmp_path: Path, journal: bytes | str, message: str) -> None:
    args, prepared, _, result = _resume_fixture(tmp_path, [0])
    if journal == "duplicate":
        second = evaluator._jsonl(result)[0]
        result.write_text(result.read_text() + json.dumps(second) + "\n")
    else:
        result.write_bytes(journal)

    with pytest.raises(ValueError, match=message):
        runner._verify_journal(args, "gsm8k", "int8", prepared)


def test_resume_rejects_sampling_configuration_drift(tmp_path: Path) -> None:
    args, prepared, config_path, _ = _resume_fixture(tmp_path, [0])
    config = json.loads(config_path.read_text())
    config["sampling"]["seed"] += 1
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="Resume config differs: sampling"):
        runner._verify_journal(args, "gsm8k", "int8", prepared)


def test_attest_arm_writes_indexed_journal_integrity(tmp_path: Path) -> None:
    args, prepared, _, _ = _resume_fixture(tmp_path, [0, 1, 2])

    runner._attest_arm(args, {"gsm8k": args.gsm8k}, "int8", {"gsm8k": prepared})

    assert args.resume is True
    integrity = json.loads((args.output / "arms" / "int8" / "integrity.json").read_text())
    assert integrity["gsm8k"]["expected_rows"] == 3
    assert integrity["gsm8k"]["unique_completed_indices"] == 3


def test_report_scores_all_arms_and_runs_saved_pairing_cli(tmp_path: Path) -> None:
    args, prepared, _, _ = _resume_fixture(tmp_path, [0, 1, 2])
    args.arms = list(runner.ARMS)
    paths = {"gsm8k": args.gsm8k}
    manifest = json.loads((args.output / "experiment_manifest.json").read_text())
    provenance = manifest["model_provenance"]
    predictions = {
        "auto": ["#### 1", "#### 2", "#### 3"],
        "int8": ["#### 9", "#### 2", "#### 3"],
        "fp8_e4m3": ["#### 1", "#### 9", "#### 3"],
        "int4": ["#### 1", "#### 2", "#### 3"],
    }
    prepared_by_task = {"gsm8k": prepared}
    for arm in runner.ARMS:
        label = arm.replace("_", "-")
        attempt = args.output / "arms" / arm / "attempt-0001"
        attempt.mkdir(parents=True, exist_ok=True)
        allocation_path = attempt / "allocation.json"
        allocation_path.write_text(json.dumps(_allocation(arm)))
        evidence = {
            "attempt": str(attempt),
            "allocation_sha256": sha256_file(allocation_path),
            "allocation_verified": True,
        }
        server_manifest = runner._server_manifest(args, arm, provenance, evidence)
        server_manifest_path = attempt / "server_manifest.json"
        server_manifest_path.write_text(json.dumps(server_manifest, indent=2) + "\n")
        config = json.loads((args.output / "runs" / "gsm8k" / "int8.config.json").read_text())
        config.update(arm=label, kv_cache_dtype=arm, server_manifest=server_manifest)
        config["server_manifest_sha256"] = sha256_file(server_manifest_path)
        (args.output / "runs" / "gsm8k" / f"{label}.config.json").write_text(json.dumps(config))
        journal = args.output / "runs" / "gsm8k" / f"{label}.jsonl"
        with journal.open("w") as stream:
            for index, row in enumerate(prepared["rows"]):
                prediction = predictions[arm][index]
                finish = "length" if arm == "int4" and index == 1 else "stop"
                response = {
                    "choices": [{"message": {"content": prediction}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                }
                usage = {
                    "local_prompt_tokens": 7,
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                }
                record = evaluator._result_record(index, row, prediction, usage, response, prepared["prompts"][index])
                stream.write(json.dumps(record) + "\n")

    runner._report(args, paths, prepared_by_task)

    summary = json.loads((args.output / "summary.json").read_text())
    scores = {row["arm"]: row["scores"] for row in summary["tasks"]}
    exhausted = {row["arm"].replace("_", "-"): row["length_exhaustion"] for row in summary["tasks"]}
    assert scores["auto"]["auto_strict_em"] == 1.0
    assert scores["int8"]["int8_flexible_em"] == pytest.approx(2 / 3)
    assert scores["fp8_e4m3"]["fp8-e4m3_strict_em"] == pytest.approx(2 / 3)
    assert scores["int4"]["int4_flexible_em"] == 1.0
    assert exhausted == {"auto": 0, "int8": 0, "fp8-e4m3": 0, "int4": 1}
    expected_flips = {
        "int8": ["down", "tie", "tie"],
        "fp8_e4m3": ["tie", "down", "tie"],
        "int4": ["tie", "tie", "tie"],
    }
    for arm, flips in expected_flips.items():
        paired = json.loads((args.output / "paired" / "gsm8k" / arm / "paired_scores.json").read_text())
        assert paired["n"] == 3
        assert [item["flip"] for item in paired["items"]] == flips
