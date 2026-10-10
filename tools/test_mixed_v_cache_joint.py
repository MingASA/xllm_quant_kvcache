# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Verify full-cycle scoring preserves replays, load levels and pairing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import report_mixed_v_cache_experiment as report


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _run(root: Path, arm: str, workload: str, concurrency: int, trace: list[dict]) -> Path:
    run = root / arm / workload / f"c{concurrency}_r0"
    run.mkdir(parents=True)
    (run / "summary.json").write_text("{}")
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "quality_protocol": "joint_full_trace",
                "concurrency": concurrency,
                "repetition": 0,
                "replay_cycles": 2,
            }
        )
    )
    responses = []
    for index, row in enumerate(trace * 2):
        changed = (arm != "auto" and index >= len(trace)) or concurrency == 16
        correct = "#### 42" if workload == "gsm8k" else "yes"
        wrong = "#### 41" if workload == "gsm8k" else "no"
        responses.append(
            {
                "index": index,
                "id": row["id"],
                "task": row["task"],
                "prompt_sha256": row["prompt_sha256"],
                "prediction": wrong if changed else correct,
                "finish_reason": "length" if changed else "stop",
            }
        )
    # Request completion order differs from arrival order.
    _jsonl(run / "responses.jsonl", list(reversed(responses)))
    return run


def test_joint_quality_keeps_both_cycles_and_each_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    traces = [
        {"id": "gsm8k/0", "task": "gsm8k", "prompt_sha256": "gsm", "row": {"answer": "#### 42"}},
        *[
            {"id": f"{task}/0", "task": task, "prompt_sha256": task, "row": {"answers": ["yes"]}}
            for task in report.assets.TASKS
        ],
    ]
    _jsonl(tmp_path / "trace_quality.jsonl", traces)
    for workload, rows in (("gsm8k", traces[:1]), ("longbench", traces[1:])):
        _jsonl(tmp_path / f"trace_{workload}.jsonl", rows)
        for arm in ("auto", "bf16_int8"):
            _run(tmp_path, arm, workload, 8, rows)
        _run(tmp_path, "auto", workload, 16, rows)
    metrics = SimpleNamespace(
        qa_f1_score=lambda prediction, answer: float(prediction == answer),
        code_sim_score=lambda prediction, answer: float(prediction == answer),
    )
    monkeypatch.setattr(report.evaluator, "_longbench_assets", lambda _: ({}, {}, metrics, ""))
    output = tmp_path / "report"
    output.mkdir()
    scores = report._quality(tmp_path, output)
    assert scores["reference_concurrency"] == 8
    assert scores["tasks"]["gsm8k"]["bf16_int8_flexible_em"] == 0.5
    assert scores["longbench_macro"]["bf16_int8"] == 0.5
    assert scores["statistics"]["gsm8k"]["bf16_int8_flexible_em"] == {
        "mean": 0.5,
        "std": 0.5,
        "observations": 2,
    }
    assert scores["length_exhaustion"]["bf16_int8"] == [0, 1]
    assert scores["by_concurrency"]["16"]["tasks"]["gsm8k"]["auto_flexible_em"] == 0
    paired = json.loads((output / "paired_gsm8k_c8_r0_cycle1.json").read_text())
    assert paired[0]["scores"] == {"auto": 1, "bf16_int8": 0}


@pytest.mark.parametrize("fault", ["duplicate", "missing", "wrong_id", "wrong_prompt"])
def test_completed_joint_run_rejects_corrupt_cycle(tmp_path: Path, fault: str) -> None:
    trace = [{"id": "gsm8k/0", "task": "gsm8k", "prompt_sha256": "gsm"}]
    run = _run(tmp_path, "auto", "gsm8k", 8, trace)
    rows = report.evaluator._jsonl(run / "responses.jsonl")
    if fault == "duplicate":
        rows[0]["index"] = rows[1]["index"]
    elif fault == "missing":
        rows.pop()
    elif fault == "wrong_id":
        rows[0]["id"] = "gsm8k/other"
    else:
        rows[0]["prompt_sha256"] = "other"
    _jsonl(run / "responses.jsonl", rows)
    with pytest.raises(ValueError):
        report._full_cycle_responses(run, trace, 0)
