# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import summarize_gsm8k_pairs as summarize


def test_binary_discordants_use_exact_two_sided_mcnemar() -> None:
    metrics = summarize._metric_block([1, 1, 1, 0], [0, 0, 0, 0])

    assert metrics["correct"] == {"auto": 3, "int8": 0}
    assert metrics["paired_outcomes"] == {"down": 3, "up": 0, "tie": 1}
    assert metrics["mcnemar_exact_two_sided_p"] == 0.25


def test_equal_paired_scores_have_zero_bootstrap_interval() -> None:
    metrics = summarize._metric_block([0, 1, 1, 0], [0, 1, 1, 0])

    assert metrics["delta_accuracy_int8_minus_auto"] == 0
    assert metrics["mcnemar_exact_two_sided_p"] == 1
    assert metrics["paired_outcomes"] == {"down": 0, "up": 0, "tie": 4}
    assert metrics["paired_bootstrap_95ci_delta_accuracy"] == [0.0, 0.0]


def test_no_complete_first_batch_emits_progress_but_no_stage(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    batch_manifest = tmp_path / "dataset_manifest.json"
    batches = []
    for batch_index in range(27):
        start = batch_index * 50
        end = min(start + 50, 1319)
        batches.append(
            {
                "batch_index": batch_index,
                "file": f"batch_{batch_index:04d}.jsonl",
                "global_start": start,
                "global_end_exclusive": end,
                "count": end - start,
                "sha256": "0" * 64,
            }
        )
    batch_manifest.write_text(
        json.dumps(
            {
                "dataset": "gsm8k test",
                "source_count": 1319,
                "source_sha256": "1" * 64,
                "batches": batches,
            }
        ),
        encoding="utf-8",
    )

    result = summarize.summarize(run_root, batch_manifest)

    assert result is None
    assert not list(run_root.glob("stage_*.json"))
    progress = json.loads((run_root / "progress.json").read_text(encoding="utf-8"))
    assert progress["complete_batch_count"] == 0
    assert len(progress["partial_or_excluded_batches"]) == 27


def test_serial_protocol_can_be_inferred_only_with_server_max_seqs_one(tmp_path: Path) -> None:
    serial_config = {"server_manifest": {"runtime_configuration": {"max_seqs_per_batch": 1}}}

    protocol = summarize._execution_protocol(serial_config, "auto", tmp_path)

    assert protocol == {
        "request_concurrency": 1,
        "max_seqs_per_batch": 1,
        "concurrency_source": "inferred_serial_runner_and_max_seqs_1",
    }


def test_paired_protocol_records_mixed_request_concurrency(tmp_path: Path) -> None:
    auto_config = {
        "request_concurrency": 1,
        "server_manifest": {"runtime_configuration": {"max_seqs_per_batch": 1}},
    }
    int8_config = {
        "request_concurrency": 4,
        "server_manifest": {"runtime_configuration": {"max_seqs_per_batch": 1}},
    }

    protocol = summarize._paired_execution_protocol(auto_config, int8_config, tmp_path)
    assert protocol["auto"]["request_concurrency"] == 1
    assert protocol["int8"]["request_concurrency"] == 4


def test_paired_protocol_records_mixed_server_max_seqs(tmp_path: Path) -> None:
    auto_config = {
        "request_concurrency": 4,
        "server_manifest": {"runtime_configuration": {"max_seqs_per_batch": 4}},
    }
    int8_config = {
        "request_concurrency": 4,
        "server_manifest": {"runtime_configuration": {"max_seqs_per_batch": 1}},
    }

    protocol = summarize._paired_execution_protocol(auto_config, int8_config, tmp_path)
    assert protocol["auto"]["max_seqs_per_batch"] == 4
    assert protocol["int8"]["max_seqs_per_batch"] == 1
