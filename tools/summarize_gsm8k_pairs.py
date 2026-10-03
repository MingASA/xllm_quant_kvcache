# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Accumulate only complete, revalidated paired GSM8K shard scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.logger import logger
from tools.eval_kv_cache_quality import _jsonl, _score_saved

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 17
BOOTSTRAP_CHUNK_SIZE = 128


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _exact_mcnemar_p(auto_correct: list[int], int8_correct: list[int]) -> float:
    discordant_auto = sum(a == 1 and b == 0 for a, b in zip(auto_correct, int8_correct))
    discordant_int8 = sum(a == 0 and b == 1 for a, b in zip(auto_correct, int8_correct))
    discordants = discordant_auto + discordant_int8
    if discordants == 0:
        return 1.0
    smaller = min(discordant_auto, discordant_int8)
    probability_numerator = 2 * sum(math.comb(discordants, k) for k in range(smaller + 1))
    return min(1.0, probability_numerator / (2**discordants))


def _paired_bootstrap_ci(auto_correct: list[int], int8_correct: list[int], seed: int = BOOTSTRAP_SEED) -> list[float]:
    import numpy as np

    sample_count = len(auto_correct)
    if sample_count == 0:
        raise ValueError("Cannot bootstrap an empty paired sample")
    auto = np.asarray(auto_correct, dtype=np.float64)
    int8 = np.asarray(int8_correct, dtype=np.float64)
    generator = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    written = 0
    while written < BOOTSTRAP_REPLICATES:
        count = min(BOOTSTRAP_CHUNK_SIZE, BOOTSTRAP_REPLICATES - written)
        indices = generator.integers(0, sample_count, size=(count, sample_count))
        deltas[written : written + count] = np.mean(int8[indices] - auto[indices], axis=1)
        written += count
    lower, upper = np.quantile(deltas, [0.025, 0.975], method="linear")
    return [float(lower), float(upper)]


def _metric_block(auto_correct: list[int], int8_correct: list[int]) -> dict[str, Any]:
    count = len(auto_correct)
    if count != len(int8_correct) or count == 0:
        raise ValueError("Paired correctness vectors must be non-empty and the same length")
    auto_right = sum(auto_correct)
    int8_right = sum(int8_correct)
    down = sum(a == 1 and b == 0 for a, b in zip(auto_correct, int8_correct))
    up = sum(a == 0 and b == 1 for a, b in zip(auto_correct, int8_correct))
    tie = count - down - up
    delta = (int8_right - auto_right) / count
    return {
        "n": count,
        "correct": {"auto": auto_right, "int8": int8_right},
        "accuracy": {"auto": auto_right / count, "int8": int8_right / count},
        "delta_accuracy_int8_minus_auto": delta,
        "delta_percentage_points": 100.0 * delta,
        "paired_outcomes": {"down": down, "up": up, "tie": tie},
        "mcnemar_exact_two_sided_p": _exact_mcnemar_p(auto_correct, int8_correct),
        "paired_bootstrap_95ci_delta_accuracy": _paired_bootstrap_ci(auto_correct, int8_correct),
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "gsm8k test" or not isinstance(manifest.get("batches"), list):
        raise ValueError(f"Not a GSM8K batch manifest: {path}")
    if manifest.get("source_count") != 1319 or len(manifest["batches"]) != 27:
        raise ValueError("Expected the frozen full GSM8K test manifest (1319 rows, 27 batches)")
    return manifest


def _execution_protocol(config: dict[str, Any], arm: str, batch_dir: Path) -> dict[str, Any]:
    server = config["server_manifest"]
    runtime_config = server.get("runtime_configuration", {})
    max_seqs = runtime_config.get("max_seqs_per_batch")
    if not isinstance(max_seqs, int) or isinstance(max_seqs, bool) or max_seqs < 1:
        raise ValueError(f"{batch_dir}: {arm} manifest lacks valid max_seqs_per_batch")
    request_concurrency = config.get("request_concurrency")
    concurrency_source = "run_config"
    if request_concurrency is None:
        if max_seqs != 1:
            raise ValueError(f"{batch_dir}: {arm} config lacks request_concurrency for max_seqs={max_seqs}")
        request_concurrency = 1
        concurrency_source = "inferred_serial_runner_and_max_seqs_1"
    if not isinstance(request_concurrency, int) or isinstance(request_concurrency, bool) or request_concurrency < 1:
        raise ValueError(f"{batch_dir}: {arm} request_concurrency must be a positive integer")
    return {
        "request_concurrency": request_concurrency,
        "max_seqs_per_batch": max_seqs,
        "concurrency_source": concurrency_source,
    }


def _paired_execution_protocol(
    auto_config: dict[str, Any], int8_config: dict[str, Any], batch_dir: Path
) -> dict[str, dict[str, Any]]:
    protocol_by_arm = {
        "auto": _execution_protocol(auto_config, "auto", batch_dir),
        "int8": _execution_protocol(int8_config, "int8", batch_dir),
    }
    return protocol_by_arm


def _validate_pair(
    run_root: Path, batch: dict[str, Any], dataset_manifest_path: Path
) -> tuple[dict[str, Any], dict[str, list[int]], dict[str, str]]:
    batch_index = batch["batch_index"]
    batch_dir = run_root / f"batch_{batch_index:04d}"
    auto_path = batch_dir / "auto.jsonl"
    int8_path = batch_dir / "int8.jsonl"
    auto_config_path = batch_dir / "auto.config.json"
    int8_config_path = batch_dir / "int8.config.json"
    saved_scores_path = batch_dir / "paired_scores.json"
    required = (auto_path, int8_path, auto_config_path, int8_config_path, saved_scores_path)
    missing = [str(path.name) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{batch_dir}: missing {', '.join(missing)}")

    auto_config = json.loads(auto_config_path.read_text(encoding="utf-8"))
    int8_config = json.loads(int8_config_path.read_text(encoding="utf-8"))
    expected_data = (dataset_manifest_path.parent / batch["file"]).resolve()
    expected_sha = batch["sha256"]
    auto_server = auto_config.get("server_manifest", {})
    int8_server = int8_config.get("server_manifest", {})
    if auto_server.get("kv_cache_mode") != "auto" or int8_server.get("kv_cache_mode") != "int8":
        raise ValueError(f"{batch_dir}: server manifests do not identify the expected auto/int8 arms")
    if any(server.get("graph_mode") != "off" for server in (auto_server, int8_server)):
        raise ValueError(f"{batch_dir}: paired cache-quality stages require graph_mode=off")
    if any(server.get("server_started") is not True for server in (auto_server, int8_server)):
        raise ValueError(f"{batch_dir}: both server manifests must attest server_started=true")

    for arm, config in (("auto", auto_config), ("int8", int8_config)):
        if config.get("task") != "gsm8k" or config.get("arm") != arm:
            raise ValueError(f"{batch_dir}: invalid {arm} task/config identity")
        data_path = Path(config["data"]).resolve()
        if data_path != expected_data:
            raise ValueError(f"{batch_dir}: {arm} config points to unexpected shard {data_path}")
        if _file_sha256(data_path) != expected_sha or config.get("input_sha256") != expected_sha:
            raise ValueError(f"{batch_dir}: {arm} input hash does not match frozen shard manifest")
        if config.get("examples") != batch["count"]:
            raise ValueError(f"{batch_dir}: {arm} example count differs from frozen shard")
        if config.get("model_impl") != "python":
            raise ValueError(f"{batch_dir}: expected the Python model implementation")
    protocol_by_arm = _paired_execution_protocol(auto_config, int8_config, batch_dir)

    auto_rows = _jsonl(auto_path)
    int8_rows = _jsonl(int8_path)
    if len(auto_rows) != batch["count"] or len(int8_rows) != batch["count"]:
        raise ValueError(f"{batch_dir}: result JSONL is not a complete shard pair")
    start = batch["global_start"]
    for local_index, (auto_row, int8_row) in enumerate(zip(auto_rows, int8_rows)):
        expected_source_index = start + local_index
        if auto_row.get("_source_index") != expected_source_index:
            raise ValueError(f"{batch_dir}: auto source index mismatch at row {local_index}")
        if int8_row.get("_source_index") != expected_source_index:
            raise ValueError(f"{batch_dir}: int8 source index mismatch at row {local_index}")
        if auto_row.get("_eval_index") != local_index or int8_row.get("_eval_index") != local_index:
            raise ValueError(f"{batch_dir}: eval index mismatch at row {local_index}")

    # Re-run the harness's paired validation/scoring into a temporary directory;
    # this checks paired configs, source-record hashes, prompts, and token counts.
    with tempfile.TemporaryDirectory(prefix="gsm8k-pair-validate-") as temp_dir:
        validated_dir = Path(temp_dir)
        args = SimpleNamespace(
            task="gsm8k",
            auto_results=auto_path,
            int8_results=int8_path,
            output=validated_dir,
            strict_token_count=True,
        )
        _score_saved(args, None)
        validated_scores = json.loads((validated_dir / "paired_scores.json").read_text(encoding="utf-8"))
    saved_scores = json.loads(saved_scores_path.read_text(encoding="utf-8"))
    if saved_scores != validated_scores:
        raise ValueError(f"{batch_dir}: saved paired_scores.json is stale or differs from revalidation")
    if validated_scores.get("n") != batch["count"]:
        raise ValueError(f"{batch_dir}: scorer output count differs from frozen shard")

    strict_auto: list[int] = []
    strict_int8: list[int] = []
    flexible_auto: list[int] = []
    flexible_int8: list[int] = []
    for row in validated_scores["items"]:
        for source, destination in (
            (row["scores"]["auto"], strict_auto),
            (row["scores"]["int8"], strict_int8),
            (row["flexible_scores"]["auto"], flexible_auto),
            (row["flexible_scores"]["int8"], flexible_int8),
        ):
            if source not in (0, 1, 0.0, 1.0):
                raise ValueError(f"{batch_dir}: GSM8K correctness score is not binary: {source}")
            destination.append(int(source))

    evidence = {
        "auto_config_sha256": _file_sha256(auto_config_path),
        "auto_results_sha256": _file_sha256(auto_path),
        "int8_config_sha256": _file_sha256(int8_config_path),
        "int8_results_sha256": _file_sha256(int8_path),
        "paired_scores_sha256": _file_sha256(saved_scores_path),
        "input_sha256": expected_sha,
        "execution_protocol": protocol_by_arm,
    }
    score_vectors = {
        "strict_auto": strict_auto,
        "strict_int8": strict_int8,
        "flexible_auto": flexible_auto,
        "flexible_int8": flexible_int8,
    }
    return validated_scores, score_vectors, evidence


def summarize(run_root: Path, dataset_manifest_path: Path) -> Path | None:
    manifest = _load_manifest(dataset_manifest_path)
    run_root.mkdir(parents=True, exist_ok=True)
    batches = manifest["batches"]
    cumulative_vectors = {name: [] for name in ("strict_auto", "strict_int8", "flexible_auto", "flexible_int8")}
    evidence: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    complete_prefix: list[int] = []
    gap_found = False

    for batch in batches:
        batch_index = batch["batch_index"]
        try:
            _, vectors, batch_evidence = _validate_pair(run_root, batch, dataset_manifest_path)
        except (OSError, KeyError, TypeError, ValueError) as error:
            partial.append({"batch_index": batch_index, "status": "incomplete_or_invalid", "reason": str(error)})
            gap_found = True
            continue
        if gap_found:
            partial.append({"batch_index": batch_index, "status": "complete_after_gap_excluded"})
            continue
        complete_prefix.append(batch_index)
        evidence.append({"batch_index": batch_index, **batch_evidence})
        for name, values in vectors.items():
            cumulative_vectors[name].extend(values)

    progress = {
        "dataset_manifest": str(dataset_manifest_path.resolve()),
        "dataset_manifest_sha256": _file_sha256(dataset_manifest_path),
        "expected_batches": len(batches),
        "complete_contiguous_batches": complete_prefix,
        "complete_batch_count": len(complete_prefix),
        "complete_example_count": len(cumulative_vectors["strict_auto"]),
        "partial_or_excluded_batches": partial,
        "warning": "Only a fully completed, strictly revalidated contiguous paired prefix enters stage statistics.",
    }
    (run_root / "progress.json").write_bytes(_json_bytes(progress))
    if not complete_prefix:
        logger.warning("No complete strictly paired GSM8K shard yet; progress saved, no stage emitted")
        return None

    strict = _metric_block(cumulative_vectors["strict_auto"], cumulative_vectors["strict_int8"])
    flexible = _metric_block(cumulative_vectors["flexible_auto"], cumulative_vectors["flexible_int8"])
    example_count = strict["n"]
    stage: dict[str, Any] = {
        "task": "gsm8k",
        "stage_examples": example_count,
        "complete_batches": complete_prefix,
        "source_sha256": manifest["source_sha256"],
        "dataset_manifest_sha256": progress["dataset_manifest_sha256"],
        "pair_evidence": evidence,
        "execution_protocol_by_batch": {str(item["batch_index"]): item["execution_protocol"] for item in evidence},
        "bootstrap": {
            "method": "paired nonparametric bootstrap, resampling questions with replacement",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "chunk_size": BOOTSTRAP_CHUNK_SIZE,
            "confidence_level": 0.95,
        },
        "strict": strict,
        "flexible": flexible,
    }
    stage_path = run_root / f"stage_{example_count:04d}.json"
    stage_bytes = _json_bytes(stage)
    if stage_path.exists():
        if stage_path.read_bytes() != stage_bytes:
            raise FileExistsError(f"Refusing to overwrite changed stage results: {stage_path}")
        logger.info("Verified existing immutable stage %s", stage_path)
    else:
        stage_path.write_bytes(stage_bytes)
        logger.info("Wrote immutable paired GSM8K stage %s", stage_path)
    return stage_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Root containing batch_0000, batch_0001, ...")
    parser.add_argument("--dataset-manifest", type=Path, required=True, help="Frozen GSM8K shard manifest")
    args = parser.parse_args()
    result = summarize(args.root, args.dataset_manifest)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
