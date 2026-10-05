# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Export compact, source-hashed evidence for the 2026-10-05 FP8 experiments."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = Path("/mnt/e/AI/xllm-eval-data")
_GSM_DATA = _DATA_ROOT / "int4-model-eval-20261002/gsm8k/data.jsonl"
_LB_TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "2wikimqa",
    "hotpotqa",
    "musique",
    "lcc",
    "repobench-p",
)
_ARMS = ("auto", "k-only-fp8", "v-only-fp8")
_LOG_PATTERN = re.compile(r"backend|scale|kv_cache_dtype|quantized|flashinfer|triton", re.IGNORECASE)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_record(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _compact_rows(source: Path, destination: Path, task: str, arm: str) -> int:
    rows = _read_jsonl(source)
    indices = [row.get("_eval_index") for row in rows]
    if any(not isinstance(index, int) for index in indices) or len(set(indices)) != len(indices):
        raise ValueError(f"invalid or duplicate _eval_index in {source}")
    compact_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["_eval_index"]):
        response = row.get("api_response") or {}
        choices = response.get("choices") or [{}]
        choice = choices[0]
        usage = row.get("usage") or {}
        compact_rows.append(
            {
                "task": task,
                "arm": arm,
                "question_id": row.get("_id", row.get("id", f"{task}:{row['_eval_index']}")),
                "eval_index": row["_eval_index"],
                "prompt_sha256": row.get("_prompt_sha256"),
                "source_record_sha256": row.get("_record_sha256"),
                "completion_text": row.get("prediction", ""),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "error": row.get("error") or response.get("error"),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", encoding="utf-8", compresslevel=9) as stream:
        for row in compact_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(compact_rows)


def _compact_pair_scores(source: Path, destination: Path) -> dict[str, Any]:
    value = json.loads(source.read_text(encoding="utf-8"))
    fields = ("index", "id", "scores", "flexible_scores", "flip", "flexible_flip")
    items = [{key: item[key] for key in fields if key in item} for item in value.get("items", [])]
    compact = {"n": value.get("n"), "scores": value.get("scores"), "items": items}
    _write_json(destination, compact)
    return compact


def _compact_vllm(source: Path, destination: Path, arm: str) -> list[dict[str, Any]]:
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value.get("records"), list):
        raise ValueError(f"unexpected vLLM result schema: {source}")
    dataset_rows = _read_jsonl(_GSM_DATA)
    compact: list[dict[str, Any]] = []
    for record in sorted(value["records"], key=lambda item: item["index"]):
        index = record["index"]
        source_row = dataset_rows[index]
        prompt = record.get("prompt", "")
        compact.append(
            {
                "task": "gsm8k",
                "arm": arm,
                "question_id": f"gsm8k:{index}",
                "eval_index": index,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "source_record_sha256": _sha256_record(source_row),
                "completion_text": record.get("output", ""),
                "prompt_tokens": record.get("prompt_tokens"),
                "completion_tokens": record.get("output_tokens"),
                "total_tokens": (record.get("prompt_tokens") or 0) + (record.get("output_tokens") or 0),
                "finish_reason": record.get("finish_reason"),
                "strict_em": record.get("strict_em"),
                "flexible_em": record.get("flexible_em"),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", encoding="utf-8", compresslevel=9) as stream:
        for row in compact:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return compact


def _index_logs(sources: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for root in sources:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.log")):
            excerpts: list[str] = []
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if _LOG_PATTERN.search(line):
                        excerpts.append(line.strip()[:320])
                        if len(excerpts) == 20:
                            break
            entries.append(
                {
                    "source_path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "backend_scale_excerpt": excerpts,
                }
            )
    return entries


def _index_source_files(sources: dict[str, Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for label, root in sources.items():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix != ".log":
                entries.append(
                    {
                        "experiment": label,
                        "source_path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
    return entries


def _copy_qdq_evidence(output: Path, gsm_source: Path, lb_source: Path) -> dict[str, Any]:
    base = output / "qdq-single-sided-fp8"
    gsm_counts: dict[str, int] = {}
    _copy_file(gsm_source / "experiment_manifest.json", base / "gsm8k/metadata/experiment_manifest.json")
    for arm in _ARMS:
        _copy_file(gsm_source / "runs/gsm8k" / f"{arm}.config.json", base / f"gsm8k/metadata/configs/{arm}.json")
        _copy_file(
            gsm_source / "arms" / arm / "server_manifest.json", base / f"gsm8k/metadata/server_manifests/{arm}.json"
        )
        gsm_counts[arm] = _compact_rows(
            gsm_source / "runs/gsm8k" / f"{arm}.jsonl", base / f"gsm8k/runs/{arm}.jsonl.gz", "gsm8k", arm
        )
    gsm_pairs = {}
    for arm in ("k-only-fp8", "v-only-fp8"):
        pair = _compact_pair_scores(
            gsm_source / f"runs/gsm8k/paired-{arm}/paired_scores.json",
            base / f"gsm8k/paired/{arm}.json",
        )
        gsm_pairs[arm] = pair["scores"]

    _copy_file(lb_source / "experiment_manifest.json", base / "longbench/metadata/experiment_manifest.json")
    lb_counts: dict[str, dict[str, int]] = {}
    lb_pairs: dict[str, dict[str, Any]] = {}
    for task in _LB_TASKS:
        lb_counts[task] = {}
        for arm in _ARMS:
            config = lb_source / "runs" / task / f"{arm}.config.json"
            if config.is_file():
                _copy_file(config, base / f"longbench/metadata/configs/{task}/{arm}.json")
            lb_counts[task][arm] = _compact_rows(
                lb_source / "runs" / task / f"{arm}.jsonl",
                base / f"longbench/runs/{task}/{arm}.jsonl.gz",
                task,
                arm,
            )
            if arm != "auto":
                lb_pairs[f"{task}/{arm}"] = _compact_pair_scores(
                    lb_source / f"runs/{task}/paired-{arm}/paired_scores.json",
                    base / f"longbench/paired/{task}/{arm}.json",
                )["scores"]
    for arm in _ARMS:
        _copy_file(
            lb_source / "arms" / arm / "server_manifest.json", base / f"longbench/metadata/server_manifests/{arm}.json"
        )
    _copy_file(lb_source / "source_snapshot.tar.gz", base / "source/source_snapshot.tar.gz")
    summary = {
        "gsm8k_rows_per_arm": gsm_counts,
        "gsm8k_pair_scores": gsm_pairs,
        "longbench_rows": lb_counts,
        "longbench_pair_scores": lb_pairs,
    }
    _write_json(base / "summary.json", summary)
    return summary


def _copy_xllm_native(output: Path, source: Path) -> dict[str, Any]:
    base = output / "xllm-native-fp8"
    arms = ("auto", "fp8-e4m3", "fp8-e5m2")
    _copy_file(source / "experiment_manifest.json", base / "metadata/experiment_manifest.json")
    for arm in arms:
        _copy_file(source / "runs" / f"{arm}.config.json", base / f"metadata/configs/{arm}.json")
        _copy_file(source / "arms" / arm / "server_manifest.json", base / f"metadata/server_manifests/{arm}.json")
    _copy_file(source / "source_snapshot_manifest.json", base / "source/source_snapshot_manifest.json")
    shutil.copytree(source / "source_snapshot", base / "source/source_snapshot")
    counts: dict[str, int] = {}
    for arm in arms:
        counts[arm] = _compact_rows(source / "runs" / f"{arm}.jsonl", base / f"runs/{arm}.jsonl.gz", "gsm8k", arm)
    pairs: dict[str, Any] = {}
    for arm in ("fp8-e4m3", "fp8-e5m2"):
        pairs[arm] = _compact_pair_scores(
            source / f"runs/paired-{arm}/paired_scores.json", base / f"paired/{arm}.json"
        )["scores"]
    summary = {"rows_per_arm": counts, "pair_scores": pairs}
    _write_json(base / "summary.json", summary)
    return summary


def _copy_vllm_reference(output: Path, source: Path) -> dict[str, Any]:
    base = output / "vllm-reference"
    _copy_file(source / "run_config.json", base / "metadata/run_config.json")
    _copy_file(source / "run_vllm_fp8_reference.py", base / "reproduction_source/run_vllm_fp8_reference.py")
    compact_arms: dict[str, list[dict[str, Any]]] = {}
    for arm, filename in (("bf16", "gsm8k-bf16.json"), ("fp8-e4m3", "gsm8k-fp8.json")):
        compact_arms[arm] = _compact_vllm(source / filename, base / f"gsm8k/{arm}.jsonl.gz", arm)
    by_index = {arm: {row["eval_index"]: row for row in rows} for arm, rows in compact_arms.items()}
    if set(by_index["bf16"]) != set(by_index["fp8-e4m3"]):
        raise ValueError("vLLM BF16/FP8 GSM8K indices differ")
    paired: list[dict[str, Any]] = []
    for index in sorted(by_index["bf16"]):
        bf, fp8 = by_index["bf16"][index], by_index["fp8-e4m3"][index]
        if bf["prompt_sha256"] != fp8["prompt_sha256"]:
            raise ValueError(f"vLLM paired prompt hash mismatch at {index}")
        paired.append(
            {
                "eval_index": index,
                "question_id": bf["question_id"],
                "prompt_sha256": bf["prompt_sha256"],
                "source_record_sha256": bf["source_record_sha256"],
                "scores": {"bf16": bf["strict_em"], "fp8-e4m3": fp8["strict_em"]},
                "flexible_scores": {"bf16": bf["flexible_em"], "fp8-e4m3": fp8["flexible_em"]},
            }
        )
    _write_json(base / "gsm8k/paired_scores.json", {"n": len(paired), "items": paired})
    summary = {
        arm: {
            "n": len(rows),
            "strict_correct": sum(row["strict_em"] for row in rows),
            "flexible_correct": sum(row["flexible_em"] for row in rows),
            "length": sum(row["finish_reason"] == "length" for row in rows),
        }
        for arm, rows in compact_arms.items()
    }
    _write_json(base / "summary.json", summary)
    return summary


def _write_readme(output: Path, qdq: dict[str, Any], native: dict[str, Any], vllm: dict[str, Any]) -> None:
    text = f"""# FP8 / KV 量化评测精简证据（2026-10-05）

此目录由 `tools/export_fp8_experiment_evidence.py` 从 E 盘原始结果只读导出。每条压缩 JSONL 仅保留题目 ID、评测索引、prompt/source record SHA256、生成文本、token 计数、finish reason 和错误标记；不包含 LongBench 输入 passage、完整 prompt、few-shot 文本或模型权重。配对分数只保留 item scores/flip。原始 E 盘数据与日志未改动；完整大日志以源路径和 SHA256 记录在 `source_log_index.json`，另摘录有限 backend/scale 行供查证。

## 结果计数摘要

- 单侧 FP8 QDQ GSM8K：`{json.dumps(qdq["gsm8k_rows_per_arm"], ensure_ascii=False)}`。配对准确率：`{json.dumps(qdq["gsm8k_pair_scores"], ensure_ascii=False)}`。
- 单侧 FP8 QDQ LongBench：每任务每臂条数见 `qdq-single-sided-fp8/summary.json`；八任务 × 50 条 × 3 臂。任务配对分数亦在该文件。
- xLLM 真实动态 scale FP8 GSM8K：`{json.dumps(native["rows_per_arm"], ensure_ascii=False)}`；逐题配对分数见其 `summary.json`。
- vLLM BF16/E4M3 GSM8K：`{json.dumps(vllm, ensure_ascii=False)}`；这是 64 题参考结果。vLLM 文件中的 `run_vllm_fp8_reference.py` 是复现源码，不是 worker 启动时冻结的 runtime snapshot；边界详见 vLLM 报告。

## 来源与复核

详细实验结论和配置见仓库 `docs/kv_cache_fp8_kv_ablation_20261005_zh.md`、`docs/xllm_fp8_gsm8k_20261005_zh.md`、`docs/vllm_fp8_reference_20261005_zh.md`、`docs/flashinfer-fp8-integration-20261005.md`。各实验原始 E 盘路径、文件 SHA256、导出 artifact SHA256、条数校验及导出脚本 hash 位于 `export_index.json`；`SHA256SUMS.txt` 对导出目录文件逐项校验（不含该 checksums 文件自身）。

QDQ 两 benchmark 共用冻结 runtime 源快照，原 tar 原样保存在 `qdq-single-sided-fp8/source/source_snapshot.tar.gz`；xLLM 真实 FP8 源码以目录形式及源 manifest 归档。vLLM reproduction source 有效性边界记录在本目录 `vllm-reference/metadata/run_config.json` 与对应报告。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _build_index(output: Path, source_roots: dict[str, Path], summaries: dict[str, Any]) -> None:
    log_index = _index_logs(list(source_roots.values()))
    _write_json(output / "source_log_index.json", log_index)
    source_index = _index_source_files(source_roots)
    _write_json(output / "source_file_index.json", source_index)
    artifact_hashes: dict[str, dict[str, Any]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"export_index.json", "SHA256SUMS.txt"}:
            rel = str(path.relative_to(output))
            artifact_hashes[rel] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    export_script = Path(__file__).resolve()
    index = {
        "created_utc": "2026-10-05",
        "source_roots": {key: str(value) for key, value in source_roots.items()},
        "source_log_index": "source_log_index.json",
        "source_file_index": "source_file_index.json",
        "export_script": str(export_script),
        "export_script_sha256": _sha256_file(export_script),
        "summaries": summaries,
        "artifacts": artifact_hashes,
        "notes": [
            "No model weights, full LongBench passages, complete prompts, or few-shot text are exported.",
            "E-drive originals are read-only inputs and were not changed.",
            "LongBench QDQ runtime source snapshot is shared with GSM8K and copied once.",
            "vLLM source file is a reproduction source, not a verified runtime snapshot.",
        ],
    }
    _write_json(output / "export_index.json", index)
    lines = [f"{entry['sha256']}  {name}" for name, entry in sorted(artifact_hashes.items())]
    lines.append(f"{_sha256_file(output / 'export_index.json')}  export_index.json")
    lines.append(f"{_sha256_file(output / 'source_log_index.json')}  source_log_index.json")
    lines.append(f"{_sha256_file(output / 'source_file_index.json')}  source_file_index.json")
    (output / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_REPO_ROOT / "results/model-fp8-evidence-20261005")
    parser.add_argument("--source-root", type=Path, default=_DATA_ROOT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence directory: {output}")
    sources = {
        "qdq_gsm8k": args.source_root / "fp8-kv-quality-ablation-gsm8k-20261005",
        "qdq_longbench": args.source_root / "fp8-kv-quality-ablation-longbench-20261005",
        "vllm_reference": args.source_root / "vllm-fp8-reference-20261005",
        "xllm_native": args.source_root / "xllm-fp8-gsm8k-20261005",
    }
    for source in sources.values():
        if not source.is_dir():
            raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        qdq = _copy_qdq_evidence(output, sources["qdq_gsm8k"], sources["qdq_longbench"])
        native = _copy_xllm_native(output, sources["xllm_native"])
        vllm = _copy_vllm_reference(output, sources["vllm_reference"])
        summaries = {"single_sided_qdq": qdq, "xllm_dynamic_fp8": native, "vllm_reference": vllm}
        _write_readme(output, qdq, native, vllm)
        _build_index(output, sources, summaries)
    except Exception:
        # Preserve partial evidence for diagnosis; never delete an existing user target.
        raise
    print(f"Evidence exported to {output}")


if __name__ == "__main__":
    main()
