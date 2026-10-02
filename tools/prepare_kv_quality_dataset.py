# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Prepare fixed LongBench and GSM8K inputs with auditable context truncation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.logger import logger  # noqa: E402
from tools.eval_kv_cache_quality import LONG_BENCH_TASKS, _token_count  # noqa: E402

CONTEXT_LIMIT = 32768
SAFETY_MARGIN = 64
SAMPLE_COUNT = 50
SAMPLE_SEED = 17


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl_bytes(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _prompt(template: str, row: dict[str, Any]) -> str:
    return template.format(context=row["context"], input=row["input"])


def _count_prompt(tokenizer: Any, prompt: str, chat: bool) -> int:
    return _token_count(tokenizer, prompt, chat)


def _truncate_context(
    tokenizer: Any,
    row: dict[str, Any],
    template: str,
    chat: bool,
    max_new_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_context = row["context"]
    original_context_tokens = tokenizer.encode(original_context, add_special_tokens=False)
    round_trip_context = tokenizer.decode(original_context_tokens, skip_special_tokens=False)
    original_prompt_tokens = _count_prompt(tokenizer, _prompt(template, row), chat)
    original_context_sha256 = _sha256(original_context.encode("utf-8"))
    audit: dict[str, Any] = {
        "original_context_sha256": original_context_sha256,
        "original_context_tokens": len(original_context_tokens),
        "original_prompt_tokens": original_prompt_tokens,
        "prompt_tokens": original_prompt_tokens,
        "head_tokens_kept": len(original_context_tokens),
        "tail_tokens_kept": 0,
        "truncated": False,
        "decoded_head_sha256": original_context_sha256,
        "decoded_tail_sha256": _sha256(b""),
        "prepared_context_sha256": original_context_sha256,
        "tokenizer_roundtrip_sha256": _sha256(round_trip_context.encode("utf-8")),
        "tokenizer_roundtrip_matches_original": round_trip_context == original_context,
    }
    limit = CONTEXT_LIMIT - SAFETY_MARGIN
    if original_prompt_tokens + max_new_tokens <= limit:
        return dict(row), audit

    low, high = 0, len(original_context_tokens)
    best_row: dict[str, Any] | None = None
    best_audit: dict[str, Any] | None = None
    while low <= high:
        retained = (low + high) // 2
        head_count = (retained + 1) // 2
        tail_count = retained // 2
        head = tokenizer.decode(original_context_tokens[:head_count], skip_special_tokens=False)
        tail = tokenizer.decode(
            original_context_tokens[len(original_context_tokens) - tail_count :] if tail_count else [],
            skip_special_tokens=False,
        )
        candidate = dict(row)
        candidate["context"] = head + "\n\n" + tail
        prompt_tokens = _count_prompt(tokenizer, _prompt(template, candidate), chat)
        if prompt_tokens + max_new_tokens <= limit:
            best_row = candidate
            best_audit = {
                **audit,
                "prompt_tokens": prompt_tokens,
                "head_tokens_kept": head_count,
                "tail_tokens_kept": tail_count,
                "truncated": True,
                "decoded_head_sha256": _sha256(head.encode("utf-8")),
                "decoded_tail_sha256": _sha256(tail.encode("utf-8")),
                "prepared_context_sha256": _sha256(candidate["context"].encode("utf-8")),
                "tokenizer_roundtrip_sha256": audit["tokenizer_roundtrip_sha256"],
                "tokenizer_roundtrip_matches_original": audit["tokenizer_roundtrip_matches_original"],
            }
            low = retained + 1
        else:
            high = retained - 1

    if best_row is None or best_audit is None:
        raise ValueError("Prompt cannot fit with empty retained context and required safety margin")
    if best_audit["prompt_tokens"] + max_new_tokens > limit:
        raise AssertionError("Truncation failed to respect the context budget")
    return best_row, best_audit


def prepare(args: argparse.Namespace) -> Path:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    config_dir = args.longbench_repo / "LongBench" / "config"
    prompt_templates = json.loads((config_dir / "dataset2prompt.json").read_text(encoding="utf-8"))
    max_lengths = json.loads((config_dir / "dataset2maxlen.json").read_text(encoding="utf-8"))
    archive_bytes = args.longbench_zip.read_bytes()
    archive_sha256 = _sha256(archive_bytes)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling": {
            "method": "uniform random sample without replacement, sorted by source row index",
            "count_per_task": SAMPLE_COUNT,
            "seed_per_task": SAMPLE_SEED,
        },
        "context_budget": {
            "max_model_tokens": CONTEXT_LIMIT,
            "safety_margin_tokens": SAFETY_MARGIN,
            "condition": "prompt_tokens + official max_new_tokens <= 32768 - 64",
            "truncate_field": "context only",
            "retention": "balanced head/tail token slices separated by two newlines",
        },
        "longbench": {
            "archive_path": str(args.longbench_zip),
            "archive_sha256": archive_sha256,
            "tasks": {},
        },
        "gsm8k": {},
    }

    with zipfile.ZipFile(args.longbench_zip) as archive:
        for task in LONG_BENCH_TASKS:
            source_name = f"data/{task}.jsonl"
            raw_bytes = archive.read(source_name)
            rows = _jsonl_bytes(raw_bytes)
            if len(rows) < SAMPLE_COUNT:
                raise ValueError(f"{task} has only {len(rows)} rows; need {SAMPLE_COUNT}")
            indices = sorted(random.Random(SAMPLE_SEED).sample(range(len(rows)), SAMPLE_COUNT))
            template = prompt_templates[task]
            max_new_tokens = int(max_lengths[task])
            chat = task not in {"lcc", "repobench-p"}
            prepared_rows = []
            audits = []
            for index in indices:
                original = rows[index]
                prepared, audit = _truncate_context(tokenizer, original, template, chat, max_new_tokens)
                prepared_rows.append(prepared)
                audits.append(
                    {
                        "source_row_index": index,
                        "source_id": original.get("_id", original.get("id", str(index))),
                        "raw_file_sha256": _sha256(raw_bytes),
                        **audit,
                    }
                )
            data_path = output / f"{task}.jsonl"
            audit_path = output / f"{task}.audit.jsonl"
            data_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared_rows),
                encoding="utf-8",
            )
            audit_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audits),
                encoding="utf-8",
            )
            truncated = sum(item["truncated"] for item in audits)
            manifest["longbench"]["tasks"][task] = {
                "source_member": source_name,
                "raw_file_sha256": _sha256(raw_bytes),
                "source_row_count": len(rows),
                "selected_row_indices": indices,
                "max_new_tokens": max_new_tokens,
                "chat_template_applied": chat,
                "selected_count": len(prepared_rows),
                "truncated_count": truncated,
                "data_file": data_path.name,
                "data_file_sha256": _sha256(data_path.read_bytes()),
                "audit_file": audit_path.name,
                "audit_file_sha256": _sha256(audit_path.read_bytes()),
            }
            logger.info("%s: %d/%d selected rows truncated", task, truncated, len(rows))

    gsm_source = args.gsm8k_test
    gsm_bytes = gsm_source.read_bytes()
    gsm_rows = _jsonl_bytes(gsm_bytes)
    if len(gsm_rows) != 1319:
        logger.warning("Expected GSM8K test set of 1319 rows, found %d", len(gsm_rows))
    gsm_output = output / "gsm8k_test.jsonl"
    gsm_output.write_bytes(gsm_bytes)
    shots_path = args.gsm8k_shots
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    if len(shots) != 5:
        raise ValueError(f"Expected exactly 5 GSM8K few-shot examples, found {len(shots)}")
    manifest["gsm8k"] = {
        "source_path": str(gsm_source),
        "source_sha256": _sha256(gsm_bytes),
        "test_count": len(gsm_rows),
        "prepared_file": gsm_output.name,
        "prepared_sha256": _sha256(gsm_output.read_bytes()),
        "shots_path": str(shots_path),
        "shots_sha256": _sha256(shots_path.read_bytes()),
        "shots_count": len(shots),
        "truncation": False,
    }

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote prepared data and audit manifest to %s", output)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longbench-zip", type=Path, required=True)
    parser.add_argument("--longbench-repo", type=Path, required=True)
    parser.add_argument("--gsm8k-test", type=Path, required=True)
    parser.add_argument("--gsm8k-shots", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
