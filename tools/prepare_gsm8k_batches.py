# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Freeze GSM8K test rows into ordered, auditable paired-evaluation batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.logger import logger

BATCH_SIZE = 50


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_batches(source_path: Path, output_dir: Path) -> Path:
    source_bytes = source_path.read_bytes()
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("GSM8K source test file is empty")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    batches: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(rows), BATCH_SIZE)):
        end = min(start + BATCH_SIZE, len(rows))
        filename = f"batch_{batch_index:04d}.jsonl"
        batch_rows = []
        for source_index in range(start, end):
            row = dict(rows[source_index])
            row["_source_index"] = source_index
            batch_rows.append(row)
        content = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in batch_rows
        ).encode("utf-8")
        (output_dir / filename).write_bytes(content)
        batches.append(
            {
                "batch_index": batch_index,
                "file": filename,
                "global_start": start,
                "global_end_exclusive": end,
                "count": end - start,
                "sha256": _sha256(content),
            }
        )

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "gsm8k test",
        "source_path": str(source_path),
        "source_sha256": _sha256(source_bytes),
        "source_count": len(rows),
        "source_indexing": "zero-based original JSONL line index; stored per row as _source_index",
        "ordering": "original source order; no shuffle or truncation",
        "batch_size": BATCH_SIZE,
        "batch_count": len(batches),
        "paired_arm_order": ["auto", "int8"],
        "paired_arm_protocol": "run both arms on identical rows in each batch; compare after each complete pair",
        "batches": batches,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Prepared %d GSM8K rows in %d batches under %s",
        len(rows),
        len(batches),
        output_dir,
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_batches(args.source, args.output_dir)


if __name__ == "__main__":
    main()
