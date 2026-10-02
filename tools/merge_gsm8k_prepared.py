# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Merge ordered prepared GSM8K shards after validating their manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def merge_prepared_shards(source_dir: Path, output_path: Path) -> int:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    expected_start = 0
    for batch in manifest["batches"]:
        if batch["global_start"] != expected_start:
            raise ValueError(f"GSM8K shard order has a gap before index {expected_start}")
        content = (source_dir / batch["file"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != batch["sha256"]:
            raise ValueError(f"GSM8K shard hash mismatch: {batch['file']}")
        shard_rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        if len(shard_rows) != batch["count"] or batch["global_end_exclusive"] != expected_start + len(shard_rows):
            raise ValueError(f"GSM8K shard count mismatch: {batch['file']}")
        for offset, row in enumerate(shard_rows):
            if row.get("_source_index") != expected_start + offset:
                raise ValueError(f"GSM8K source index mismatch in {batch['file']} at offset {offset}")
        rows.extend(shard_rows)
        expected_start += len(shard_rows)
    if expected_start != manifest["source_count"] or expected_start != 1319:
        raise ValueError(f"Expected all 1,319 GSM8K source rows; validated {expected_start}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = merge_prepared_shards(args.source_dir, args.output)
    print(f"Validated and merged {count} GSM8K rows into {args.output}")


if __name__ == "__main__":
    main()
