# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Copy verified frozen traces to a fresh experiment directory on another host."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def _prepare(source: Path, output: Path, smoke_samples: int = 0) -> None:
    manifest_path = source / "trace_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    blobs = {}
    for name, entry in manifest["traces"].items():
        if name not in ("gsm8k", "longbench", "quality"):
            raise ValueError(f"Unknown trace name: {name}")
        blob = (source / f"trace_{name}.jsonl").read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source trace hash mismatch: {name}")
        rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
        if len(rows) != entry["requests"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Source trace count or IDs mismatch: {name}")
        for row in rows:
            if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
                raise ValueError(f"Prompt changed: {row['id']}")
        if smoke_samples:
            rows = rows[:smoke_samples]
            blob = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
            manifest["traces"][name] = {"requests": len(rows), "sha256": hashlib.sha256(blob).hexdigest()}
        blobs[name] = blob
    if smoke_samples:
        manifest["performance_scope"] = "smoke_only_not_quality_or_performance_evidence"
    output.mkdir(parents=True, exist_ok=False)
    for name, blob in blobs.items():
        (output / f"trace_{name}.jsonl").write_bytes(blob)
    (output / "trace_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output / "trace_import.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "smoke_samples": smoke_samples,
                "historical_results_copied": False,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("Verified frozen traces imported into %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-samples", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_samples < 0:
        parser.error("smoke-samples must be nonnegative")
    _prepare(args.source, args.output, args.smoke_samples)


if __name__ == "__main__":
    main()
