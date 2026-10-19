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


def _prepare(
    source: Path, output: Path, smoke_samples: int = 0, model: Path | None = None, model_provenance: Path | None = None
) -> None:
    manifest_path = source / "trace_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tokenizer = None
    if model is not None:
        from transformers import AutoTokenizer

        from tools.qwen25_experiment_model import checkpoint_provenance, read_profile, sha256_file

        profile = read_profile(model)
        provenance = json.loads(model_provenance.read_text()) if model_provenance else checkpoint_provenance(model)
        if provenance["metadata_sha256"]["config.json"] != sha256_file(model / "config.json"):
            raise ValueError("Checkpoint provenance config mismatch")
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        manifest["source_model_provenance"] = manifest.get("model_provenance")
        manifest["model_provenance"] = provenance
        manifest["checkpoint"] = str(model.resolve())
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
        if tokenizer is not None:
            from tools.eval_kv_cache_quality import _token_count

            for row in rows:
                row["prompt_tokens"] = _token_count(tokenizer, row["prompt"], row["chat"])
                if row["prompt_tokens"] + row["max_tokens"] > min(32768, profile["max_position_embeddings"]):
                    raise ValueError(f"Target model context exceeded: {row['id']}; no truncation")
        if smoke_samples:
            rows = rows[:smoke_samples]
        if smoke_samples or tokenizer is not None:
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
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-provenance", type=Path)
    args = parser.parse_args()
    if args.smoke_samples < 0:
        parser.error("smoke-samples must be nonnegative")
    _prepare(args.source, args.output, args.smoke_samples, args.model, args.model_provenance)


if __name__ == "__main__":
    main()
