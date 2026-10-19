# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Reproduce Qwen2.5 quality ablations and frozen mixed-V service workloads."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.qwen25_experiment_model import checkpoint_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-source", type=Path, required=True)
    parser.add_argument("--govreport-source", type=Path)
    parser.add_argument("--gates", type=Path)
    parser.add_argument("--stage", choices=("quality", "performance", "all"), default="all")
    args = parser.parse_args()
    if args.stage != "quality" and (args.gates is None or not args.gates.is_file()):
        parser.error("Performance requires fresh target-device --gates")
    provenance = checkpoint_provenance(args.model.resolve())
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = args.output / "model_provenance.json"
    identity.write_text(json.dumps(provenance, indent=2) + "\n")
    env = {
        **os.environ,
        "XLLM_MODEL_PATH": str(args.model.resolve()),
        "XLLM_MODEL_ID": args.model.resolve().name,
        "XLLM_MODEL_PROVENANCE": str(identity),
    }
    commands = []

    def run(script: str, *options: str | Path) -> None:
        command = [sys.executable, "-B", str(ROOT / "tools" / script), *map(str, options)]
        commands.append(command)
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    if args.stage != "performance":
        for cycle in range(2):
            run("run_kv_fp8_quality_ablation.py", "all", "--output-root", args.output / f"quality-cycle{cycle}")
    if args.stage != "quality":
        jobs = [
            ("gsm8k", args.trace_source, "gsm8k", [128, 160, 208, 256], True),
            ("longbench", args.trace_source, "longbench", [8, 16, 32, 64, 128], False),
        ]
        if args.govreport_source:
            jobs.append(("govreport", args.govreport_source, "longbench", [8, 16, 32, 64, 128], True))
        for name, source, workload, concurrency, defaults in jobs:
            destination = args.output / name
            run(
                "prepare_mixed_v_reproduction.py",
                "--source",
                source.resolve(),
                "--output",
                destination,
                "--model",
                args.model.resolve(),
                "--model-provenance",
                identity,
            )
            options = [
                "run",
                "--output",
                destination,
                "--gates",
                args.gates.resolve(),
                "--attention-backend",
                "flashinfer_mixed",
                "--workloads",
                workload,
                "--concurrency",
                *map(str, concurrency),
                "--repetitions",
                "1",
                "--replay-cycles",
                "2",
                "--max-seqs-per-batch",
                "256" if defaults else "128",
                "--max-concurrent-requests",
                "512" if defaults else "200",
                "--max-memory-utilization",
                "0.93",
                "--joint-quality",
            ]
            if defaults:
                options.append("--use-default-token-budget")
            run("run_mixed_v_cache_experiment.py", *options)
            run("report_mixed_v_reproduction.py", "--output", destination)


if __name__ == "__main__":
    main()
