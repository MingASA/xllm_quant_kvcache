# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Run the prepared full suite; --smoke only validates runner/backend integration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    saved = json.loads((HERE / "runtime_environment.json").read_text())
    env = dict(os.environ, **{key: value for key, value in saved.items() if value is not None})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-B",
        str(ROOT / "tools/run_mixed_v_cache_experiment.py"),
        "run",
        "--output",
        str(HERE / ("runner-smoke" if args.smoke else "full-suite")),
        "--gates",
        str(HERE / "gates-final/correctness_gates.json"),
        "--attention-backend",
        "flashinfer_mixed",
        "--max-memory-utilization",
        "0.93",
        "--port",
        "18996",
    ]
    if args.smoke:
        command += [
            "--arms",
            "auto",
            "--workloads",
            "gsm8k",
            "--no-joint-quality",
            "--concurrency",
            "8",
            "--repetitions",
            "1",
            "--replay-cycles",
            "1",
            "--warmup",
            "2",
        ]
    else:
        command += [
            "--arms",
            "auto",
            "bf16_int8",
            "bf16_fp8_e4m3",
            "bf16_int4",
            "--workloads",
            "gsm8k",
            "longbench",
            "--joint-quality",
            "--concurrency",
            "8",
            "16",
            "32",
            "64",
            "128",
            "--repetitions",
            "3",
            "--replay-cycles",
            "2",
            "--warmup",
            "2",
        ]
    os.chdir(ROOT)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
