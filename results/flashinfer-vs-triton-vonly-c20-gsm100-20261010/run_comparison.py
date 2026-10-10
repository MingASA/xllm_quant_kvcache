# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Run immutable 100-question GSM8K service comparison, C20, 3 repetitions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    saved = json.loads((ROOT / "results/flashinfer-mixed-v2-20261010/runtime_environment.json").read_text())
    env = dict(os.environ, **{key: value for key, value in saved.items() if value is not None})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for backend in ("triton", "flashinfer_mixed"):
        gates = (
            HERE / "triton_gates.json"
            if backend == "triton"
            else ROOT / "results/flashinfer-mixed-v2-20261010/gates-final/correctness_gates.json"
        )
        command = [
            sys.executable,
            "-B",
            str(ROOT / "tools/run_mixed_v_cache_experiment.py"),
            "run",
            "--output",
            str(HERE / backend),
            "--gates",
            str(gates),
            "--attention-backend",
            backend,
            "--arms",
            "bf16_int8",
            "bf16_int4",
            "--workloads",
            "gsm8k",
            "--no-joint-quality",
            "--concurrency",
            "20",
            "--repetitions",
            "3",
            "--replay-cycles",
            "1",
            "--warmup",
            "2",
            "--max-memory-utilization",
            "0.93",
            "--port",
            "18997",
        ]
        with (HERE / f"service_{backend}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
