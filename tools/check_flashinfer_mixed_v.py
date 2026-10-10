# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Attest the native mixed-V backend using dense-reference and sanitizer tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger

_ARMS = ("auto", "bf16_int8", "bf16_fp8_e4m3", "bf16_int4")
_TEST = "tests/python/test_flashinfer_mixed_attention.py"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_results(path: Path) -> int:
    cases = list(ET.parse(path).iter("testcase"))
    if len(cases) < 36 or any(
        any(case.find(kind) is not None for kind in ("failure", "error", "skipped")) for case in cases
    ):
        raise ValueError(f"Incomplete or failing GPU correctness tests: {path}")
    for arm in _ARMS:
        if not any(f"[{arm}-" in case.attrib["name"] for case in cases):
            raise ValueError(f"Missing correctness tests for {arm}")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allocator-gates",
        type=Path,
        required=True,
        help="Prior tested allocator evidence; unchanged source/binary hashes are required",
    )
    parser.add_argument("--sanitizer", choices=("memcheck", "racecheck"), default="memcheck")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.allocator_gates.read_text())
    if not all(base.get(arm, {}).get("passed") for arm in _ARMS):
        raise ValueError("Allocator gates must cover all four arms")
    binary = ROOT / "build/cmake.cuda-x86/xllm/xllm"
    if _hash(binary) != base["binary_sha256"]:
        raise ValueError("Server binary changed since allocator correctness tests")
    # The previous backend entry-point is deliberately replaced. Codec, writer
    # and allocator are inherited only when their tested source is unchanged.
    inherited = {
        name: digest
        for name, digest in base["source_sha256"].items()
        if name != "xllm/python/model_executor/executor.py"
    }
    for name, digest in inherited.items():
        if _hash(ROOT / name) != digest:
            raise ValueError(f"Allocator/codec evidence is stale: {name}")
    xml = args.output / "correctness.xml"
    command = [sys.executable, "-B", "-m", "pytest", _TEST, "-q", f"--junitxml={xml.resolve()}"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    logger.info("Checking native mixed-V backend, all formats and prefill/decode")
    with (args.output / "correctness.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    count = _check_results(xml)
    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        raise ValueError("compute-sanitizer must be on PATH")
    sanitizer_xml = args.output / "sanitizer.xml"
    env["XLLM_MIXED_SANITIZER"] = "1"
    command = [
        sanitizer,
        "--tool",
        args.sanitizer,
        "--error-exitcode",
        "99",
        sys.executable,
        "-B",
        "-m",
        "pytest",
        _TEST,
        "-q",
        f"--junitxml={sanitizer_xml.resolve()}",
    ]
    with (args.output / "sanitizer.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    _check_results(sanitizer_xml)
    # Stock FlashInfer's own native-run symbol warning under CUDA 13 is tested
    # separately; sanitizer run exercises only our independent CUDA namespace.
    import torch

    from xllm.python.attention.flashinfer_mixed_build import load_library

    library, native = load_library(torch.device("cuda", torch.cuda.current_device()))
    paths = [
        ROOT / _TEST,
        ROOT / "tools/check_flashinfer_mixed_v.py",
        ROOT / "xllm/python/model_executor/executor.py",
        ROOT / "xllm/python/attention/flashinfer_mixed.py",
        ROOT / "xllm/python/attention/flashinfer_mixed_build.py",
        *[
            path
            for path in (ROOT / "xllm/python/kernels_cuda/flashinfer_mixed").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ],
    ]
    source_hashes = {**inherited, **{str(path.relative_to(ROOT)): _hash(path) for path in paths}}
    evidence = [
        {"path": str(args.output / "correctness.log"), "result": f"{count} GPU tests passed"},
        {"path": str(args.output / "sanitizer.log"), "result": f"{args.sanitizer}: zero errors"},
        {
            "path": str(args.allocator_gates),
            "sha256": _hash(args.allocator_gates),
            "result": "Inherited unchanged binary/allocator/codec correctness evidence",
        },
    ]
    gates = {arm: {"passed": True, "evidence": evidence} for arm in _ARMS}
    gates.update(
        attention_backend="flashinfer_mixed",
        native_build=native,
        binary_sha256=_hash(binary),
        source_sha256=source_hashes,
    )
    (args.output / "correctness_gates.json").write_text(json.dumps(gates, indent=2) + "\n")
    logger.info("All four arms attested: %s", args.output / "correctness_gates.json")


if __name__ == "__main__":
    main()
