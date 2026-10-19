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
    if len(cases) < 108 or any(
        any(case.find(kind) is not None for kind in ("failure", "error", "skipped")) for case in cases
    ):
        raise ValueError(f"Incomplete or failing GPU correctness tests: {path}")
    for arm in _ARMS:
        if not any(f"[{arm}-" in case.attrib["name"] for case in cases):
            raise ValueError(f"Missing correctness tests for {arm}")
    for q, kv in ((12, 2), (28, 4), (40, 8)):
        for arm in _ARMS:
            if not any(f"[{arm}-q{q}-kv{kv}-" in case.attrib["name"] for case in cases):
                raise ValueError(f"Missing model-shape correctness tests: {arm}, {q}/{kv}")
    return len(cases)


def _fresh_allocator_gates(output: Path, binary: Path) -> dict:
    """Revalidate rebuilt allocators/codecs on the current GPU, not a prior host."""
    import torch

    if not torch.cuda.is_available():
        raise ValueError("Fresh allocator gates require a CUDA device")
    evidence = []
    tests = binary.parents[1] / "tests/core/framework/kv_cache"
    for name, minimum in (("kv_cache_estimation_test", 25), ("kv_cache_test", 26)):
        xml = (output / f"{name}.xml").resolve()
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(
                [str(tests / name), f"--gtest_output=xml:{xml}"], check=True, stdout=log, stderr=subprocess.STDOUT
            )
        cases = list(ET.parse(xml).iter("testcase"))
        passed = 0
        for case in cases:
            if case.find("failure") is not None or case.find("error") is not None:
                raise ValueError(f"Allocator test failed: {name}")
            skipped = case.find("skipped")
            if skipped is not None:
                if "MLU" not in ET.tostring(skipped, encoding="unicode"):
                    raise ValueError(f"Unexpected skipped allocator test: {name}")
            elif case.get("status") == "run":
                passed += 1
        if passed < minimum:
            raise ValueError(f"Incomplete allocator tests: {name}: {passed}")
        evidence.append({"path": str(xml), "result": f"{passed} passed", "sha256": _hash(xml)})
    xml = (output / "codec.xml").resolve()
    with (output / "codec.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "tests/python/test_quantized_attention.py",
                "-q",
                f"--junitxml={xml}",
            ],
            cwd=ROOT,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    cases = list(ET.parse(xml).iter("testcase"))
    if len(cases) < 215 or any(
        any(case.find(kind) is not None for kind in ("failure", "error", "skipped")) for case in cases
    ):
        raise ValueError("Fresh CUDA codec/writer tests failed or skipped")
    evidence.append({"path": str(xml), "result": f"{len(cases)} passed", "sha256": _hash(xml)})
    paths = [
        ROOT / "xllm/python/attention/quantized.py",
        ROOT / "xllm/python/attention/quantized_triton.py",
        *[p for p in (ROOT / "xllm/core/framework/kv_cache").rglob("*") if p.is_file()],
    ]
    base = {arm: {"passed": True, "evidence": evidence} for arm in _ARMS}
    base.update(
        binary_sha256=_hash(binary),
        device={"name": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())},
        source_sha256={str(p.relative_to(ROOT)): _hash(p) for p in paths},
    )
    (output / "allocator_gates.json").write_text(json.dumps(base, indent=2) + "\n")
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allocator-gates",
        type=Path,
        help="Optional prior allocator evidence from the same GPU architecture; otherwise run fresh gates",
    )
    parser.add_argument("--sanitizer", choices=("memcheck", "racecheck"), default="memcheck")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = Path(os.environ.get("XLLM_SERVER_BINARY", str(ROOT / "build/cmake.cuda-x86/xllm/xllm"))).resolve()
    base = (
        json.loads(args.allocator_gates.read_text())
        if args.allocator_gates
        else _fresh_allocator_gates(args.output, binary)
    )
    import torch

    capability = list(torch.cuda.get_device_capability())
    if base.get("device", {}).get("capability", [12, 0]) != capability:
        raise ValueError("Allocator gates belong to another GPU architecture; rerun fresh gates")
    if not all(base.get(arm, {}).get("passed") for arm in _ARMS):
        raise ValueError("Allocator gates must cover all four arms")
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
            "path": str(args.allocator_gates or args.output / "allocator_gates.json"),
            "sha256": _hash(args.allocator_gates or args.output / "allocator_gates.json"),
            "result": "Inherited unchanged binary/allocator/codec correctness evidence",
        },
    ]
    gates = {arm: {"passed": True, "evidence": evidence} for arm in _ARMS}
    gates.update(
        attention_backend="flashinfer_mixed",
        supported_shapes=[[12, 2, 128], [28, 4, 128], [40, 8, 128]],
        native_build=native,
        binary_sha256=_hash(binary),
        source_sha256=source_hashes,
    )
    (args.output / "correctness_gates.json").write_text(json.dumps(gates, indent=2) + "\n")
    logger.info("All four arms attested: %s", args.output / "correctness_gates.json")


if __name__ == "__main__":
    main()
