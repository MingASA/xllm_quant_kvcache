# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Build a separate, pinned FlashInfer extension without modifying FlashInfer."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import flashinfer
import torch

from scripts.logger import logger

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "kernels_cuda/flashinfer_mixed"


def load_library(device: torch.device) -> tuple[ctypes.CDLL, dict]:
    """Compile once at backend startup; every artifact has a source-addressed path."""
    manifest = json.loads((SOURCE_ROOT / "upstream.json").read_text())
    if flashinfer.__version__ != manifest["flashinfer_version"]:
        raise ValueError("Mixed-V FlashInfer requires the pinned version in upstream.json")
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (12, 0) or properties.shared_memory_per_multiprocessor != 102400:
        raise ValueError("Mixed-V FA2 specialization requires SM120 with 100 KiB shared memory per SM")
    data = Path(flashinfer.__file__).resolve().parent / "data"
    include = data / "include"
    for name, expected in manifest["upstream_sha256"].items():
        actual = hashlib.sha256((include / "flashinfer/attention" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Installed FlashInfer source differs from the pinned source: {name}")
    nvcc = os.environ.get("XLLM_FLASHINFER_MIXED_NVCC") or shutil.which("nvcc")
    if nvcc is None:
        raise ValueError("nvcc must be on PATH or set XLLM_FLASHINFER_MIXED_NVCC")
    compiler = subprocess.check_output([nvcc, "--version"], text=True)
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(SOURCE_ROOT.iterdir())
        if path.is_file()
    }
    source_hashes["flashinfer_mixed_build.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Include all upstream headers, because the modified headers depend on them.
    upstream_headers = {
        str(path.relative_to(include)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(include.rglob("*"))
        if path.is_file()
    }
    cutlass_headers = {
        str(path.relative_to(data / "cutlass/include")): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((data / "cutlass/include").rglob("*"))
        if path.is_file()
    }
    provenance = dict(
        cutlass_headers=cutlass_headers,
        source_sha256=source_hashes,
        upstream_headers=upstream_headers,
        compiler=compiler,
        architecture="sm_120",
        flags=["-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC"],
    )
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    build_root = (
        Path(os.environ.get("XLLM_FLASHINFER_MIXED_BUILD_DIR", str(Path.home() / ".cache/xllm/flashinfer_mixed"))) / key
    )
    build_root.mkdir(parents=True, exist_ok=True)
    artifact = build_root / "mixed_prefill.so"
    with (build_root / "build.lock").open("a") as build_lock:
        fcntl.flock(build_lock, fcntl.LOCK_EX)
        if not artifact.exists():
            # Upstream headers use relative includes; copy them into the private build.
            shutil.copytree(include, build_root / "include", dirs_exist_ok=True)
            overlay = build_root / "include/flashinfer/attention"
            overlay.mkdir(parents=True, exist_ok=True)
            for name in ("decode.cuh", "prefill.cuh"):
                shutil.copyfile(SOURCE_ROOT / name, overlay / name)
            temporary = build_root / f"mixed_prefill.{os.getpid()}.so"
            command = [
                nvcc,
                *provenance["flags"],
                "-gencode",
                "arch=compute_120,code=sm_120",
                f"-I{build_root / 'include'}",
                f"-I{include}",
                f"-I{data / 'cutlass/include'}",
                str(SOURCE_ROOT / "mixed_prefill.cu"),
                "-o",
                str(temporary),
            ]
            logger.info("Building independent mixed-V FlashInfer extension: %s", build_root)
            with (build_root / f"build.{os.getpid()}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            temporary.replace(artifact)
    provenance.update(
        build_key=key,
        artifact=str(artifact),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        flashinfer_version=flashinfer.__version__,
    )
    (build_root / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return ctypes.CDLL(str(artifact)), provenance
