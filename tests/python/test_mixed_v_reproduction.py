# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Architecture policy and immutable experiment transport regressions."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.eval_kv_cache_quality import _score_pair
from tools.prepare_mixed_v_reproduction import _prepare
from xllm.python.attention.flashinfer_mixed_build import _architecture_config


def test_architecture_rejects_unvalidated_devices() -> None:
    assert _architecture_config((12, 0), 102400, 101376) == ("120", [])
    assert _architecture_config((9, 0), 233472, 232448) == ("90", ["-DXLLM_MIXED_V_SM90=1"])
    for capability, shared_sm, shared_block in [
        ((8, 0), 167936, 166912),
        ((12, 0), 233472, 232448),
        ((9, 0), 102400, 101376),
        ((9, 0), 233472, 49152),
    ]:
        with pytest.raises(ValueError):
            _architecture_config(capability, shared_sm, shared_block)


def test_trace_transport_preserves_bytes_and_rejects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    row = {"id": "gsm8k/0", "prompt": "question", "prompt_sha256": hashlib.sha256(b"question").hexdigest()}
    blob = (json.dumps(row) + "\n").encode()
    trace = source / "trace_gsm8k.jsonl"
    trace.write_bytes(blob)
    manifest = {"traces": {"gsm8k": {"requests": 1, "sha256": hashlib.sha256(blob).hexdigest()}}}
    (source / "trace_manifest.json").write_text(json.dumps(manifest))
    _prepare(source, tmp_path / "copy")
    assert (tmp_path / "copy/trace_gsm8k.jsonl").read_bytes() == blob
    with pytest.raises(FileExistsError):
        _prepare(source, tmp_path / "copy")
    trace.write_bytes(blob + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        _prepare(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_govreport_uses_rouge_not_qa_f1() -> None:
    metrics = SimpleNamespace(rouge_score=lambda p, a: 0.75, qa_f1_score=lambda p, a: 0.25)
    scores, _ = _score_pair("gov_report", [{"answers": ["reference"]}], {"auto": ["summary"]}, metrics)
    assert scores == {"auto": 0.75}
    scores, _ = _score_pair("qasper", [{"answers": ["reference"]}], {"auto": ["answer"]}, metrics)
    assert scores == {"auto": 0.25}
