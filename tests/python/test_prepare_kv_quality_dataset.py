# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import pytest

from tools import prepare_kv_quality_dataset as prepare


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()

    def decode(self, tokens: list[str], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is False
        return " ".join(tokens)

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[str]]:
        del add_special_tokens
        return {"input_ids": self.encode(text)}


def test_truncation_keeps_balanced_edges_and_fits_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "CONTEXT_LIMIT", 16)
    monkeypatch.setattr(prepare, "SAFETY_MARGIN", 2)
    tokenizer = _WhitespaceTokenizer()
    row = {"context": " ".join(f"t{i}" for i in range(20)), "input": "question", "answers": ["x"]}

    prepared, audit = prepare._truncate_context(tokenizer, row, "{context} {input}", False, max_new_tokens=2)

    assert audit["truncated"] is True
    assert audit["prompt_tokens"] + 2 <= 14
    assert audit["head_tokens_kept"] == 6
    assert audit["tail_tokens_kept"] == 5
    assert prepared["context"].startswith("t0 t1 t2 t3 t4 t5\n\n")
    assert prepared["context"].endswith("t15 t16 t17 t18 t19")
    assert prepared["input"] == row["input"]
    assert prepared["answers"] == row["answers"]


def test_context_within_budget_is_not_modified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "CONTEXT_LIMIT", 16)
    monkeypatch.setattr(prepare, "SAFETY_MARGIN", 2)
    tokenizer = _WhitespaceTokenizer()
    row = {"context": "short context", "input": "question", "answers": ["answer"]}

    prepared, audit = prepare._truncate_context(tokenizer, row, "{context} {input}", False, max_new_tokens=2)

    assert prepared == row
    assert audit["truncated"] is False
    assert audit["head_tokens_kept"] == 2
    assert audit["tail_tokens_kept"] == 0


def test_empty_context_that_cannot_fit_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "CONTEXT_LIMIT", 16)
    monkeypatch.setattr(prepare, "SAFETY_MARGIN", 2)
    tokenizer = _WhitespaceTokenizer()
    row = {"context": "", "input": "question"}

    with pytest.raises(ValueError, match="cannot fit"):
        prepare._truncate_context(tokenizer, row, "{context} {input}", False, max_new_tokens=14)
