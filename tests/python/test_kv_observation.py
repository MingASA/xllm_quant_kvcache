# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU math/lifecycle checks; these do not certify GPU Graph execution."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from xllm.python.attention.kv_observation import KVRangeObserver


def test_observations_cover_all_calls_original_layer_ids_and_both_phases() -> None:
    observer = KVRangeObserver([3, 7], 1, 2, torch.device("cpu"))
    # Invalid padding contains both huge and nonfinite values and must not leak.
    key = torch.tensor([[[float("nan"), float("inf")]], [[1.0, 10.0]], [[1e9, -1e9]]])
    value = torch.tensor([[[float("nan"), float("inf")]], [[2.0, 3.0]], [[1e9, -1e9]]])
    slots = torch.tensor([0, 128, -1])
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("Host read in observe")):
        observer.observe(3, True, key, value, slots)
        observer.observe(7, True, key, value, slots)
        key[1, 0, 1] = 100.0
        observer.observe(3, True, key, value, slots)
        observer.observe(7, False, key, value, slots)
    result = observer.snapshot()
    assert result["layer_ids"] == [3, 7]
    assert result["observed_absmax"] == {"key": 100.0, "value": 3.0}
    assert result["phase_calls_by_layer"] == [[2, 0], [1, 1]]
    assert result["FP8_E4M3_candidate_scales"]["key"] == 0.25
    assert result["activation_tensors_retained"] is False
    assert sum(row["finite_values"] for row in result["rows"]) == 16
    assert result["counter_storage_bytes"] == 2 * 2 * 2 * 1 * (4 + 8 + 8) + 2 * 2 * 8


def test_observer_reset_and_invalid_actual_values_cannot_certify_scales() -> None:
    observer = KVRangeObserver([63], 1, 2, torch.device("cpu"))
    key = torch.tensor([[[1.0, 2.0]]])
    observer.observe(63, False, key, key, torch.tensor([128]))
    observer.reset()
    with pytest.raises(ValueError, match="empty"):
        observer.snapshot()
    key[0, 0, 0] = float("inf")
    observer.observe(63, True, key, key, torch.tensor([128]))
    with pytest.raises(ValueError, match="Nonfinite"):
        observer.snapshot()


def test_observer_does_not_retain_an_input_autograd_graph() -> None:
    observer = KVRangeObserver([3], 1, 2, torch.device("cpu"))
    key = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    observer.observe(3, True, key, key, torch.tensor([128]))
    assert not observer._maxima.requires_grad
    with torch.no_grad():
        key.mul_(100)
    assert observer.snapshot()["observed_absmax"] == {"key": 2.0, "value": 2.0}
