# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Untimed independent BF16 Q/K/V clipping diagnostics; never formal serving.

Only statistics persist. Reset/export reuse the existing quiescent filesystem
protocol. Device reductions are Graph-compatible, with no host reads in forward.
The BF16 attention output is unaffected by the hypothetical FP8 conversions.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import threading
import time
from itertools import accumulate
from typing import TYPE_CHECKING

import torch

from scripts.logger import logger
from xllm.python.attention.backend import AttentionMetadata, LayerCache
from xllm.python.attention.flashinfer import FlashInferBackend

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


METHOD = "independent_BF16_QKV_FP8_clipping_error_v1"
_RECIPES = (("key", .125, 4), ("value_current", .0625, 4),
            ("value_candidate", .5, 4), ("query", 1., 24))


def _publish(path: Path, value: dict) -> str:
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(data) > 2**20:
        raise ValueError("Numeric diagnostic export exceeded one MiB")
    prepared = path.with_name("." + path.name + ".prepared")
    with prepared.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(prepared, path, follow_symlinks=False)
    return hashlib.sha256(data).hexdigest()


class NumericFlashInferBackend(FlashInferBackend):
    """Explicit diagnostic-only selection; production branches omit this class."""

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        super().bind_kv_caches(kv_caches)
        config = Path(os.environ["XLLM_FP8_RANGE_DIAGNOSTICS_CONFIG"])
        if config.resolve(strict=True) != config or config.stat().st_uid != os.getuid() or config.stat().st_size > 65536:
            raise ValueError("Numeric diagnostics require an owned bounded config")
        self._config = json.loads(config.read_text())
        output = Path(self._config["output_dir"])
        if (self._config.get("method") != METHOD or output.resolve(strict=True) != output
                or output.stat().st_uid != os.getuid() or self.dtype != torch.bfloat16
                or (self.num_heads, self.num_kv_heads, self.head_dim) != (24, 4, 256)):
            raise ValueError("Numeric diagnostics require owned output and BF16 Q24/KV4/D256")
        self._layer_ids = tuple(i for i, cache in enumerate(kv_caches) if cache.key is not None)
        if self._layer_ids != tuple(range(3, 64, 4)):
            raise ValueError("Numeric diagnostics require the frozen Qwen hybrid full-attention layers")
        device = self._decode_workspace.device
        self._stats = {}
        for name, _, heads in _RECIPES:
            shape = (len(self._layer_ids), 2, heads)
            self._stats[name] = (
                torch.zeros((*shape, 4), dtype=torch.int64, device=device),
                torch.zeros((*shape, 2), dtype=torch.float64, device=device),
                torch.zeros(shape, dtype=torch.float32, device=device),
            )
        self._calls = torch.zeros((len(self._layer_ids), 2), dtype=torch.int64, device=device)
        self._step_counts = {"prefill_plan_calls":0, "mixed_plan_calls":0,
                             "decode_rows_in_mixed":0, "prefill_tokens_in_mixed":0,
                             "decode_plan_calls":0, "decode_graph_plan_calls":0}
        self._diagnostic_tensors = (*[t for group in self._stats.values() for t in group], self._calls)
        logger.info("FP8 numerical diagnostics enabled: method=%s; untimed BF16 only", METHOD)
        threading.Thread(target=self._control, args=(output,), daemon=True, name="FP8-quiescent-diagnostics").start()

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        super().prepare(metadata, graph_mode=graph_mode)
        counts = getattr(self, "_step_counts", None)
        if counts is None:
            return
        if metadata.is_prefill or metadata.is_chunked_prefill:
            counts["prefill_plan_calls"] += 1
            ends = metadata.q_cu_seq_lens_host_values
            if not ends:
                host = getattr(metadata, "q_seq_lens_host", None)
                if host is not None:
                    if host.device.type != "cpu":
                        raise ValueError("Numerical diagnostics cannot read query lengths from device")
                    ends = host.tolist()
                    if len(ends) == metadata.paged_kv_last_page_len.numel():
                        ends = list(accumulate(ends, initial=0))
            if ends:
                if len(ends) == metadata.paged_kv_last_page_len.numel():
                    ends = [0, *ends]
                lengths = [b-a for a,b in zip(ends, ends[1:])]
                decodes = sum(length == 1 for length in lengths)
                prefills = sum(length for length in lengths if length > 1)
                if decodes and prefills:
                    counts["mixed_plan_calls"] += 1
                    counts["decode_rows_in_mixed"] += decodes
                    counts["prefill_tokens_in_mixed"] += prefills
        else:
            counts["decode_plan_calls"] += 1
            counts["decode_graph_plan_calls"] += int(graph_mode)

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        self._observe(q, k, v, layer.layer_id)
        return super().execute(q, k, v, layer)

    def _observe(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int) -> None:
        metadata = self._metadata
        if metadata is None:
            raise RuntimeError("Numeric diagnostic prepare() was not called")
        row, phase = (layer_id - 3) // 4, int(not (metadata.is_prefill or metadata.is_chunked_prefill))
        valid = (metadata.slot_mapping > 0)[:, None, None]
        for (name, scale, heads), source in zip(_RECIPES, (k, v, v, q)):
            counts, errors, maxima = self._stats[name]
            values = source.view(-1, heads, self.head_dim).detach().float()
            finite = torch.isfinite(values)
            selected = valid & finite
            safe = torch.where(selected, values, 0)
            quantized = (safe / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
            counts[row, phase, :, 0].add_(selected.sum(dim=(0, 2)))
            counts[row, phase, :, 1].add_((selected & (safe.abs() > 448 * scale)).sum(dim=(0, 2)))
            counts[row, phase, :, 2].add_((selected & (safe != 0) & (quantized == 0)).sum(dim=(0, 2)))
            counts[row, phase, :, 3].add_((valid & ~finite).sum(dim=(0, 2)))
            errors[row, phase, :, 0].add_((safe - quantized).square().sum(dim=(0, 2), dtype=torch.float64))
            errors[row, phase, :, 1].add_(safe.square().sum(dim=(0, 2), dtype=torch.float64))
            maxima[row, phase].copy_(torch.maximum(maxima[row, phase], safe.abs().amax(dim=(0, 2))))
        self._calls[row, phase].add_(valid.any().to(torch.int64))

    def _snapshot(self) -> dict:
        torch.cuda.synchronize(self._decode_workspace.device)
        rows = []
        for name, scale, heads in _RECIPES:
            counts, errors, maxima = (t.cpu().tolist() for t in self._stats[name])
            for i, layer in enumerate(self._layer_ids):
                for phase in range(2):
                    for head in range(heads):
                        rows.append({"recipe":name, "scale":scale, "layer_id":layer,
                                     "phase":"prefill" if phase == 0 else "decode", "head":head,
                                     "finite":counts[i][phase][head][0], "clipped":counts[i][phase][head][1],
                                     "rounded_to_zero":counts[i][phase][head][2], "nonfinite":counts[i][phase][head][3],
                                     "squared_error":errors[i][phase][head][0],
                                     "squared_signal":errors[i][phase][head][1], "absmax":maxima[i][phase][head]})
        return {"method":METHOD, "rows":rows, "phase_calls":self._calls.cpu().tolist(),
                "host_plan_counts":dict(self._step_counts), "graph_plan_counts_are_not_actual_replay_proof":True,
                "persistent_counter_bytes":sum(t.numel()*t.element_size() for t in self._diagnostic_tensors),
                "activation_tensors_retained":False, "host_reads_in_forward":False,
                "BF16_attention_outputs_transformed":False, "query_decode_FP8_error_is_hypothetical":True,
                "performance_claim":False, "quality_acceptance_claim":False}

    def _control(self, output: Path) -> None:
        try:
            for action in ("reset", "export"):
                command = output / ("numeric_" + action + ".json")
                while not command.exists():
                    time.sleep(.2)
                if (command.resolve(strict=True) != command or command.stat().st_uid != os.getuid()
                        or command.stat().st_size > 65536
                        or json.loads(command.read_text()) != {"action":action,"request_sha256":self._config["request_sha256"]}):
                    raise ValueError("Numeric diagnostic command identity mismatch")
                if action == "reset":
                    snapshot_sha256 = None
                    torch.cuda.synchronize(self._decode_workspace.device)
                    for tensor in self._diagnostic_tensors:
                        tensor.zero_()
                    self._step_counts = {key:0 for key in self._step_counts}
                    torch.cuda.synchronize(self._decode_workspace.device)
                else:
                    snapshot_sha256 = _publish(output/"numeric_snapshot.json", {**self._snapshot(),
                        "run_id":self._config["run_id"], "request_sha256":self._config["request_sha256"]})
                _publish(output/("numeric_"+action+"_ack.json"),
                         {"action":action,"request_sha256":self._config["request_sha256"],
                          "snapshot_sha256":snapshot_sha256})
        except (OSError, ValueError, RuntimeError) as error:
            _publish(output/"numeric_ERROR.json", {"error":type(error).__name__+": "+str(error)})
            logger.exception("Independent numerical diagnostic failed")
            raise
