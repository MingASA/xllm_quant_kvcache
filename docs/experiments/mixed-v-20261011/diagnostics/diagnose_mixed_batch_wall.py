# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Reuse existing mixed-V calibration cases with eager events, outside formal runs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/mingasa/projects/xllm_quant_kvcache')
sys.path.insert(0, str(ROOT))
formal_env = json.loads((ROOT/'results/flashinfer-govreport-seq256-20261011/formal-two-passes/environment.json').read_text())['runtime_environment']
for key, value in formal_env.items():
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
from tools import benchmark_flashinfer_mixed_v as calibration
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--batches', type=int, nargs='+', default=[8,16,24,29,32,34,38,48,64])
parser.add_argument('--contexts', type=int, nargs='+', default=[8192,12288])
parser.add_argument('--repetitions', type=int, default=5)
parser.add_argument('--fixed-split-pages', type=int)
args = parser.parse_args()
if args.output.exists():
    parser.error('Use a fresh output')
gate_path = ROOT/'results/flashinfer-mixed-v2-20261010/gates-final/correctness_gates.json'
gates = json.loads(gate_path.read_text())
if not all(gates.get(arm, {}).get('passed') for arm in calibration._ARMS):
    raise RuntimeError('Missing correctness gate')
for name, expected in gates['source_sha256'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Stale source gate: '+name)
plans = []
original_prepare = calibration.FlashInferMixedVBackend.prepare

def _prepare(self, metadata, **kwargs):
    original_plan = self._planner.plan
    if args.fixed_split_pages is not None:
        def forced_plan(*plan_args, **plan_kwargs):
            plan_kwargs['fixed_split_size'] = args.fixed_split_pages
            return original_plan(*plan_args, **plan_kwargs)
        self._planner.plan = forced_plan
    try:
        result = original_prepare(self, metadata, **kwargs)
    finally:
        self._planner.plan = original_plan
    plans.append({'mode':self._mode, 'plan_info':list(self._planner._plan_info)})
    return result

def _warm(call):
    for _ in range(30):
        call()
    torch.cuda.synchronize()
    return None

def _measure(unused, call):
    # No graph, no planner in timed interval; includes eager launch gaps.
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    before = time.perf_counter()
    start.record()
    for _ in range(20):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)*1000/20, (time.perf_counter()-before)*1e6/20

calibration.FlashInferMixedVBackend.prepare = _prepare
calibration._capture = _warm
calibration._measure = _measure
result = {'scope':'independent eager attention-only diagnostic; not formal service performance',
          'uniform_contexts':True, 'fixed_split_pages':args.fixed_split_pages, 'page_permutation':'existing case builder; identical across arms',
          'timing':'30 warm calls/arm, randomized arm order, eager CUDA interval over20 calls;5 repeats by default',
          'native_build':gates['native_build'], 'torch':torch.__version__,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases':[]}
args.output.parent.mkdir(parents=True,exist_ok=True)
for length in args.contexts:
    for batch in args.batches:
        plans.clear()
        case = calibration._case(batch, 1, length, args.repetitions, gates)
        case['plans'] = list(plans)
        case['gpu_snapshot'] = subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader'],text=True).strip()
        result['cases'].append(case)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'batch':batch,'context':length,'median':case['median']}),flush=True)
        gc.collect()
        torch.cuda.empty_cache()
