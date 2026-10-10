# BF16 K + compressed V experiment

Branch: `feat/mixed-v-kv-cache`. Scope: ordinary CUDA MHA/GQA in the Python eager executor. Native CUDA attention and CUDA Graph are outside this experiment.

Current protocol (updated 2026-10-10): jointly measure performance and quality from full GSM8K (1319) and existing LongBench (400) traces, replayed twice per run. Scan concurrency 8, 16, 32, 64, 128 with three repetitions after warmup. The earlier 64-sample/separate-quality suite is superseded and retained independently.

The cache modes are `auto`, `bf16_int8`, `bf16_fp8_e4m3`, and `bf16_int4`. Mixed modes require BF16 model weights/activations. The existing codec supplies dynamic symmetric FP32 scales per token/head. K has its original BF16 layout and no scale; V owns INT8 bytes, FP8 E4M3 encoded bytes, or signed packed INT4 nibbles. Attention dequantizes V inside the paged kernel, without a persistent or temporary full BF16 V cache. Block copies include V scales and preserve each tensor's dtype and stride. Existing uniform quantized modes remain available.

For H KV heads and logical head dimension D, the bytes/token/layer are:

| Mode | Formula | Qwen2.5-1.5B (H=2, D=128) | Relative to BF16 |
|---|---|---:|---:|
| BF16 | 4HD | 1024 | 1.000000 |
| BF16 K + INT8 V | H(3D + 4) | 776 | 0.7578125 |
| BF16 K + FP8 V | H(3D + 4) | 776 | 0.7578125 |
| BF16 K + INT4 V | H(2D + ceil(D/2) + 4) | 648 | 0.6328125 |

These formulas include the V scale and INT4 padding. The server records actual per-layer tensor shapes/dtypes/nbytes in `KV_ALLOCATION`, which the runner saves as `allocation.json`. At a fixed VRAM budget the predicted token-capacity multipliers are 1.31959 and 1.58025 before block rounding. Allocation measurements, rather than these ratios, determine the reported capacities.

## Reproduction

Use the CUDA-capable runtime `/home/mingasa/venvs/vllm/bin/python`. The `xllm` venv on this machine supplies clang-format 20.1.6 and build tools but lacks torch. Source the existing `build/cmake.cuda-x86/resume-env.sh` for the server's native dependencies, and add the repository root to `PYTHONPATH` when running scripts.

```bash
source build/cmake.cuda-x86/resume-env.sh
export PATH="/usr/lib/wsl/lib:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
/home/mingasa/venvs/vllm/bin/python tools/run_mixed_v_cache_experiment.py prepare \
  --output results/mixed-v-cache-20261010/joint-suite --performance-samples 0
/home/mingasa/venvs/vllm/bin/python tools/run_mixed_v_cache_experiment.py run \
  --output results/mixed-v-cache-20261010/joint-suite \
  --gates results/mixed-v-cache-20261009/correctness_gates.json \
  --arms auto bf16_int8 bf16_fp8_e4m3 bf16_int4 \
  --workloads gsm8k longbench --joint-quality \
  --concurrency 8 16 32 64 128 --repetitions 3 --replay-cycles 2
/home/mingasa/venvs/vllm/bin/python tools/report_mixed_v_cache_experiment.py \
  --output results/mixed-v-cache-20261010/joint-suite
```

`prepare` freezes and hashes the trace and checks every prompt against the checkpoint's context limit. Existing GSM8K five-shot prompts, the full 1319 test questions, and the fixed LongBench 8-task n50 protocol are reused. The default `--performance-samples 0` selects all rows; joint scoring rejects a subset. Each timed GSM8K run contains 2638 requests and each LongBench run contains 800 requests. These same responses supply quality scores, with no extra independent quality replay. Benchmark generation limits and EOS are preserved; no fixed token length, output padding, or truncation is introduced. The LongBench trace interleaves tasks in their existing fixed order. `trace_quality.jsonl` remains an immutable reference of all 1719 unique rows, not an additional requested workload.

The runner checks correctness gates before starting each arm. After API readiness it sends one unmeasured startup probe, because the Python backend binds and reports actual cache tensors on its first forward. The four-format GSM8K/LongBench c1/c4 service smoke has already passed; formal runs proceed BF16, V-INT8, V-FP8, V-INT4. Each group warms up at its tested concurrency with at least one full wave of requests before starting measurement. Client concurrency is capped at 128, matching the unchanged scheduler limit. If this scan does not bracket saturation or overload, report it as unconfirmed rather than automatically going beyond the requested cap. Closed-loop request timing depends on completion times; offered arrival rates are not inferred from concurrency. A legacy separate-quality or subset smoke run requires `--no-joint-quality` explicitly.

The fair BF16 baseline explicitly sets `XLLM_QUANTIZED_BACKEND=triton`, which selects the same paged algorithm for `auto` as for the compressed modes. All four modes use identical GQA grouping, dot strategy, tiling, and split policies. The ordinary optimized BF16 backend is available by leaving that environment variable unset; its results must be labeled separately. All arms disable graphs and prefix caching, use `max_memory_utilization=0.93` (about 15000–15300 MiB total VRAM on this card, retaining roughly 1 GiB headroom), and use the same 4096-token batching limit. Disabling prefix caching makes repeated traces exercise prefill rather than retained shared prefixes.

## Measurement and artifacts

Each completed run saves raw SSE text timestamps, authoritative API token counts, finish reasons, benchmark responses, latency percentiles, request throughput, generated-token throughput, and elapsed time. First nonempty text is the TTFT boundary, because decoded text can buffer multiple generated tokens. TPOT is `(E2E - TTFT)/(API completion_tokens - 1)`, including the stream finish/stop buffering tail. The older first-to-last text span is saved separately as `text_span_tpot_s`, a proxy. These are client metrics, not engine token timestamps. Latency starts after acquiring a closed-loop concurrency slot; client slot waiting is excluded. Makespan includes all timed requests and excludes startup/warmup. Client peak inflight, CPU fraction, event-loop lag, absolute timing boundaries and metric sample counts are saved to audit load-generator overhead. No retokenization estimate is substituted for API token counts. Incomplete streams or prompt-token mismatches fail the run. Attention page-table stride is a runtime kernel argument so new prompt lengths do not compile a variant per page count. The runner checks local Triton attention metadata after each timed replay; fresh compilation invalidates the cell, preserves raw data, and stops the suite until warmup is fixed.

Low-overhead one-second sampling records NVIDIA GPU utilization, VRAM use, memory-controller utilization, optional power, and BRPC Prometheus counters (`num_running_requests` and `num_waiting_requests`) at `/brpc_metrics`. Unsupported values remain unavailable in the raw log. Sampled peak VRAM is not a profiler's allocation high-water mark. NVIDIA memory utilization measures activity and is not a measurement of bytes/s or percentage of physical peak bandwidth. The hardware is GDDR7; reports use GPU VRAM and VRAM bandwidth terminology. A separate Nsight run is needed to quantify traffic and dequantization cost.

The output saves the source commit, dirty patch when present, source and binary hashes, GPU environment, exact server command, and immutable dataset/trace hashes. Completed runs can be skipped when resuming with the same source environment; an incomplete request journal is preserved and requires a separate output path or explicit investigation rather than silently being overwritten. Keep the launch arguments with the run so repetition counts and trace cycles remain reviewable.

The report script generates memory/capacity, throughput, tail-latency and quality figures from the same completed timed runs, using existing GSM8K strict/flexible EM and LongBench scorers. Response indices retain both full replay cycles even when completion order differs. Pairing compares identical sample IDs at the same concurrency, repetition and cycle; each cycle is scored individually before aggregation. Per-task paired scores, improvement/regression counts, length exhaustion, and eight-task macro remain available for every cycle. Quality means and standard deviations describe cycle/repetition variation, with six full-cycle observations at each fully completed load (three repetitions times two cycles); they are not six independent datasets. `quality.json` retains results at every concurrency. The quality figure shows the lowest completed concurrency, explicitly labeled, and the final table uses quality at its reported high load. Missing results stay pending. Max tested concurrency is not reported as a proven stable concurrency, and a throughput plateau without waiting/tail evidence is not treated as a confirmed saturation point.

The memory budget is configurable with `--max-memory-utilization`; the current authorized default is 0.93. Earlier 0.87 validation runs remain archived separately. Recheck LongBench at concurrency 128 after changing the budget, and use the same value for every formal arm.
