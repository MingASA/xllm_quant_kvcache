# FlashInfer mixed-V eager backend

Use `XLLM_QUANTIZED_BACKEND=flashinfer_mixed` with `kv_cache_dtype=auto`, `bf16_int8`, `bf16_fp8_e4m3`, or `bf16_int4`. All four formats use the same FlashInfer FA2 paged Tensor Core pipeline for prefill and decode. This is a separate extension; the installed FlashInfer package and existing backend selection remain available.

The specialization targets Qwen2.5-1.5B-Instruct on SM120 and SM90 (H800/H200): BF16 queries/keys, 12 query heads, 2 KV heads, head dimension 128, full causal attention, eager execution. Its launch geometry matches the pinned FlashInfer BF16 heuristic: the original 100-KiB SM120 path is preserved, and SM90 has a separate 228-KiB tile policy. See the [Hopper reproduction README](hopper-mixed-v/README.md) for fresh-machine gates, frozen data transport, full experiment commands, and validation limitations. Unsupported shapes, sliding windows, devices and graph execution fail explicitly.

V remains in its allocated INT8, FP8 E4M3 byte, or packed signed INT4 format with FP32 per-token/head scales. The original codec, writer, allocator and block-copy paths are reused. Only the current attention tile is dequantized in shared memory before the original BF16 MMA operations. No global BF16 V tensor or BF16 V Cache is created.

The private headers retain upstream Apache notices; `upstream.json` pins FlashInfer 0.6.18.post1 and original header hashes. The loader copies headers into a source-addressed private build directory, compiles only at backend startup, isolates CUDA symbols, and records compiler, upstream/CUTLASS sources and artifact hashes. The 128 MiB planner workspace is allocated before KV budgeting and reused across layers and steps.

Validate before running experiments:

```bash
python tools/check_flashinfer_mixed_v.py \
  --output results/new-native-gates \
  --allocator-gates results/mixed-v-cache-20261010/correctness_gates.json
python tools/run_mixed_v_cache_experiment.py run \
  --output results/new-full-suite \
  --gates results/new-native-gates/correctness_gates.json \
  --attention-backend flashinfer_mixed \
  --concurrency 8 16 32 64 128 --repetitions 3 --replay-cycles 2
```

Prepare immutable full benchmark traces in the output directory first with the runner's `prepare` action. Set `nvcc`/`compute-sanitizer` on PATH, use the existing CUDA Python environment, and keep graph execution disabled. `XLLM_FLASHINFER_MIXED_BUILD_DIR` and `XLLM_FLASHINFER_MIXED_NVCC` optionally select the persistent build cache and compiler. Kernel artifacts must match the correctness gate. Attention/writer compilation during timed requests invalidates a run.

The integrated experiment report records the selected backend. The optional kernel diagnostic `tools/benchmark_flashinfer_mixed_v.py` calibrates against untouched FlashInfer BF16 and saves eager plus CUDA-event timings; its graph timings are not server latency measurements.

[Second-version validation and calibration](../results/flashinfer-mixed-v2-20261010/README.md)

[V-only INT8/INT4 versus Triton: batch-20 kernels and GSM8K 100-question service comparison](../results/flashinfer-vs-triton-vonly-c20-gsm100-20261010/README.md)

[Fixed-history layer/logit checks and GPU/eager/planning latency decomposition](../results/flashinfer-numerics-20261010/README.md)
