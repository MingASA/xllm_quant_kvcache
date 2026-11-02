# Native KV cache on SM90

Frozen implementation: `da7c6446ff4361c8060633b04d6bfb77e6f737ff`, based on official xLLM `261274e5f45449e98d7740144c3a958a29f1fd3e`. The CUDA/NPU build guard (`721afdab`), shared VMM page geometry (`a8c8f0d4`), and lazy BF16 kernel import (`55457179`) are separate commits. No old research branch was merged.

| Format | Resident bytes per K or V head, D=256 | Prefill | Decode |
|---|---:|---|---|
| BF16 (`auto`) | 512 | Official FlashInfer selection | Official FlashInfer selection |
| `fp8_e4m3` | 256 | FA3, FP8 query; explicit FA2/BF16-query candidate | XQA, BF16 query; explicit FlashInfer candidate |
| `nvfp4` | 144 | FlashInfer native FA2 | FlashInfer native FA2 tensor-core wrapper |

Native formats own byte tensors with typed aliases. Each NVFP4 page contains packed E2M1 data followed by E4M3 block-16 scales, passed to FlashInfer with independent page strides. Whole-page prefix copies include both planes. There is no full-cache BF16 shadow. GDN convolution/recurrent states retain their official dtypes, state interface, and budgets.

Use `--kv_cache_dtype=fp8_e4m3` or `--kv_cache_dtype=nvfp4` with the CUDA Python Executor. Quantized paths alone accept `--kv_cache_prefill_backend=auto|fa2|fa3` and `--kv_cache_decode_backend=auto|flashinfer|xqa`. `--kv_cache_k_scale` / `--kv_cache_v_scale` are fixed positive normal FP32 decode scales; NVFP4 also has per-block scales. The measured Qwen3.8-27B FP8 configuration used FA3/XQA and K/V scales 0.125/0.0625; NVFP4 used FA2/FlashInfer and scales 1/1. These are model-specific evaluated settings, not a universal calibration recipe. Backend incompatibility fails explicitly; there is no failure-driven fallback.

Scope: SM90, dense full attention with logical head dimension 128/256, GQA groups >=4, and the official hybrid LayerCache/GDN interface. Actual H800/model evidence is Q24/KV4/D256/page128; D128 admission is not a model qualification. MLA/indexer, padded dimensions, PD transfer, and host offload are excluded. The cache-budget calculation uses actual resident bytes, independent of virtual reservation.

Official CUDA rejects `enable_vmm_kvcache` before model loading (the feature is NPU-only at the frozen base). This unchanged guard failed the dedicated BF16 VMM attempt; FP8/NVFP4 VMM were not attempted. Typed byte/page alignment has host geometry tests only. VMM mapping/reclaim, elastic capacity, and XQA with unmapped pages are not qualified; explicit XQA/VMM is rejected. The default allocator is the official CUDA production path.

H800 has no Blackwell FP4 Tensor Core compute. FlashInfer SM90 NVFP4 unpacks tiles into shared/register BF16 for MMA; it does not materialize a global full-cache BF16 staging buffer. Direct-library timings expose substantial NVFP4 compute cost. Compression alone is not a throughput claim.

The fixed H800 campaign independently checks actual storage/backend bindings, Writer/Prefill/Decode, changed-input Graph, page copies/negative slots, real cross-request prefix reuse and GDN restore, same-version core quality, native multi-turn tools, and real completed service requests. FA3's original dense 0.02 tolerance failure remains recorded; independent direct FlashInfer reproduces it and native/direct results match bitwise. Successful Graph APIs during qualification do not establish complete formal mixed-batch Graph coverage. No Scheduler/Graph rewrite is included.

Full results, exact source/binary identities, paired uncertainty, latency tradeoffs, and limitations: [production A/C report](https://github.com/MingASA/kv_research_agent/blob/main/reports/experiments/upstream-production-kv-20261102/REPORT.md). The implementation commit above remains the measured source even if this documentation is updated later.
