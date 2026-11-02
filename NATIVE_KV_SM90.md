# Native SM90 KV cache (qualification in progress)

Base: official xLLM `261274e5f45449e98d7740144c3a958a29f1fd3e`.
The public VMM page-alignment correction is isolated in `cdbfbd1c`.
This branch selectively adds compressed full-attention storage; it preserves
upstream scheduling and the twelve-field LayerCache/GDN state interface.

| Format | Resident bytes per K or V head, D=256 | Prefill | Decode |
|---|---:|---|---|
| BF16 (`auto`) | 512 | Unmodified official selection | Unmodified official selection |
| `fp8_e4m3` | 256 | FA3 FP8 Q by default; explicit FA2 BF16 Q candidate | BF16 Q XQA for non-VMM full attention; FlashInfer for VMM |
| `nvfp4` | 144 | FlashInfer native FA2 | FlashInfer native FA2 tensor-core wrapper |

NVFP4 owns one byte tensor for each K/V cache. Every page contains a packed
E2M1 data plane followed by its linear E4M3 block-16 scale plane. FlashInfer
receives aliases with independent page strides. Whole-page prefix copying and
VMM mapping/reclaim preserve both planes. No BF16 KV shadow is allocated.
GDN convolution and recurrent states retain their upstream dtypes and budgets.

For native formats, `--kv_cache_k_scale` and `--kv_cache_v_scale` are immutable
positive normal FP32 decode scales. NVFP4 also stores native per-block scales.
`--kv_cache_prefill_backend=auto|fa2|fa3` and
`--kv_cache_decode_backend=auto|flashinfer|xqa` affect quantized paths only.
NVFP4 on SM90 admits FA2/FlashInfer only. Explicit XQA with VMM is rejected:
this integration has not established finite initialization of unmapped pages.
There is no retry or failure-driven backend substitution. CUDA Python Executor,
SM90, logical head dimensions 128/256 and GQA groups >=4 are the initial scope;
MLA/indexer, padded head dimensions, PD transfer and host offload are excluded.

H800 does not provide Blackwell native FP4 Tensor Core computation. NVFP4 here
compresses resident storage and is decoded by FlashInfer's SM90 attention path.
With the default 2 MiB VMM hardware pages, D=256/KV-heads=4/page-size=128,
NVFP4 maps groups of 256 token blocks (288 physical pages across 16 FA layers).
This allocation granularity must be included in measured capacity conclusions.

Validation so far: 8 host VMM layout tests, 2 CPU tensor alias/copy tests and
an offline compilation of the fused FP8 writer for SM90. C++ allocation tests
and real H800 operators/model/Graph/performance/quality remain pending.
These CPU checks do not establish serving correctness or production readiness.
