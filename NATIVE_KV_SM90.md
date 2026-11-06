# Native KV cache on SM90

This branch adds OCP MXFP4 resident KV storage. **The implementation is qualified within the narrow scope below; its measured model quality fails the frozen engineering gate and remains research-only.**

Measured Python and compiled xLLM source: [`ba5eaec0a1b5b941da4b8ded0343d43f2c2b4158`](https://github.com/MingASA/xllm_quant_kvcache/commit/ba5eaec0a1b5b941da4b8ded0343d43f2c2b4158), derived from measured native C `da7c6446ff4361c8060633b04d6bfb77e6f737ff` and official xLLM `261274e5f45449e98d7740144c3a958a29f1fd3e`. BF16 A is `721afdabc1de1af8e594b4da3e6fa926def50821`. The CUDA/NPU build guard, byte-page geometry and lazy BF16 kernel import remain separate changes; no old research branch was merged. A subsequent documentation-only commit does not replace these measured source or binary identities. Executed binary and JIT hashes remain in the run manifests.

| Format | Resident bytes per K or V head, D=256 | Prefill | Pure decode |
|---|---:|---|---|
| BF16 (`auto`) | 512 | Official FlashInfer selection | Official FlashInfer selection |
| `fp8_e4m3` | 256 | FA3 with FP8 query | XQA with BF16 API query |
| `nvfp4` | 144 | FlashInfer native FA2 | FlashInfer FA2 tensor-core wrapper |
| `mxfp4` | 136 | Pinned private FlashInfer FA2 adaptation | Same paged FA2 adaptation, one query/request |

The inherited FP8/NVFP4 paths have separate historical evidence; MXFP4 qualification does not extend it. Backend incompatibility fails explicitly. There is no failure-driven fallback.

Use the CUDA Python Executor with `--kv_cache_dtype=mxfp4`, `--kv_cache_prefill_backend=auto|fa2`, `--kv_cache_decode_backend=auto|flashinfer`, and `--kv_cache_k_scale=1 --kv_cache_v_scale=1`. MXFP4 admission requires **SM90, BF16 input/output, Q24/KV4/D256, page128, paged NHD full attention, no sliding window and no VMM**. Other global scales are rejected. D128/FP16 admission in the shared native backend does not apply to MXFP4. Ragged attention, MLA/indexer, padded dimensions, PD transfer and host offload are outside this qualification.

MXFP4 uses E2M1 elements with one E8M0 scale per 32 consecutive channels, as specified by [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). This differs from NVFP4's E4M3 block-16 scales. E2M1 magnitudes are `0, .5, 1, 1.5, 2, 3, 4, 6`; the earlier channel occupies the low nibble. For finite nonzero blocks, the writer selects `e=clamp(floor(log2(absmax))-2,-127,127)`, stores E8M0 code `e+127`, and rounds elements to nearest, ties to even, with saturation. Signed zero is retained; an all-zero block uses code 0, meaning scale `2^-127`. A block containing a nonfinite input receives scale code 255 and zero payload, yielding a NaN block on reconstruction. Infinity preservation is not supported.

Native formats own contiguous byte tensors with typed/strided aliases. Each MX page stores the whole packed-data plane followed by the E8M0 scale plane; whole-page prefix copies include both. At D256 this is `128+8=136` bytes/head, or 4.25 bits/value. For Q24/KV4, K+V cost 1,088 bytes/token/FullAttention layer. The model qualification allocated 5,069 pages across 16 FullAttention layers: **11,294,867,456 resident bytes and 648,832 pooled physical token slots (648,704 usable after reserving page 0)**. The `64/17` capacity ratio versus BF16 applies only to an equal FullAttention KV budget, not whole-model capacity or a per-request context limit. GDN convolution/recurrent dtypes, state interface and separate budget are unchanged.

The adapter pins FlashInfer 0.6.18.post1's attention header (`f93cbea96e3a…`; adapted `b06c014c4f93…`) and rejects an incompatible header. H800 has no Blackwell FP4 Tensor Core compute. E8M0 bytes expand only within the current tile; scales are decoded in FP32, converted to BF16, and multiplied with BF16 E2M1 fragments for BF16 MMA with FP32 accumulation. There is **no FP16 scale conversion, global expanded-scale plane or full-cache BF16 shadow**. All finite E8M0 scales, including values below `2^-24` and above `2^15`, are representable in BF16. Full-attention overflow/nonfinite behavior remains bounded by the BF16/FP32 compute path; writer edge tests do not qualify arbitrary extreme attention inputs.

This adaptation retains FA2 planning and scratch memory. Compared with native NVFP4, it uses two writer launches, synchronous tile scale loading/duplication, and additional Graph metadata staging; pure decode also retains the prefill wrapper's uint8-KV split-KV safety restriction. Their service-time contribution is unmeasured. Compression alone establishes no throughput advantage.

Official CUDA at the frozen base rejects `enable_vmm_kvcache` before model loading. The historical BF16 VMM attempt failed at that guard; MXFP4 rejects VMM explicitly. Host page-geometry tests do not qualify mapping/reclaim, elastic capacity or unmapped pages. The default resident allocator is used.

The existing H800 operator/model archives contain **217 SHA/size-verified files (93+124)**. Writer bytes and edge cases match an independent CPU reference. Prefill/decode max absolute errors against independently reconstructed KV and dense attention are `0.008329153060913086` / `0.0020748376846313477`, within the unchanged `atol=rtol=.02` gate. Changed-input Graph replay matches direct execution bitwise; whole-page copies preserve data and scales. Model qualification passed 16 known requests, real cross-request prefix reuse/GDN restoration, and a fresh three-request observation window with six successful CUDA Graph launches. This is not all-request or formal mixed-batch Graph coverage. Shared Scheduler/Graph/GDN code was not rewritten.

The formal quality comparison retains all 1,720 core generations, original EOS/caps and denominators, plus 32 native multi-turn tool tasks. Scores below are rounded percentages; tools are correct tasks.

| Arm | GSM8K flexible EM | LongBench eight-task macro | RepoBench-P | Tools |
|---|---:|---:|---:|---:|
| BF16 A | 96.7399545110 | 53.8984440135 | 43.92 | 26/32 |
| Original FP8 C | 96.5883244882 | 53.2642144417 | 39.74 | 26/32 |
| Original NVFP4 C | 96.5883244882 | 53.8296563882 | 39.68 | 27/32 |
| MXFP4 | 95.9818043973 | 52.9147022788 | 36.48 | 23/32 |

Against A/original FP8, MXFP4 loses **0.758150/0.606520 percentage points on GSM8K, 7.44/3.26 on RepoBench-P, and 3/3 tool tasks**, exceeding the respective frozen limits of .5 points, 2 points and one task. LongBench macro losses .983742/.349512 are within its one-point limit, but the paired 95% interval versus A is `[-2.691894, .540221]` points. RepoBench-P's interval versus A is `[-14.821, -.72]`; statistical noninferiority is not established. These are engineering thresholds, not user-approved SLOs. RepoBench-P measures completion similarity, not code execution success; the tools subset is not the full BFCL leaderboard. The single long supplement scored 50% for every arm and is excluded from the eight-task macro.

Exact quality values, paired comparisons, preserved failures, source/binary receipts and audited service results belong to the [fork research report branch](https://github.com/MingASA/kv_research_agent/tree/research/mxfp4-fp8-20261105/reports/experiments/mxfp4-fp8-20261105). The [historical A/C report](https://github.com/MingASA/kv_research_agent/blob/main/reports/experiments/upstream-production-kv-20261102/REPORT.md) remains separate. This document makes no production acceptance or service speedup claim.

The independent raw-service audit completed all 512/512/64 original requests with natural EOS and unchanged caps on the same H800 GPU as A. Output tokens/s were 599.6924100622, 663.1120130317, 113.2293250025 for ShareGPT C32/C64 and GovReport C32. All three were below BF16 A; this implementation is not a production performance recommendation. Real cross-request prefix hits were zero in these formal workloads, distinct from the successful reuse qualification.
