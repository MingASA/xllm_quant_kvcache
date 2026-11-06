<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# FP8 core review and narrow dispatch fix — 2026-11-06

Review base: `d9f17a66a34a05d37b8474591961989cbe2f1c2a`, on official
`261274e5f45449e98d7740144c3a958a29f1fd3e`; actual A remains `721afdabc`,
original measured C remains `da7c6446`. No rebase or kernel retuning.

## Strengths

- `flashinfer_native_kv.py:89`: XQA workspace initialization runs once before
  capture. The inspected FlashInfer 0.6.18.post1 `decode.py` SHA256
  `d82d107a644596a9349780b839b34e690c50169ea4cea3d0ee02b78e9dc88c1a`
  matches the recorded H800 dependency. Its XQA API explicitly requires
  first-use zero initialization and reserves the first 8 MiB for semaphores.
  The unchanged graph runner waits on the constructor's current stream before
  capture (`decode_cuda_graph.py:141`). This supports the fix; it does not
  establish that initialization caused every historical numerical mismatch.
- `mixed_native_kv.py:34`: splitting consumes existing host query/page lengths,
  preserves query and page order, and leaves scheduler/GDN state unchanged.
  Planning is shared across the 16 observed FullAttention layers.
- The failed operator mirror (83 members) and independent numerical mirror
  (104 members) were rechecked read-only: every member SHA256 and size matched.
  The original `.001/.001` mixed-conformance failure remains a failure.

## Issues

### Important — FA2 query dtype regression, fixed here

`flashinfer_native_kv.py:138` previously selected mixed splitting for explicit
FP8 FA2+XQA, although `execute` unconditionally converts its prefill subset to
FP8. The FA2 planner declares BF16/FP16 Q, and FlashInfer rejects the mismatch.
The fix restricts this optimization to the frozen FA3+XQA combination and
updates the runtime report at line 267. FA2 keeps its existing whole-batch
prefill route and pure decode still uses XQA. This is selection before
execution, with no exception-driven fallback.

The frozen FA3+XQA experiment can continue on d9. This patch is a separate
support-boundary correction, with no new kernel, scale, or benchmark shape.

### Important — microbenchmark gains do not establish model speedup

`query_quantization.py:26` supports strided input, but the target CUDA model
inherits `Qwen3_5Attention.forward`: Q normalization and partial RoPE end in
`torch.cat` (`layers/qwen3_5/common.py:160`) before flattening. Source inspection
therefore supports contiguous Q for this model. The six recorded query
fixtures are bitwise correct; contiguous ratios are about 1.001–1.007,
whereas the large strided fixtures improve. This change has no demonstrated
target-model service gain and does not remove Query conversion itself.

`flashinfer_native_kv.py:225` executes gathered XQA and FA3 work in sequence,
then scatters both outputs before downstream layers. It cannot let a decode
request bypass the rest of the mixed model step. The unchanged
`DecodeCUDAGraphRunner.can_execute` excludes prefill/chunked batches
(`model_executor/runners/decode_cuda_graph.py:124`), and CUDA GDN sends the
whole mixed batch through `_prefill` (`layers/cuda/qwen3_5/gated_delta_net.py:230`).

## Recommendations

1. Finish d9's frozen direct-FI correctness and three-shape performance gates.
   Keep `.001/.001`, query ratio <= 1.05, and mixed geometric mean <= 0.97 /
   worst case <= 1.15, including prepare amortized across 16 layers. If mixed
   fails its gate, reject that direction; do not tune shapes or tolerance.
2. Keep only the already frozen numerical candidate K=.125 / V=.5 / Q=1.
   Independent calibration shows lower V MSE and no observed clipping, but
   more small values round to zero. Formal quality remains required.
3. Do not add kernel tuning now. In the old same-base 8K operator evidence,
   Query conversion is about 120.75 / 1343.29 us (8.99% of that operator),
   and the whole native/direct gap is about 11.35%. These are operator costs,
   not the model's time distribution. The present strided conversion patch
   does not recover that whole cost on the target contiguous-Q path.
4. Public scheduling/Graph/GDN work requires separate evidence and attribution.
   The minimum useful evidence is representative step/phase time, host gaps,
   FullAttention/GDN/other compute time, and observed graph selection tied to
   tail-token delays. Counts alone cannot decide the bottleneck. The untimed
   calibration has 2,344 mixed decode rows and 98,797 pure-decode observations:
   2.31756% of rows, not a service-time fraction or speedup bound. Any later
   shared change needs its own BF16 regression and paired serving comparison.

## Validation

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B
tests/python/test_native_kv_prefill_dispatch_cpu.py -v`: **2 tests / 8 subcases
passed**. The tests load the real method AST and exercise host dispatch/report
behavior with opaque dtype/metadata and a mock external planner. They neither
import Torch nor validate kernels. A separate original-vs-fixed check
reproduced the old FA2 routing defect for BF16/FP16 and prefill/chunked inputs.
AST syntax and `git diff --check` also passed.

Bounded local attempts to import Torch and collect an initial tensor-based
test reached 45 s, 30 s (instrumented), and 180 s limits without results.
Those are incomplete environment checks, not test passes or code failures.
The retained test is the bounded stdlib check above. No local GPU task,
weights, model download, remote mutation, or experiment launch was performed.

The additional support claim still needs one H800 FA2+XQA mixed fixture using
the existing geometry: Q24/KV4/D256/P128, context129 and lengths [1,33,1].
Require the full FA2 route, declared Q dtype, finite output, and direct FA2
agreement at `.001/.001`; pure decode remains the existing XQA path. Do not
change the frozen FA3+XQA qualification or run a new performance search for
this support fix. The Controller owns any such qualification and publication.

## Assessment

**Ready to merge? No production-acceptance claim.** The narrow dispatch fix has
host-contract evidence and awaits the affected configuration's GPU check.
The frozen optimization still needs operator/model/quality/service gates.
Stopping kernel tuning is the present evidence-based decision; the public
execution path is a plausible next investigation, not an established bottleneck.
