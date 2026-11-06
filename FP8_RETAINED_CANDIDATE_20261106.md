<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Isolated FP8 candidate after mixed-performance rejection

The frozen mixed split is rejected. Its corrected d9 operator passed direct
FlashInfer conformance with max absolute error 0 and unchanged GDN state, but
the three fixed latency ratios were **1.2223488814140813**, 0.3866250406271028,
and 0.7877233187379983. The geometric mean was 0.7193710450860855; the first
case exceeds the frozen 1.15 maximum. `mixed_gate=false` remains unchanged.

The `fp8-fixed-operator-0001` mirror was independently checked: all 91 members
matched both SHA256 and size. Acceptance SHA256:
`23b1d2e1779cc2622ec7d1ebd7daafe4b0b987055df22edac79a419c50fdc07f`.
Manifest SHA256:
`11970f48d4909fdbdb1b799cd8eed004e4da1114fd921f20d396d842beb838da`.
The earlier r0 correctness failure also remains recorded.

## Source decision

- Remove mixed subset construction, gathering/scattering, dispatch state, and
  the rejected direction's dedicated implementation/tests. Mixed/prefill work
  again uses the full packed FA3 batch; pure decode still uses XQA.
- Retain d9's first-use workspace zero initialization and the already frozen
  query conversion. No new kernel, shape, backend, or tuning threshold.
- Select K=.125 / V=.5 / Q=1 through the existing explicit configuration.
  Universal scale defaults are unchanged. V=.5 is the sole independently
  calibrated candidate, with actual quality still unproven.
- `prepare` and `execute` method ASTs match original C (`a5ce9684`; those
  attention files are identical to measured `da7c6446`). Query conversion,
  `xqa_decode.py`, and the KV writer remain byte-identical to d9.
- Runtime reporting returns `mixed_FP8_decode=null`. Query dtype in that
  report describes the API tensor, not XQA's internal MMA precision; see
  [the scale trace](FP8_XQA_SCALE_TRACE_20261106.md).

Branch `research/fp8-final-isolated-20261106` descends from `be597590`; the d9
measurement branch and the separate be59 boundary-fix ref are preserved.
No shared Scheduler, Graph runner, or GDN code changes are included.

## Qualification handoff

Use a distinct retained-candidate operator contract; never reinterpret the
old mixed gate as passed.

1. Reuse the original six query fixtures: bitwise independent/original-query
   agreement and the original <=1.05 limit. Do not introduce a new search.
2. At the existing Q24/KV4/D256/P128, context129, query-length [1,33,1] fixture,
   verify normal mixed work takes the full FA3 route and agrees with direct
   FA3 at `.001/.001`. Verify pure XQA decode against direct XQA using the
   existing decode rows. Test only V=.0625 and the frozen V=.5, with K=.125.
3. Check native K/V bytes against independently quantized source tensors at
   both scales before output comparison. An oracle reading only the same
   native-written pages cannot independently detect a writer-scale error.
   Preserve first-use zero-workspace assertions for backend and oracle.
4. Source-matched model qualification must still establish real Graph replay,
   changed-input correctness, prefix/GDN behavior, then the frozen quality and
   service matrix. Reuse an already planned Graph check rather than adding a
   separate performance matrix. No mixed service-Graph claim is made.

The small standalone host test runs without Torch:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B tests/python/test_native_kv_prefill_dispatch_cpu.py -v`.
**4 tests / 18 subcases passed** for packed prefill routing, both pure-decode
plan modes, forwarding both fixed scale configurations, and the runtime
report. These execute real method bodies with mocked device boundaries;
they are not kernel, Graph, model, or serving evidence. Syntax/whitespace and
the source-equivalence checks above passed. No local GPU work was performed.
