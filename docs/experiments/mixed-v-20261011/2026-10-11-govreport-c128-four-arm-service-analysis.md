# GovReport c128 four-arm service comparison

This is the completed c128 service group from `flashinfer-govreport-seq256-20261011/formal-two-passes`, using the frozen GovReport trace (200 prompts replayed twice, 400 responses per arm) and normal EOS. It summarizes the existing `matched_c128_analysis.json` and per-cell summaries; it does not inspect or reproduce model answers. Quality scoring is pending and must be added separately.

| Arm | Effective max seqs/batch | req/s | tok/s | Mean generated tokens | p95 TTFT (s) | p95 TPOT (ms) | p95 E2E (s) | Length exits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 baseline (`auto`) | 128 | 0.5570 | 371.9 | 667.7 | 192.48 | 89.3 | 264.77 | 43 |
| INT8 | 256 | 0.4952 | 333.6 | 673.6 | 205.12 | 133.0 | 308.01 | 47 |
| FP8 E4M3 | 256 | 0.4906 | 333.3 | 679.3 | 204.53 | 130.3 | 303.27 | 43 |
| INT4 | 256 | 0.5714 | 323.0 | 565.2 | 170.68 | 164.2 | 288.29 | 37 |

Relative to the BF16 baseline, INT8 and FP8 delivered about 10.3% fewer generated tokens/s, with 46–49% higher p95 TPOT and 14.5–16.3% higher p95 E2E. INT4 delivered 2.6% more requests/s but 13.2% fewer generated tokens/s; its mean output was about 15.4% shorter, p95 TTFT 11.3% lower, p95 TPOT 83.8% higher, and p95 E2E 8.9% higher. The request-throughput gain therefore does not establish a user-visible quality or latency gain; quality scores remain pending.

A scheduler confound must remain explicit: the reused BF16 c128 cell used `max_seqs_per_batch=128`; all three quantized cells used 256. All cells used token budgets 10240/10240 and client concurrency 128. Thus this is not a format-only controlled comparison. Quantized-arm telemetry reported mean KV utilization 84.5–85.9%, peaks 91.9–92.4%, zero preemptions, mean running 37.4–44.2 (peaks 46–54), and mean waiting 71.8–77.7 (peaks 123–124). BF16 showed mean utilization 85.6%, peak 92.0%, zero preemptions, mean running 28.6 (peak 37), and mean waiting 84.5 (peak 125). Per-format allocator block counts are not byte-comparable. Waiting and utilization alone do not prove a KV-admission cause.

Interpretation is descriptive at this single concurrency. Scheduler-cap differences and different output-length distributions confound attributing changes to the cache codec. Compare each subsequent same-concurrency group independently and retain request TPOT separately from rolling inter-token telemetry.

## Root judgment: load trend and scheduler evidence

c32→c64→c128 tok/s is approximately BF16 368.7→367.9→371.9, INT8 337.5→332.8→333.6, FP8 340.7→335.0→333.3, INT4 308.4→319.9→323.0. Raising offered concurrency did not lift the INT8/FP8 throughput plateau above BF16. Their TTFT advantage at c32 disappeared at c64 and reversed at c128; additional admission reduces waiting count but does not compensate for lower service completion rate.

INT4 completed the same 400 requests in 700.0s versus BF16 718.1s, a real 2.5% reduction in total completion time. This is retained as a positive request-level result, alongside shorter outputs, pending quality and worse token/tail metrics. Its median E2E is 210.17s versus 224.16s (−6.2%), whereas p95 E2E is 288.29 versus 264.77s (+8.9%). Therefore the outcome is distribution-dependent; it would be incorrect to claim that every user's latency worsened or that the median improvement proves preserved-quality acceleration. Previous c32/c64 INT4 ROUGE-L decline and output shortening require checking c128 quality before acceptance.

The reused BF16 cap128 vs quant cap256 remains a provenance/config difference. Observed running peaks37–54 are below both caps, and the default latency profiler is off; there is no evidence of direct cap truncation in this group. That difference is not a demonstrated explanation for the observed slowdown. Indirect allocation/workspace effects and separate-run variability are not isolated by this comparison.

Independent event data and scheduler source support a mechanism rather than a proven scheduler bug: decode is admitted first, leftover token budget admits large prefills, and batch output follows full execution. Prefill linear/MLP work remains expensive regardless of V precision; bigger active decode batches increase per-step work. Token-based default budgets do not enforce a time limit, so large mixed rounds can stall users. Changing scheduling could alter the throughput/latency tradeoff, but cannot eliminate required prefill computation. See `2026-10-11-prefill-decode-scheduler-analysis.md` for phase timings, admission reserve and precise limits. No formal configuration changed to force a positive result.

## Quality score follow-up

The existing scorer has now completed both full 200-row cycles per arm. It used official LongBench ROUGE-L F1 on the selected `GovReport-1024-output-cap-extension` protocol; three prompts required the documented head/tail context handling. The two cycles are replay passes, not independent benchmark repetitions.

| Arm | Cycle 0 ROUGE-L | Cycle 1 ROUGE-L | Paired mean delta vs BF16 (cycle 0 / cycle 1) | Paired improved / regressed (cycle 0 / cycle 1) |
|---|---:|---:|---:|---:|
| BF16 | 0.32035 | 0.32268 | — | — |
| INT8 | 0.31929 | 0.32159 | -0.00106 / -0.00110 | 103/93 / 98/101 |
| FP8 E4M3 | 0.32201 | 0.32233 | +0.00167 / -0.00035 | 100/99 / 94/106 |
| INT4 | 0.30594 | 0.30729 | -0.01440 / -0.01539 | 82/118 / 67/133 |

The small INT8/FP8 differences change direction or remain near zero across the two passes. INT4 is lower in both cycles, with more paired regressions than improvements. This does not support treating the INT4 request-throughput/TTFT improvement as a net quality-preserving benefit. The scheduler-cap mismatch noted above also remains: BF16 used 128 max sequences per batch while quantized arms used 256.
