# GovReport c8 four-arm service comparison

This completed group used the frozen GovReport trace (200 prompts replayed twice, 400 responses per arm), normal EOS, and the `GovReport-1024-output-cap-extension` protocol. Values come from the existing `matched_c8_analysis.json` and cell summaries; no answer text was read. Quality scoring for this group is pending.

| Arm | max seqs/batch | req/s | tok/s | Mean output tokens | p95 TTFT (s) | p95 TPOT (ms) | p95 E2E (s) | Length exits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 (`auto`) | 128 | 0.3324 | 218.3 | 656.6 | 2.79 | 41.7 | 37.58 | 36 |
| INT8 | 256 | 0.3224 | 218.7 | 678.3 | 3.36 | 42.0 | 38.30 | 54 |
| FP8 E4M3 | 256 | 0.3258 | 220.9 | 678.2 | 3.30 | 41.5 | 38.69 | 46 |
| INT4 | 256 | 0.3699 | 208.2 | 562.7 | 3.19 | 46.4 | 39.06 | 44 |

Relative to BF16, INT8 and FP8 have slightly lower request throughput (-3.0% / -2.0%) while generated-token throughput is essentially flat (+0.2% / +1.2%). Their p95 TTFT and E2E are modestly higher. INT4 has 11.3% higher request throughput but 4.6% lower token throughput, 14.2% higher p95 TTFT, 11.1% higher p95 TPOT, and 3.9% higher p95 E2E; its mean output is 14.3% shorter. These service numbers alone do not establish a user benefit; quality is pending.

The reused BF16 baseline used `max_seqs_per_batch=128`; quantized cells used 256. All four use token budgets 10240/10240 and client concurrency 8. This remains a scheduler-cap confound rather than a format-only comparison. Telemetry shows low mean KV utilization (15.6–24.0%, peaks 24.3–36.6%), zero preemptions, mean running about 7.9 across arms, mean waiting zero and observed waiting peaks 5–6. No evidence here points to sustained KV pressure at c8. This single low-load group should not be generalized to higher concurrency.

## Root interpretation across load levels

At c8, INT8/FP8 generated tok/s differences of +0.2%/+1.2% are small, while request throughput is lower and outputs about3.3% longer. Two replay passes do not provide independent timing repeats, so these small differences cannot establish a reproducible acceleration. At c32–128, INT8/FP8 instead show a clear roughly8–10% generated-throughput deficit and substantially worse request TPOT. The pattern supports a workload/batch-dependent cost, rather than a universal codec slowdown.

At c8 there is no persistent waiting and only8 active sequences, so neither the128 nor256 sequence ceiling directly binds. Their provenance differs and indirect workspace/allocation effects are not isolated, but this cap difference is not evidence for direct scheduler truncation. Low-load results cannot establish how much of the higher-load loss is codec versus planner/mixed-round shape; independent same-shape eager measurements are still needed.

INT4's +11.3% request-rate result is preserved. It also produces14.3% fewer output tokens per request and worsens the three reported p95 latency metrics. c128 quality has already shown mean ROUGE-L32.15→30.66 forINT4, so c8 quality must be checked before claiming preserved-quality service gains. No formal settings or completed results were changed.

## Quality score follow-up

The scorer has completed both 200-row replay cycles per arm using official LongBench ROUGE-L F1. These two cycles are replay passes, not independent benchmark repetitions.

| Arm | Cycle 0 ROUGE-L | Cycle 1 ROUGE-L | Paired mean delta vs BF16 | Paired improved / regressed (cycle 0 / cycle 1) |
|---|---:|---:|---:|---:|
| BF16 | 0.32143 | 0.31860 | — | — |
| INT8 | 0.32219 | 0.31860 | +0.00076 / +0.00001 | 93/105 / 97/101 |
| FP8 E4M3 | 0.32172 | 0.32553 | +0.00029 / +0.00694 | 102/94 / 115/84 |
| INT4 | 0.30358 | 0.30595 | -0.01785 / -0.01265 | 76/123 / 80/120 |

INT8 is near baseline and FP8 is above baseline in both observed cycles, though paired directions are less decisive for INT8 and the c8 comparisons retain the max-sequence-cap mismatch. INT4 shows lower ROUGE-L in both cycles and more regressions than improvements, so its higher request throughput does not amount to a quality-preserving gain here.
