# GovReport c16 four-arm service comparison

This completed group used the frozen GovReport trace (200 prompts replayed twice, 400 responses per arm), normal EOS, and the `GovReport-1024-output-cap-extension` protocol. Metrics come from the completed `matched_c16_analysis.json`; answer text was not read. Quantized-arm quality scoring is pending.

| Arm | max seqs/batch | req/s | tok/s | Mean output tokens | p95 TTFT (s) | p95 TPOT (ms) | p95 E2E (s) | Length exits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 (`auto`) | 128 | 0.4731 | 313.45 | 662.5 | 3.862 | 58.2 | 52.16 | 39 |
| INT8 | 256 | 0.4700 | 315.77 | 671.9 | 3.658 | 60.9 | 53.57 | 43 |
| FP8 E4M3 | 256 | 0.4640 | 312.08 | 672.6 | 3.700 | 59.7 | 53.75 | 47 |
| INT4 | 256 | 0.5186 | 285.35 | 550.2 | 3.562 | 67.0 | 58.70 | 42 |

Relative to BF16, INT8 is nearly flat on request throughput (-0.7%) and slightly higher on token throughput (+0.7%), with p95 TTFT -5.3%, TPOT +4.6%, and E2E +2.7%. FP8 has request/token throughput -1.9%/-0.4%, TTFT -4.2%, TPOT +2.4%, and E2E +3.0%. INT4 has 9.6% higher request throughput but 9.0% lower token throughput and mean output about 17.0% shorter; its TTFT is 7.8% lower but TPOT 15.0% higher and E2E 12.5% higher. Quality is not yet available, so the INT4 request-throughput gain is not evidence of a net user benefit.

The reused BF16 baseline ran with `max_seqs_per_batch=128`; quantized cells used 256. All cells used token budgets 10240/10240 and client concurrency 16. This remains a scheduler-cap confound. Mean KV utilization was 46.9% BF16 and 30.6–36.3% quantized, with peaks 66.1% and 41.9–50.6%, respectively. Mean running was 15.6 for all arms; mean waiting was about 0.1 with peaks 13–14; preemptions were zero. These observations show low queueing at c16 and no preemption, but do not by themselves prove why utilization differs. Allocator block counts are not comparable across cache formats.

## Quality score follow-up

The final scorer completed both full 200-row replay cycles per arm using official LongBench ROUGE-L F1 on `GovReport-1024-output-cap-extension` (three rows used documented head/tail context handling). The two cycles are replay passes, not independent benchmark repetitions.

| Arm | Cycle 0 ROUGE-L | Cycle 1 ROUGE-L | Paired mean delta vs BF16 (cycle 0 / cycle 1) | Paired improved / regressed (cycle 0 / cycle 1) |
|---|---:|---:|---:|---:|
| BF16 | 0.32102 | 0.32016 | — | — |
| INT8 | 0.32242 | 0.31986 | +0.00140 / -0.00030 | 104/92 / 98/98 |
| FP8 E4M3 | 0.31886 | 0.32055 | -0.00216 / +0.00040 | 99/100 / 97/102 |
| INT4 | 0.30818 | 0.30571 | -0.01284 / -0.01444 | 73/127 / 85/115 |

INT8 and FP8 change direction or stay close to baseline across the two replay passes. INT4 is lower in both passes and has more paired regressions than improvements, so the observed request-throughput gain does not establish a quality-preserving user benefit.
