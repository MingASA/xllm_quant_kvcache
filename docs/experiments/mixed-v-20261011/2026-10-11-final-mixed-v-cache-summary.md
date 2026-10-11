# Final mixed-V KV-cache experiment summary

## Scope and provenance

- Both suites completed with exit code 0 under commit `f944f205409db0dab1751abcf8962adad42caebb`, xLLM binary SHA-256 `d40a5d5622c3f04e9eaf2bc2a43cd74b888451463d6c6f1ac34f5c3c7522cd87`, Python 3.12.13, PyTorch 2.13.0+cu132, and backend `flashinfer_mixed`. Model: `/mnt/e/AI/models/Qwen2.5-1.5B-Instruct`; final gates: `/mnt/xllm-e/data/home/mingasa/projects/xllm_quant_kvcache/results/flashinfer-mixed-v2-20261010/gates-final/correctness_gates.json` (SHA-256 `edbfddaca2d0ca9244ca4b1c3473acb97a9bfec996d144019497e6d758857b1e`).
- Effective scheduler token budget was 10,240 tokens/batch with default prefill chunk resolving to 10,240, sequence cap 256, service admission cap 512, and memory utilization 0.93. GSM8K used client concurrency 128/160/208; GovReport used 8/16/32/64/128. Both used normal EOS and the frozen full traces.
- GSM8K E archive: `/mnt/xllm-e/data/home/mingasa/projects/xllm_quant_kvcache/results/flashinfer-service-seq256-20261011/gsm8k-default-token-budget-seq256-c128-160-208-256`. Twelve cells (four arms × three loads) completed, each with 2,638 responses (1,319 prompts × two replay cycles), totaling 31,656. No missing, invalid, or in-window attention-JIT cells.
- GovReport E archive: `/mnt/xllm-e/data/home/mingasa/projects/xllm_quant_kvcache/results/flashinfer-govreport-seq256-20261011/formal-two-passes`. Twenty cells (four arms × five loads) completed, each with 400 responses (200 prompts × two cycles), totaling 8,000. All quality scores cover 200 requests per arm and cycle. No missing, invalid, or in-window attention-JIT cells.
- GovReport used the preselected `GovReport-1024-output-cap-extension`: the BF16 pilot exhausted the 512-token cap on 18/20 prompts. Three of the 200 full prompts required the documented head/tail context handling. This is an output-cap extension and should not be presented as native 512-token LongBench scoring. The two full-trace cycles are replay passes, not independent repetitions.
- Reused GovReport BF16 baselines used `max_seqs_per_batch=128`; quantized cells used 256. Client concurrency and token budgets were held as configured, but this sequence-cap difference means the arm comparisons are not format-only controlled. At c64/c128, observed peak running sequences were 37–54, below either configured sequence cap; this does not establish a hardware ceiling. No hardware bottleneck is inferred here.

## GSM8K high-load result: client concurrency 208

Each arm served all 2,638 requests. Service values are aggregate across the two replay cycles; scores below remain cycle-specific.

| Arm | req/s | tok/s | Mean output | p95 TTFT (s) | p95 TPOT (ms) | p95 E2E (s) | Length exits | Strict EM cycle 0 / 1 | Flexible EM cycle 0 / 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 12.633 | 1,959.6 | 155.1 | 5.44 | 107.56 | 33.75 | 27 | 0.4405 / 0.4382 | 0.6513 / 0.6490 |
| INT8 | 12.298 | 1,918.1 | 156.0 | 5.26 | 109.55 | 35.57 | 32 | 0.4390 / 0.4337 | 0.6437 / 0.6429 |
| FP8 E4M3 | 12.387 | 1,912.8 | 154.4 | 5.30 | 107.80 | 35.30 | 28 | 0.4390 / 0.4466 | 0.6444 / 0.6437 |
| INT4 | 12.587 | 1,906.3 | 151.5 | 5.33 | 109.46 | 32.97 | 23 | 0.3942 / 0.4079 | 0.6338 / 0.6346 |

At c208 INT4 request throughput was 0.4% below BF16, not a gain; token throughput was 2.7% lower. Its p95 E2E was 2.3% lower, with similar TTFT/TPOT, but output length was shorter and strict accuracy was 3.3–4.6 percentage points lower; flexible accuracy was 1.4–1.5 points lower. Paired results also had more INT4 regressions than improvements. INT8/FP8 stayed near BF16 accuracy with modestly lower throughput and higher p95 E2E. Across all three GSM loads, INT4 request throughput remained below BF16; do not generalize request-rate gains observed in GovReport to this workload.

## GovReport service and quality by concurrency

Service cells show `req/s / tok/s; p95 TTFT / TPOT / E2E`, with latency units seconds / milliseconds / seconds. Quality columns give paired ROUGE-L mean deltas against BF16 for replay cycle 0 / cycle 1. The BF16 baseline ROUGE-L scores are included for context. These cycles are not independent repetitions.

| Client c | BF16 baseline req/s / tok/s; p95 TTFT / TPOT / E2E | INT8 req/s / tok/s; p95 TTFT / TPOT / E2E | FP8 req/s / tok/s; p95 TTFT / TPOT / E2E | INT4 req/s / tok/s; p95 TTFT / TPOT / E2E | BF16 ROUGE-L c0 / c1 | Paired ROUGE-L Δ INT8 c0/c1 | FP8 c0/c1 | INT4 c0/c1 |
|---:|---|---|---|---|---:|---:|---:|---:|
| 8 | .332 / 218.3; 2.79 / 41.7 / 37.58 | .322 / 218.7; 3.36 / 42.0 / 38.30 | .326 / 220.9; 3.30 / 41.5 / 38.69 | .370 / 208.2; 3.19 / 46.4 / 39.06 | .32143 / .31860 | +.00076 / +.00001 | +.00029 / +.00694 | -.01785 / -.01265 |
| 16 | .473 / 313.4; 3.86 / 58.2 / 52.16 | .470 / 315.8; 3.66 / 60.9 / 53.57 | .464 / 312.1; 3.70 / 59.7 / 53.75 | .519 / 285.4; 3.56 / 67.0 / 58.70 | .32102 / .32016 | +.00140 / -.00030 | -.00216 / +.00040 | -.01284 / -.01444 |
| 32 | .557 / 368.7; 19.40 / 87.2 / 86.20 | .512 / 337.5; 9.78 / 112.6 / 96.13 | .511 / 340.7; 12.40 / 110.7 / 98.24 | .552 / 308.4; 11.60 / 122.9 / 101.77 | .31951 / .32120 | -.00032 / -.00352 | +.00222 / +.00043 | -.01401 / -.01133 |
| 64 | .556 / 367.9; 81.47 / 89.0 / 146.71 | .501 / 332.8; 78.94 / 130.8 / 178.15 | .497 / 335.0; 75.92 / 127.9 / 178.33 | .569 / 319.9; 48.94 / 164.8 / 177.46 | .31986 / .32067 | -.00047 / -.00239 | -.00054 / +.00377 | -.01447 / -.01294 |
| 128 | .557 / 371.9; 192.48 / 89.3 / 264.77 | .495 / 333.6; 205.12 / 133.0 / 308.01 | .491 / 333.3; 204.53 / 130.3 / 303.27 | .571 / 323.0; 170.68 / 164.2 / 288.29 | .32035 / .32268 | -.00106 / -.00110 | +.00167 / -.00035 | -.01440 / -.01539 |

INT8 and FP8 generally retained ROUGE-L near BF16, but service performance separated at larger loads: at c64/c128 both lost roughly 9–10% generated-token throughput and had substantially higher TPOT/E2E, while c8/c16 were closer. Their quality deltas vary by replay cycle, so small gains are not stable enough to claim as quality improvements.

INT4 showed real request-rate gains in GovReport at c8 (+11.3%), c16 (+9.6%), c64 (+2.4%), and c128 (+2.6%); c32 was slightly below BF16. Those gains came with shorter outputs (about 14–17% lower mean generated length) and lower token throughput (about 4.6–13.2% lower). p95 TTFT improved at c16 and c32–c128, while TPOT and E2E were worse at every load. INT4 ROUGE-L was lower in both replay cycles at all five loads, by about 0.011–0.018, and paired regressions exceeded improvements. The request-rate increases are real measurements but do not establish a quality-preserving user benefit.

## Archive and outstanding caveats

The formal jobs and reports have `completed_with_report`, exit 0; the E archive includes all response, telemetry, summary, environment, and report artifacts. GSM traces are 1,319 prompts and GovReport traces are 200 prompts, each replayed twice. Do not combine the distinct GSM and GovReport protocols or treat replay cycles as independent statistical repetitions. Hardware cause/ceiling analysis is intentionally left to the separate controlled-kernel work.
