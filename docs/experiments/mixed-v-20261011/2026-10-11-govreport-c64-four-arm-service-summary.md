# GovReport c64: four-arm service observations

Completed cells are 400 requests each (200 frozen prompts × 2 cycles), client concurrency 64, normal EOS and output cap 1024. This summary contains completed service metrics and official paired ROUGE-L scores (updated below). GPU time attribution remains an independent diagnostic; no profiling data enters these service metrics.

| Arm | req/s | tok/s | Mean output tokens | p95 TTFT s | p95 TPOT ms | p95 E2E s | Length finishes / 400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reused BF16 | 0.5555 | 367.9 | 662.3 | 81.47 | 89.0 | 146.71 | 38 |
| INT8 | 0.5008 | 332.8 | 664.4 | 78.94 | 130.8 | 178.15 | 47 |
| FP8 E4M3 | 0.4971 | 335.0 | 673.9 | 75.92 | 127.9 | 178.33 | 44 |
| INT4 | 0.5688 | 319.9 | 562.3 | 48.94 | 164.8 | 177.46 | 40 |

All four recorded no fresh attention compilations; client CPU was 1.76–2.06% of one core, event-loop lag p95/p99 1.52–1.59/2.12–2.16 ms, max 5.08–6.66 ms, and no >20 ms stalls. Telemetry sampled KV-utilization mean/p90/sample-max was: BF16 85.58/90.94/92.17%; INT8 85.58/91.05/93.65%; FP8 85.49/90.95/92.47%; INT4 84.26/90.87/91.77%. Sampled running sequences mean/max: 28.63/37, 37.74/46, 37.34/48, 43.84/54; waiting mean/max: 32.24/61, 23.86/61, 24.22/62, 17.70/61. Sampled free-block minima were 230, 246, 292, and 382; sampled maximum used-block values were 2707, 3631, 3585, and 4261; reported preempt counter remained zero in each. Free/used block counts are allocator-specific and should not be compared as byte capacity across dtypes. Nonzero free blocks and zero preemption do not rule out full-prompt-footprint admission waiting.

Observed pattern: quantized arms had lower p95 TTFT and higher mean running concurrency, but also lower request throughput for INT8/FP8 and lower token throughput plus much higher TPOT/E2E for all quantized arms. INT4's req/s was 2.4% above BF16 while mean output was 15.1% shorter and tok/s 13.1% lower. These are concurrent-service observations, not a causal codec ranking. Reused BF16 retains its original sequence cap 128 while quant arms use 256; observed running peaks stayed below both limits, but the baseline has different provenance/config. Quality and paired flips are intentionally pending until the scorer resumes; do not infer score from response length or finish reason.

## 执行归因补充（主线程）

INT4 req/s比BF16约高2.4%，但平均输出562而非662（−15.1%），p95 E2E却高约21%、TPOT高约85%。不能把完成更多较短响应当成同等服务体验提升。各量化arm确实降低排队（INT4 waiting均值32.2→17.7，TTFT约−40%），然而后续生成等待变长，最终E2E不改善。

同构建同形状算子计时显示长上下文decode attention量化更快约9%～14%，而长query prefill量化更慢约16%～19%。真实INT8每步token工作更大，因此服务变慢应分解为实际批次/混批组成变化、prefill成本、writer/attention与其他模型工作；不能以解量化一句话解释。详见 `2026-10-11-quantization-service-time-attribution.md`。

两次Nsight均缺少GPU kernel事件（硬件和软件trace都失败），不采信其GPU耗时占比。正式已恢复，隔离的CUDA event配对诊断已准备，将在下一有效完整cell后进行；它带有插桩开销，结果仅用于归因，不能替代正式服务指标。

## Official ROUGE-L scoring update

Scoring completed at 2026-10-11 12:32:06 UTC; both cycles contain 200/200 rows for all four arms. Values are mean ± population SD; paired Δ is the per-example ROUGE-L difference from BF16 on that same cycle.

| Arm | Cycle 0 | Paired Δ vs BF16 | Cycle 1 | Paired Δ vs BF16 |
|---|---:|---:|---:|---:|
| BF16 | 0.31986 ± 0.05366 | — | 0.32067 ± 0.05972 | — |
| INT8 | 0.31939 ± 0.05634 | −0.00047 (95 up / 104 down / 1 tie) | 0.31828 ± 0.05552 | −0.00239 (94 / 104 / 2) |
| FP8 E4M3 | 0.31932 ± 0.05599 | −0.00054 (98 / 101 / 1) | 0.32444 ± 0.05896 | +0.00377 (109 / 89 / 2) |
| INT4 | 0.30539 ± 0.05427 | −0.01447 (64 / 135 / 1) | 0.30773 ± 0.05457 | −0.01294 (72 / 128 / 0) |

This confirms INT4's paired quality regression on both cycles, while INT8 is near baseline and FP8 is mixed across the two cycles. It does not alter the service measurements above or establish quality causation.
