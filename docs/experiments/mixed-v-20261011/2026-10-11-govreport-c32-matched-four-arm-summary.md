# GovReport matched service group: client concurrency 32

Source: `results/flashinfer-govreport-seq256-20261011/formal-two-passes/report/matched_c32_analysis.json`. Four arms each completed the same 200 GovReport prompts × 2 cycles (400 requests), normal EOS, named output-cap1024 extension, token batch/chunk10,240. BF16 is the intentionally reused earlier baseline at original seq128; quant arms use seq256. Since every arm's observed running peak was 32, neither configured seq ceiling bound this load, but run time and exact scheduler setting differ. Quality scoring remains pending.

| Arm | req/s | tok/s | mean output | p95 TTFT / TPOT / E2E | sampled KV mean / max | wait mean / max | length-exhausted |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 auto | .5568 | 368.7 | 662.1 | 19.40s / 87.2ms / 86.20s | 84.6 / 92.4% | 2.97 / 30 | 35/400 |
| INT8 V | .5117 | 337.5 | 659.5 | 9.78s / 112.6ms / 96.13s | 70.5 / 88.8% | .56 / 29 | 38/400 |
| FP8 E4M3 V | .5108 | 340.7 | 666.9 | 12.40s / 110.7ms / 98.24s | 70.4 / 90.0% | .59 / 29 | 40/400 |
| INT4 V | .5523 | 308.4 | 558.4 | 11.60s / 122.9ms / 101.77s | 59.1 / 75.7% | .61 / 30 | 34/400 |

All arms ran with 32 observed peak in-flight service requests; sampled preemptions were zero. Per-arm comparisons and scheduler admission context are documented in the adjacent `govreport-c32-{int8,fp8,int4}-vs-reused-bf16.md` notes. Summary: no quantized arm exceeded BF16 request throughput; INT4 nearly matched requests/s but generated fewer tokens and had the slowest p95 TPOT/E2E. INT8/FP8 had lower p95 TTFT but also lower request/token throughput and higher TPOT/E2E. Do not treat shorter INT4 output as a quality-equivalent speedup until scoring finishes. Source scheduler performs full-prompt-footprint admission for fresh chunked-prefill requests (`xllm/core/scheduler/scheduler_policy.cpp:279-299`); saved free-block samples alone do not rule out admission waits. No admission-denial counter was captured, so that remains a possible explanation for TTFT differences, not a confirmed cause.

## 原因判断（root）

BF16 running采样均值28.16、waiting均值2.97；INT8/FP8/INT4 running均值30.77/30.76/30.63、waiting0.56/0.59/0.61。在同客户端32下，量化确实关联更多请求处于运行而非等待，以及较低TTFT，属于观测到的首token等待改善。但吞吐未上升、TPOT和E2E更差，不能称整体服务加速。平均运行批次增大本身可加长每步执行；量化读取/解量化、prefill/decode交错与不同请求上下文分布也可能贡献。没有隔离每步/每kernel的测量，无法把TPOT增加全部归因解量化。INT4输出缩短约16%，须与ROUGE结果合读。

决定：继续同配置64/128验证饱和负载的服务取舍，不调token预算/不固定生成长度；独立评分后台进行，结果到齐后补此文档。原BF16seq128与quantseq256差异保留并披露，本档均未碰序列上限；复用基线不能消除跨时段运行噪声。

### Scored quality update

The official ROUGE-L scorer finished both full cycles (200 examples per cycle, 400 matched items/arm total). These cycles are the two replays of the same trace, **not two independent repetitions**; keep per-cycle distributions separate. Mean ± item-level standard deviation, and paired per-item Δ vs BF16 auto:

| Arm | Cycle 0 ROUGE-L mean ± SD | Cycle 0 paired Δ | Cycle 1 ROUGE-L mean ± SD | Cycle 1 paired Δ |
|---|---:|---:|---:|---:|
| BF16 auto | .3195 ± .0563 | — | .3212 ± .0563 | — |
| INT8 V | .3192 ± .0586 | −.0003 | .3177 ± .0564 | −.0035 |
| FP8 E4M3 V | .3217 ± .0564 | +.0022 | .3216 ± .0575 | +.0004 |
| INT4 V | .3055 ± .0536 | −.0140 | .3099 ± .0570 | −.0113 |

Interpret with the service table and the reused BF16/seq128 versus quant/seq256 note above. INT4 has materially shorter outputs; its lower token rate is not evidence of a quality-preserving speedup. Per-cycle score evidence is in `report/govreport_items_c32_cycle0.jsonl` and `...cycle1.jsonl`; the scorer remains active for later concurrency groups.
