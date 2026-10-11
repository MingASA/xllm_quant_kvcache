# GSM8K 128 并发（seq256）分析

结论：未观察到整体服务收益。量化吞吐低于BF16，TPOT较高；TTFT或INT4 E2E的小幅改善需与生成长度、质量一起看。

同一模型/FlashInfer后端/冻结1319题两遍/正常EOS；batch与prefill默认10240，scheduler256，接入512，VRAM预算0.93。两个重放不视为独立实验重复。

|格式|req/s|tok/s|TTFT p95(s)|TPOT p95(ms)|E2E p95(s)|平均输出|Strict/Flexible EM|
|---|---:|---:|---:|---:|---:|---:|---:|
|auto|12.031|1864.9|0.631|68.38|22.38|155.0|43.18% / 65.05%|
|bf16_int8|11.755|1820.3|0.587|70.47|22.40|154.8|43.71% / 64.78%|
|bf16_fp8_e4m3|11.723|1810.2|0.600|70.01|22.85|154.4|44.66% / 64.44%|
|bf16_int4|11.748|1798.1|0.610|71.13|21.81|153.1|39.73% / 64.67%|

证据支持的原因：BF16 KV采样峰仅32.3%，抢占0，running峰128，seq256未触顶。没有缓存不足被量化解除的证据。瞬时waiting峰不是稳态排队，不能单凭峰值认定调度瓶颈。

待验证：量化读取/解量化及其他模型计算、权重访存、prefill是否抵消V访存节省；未隔离profile，不宣称唯一原因。

下一步：保留本组无收益结果，结合更高并发和GovReport长上下文较长输出的同档对照；不改EOS、不强制等长生成。

原始依据：/home/mingasa/projects/xllm_quant_kvcache/results/flashinfer-service-seq256-20261011/gsm8k-default-token-budget-seq256-c128-160-208-256/report/matched_c128_analysis.json，以及gsm8k_quality.json。
