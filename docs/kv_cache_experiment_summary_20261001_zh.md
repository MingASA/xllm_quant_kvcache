# KV Cache 量化实验阶段总结

日期：2026-10-01（最新 INT8 kernel 服务 smoke 时间为 UTC 2026-10-02；对应本地 PDT 仍为 2026-10-01）。本轮实验已暂停，相关服务已关闭；本文只汇总现有结果，不代表新增实验或生产验收。

## 结论

缓存容量收益明确：在本次 per-token/head scale 布局中，INT8/FP8 的 KV 实际占用约为 BF16 的 1/1.94，即节省约 48.4%；packed INT4 约为 BF16 的 1/3.76，即节省约 73.4%。这是 KV cache payload 与 scale 的容量比较，不是进程总 HBM、整机峰值显存或可承载服务并发的等比例变化。

完整 GSM8K 配对评测显示，旧版 INT8 路径低于 BF16：Strict EM 低 7.73 个百分点，Flexible EM 低 5.46 个百分点。LongBench 八任务抽样的 INT8 macro 高 2.716 分，但样本、并发配置和任务指标均有限，不能据此宣称量化提升模型能力。最新 kernel 对特定长上下文混合 prefill/decode 形状有显著单层提速；纯 decode 与短 prefill未见收益，且尚未重跑完整 LongBench/GSM8K。现有证据支持继续把它视为实验性路径，不足以宣称生产成熟或默认替换 BF16。

## 主要结果

| 项目 | BF16 | INT8 | FP8 E4M3 | FP8 E5M2 | INT4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| KV cache（MiB） | 64 | 33 | 33 | 33 | 17 |
| allocated 峰值（MiB） | 164.14 | 133.39 | 133.39 | 133.39 | 117.77 |
| 相对 setup 增量（MiB） | 3.26 | 3.51 | 3.51 | 3.51 | 3.89 |
| attention 输出 relative L2 | 0% | 0.95% | 3.75% | 7.43% | 16.96% |

上表是 context 4096、batch 4、query 16、Hkv=8、head dim=128 的历史 Python 单层合成原型。峰值包含输入张量与 allocator 分配，不是整机/模型服务 HBM；模型权重、运行时其他缓存和服务 workspace 不在此表。误差是固定随机输入下相对 BF16 的 attention 输出误差，不是生成质量。它与后续融合、最新 kernel 属于不同阶段，不能相互替代。详见[合成基准](kv_cache_benchmark_zh.md)及[原始结果](../results/20260929T080019Z_de1782a3)。

| GSM8K 指标（1,319 题/臂） | BF16 | INT8 | INT8−BF16 |
| --- | ---: | ---: | ---: |
| Strict EM | 575/1,319（43.59%） | 473/1,319（35.86%） | −7.73pp |
| Flexible EM | 856/1,319（64.90%） | 784/1,319（59.44%） | −5.46pp |

质量对照发生在最后一轮 kernel 优化前，不能把差异归因于单一因素。GSM8K 分阶段并发为前 50 题 serial、50–949 为 BF16 C10/INT8 C16、950–1318 为 C32/C50；该运行速度不是统一并发对照。完整配置见[GSM8K 报告](kv_cache_gsm8k_full_20261001_zh.md)。

下表使用 Qwen 的真实 attention 形状（Hq=12、Hkv=2、D=128、page size=128），时间为 CUDA event p50。混合负载和纯 decode 的 batch 为 16，纯 prefill 的 batch 为 1。4K 负载预热 10 次、采样 50 次；长上下文预热 2 次、采样 5 次。

| 最新 INT8 kernel 单层负载 | 旧 ms | 新 ms | 加速 |
| --- | ---: | ---: | ---: |
| 4K mixed `[1024]+15×[1]` | 37.455 | 2.965 | 12.63× |
| 16K mixed `[4096]+15×[1]` | 600.806 | 33.799 | 17.78× |
| 32K mixed `[4096]+15×[1]` | 1198.355 | 70.989 | 16.88× |
| 16K 纯 prefill Q4096 | 40.457 | 31.826 | 1.27× |
| 4K 纯 prefill Q128 | 0.498 | 0.517 | 无明显改善 |
| 4K 纯 decode Q1 | 0.404 | 0.431 | 无改善 |

混合负载优化针对空 query tile 仍扫描全 cache，消除填充 tile 的空扫描，裁剪 causal/window 范围，并增加 GQA=3/6 复用及 64-key 跨页读取。旧新使用相同 Q/K/V、布局和 cache hash；缓存占用及量化规则不变。输出差异 max abs 0.0004883、relative L2≤9.87e-5（相对旧 INT8，不是 BF16）。165 项量化测试全部通过，含新增 GQA=6、混合 query 和 NaN 尾页回归。见[kernel 报告](int8-kernel-optimization-20261001.md)。

| 旧版 LongBench 服务窗口 | 请求并发 | 窗口秒 | generated tokens | tok/s |
| --- | ---: | ---: | ---: | ---: |
| BF16 | C8 | 404 | 11,526 | 28.51 |
| INT8 | C16 | 3,920 | 10,146 | 2.59 |

两臂并发与生成长度不同，吞吐只能描述此次配置，不能单独归因于 KV 格式或 kernel。见[LongBench 报告](longbench-n50-bf16-int8-20261001.md)；该质量实验为每臂 400 条，旧服务并发为 BF16 C8/INT8 C16。逐任务分数见下表。

优化后 NarrativeQA C16 smoke：16/16 完成，200 tokens/140.05 秒（1.43 tok/s），p50/p95 105.83/139.49 秒；只验证新路径与 prompt token 数匹配，不提供质量结论。它不是旧 LongBench 的等负载比较。400 条 LongBench 尚未用新 kernel 重跑；单层 query-token/s 也不是模型生成 token/s。服务记录位于 `/mnt/e/AI/xllm-eval-data/int8-kernel-optimization-20261001/20261002T043000Z_narrativeqa16_int8_smoke/`。

历史 FlashInfer 对照在 context 4096、B4、Q16 的单层实验中，INT8 full-call 优化前→候选2为 0.9982→0.5983 ms；同期 FlashInfer BF16 为 0.3734 ms，INT8 仍慢 1.60×。最新 kernel 未与 FlashInfer 重测，不能据此声称已经追平。详情见[kernel 复测报告](kv_cache_kernel_optimization_zh.md)。

## LongBench 逐任务分数

以下是各任务 50 条样本的官方对应指标分数（报告中的小数乘 100 展示）；QA 任务为 QA F1，lcc 与 repobench-p 为代码相似度。混合指标的等权平均只作本次抽样摘要，不是 LongBench 官方全套总分。NarrativeQA 中 23/50 条输入对 context 做了头尾截断，使 prompt 与生成预算之和不超过 32,704 token；两臂使用完全相同的处理后输入，其余七项无需截断。

| 任务 | 指标 | BF16 | INT8 | INT8−BF16 |
| --- | --- | ---: | ---: | ---: |
| narrativeqa | QA F1 | 19.985 | 23.243 | +3.258 |
| qasper | QA F1 | 36.829 | 36.804 | −0.025 |
| multifieldqa_en | QA F1 | 43.864 | 48.592 | +4.728 |
| 2wikimqa | QA F1 | 30.219 | 30.458 | +0.239 |
| hotpotqa | QA F1 | 40.762 | 46.266 | +5.504 |
| musique | QA F1 | 14.566 | 19.089 | +4.523 |
| lcc | 代码相似度 | 45.880 | 49.060 | +3.180 |
| repobench-p | 代码相似度 | 46.240 | 46.560 | +0.320 |
| 八任务等权 macro | 混合指标 | 34.793 | 37.509 | +2.716 |

GSM8K 配对翻转按最终分数核对为：Strict BF16-only 207、INT8-only 105；Flexible BF16-only 173、INT8-only 101。差异方向与总分一致；逐题翻转不改变宏观结论。

## 阶段与适用范围

环境为 Qwen2.5-1.5B-Instruct，RTX 5060 Ti 16 GB；模型权重为 BF16，量化仅作用于 KV cache，并非权重量化。原生 BF16、Python BF16、Python INT8 服务路径均有成功运行记录，但 smoke 成功仅说明对应配置可工作。早期单层原型的量化误差不能用作后续 kernel 数值差异；最新新旧 kernel 输出对照也只是相对旧 INT8。

容量证据成立，INT8 在质量上有 GSM8K 回退信号，LongBench 小样本结果则任务间混合且不支持能力提升结论。最新优化改善了混合长 prefill 负载的 attention kernel，但尚未证明追平 FlashInfer BF16，也未证明相同完整 HBM 预算下的服务容量或端到端吞吐收益。FP8/INT4 没有完整模型质量评测；多格式量化仍是实验路径。基于这些边界，本轮结论限定为“容量节省明确、性能收益依负载而定、质量与生产服务价值尚未闭环”。
