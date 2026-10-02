# INT4 KV cache 模型质量评测（2026-10-02）

Copyright 2026 The xLLM Authors.

## 结果摘要

在当前 Python eager INT4 serving 实现下，模型成功完成 GSM8K 全部 1319 题，以及 LongBench 八任务各 50 题。生成请求均有持久化记录；LongBench 正式 corrected attempt 与 BF16 的配对校验通过。质量明显退化：GSM8K 严格准确率为 0/1319（0.00%），宽松准确率为 14/1319（1.06%）；LongBench 八项分数均大幅低于现有 BF16 结果。该实验只表明当前完整 serving 路径下观察到的质量，不足以把退化归因于 INT4 格式本身，也未排除实现/kernel 问题。

| 数据集/任务 | 样本数 | BF16 分数 | INT4 分数 | INT4 生成上限 | INT4 `finish_reason=length` |
|---|---:|---:|---:|---:|---:|
| GSM8K strict EM | 1319 | 575/1319 (43.59%) | 0/1319 (0.00%) | 512 | 1319/1319 |
| GSM8K flexible EM | 1319 | 856/1319 (64.90%) | 14/1319 (1.06%) | 512 | 1319/1319 |
| NarrativeQA | 50 | 0.19985 | 0.00469 | 128 | 48/50 |
| Qasper | 50 | 0.36829 | 0.00477 | 128 | 49/50 |
| MultifieldQA-en | 50 | 0.43864 | 0.00401 | 64 | 50/50 |
| 2WikiMQA | 50 | 0.30219 | 0.00000 | 32 | 49/50 |
| HotpotQA | 50 | 0.40762 | 0.00250 | 32 | 50/50 |
| MuSiQue | 50 | 0.14566 | 0.00769 | 32 | 50/50 |
| LCC | 50 | 0.45880 | 0.05740 | 64 | 50/50 |
| RepoBench-P | 50 | 0.46240 | 0.07260 | 64 | 50/50 |

LongBench 合计 397/400 条以输出上限结束。GSM8K 的 1319 条也全部生成满 512 token；输出中可见重复/退化文本，因此其生成吞吐不能解读为有用答案吞吐或服务收益。

| 正式结果对照 | GSM8K strict EM | GSM8K flexible EM | LongBench 八任务 macro |
|---|---:|---:|---:|
| BF16 | 575/1319 (43.59%) | 856/1319 (64.90%) | 0.34793 |
| INT8 | 473/1319 (35.86%) | 784/1319 (59.44%) | 0.37509 |
| 本次 INT4 | 0/1319 (0.00%) | 14/1319 (1.06%) | 0.01921 |

LongBench macro 是上表八个任务官方 metric 等权算术平均；按 LongBench 的任务 metric 定义复算旧 BF16/INT8 paired scores。GSM8K strict/flexible 是 evaluator 输出的答案匹配计数。

## 协议与可比性

- 模型为本地 `Qwen2.5-1.5B-Instruct`，BF16 权重、INT4 KV cache；checkpoint SHA256 为 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`。INT4 使用 Python model implementation、eager 路径，graph 关闭。服务端 `max_seqs_per_batch=30`（LongBench）或 64（GSM8K），客户端请求并发相同；max memory utilization 为 0.87、max tokens per batch 为 4096。
- GSM8K 使用既有正式 prepared 数据合并为源顺序 0–1318 的完整集合；未重采样或重新生成题目。沿用 5-shot prompt、seed 17、temperature 0、max-new 512、官方 stop strings；请求未显式传 stop token IDs，服务端默认 EOS 为 151645。
- LongBench 使用 manifest `2ae099ff6607c64cec7bb1026d19f1ee787be5f9d9a9edf99b32430f6fea3695` 指向的八任务 prepared n=50 输入；按官方 revision `2e00731f8d0bff23dc4325161044d0ed8af94c1e` 评分。每任务 seed 17、temperature 0、stop IDs `[151645,151643]`、无 stop strings，按任务使用 32/64/128 的官方输出上限。BF16 配对逐样本输入及 prompt-token 校验均匹配。
- GSM8K 历史 BF16/INT8 结果使用既有正式输出；其中旧 BF16/INT8 的首 50 题配置额外显式包含 stop ID 151643，而 INT4 本次用默认 EOS 151645。余下 1269 题旧配置的 stop ID 为 null，与本次一致。未为消除这项历史配置差异而重跑 BF16/INT8，故整套 GSM 分数是正式结果对照，但不是每一题完全相同 stop 元数据的严格配对实验。
- LongBench 旧 BF16/INT8 的执行并发与本次不同（BF16 C8、INT8 C16、INT4 C30）；加之 attention 实现路径不同，质量可按相同输入作观察性对照，吞吐不能作纯精度或并发因果比较。
- 首次 LongBench NQA attempt 因遗漏显式 stop IDs 被保留在单独目录并标记 excluded；该 attempt 没有 durable 样本，不计入以上 400 条 corrected 结果或评分。

## 生成耗时与吞吐边界

下表的生成 token 数是实际 completion token 总数；时间窗为本任务所有请求中最早 `request_started_utc` 至最晚 `request_finished_utc`，包含并发请求的重叠时间。吞吐为该总 token 数除以此 HTTP 请求墙钟窗，不是单请求解码速度。各任务依次执行，汇总墙钟窗为各任务窗之和；不包含模型启动、任务间准备和离线评分。延迟为客户端记录的逐请求延迟分位数。

| LongBench 任务 | 生成 token | 请求墙钟窗 (s) | completion tok/s | client latency p50 / p95 (s) |
|---|---:|---:|---:|---:|
| NarrativeQA | 6289 | 1676.1 | 3.75 | 1238.0 / 1644.9 |
| Qasper | 6356 | 185.9 | 34.20 | 125.6 / 165.9 |
| MultifieldQA-en | 3200 | 275.5 | 11.61 | 165.7 / 256.3 |
| 2WikiMQA | 1579 | 336.9 | 4.69 | 182.2 / 279.9 |
| HotpotQA | 1600 | 746.3 | 2.14 | 413.0 / 580.8 |
| MuSiQue | 1600 | 1004.6 | 1.59 | 546.0 / 705.2 |
| LCC | 3200 | 89.3 | 35.84 | 51.2 / 78.0 |
| RepoBench-P | 3200 | 611.6 | 5.23 | 312.7 / 598.0 |
| **合计（逐任务窗求和）** | **27024** | **4926.2** | **5.49** | — |

GSM8K 本次实际生成 675,328 token；最早请求开始至最晚请求结束为 737.6 秒，对应 915.5 completion tok/s；逐请求 client latency p50/p95 为 34.40/36.85 秒。所有 GSM 输出都撞到 512 token 上限。以上仅描述本次请求时间窗；满上限的重复退化输出不能视为有效服务收益。

## 解释边界与后续定位

当前观察到的退化机制尚未确定。候选问题包括 KV codec 的数值误差、packed 存取/scale 布局或 attention kernel 行为；不能仅凭任务分数判断是哪一种。当前 serving 使用 per-token/per-head 对称 ±7 单 scale 量化，可能受离群值影响，但仍需对真实 RoPE 后 K/V 验证。仓库已有独立 codec lab `tools/kv_cache_codec_lab.py` 支持 rotation/outlier 变体，但这套探索工具没有接入当前 serving 路径。CPU 检查发现若干层的 K projection bias 存在较大 head 内离群分量；bias 不是实际 RoPE 后 K 分布的替代品，不能据此断言因果或排除 kernel bug。

速度对比尤其不能解释为纯 bit-width 效应：当前 INT4 prefill 使用 BF16 query，但点积走 FP32/TF32x3 路径且 `grouped_heads=1`；INT8 则走 BF16 hi/lo tensor-core 路径并支持 GQA6。两者并非同一 attention kernel 框架下只改变 KV 位宽的对照。本报告未修改生产 kernel，也未运行额外 GPU 正确性实验。

## 结果文件

完整结果根目录：`/mnt/e/AI/xllm-eval-data/int4-model-eval-20261002`。GSM 原始逐题输出为 `gsm8k/int4/int4.jsonl`，LongBench 每任务输出位于 `longbench/<task>/int4_corrected/int4.jsonl`，配对评分为 `longbench/<task>/paired_score_corrected/paired_scores.json`。被排除的首次 LongBench attempt 仍单独保留，未删除、未混入 corrected 输出。实验服务已停止。
