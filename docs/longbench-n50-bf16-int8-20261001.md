# LongBench KV cache 八任务实验（2026-10-01）

两种配置均完成八个任务、每任务抽样 50 条：每臂共 400 条配对结果，每个任务 ID 均唯一；配对评分核验了所有记录和 prompt token 数。请求无失败、无客户端超时。推理结束后已关闭本次启动的 xLLM 服务。

结果目录：[`/mnt/e/AI/xllm-eval-data/longbench-n50-bf8-int16-20261001/`](/mnt/e/AI/xllm-eval-data/longbench-n50-bf8-int16-20261001/)。准备数据：[`seed17_n50_contextcap32704/`](/mnt/e/AI/xllm-eval-data/longbench-5e628be450b7e67fb7ae6e201bd6d8f7056f7672/prepared/seed17_n50_contextcap32704/)；manifest SHA256 为 `2ae099ff6607c64cec7bb1026d19f1ee787be5f9d9a9edf99b32430f6fea3695`。

## 配置与结果解读

- 模型为 Qwen2.5-1.5B-Instruct，使用 Python 模型实现；两臂 checkpoint SHA256 相同：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`。实验使用一张 RTX 5060 Ti 16 GB GPU。
- BF16 KV cache 的请求并发数 / 最大批内序列数为 `8/8`；INT8 KV cache 为 `16/16`。两臂均关闭 graph，`max_tokens_per_batch=4096`，显存利用率 0.87，使用自动 cache 预算；采样参数相同（`temperature=0`、`top_p=1`、`seed=17`、stop IDs `151645` 和 `151643`）。保留 LongBench 官方各任务输出长度设置。
- 下表 macro 是这八个抽样任务各自平均分的等权平均。LongBench QA 任务使用官方 QA F1；lcc 和 repobench-p 使用代码相似度。由于汇总了不同指标，这不是 LongBench 全套官方总分。
- NarrativeQA 的 50 条中有 23 条仅对 context 做 head/tail 截断，以满足 `prompt_tokens + task_max_output <= 32,704`；其他七项未截断。因此这是有长度上限的抽样实验，不是公开 LongBench 全量复现。
- HTTP 窗口按每任务最早请求开始至最晚请求结束计算，时间为本地 PDT。窗口吞吐为该任务生成 token 总数除以该墙钟时间；p50/p95 为逐请求客户端时延。两臂并发数不同，速度结果只反映本次两种运行配置，不能据此单独归因于 KV 格式或 kernel。
- 已确认的 kernel 性能限制：Qwen GQA ratio 为 6，而 Triton prefill 的 grouped-head 优化只覆盖 1、2、4、8，因此回落到 `grouped_heads=1`，没有跨 6 个 query heads 复用 K/V tile。这是已确认的限制，但未单独证明它解释了全部慢速现象。

## 官方逐任务评分

| 任务（每项 50 条） | 指标 | BF16 KV | INT8 KV |
| --- | --- | ---: | ---: |
| narrativeqa | QA F1 | 0.19985 | 0.23243 |
| qasper | QA F1 | 0.36829 | 0.36804 |
| multifieldqa_en | QA F1 | 0.43864 | 0.48592 |
| 2wikimqa | QA F1 | 0.30219 | 0.30458 |
| hotpotqa | QA F1 | 0.40762 | 0.46266 |
| musique | QA F1 | 0.14566 | 0.19089 |
| lcc | 代码相似度 | 0.45880 | 0.49060 |
| repobench-p | 代码相似度 | 0.46240 | 0.46560 |
| 八任务等权 macro | 混合任务指标 | **0.34793** | **0.37509** |

400 条逐样本配对分数中，INT8 分数较高 80 条、较低 67 条、持平 253 条。小样本且并发不同，不应将 macro 差异解释为 INT8 KV 格式本身更优。

## 推理窗口与时延

窗口均为本地 PDT。窗口 tok/s 用生成 token 总数除以任务 HTTP 墙钟窗口；p50/p95 单位为秒。

| 配置 | 任务 | HTTP 窗口（PDT） | 生成 tokens | 窗口 tok/s | p50 / p95 时延（秒） |
| --- | --- | --- | ---: | ---: | ---: |
| BF16 C8 | narrativeqa | 19:19:42–19:21:25 | 738 | 7.13 | 12.28 / 37.87 |
| BF16 C8 | qasper | 19:21:32–19:21:53 | 995 | 48.49 | 2.39 / 9.14 |
| BF16 C8 | multifieldqa_en | 19:21:59–19:22:29 | 1,582 | 53.03 | 4.21 / 9.28 |
| BF16 C8 | 2wikimqa | 19:22:35–19:23:09 | 555 | 16.34 | 4.25 / 13.73 |
| BF16 C8 | hotpotqa | 19:23:16–19:24:19 | 602 | 9.57 | 8.50 / 17.90 |
| BF16 C8 | musique | 19:24:27–19:25:46 | 654 | 8.27 | 11.10 / 22.46 |
| BF16 C8 | lcc | 19:25:52–19:26:11 | 3,200 | 165.58 | 2.92 / 3.89 |
| BF16 C8 | repobench-p | 19:26:18–19:27:13 | 3,200 | 57.95 | 8.06 / 13.09 |
| INT8 C16 | narrativeqa | 19:29:02–19:45:45 | 558 | 0.56 | 265.67 / 534.61 |
| INT8 C16 | qasper | 19:45:51–19:49:20 | 987 | 4.72 | 60.52 / 150.38 |
| INT8 C16 | multifieldqa_en | 19:49:26–19:56:03 | 1,144 | 2.88 | 84.76 / 289.41 |
| INT8 C16 | 2wikimqa | 19:56:09–19:59:31 | 289 | 1.43 | 62.66 / 76.20 |
| INT8 C16 | hotpotqa | 19:59:38–20:07:03 | 356 | 0.80 | 127.85 / 211.67 |
| INT8 C16 | musique | 20:07:11–20:17:02 | 412 | 0.70 | 170.21 / 262.42 |
| INT8 C16 | lcc | 20:17:07–20:18:34 | 3,200 | 36.88 | 25.79 / 38.25 |
| INT8 C16 | repobench-p | 20:18:41–20:35:07 | 3,200 | 3.24 | 345.05 / 479.83 |

按各任务 HTTP 窗口跨度求和，BF16 共生成 11,526 tokens、窗口 404 秒（28.51 tok/s）；INT8 共生成 10,146 tokens、窗口 3,920 秒（2.59 tok/s）。INT8 C16 下长 prefill（尤其 NarrativeQA 与 RepoBench prompt）伴随较长时延；并发配置和 GQA6 分组限制是已知背景，实验没有隔离其各自因果贡献。

官方 LongBench 评分代码版本为 `2e00731f8d0bff23dc4325161044d0ed8af94c1e`。逐任务评分文件位于 `scores/<task>/paired_scores.json`；逐题原始结果和配置位于 `bf16/<task>/`、`int8/<task>/`。
