# Qwen2.5-1.5B-Instruct KV Cache smoke 报告

## 结论范围

已完成 Python executor 的 LongBench 39 题 auto/INT8 对照和按官方停止条件修正后的 GSM8K 10 题 auto/INT8 对照。各自使用相同 BF16 checkpoint、数据行、prompt、seed 和生成配置；配对内只有 KV cache 模式不同。LongBench 与 GSM8K 是两套独立结果，共 49 个评测样例。它们是固定小样本的描述性对照，不是正式 benchmark，也不能用来声称 INT8 有普遍质量提升或复现论文结果。LongBench 汇总仅作八任务等权的样本内摘要，不按任务题数加权来复现论文表格；GSM8K 单独报告，不与 LongBench 合并。

首个请求包含服务启动后的冷编译/初始化开销；本次没有设计稳态吞吐或 KV 容量实验，不据此报告延迟、吞吐或容量结论。

## 实验配置与核验

- 模型：`/mnt/e/AI/models/Qwen2.5-1.5B-Instruct`，Python executor，两臂模型权重均为 BF16；KV cache 分别为 `auto` 与 `int8`；CUDA graph、prefill piecewise graph 和 Python graph backend 均关闭。服务绑定 `127.0.0.1:18994`，每次只运行一个服务。
- Checkpoint 权重 SHA256：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`。两份服务 manifest、config SHA 和 chat-template SHA 一致。
- 两臂请求统一 `temperature=0`、`top_p=1`、`seed=17`，显式 `stop_token_ids=[151645,151643]`。LongBench 使用官方各任务 prompt、最大生成长度和 `metrics.py`；对参考答案取最高分。LCC 与 RepoBench-P 使用 `/v1/completions`，其余任务及 GSM8K 使用 `/v1/chat/completions`。GSM8K 另按官方 lm-evaluation-harness `until` 传入 stop 字符串 `Question:`、`</s>`、`<|im_end|>`。
- LongBench 数据来自固定 revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，使用每个任务按源文件顺序的前 5 行，并按输入加最大输出不超过 32,768 token 筛选：narrativeqa 4 题，其余 7 项各 5 题，共 39 题。修正协议后的 GSM8K 为固定 5-shot 下官方 test 前 10 题。总计 LongBench 39 + GSM8K 10。
- 所有运行启用 `--strict-token-count`；LongBench chat/completion 短题预检与 GSM8K 完整 10 题的 API/local prompt token 数均匹配。离线配对评分启用 strict 检查并核对 task、model、checkpoint/server manifest、输入 SHA、逐题记录 SHA、prompt SHA 和 generation 参数；没有配对不匹配。

## LongBench 结果

分数为 0–100 百分制，flip 顺序为 auto→int8。`文本相同` 是逐题 prediction 字符串完全相同的数量。

| 任务 | n | auto | int8 | 差值 | up/down/tie | 文本相同 |
|---|---:|---:|---:|---:|---:|---:|
| narrativeqa | 4 | 13.39 | 13.92 | +0.53 | 1/1/2 | 1/4 |
| qasper | 5 | 39.55 | 37.84 | -1.71 | 0/2/3 | 3/5 |
| multifieldqa_en | 5 | 45.33 | 52.49 | +7.15 | 2/1/2 | 2/5 |
| 2wikimqa | 5 | 1.60 | 20.00 | +18.40 | 1/0/4 | 3/5 |
| hotpotqa | 5 | 14.67 | 14.67 | +0.00 | 0/0/5 | 3/5 |
| musique | 5 | 0.00 | 0.00 | +0.00 | 0/0/5 | 1/5 |
| lcc | 5 | 26.80 | 44.80 | +18.00 | 1/1/3 | 2/5 |
| repobench-p | 5 | 32.40 | 46.60 | +14.20 | 1/1/3 | 0/5 |
| **八任务等权均值（描述性）** | **39** | **21.72** | **28.79** | **+7.07** | **6/6/27** | **15/39** |

八任务等权均值不是完整 LongBench 测量，也不能与论文值直接横比；每任务样本极少，局部翻转尤其容易影响均值。

## GSM8K 官方停止条件修正版

首次 GSM8K 10 题运行没有传官方 generation stop strings，模型在多题上继续续写并达到 512-token 上限。因此那一轮的 GSM8K 分数不用于能力/质量比较，只保留为协议失败诊断；原始数据未覆盖，位于旧 LongBench 49 题结果目录的 `gsm8k/` 子目录。修正版在新的 E 盘目录独立重跑 auto 与 INT8 各 10 题，并记录停止条件到 generation config。

| 指标 | auto | int8 | 差值 | up/down/tie |
|---|---:|---:|---:|---:|
| Strict exact match | 40% (4/10) | 20% (2/10) | -20 个百分点 | 0/2/8 |
| Flexible exact match | 70% (7/10) | 50% (5/10) | -20 个百分点 | 0/2/8 |

strict 和 flexible 是不同答案提取规则，不合并成一个分数。逐题 strict 分数保留在 `scores`/`flip`，flexible 结果单独记录在 `flexible_scores`/`flexible_flip`。修正版两臂 10/10 均以 `finish_reason=stop` 结束，生成长度范围分别为 auto 47–342、INT8 52–408 token（均未到 512 上限）；两个 arm 的 prediction 文本逐题完全相同数为 0/10。分数方向在这个小集合中两种抽取均为 2 题 down、8 题 tie，不据此推断普遍退化或提升。

旧协议诊断中，auto 有 4 题 strict 首个 `####` 命中而 flexible 最后数值不命中；逐题重算与 scorer 一致，没有发现 scorer 实现/汇总 bug：

| test 行索引 | 目标答案 | strict 首个 `####` 抽取 | flexible 最后数值 | finish / 输出 token | prediction 末尾摘要 |
|---:|---:|---:|---:|---:|---|
| 1 | 3 | 3 | 12 | length / 512 | 生成转为重复的“昨天读 12 页”故事，并在总页数回答处截断 |
| 3 | 540 | 540 | 12 | length / 512 | 继续复述“昨天 Julie 读 12 页”，未完成当前答案 |
| 5 | 64 | 64 | 5 | length / 512 | 生成另一个 Betty 钱包题的计算，末尾只有孤立 `####` |
| 6 | 260 | 260 | 12 | length / 512 | 重复读书题内容，在 `Answer` 处截断 |

四条输出均是 512 token 截断，说明漏掉官方停止串会使 strict 与 flexible 抽取分歧；这正是旧分数不应作为协议有效 GSM8K 结果的原因。修正版仍只是 10 题 smoke，不能据此声称 INT8 推理能力提升。

## 产物

LongBench 产物位于 `/mnt/e/AI/huggingface/xllm-kv-quality/qwen25-1.5b-smoke-49-20261001T115000Z/`；修正版 GSM8K（应作为唯一有效 GSM8K 分数）位于 `/mnt/e/AI/huggingface/xllm-kv-quality/qwen25-1.5b-gsm8k-official-stops-20261001T141330Z/gsm8k/`。每臂逐题 JSONL 含原始 API 响应、token usage、记录和 prompt SHA，另有 run config、paired score 和服务日志。输入来源、样本 ID 和文件 SHA 在 `/mnt/e/AI/xllm-eval-data/smoke_manifest.json`；实际 auto/int8 manifest 位于 `/mnt/e/AI/xllm-eval-data/auto_server_manifest.json` 与 `int8_server_manifest.json`。manifest 的 `server_started=true` 基于成功 smoke 响应；每次新运行前仍需验证当前服务 ready。

LongBench 评分使用官方本地 checkout revision `2e00731f8d0bff23dc4325161044d0ed8af94c1e`。GSM8K prompt、答案抽取与停止条件来自 lm-evaluation-harness GSM8K 主配置；其官方模型分数和论文 LongBench 均值采用各自完整评测协议，本报告不声称复现它们。

## 独立接入冒烟

native BF16 接入路径也完成一条独立请求，返回 `xLLM smoke test passed.`（8 completion tokens，finish=`stop`）；日志与原始响应为 `build/cmake.cuda-x86/native-bf16-smoke-final.log` 和 `native-bf16-smoke-final-response.json`。该项不是 auto/INT8 KV 配对，也不代表长上下文或 graph 路径验证。Python 服务实验用 FlashInfer `0.6.18.post1`；native 一条请求的成功只说明该次本地依赖/构建配置可完成请求，不能外推到其他依赖组合。
