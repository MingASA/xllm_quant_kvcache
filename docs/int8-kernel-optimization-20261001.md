# INT8 分页 Attention Kernel 优化实验

## 实验范围

在 RTX 5060 Ti 16 GB 上测量 Qwen2.5-1.5B 的 attention 形状：12 个 Q head、2 个 KV head、head dim 128、仓库默认 page size 128。合成实验只测 attention，不加载模型。CUDA event 与墙钟样本、Q/K/V 和页表哈希、缓存哈希及输出均保存在 [E 盘结果目录](/mnt/e/AI/xllm-eval-data/int8-kernel-optimization-20261001/)。

旧 kernel SHA-256：`e68ec3155ca6ed4ff42cc1f54094135321525167517a5a19dcbfb4fa09cbfbf5`。

实测新 kernel SHA-256：`d8a92f29915d202e74c6efd5954c359fe575eb220a9612b091cdba062a6429e6`。

旧、新实验使用完全相同的 Q/K/V、页表、布局和量化缓存。每组缓存占用及 payload/scale 哈希均相同。输出最大绝对差异为 0.0004883，相对 L2 差异不超过 9.87e-5；这是相对旧 INT8 实现的差异，并非量化相对 BF16 的误差。

## 单层 Kernel 时间与吞吐

以下为 CUDA event p50，括号内为墙钟 p50。4K 负载预热 10 次、采样 50 次；长上下文及纯长 prefill 预热 2 次、采样 5 次。C 表示 batch/concurrency，Q 表示本次 query token 数。

| 负载 | 旧 ms | 新 ms | 变化 |
|---|---:|---:|---:|
| C1, Q128, ctx4K | 0.498 (0.565) | 0.517 (0.566) | 未见明显收益 |
| C16, Q1, ctx4K | 0.404 (0.467) | 0.431 (0.543) | 未见 decode 加速 |
| C16 混合 `[1024] + 15×[1]`, ctx4K | 37.455 (37.664) | 2.965 (3.040) | 快 12.63× |
| C16 混合 `[4096] + 15×[1]`, ctx16K | 600.806 (601.018) | 33.799 (34.023) | 快 17.78× |
| C16 混合 `[4096] + 15×[1]`, ctx32K | 1198.355 (1198.597) | 70.989 (71.167) | 快 16.88× |
| C1, Q4096, ctx16K | 40.457 (40.611) | 31.826 (32.031) | 快 1.27× |

主要问题：旧混合 prefill 的 grid 按批次最长 query 分配，短 query 的填充 tile 仍遍历整份缓存。新 kernel 让无有效 query 的 tile 不再扫描缓存，按因果/滑窗边界裁剪读取范围，支持向量页 ID 与有效性 mask，并以 64-key tile 跨页读取；同时增加 GQA=3/6 的 K/V 复用。纯 decode 本轮没有加速。

按实际处理 query token 数除以 CUDA event p50 换算，单层处理吞吐如下。这不是模型生成吞吐，也不包含缓存写入、MLP 和调度等开销。

| 负载 | 实际 query tokens | 旧 tokens/s | 新 tokens/s |
| --- | ---: | ---: | ---: |
| 4K 混合 | 1,039 | 27,741 | 350,422 |
| 16K 混合 | 4,111 | 6,842 | 121,631 |
| 32K 混合 | 4,111 | 3,431 | 57,910 |
| 16K 纯 prefill | 4,096 | 101,243 | 128,706 |

缓存量化格式、scale 和存储字节不变，保留分页缓存，不解压整份 KV cache。165 项量化测试全部通过，包含新增混合 query、GQA=6、NaN 尾页回归；代码格式与 Ruff 检查通过。

## 真实 xLLM 小规模验证

启动新的 Python INT8 xLLM 服务，关闭 graph，设置 `max_tokens_per_batch=4096`、`max_seqs_per_batch=16`、`max_memory_utilization=0.87`。使用先前准备的 NarrativeQA 前 16 题，包含约 32K token 的输入；沿用官方生成设置（`temperature=0`、`top_p=1`、seed 17、最多 128 token、stop IDs 151645/151643），并发 16，严格核对 prompt token 数。

16/16 请求完成且响应非空，API prompt token 数全部与本地分词一致。并发请求窗口为 140.05 秒，共生成 200 token，生成吞吐约 1.43 token/s；客户端延迟 p50/p95 为 105.83/139.49 秒。这只是服务执行验证，不是质量对照，也不能与先前 50 题的服务吞吐作等负载比较。完成后已关闭服务，18994 端口无监听。

启动记录、就绪证据和逐题输出保存在 [真实服务验证目录](/mnt/e/AI/xllm-eval-data/int8-kernel-optimization-20261001/20261002T043000Z_narrativeqa16_int8_smoke/)。原 LongBench 400 题结果未覆盖。本轮尚未证明新 INT8 路径追平 FlashInfer BF16，也未重跑完整质量评测。
