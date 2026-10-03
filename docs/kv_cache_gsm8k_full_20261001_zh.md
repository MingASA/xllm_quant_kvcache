# GSM8K KV cache 全量配对实验（2026-10-01）

## 结果

- 数据：GSM8K test 共 1,319 题；每题 BF16（auto KV cache）与 INT8 KV cache 各一条正式响应。
- Strict EM：BF16 575/1,319（43.6%），INT8 473/1,319（35.9%）；差 102 题（7.7 个百分点）。配对翻转：BF16-only 正确 207、INT8-only 正确 105、两者相同 1,007。
- Flexible EM：BF16 856/1,319（64.9%），INT8 784/1,319（59.4%）；差 72 题（5.5 个百分点）。配对翻转：BF16-only 正确 173、INT8-only 正确 101、两者相同 1,045。
- 这是实际配置路径上的质量对比，不是单一 kernel 或并发参数的因果实验；各阶段并发不同，不能将质量差异归因于单因素。
- 未声称这是公开官方结果的复现。吞吐量仅作为各运行配置的描述值，不作跨并发或格式的等条件性能结论。

## 执行配置

- 固定模型：Qwen2.5-1.5B-Instruct；GSM8K 官方 5-shot；greedy（temperature=0，seed=17），max_new_tokens=512；使用官方 stop strings 与 EOS；graph 关闭。
- 阶段并发：source index 0–49 为 serial（BF16/INT8 各 1）；50–949 为 BF16 C10 / INT8 C16；950–1318 为 BF16 C32 / INT8 C50。请求并发与 server max_seqs_per_batch 按阶段记录，实际配置见各 batch config/stage 文件。
- 早期设置曾使用 1 GiB max cache；后续全量 batch runner 使用 `max_cache_size=0`（自动容量），KV cache mode 为 auto。缓存容量与并发均属于实际运行条件，故不同阶段不合并作性能因果比较。

## 数据与完整性

- 旧首批 50 题及其配对结果：`/mnt/e/AI/huggingface/xllm-kv-quality/gsm8k-batch50-full-20261001/batch_0000/`（source index 0–49）。
- 后续 1,269 题及分批配对结果：`/mnt/e/AI/huggingface/xllm-kv-quality/gsm8k-batch100-bf10-int16-20261001/`（source index 50–1318；12 个 100 题批次及末批 69 题）。
- 最终累计计数见 `/mnt/e/AI/huggingface/xllm-kv-quality/gsm8k-batch100-bf10-int16-20261001/batch_1250_1318/stage.json`。
- 核对正式输出行数：旧批 BF16/INT8 各 50 行；后续批次 BF16/INT8 各 1,269 行，合计每臂 1,319 条，覆盖 source index 0–1318。隔离的旧配置尝试与独立 concurrency scan 不计入以上结果。
