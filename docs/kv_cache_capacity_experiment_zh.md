# INT8 KV 容量实验

## 成功标准

量化先评价缓存容量，再评价固定容量下可承载请求数、单层吞吐和延迟。
同负载更慢不等于没有容量价值；但缓存压缩也不等于服务吞吐提升。
本阶段不加载真实模型，不报告任务准确率、TTFT 或生产 QPS。

## 三组对照

1. `--bf16-backend triton`：BF16 与 INT8 共用分页遍历、GQA 映射、tile 和 online softmax 实现，只有读取存储格式及 scale 不同。BF16 仍独立对照 FlashInfer 检查输出。
2. 默认 `--bf16-backend flashinfer`：原版 FlashInfer BF16 与融合 Triton INT8 的实际单层执行比较。
3. `--kv-budget-mib 64`：按每种格式的真实页对齐 payload 和 FP32 scales 算出给定 context 下最大 batch，测全部三个 stage。预算只限制单层 cache，不限制输入、workspace、allocator 或模型权重，不代表整机最大并发。

三个实验使用相同的默认 GQA 形状、随机种子、分页布局规则和计时范围。
容量实验各格式 batch 不同，输入 shape 也不同，不能把它解读成同负载速度比较。
预热 10 次、采样 100 次、重复 3 轮，GPU 实验串行运行；报告每轮 p50 的中位数。
计时是包含 CPU dispatch 和同步的墙钟时间，plan 和参考误差计算不计入。

```bash
source /home/mingasa/venvs/vllm/bin/activate
python tools/benchmark_kv_cache.py --device cuda --contexts 1024 4096 --formats bf16 int8 --batches 1 4 --query-lengths 1 16 --bf16-backend triton --warmup 10 --iterations 100 --rounds 3 --output-dir results
python tools/benchmark_kv_cache.py --device cuda --contexts 1024 4096 --formats bf16 int8 --batches 1 4 --query-lengths 1 16 --bf16-backend flashinfer --warmup 10 --iterations 100 --rounds 3 --output-dir results
python tools/benchmark_kv_cache.py --device cuda --contexts 1024 4096 --formats bf16 int8 --query-lengths 1 16 --kv-budget-mib 64 --warmup 10 --iterations 100 --rounds 3 --output-dir results
```

## 实施顺序与边界

先补公平对照、正确性测试和容量扫描，再依据分项结果优化 INT8。
当前融合 kernel 已在 GPU 内读取 block table、反量化和 online softmax，无全 cache 解压副本；decode 使用 split-KV 分区和 FP32 partial softmax 合并。它仍是单层实验实现，不能称为生产优化方案。
CUDA Graph、原版 cache 写入算子的完整构建和真实服务调度验证仍是独立待办。
FP8 与 INT4 先保留既有实验，不以它们分散当前 INT8 调优工作。

低延迟路线必须追上 FlashInfer BF16；容量路线需证明相同完整设备预算下，在可接受延迟内提升吞吐/可接纳请求数。
本文件的 KV-only 容量实验只能为后者提供单层证据，不能代替完整服务验收。

## 本轮结果：RTX 5060 Ti，单层合成实验

测试使用 venv `vllm`、CUDA 13.2、PyTorch 2.13.0+cu132、Triton、FlashInfer 0.6.18，形状为 32 个 query heads、8 个 KV heads、head_dim 128、page_size 128。测试模块 `tests/python/test_quantized_attention.py`：99 passed；包含 INT8 split decode 空分区和 BF16 context length 1025。

三组 split kernel 数据如下，完整原始采样保留在仓库结果目录：

| 组别 | 结果目录 | BF16 attention | INT8 attention |
| --- | --- | --- | --- |
| 同一 Triton attention 框架 | `results/20260930T112152Z_3963e30a` | Triton | Triton split-KV |
| 64 MiB 单层 KV 预算 | `results/20260930T112211Z_07469bd8` | FlashInfer | Triton split-KV |
| 固定 batch FlashInfer 对照 | `results/20260930T112231Z_ba5e01ea` | FlashInfer | Triton split-KV |

每个配置先预热 10 次、采样 100 次并运行 3 轮。下表的 latency ratio 是各 shape 三轮 p50 比值的中位数，范围覆盖 8 个 context/batch/query 组合。小于 1 表示 INT8 更快。

| 比较 | attention INT8/BF16 | full-call INT8/BF16 |
| --- | ---: | ---: |
| 同一 Triton 框架 | 1.06×（0.74–1.34×） | 1.11×（0.76–1.14×） |
| FlashInfer BF16 对 Triton INT8 | 1.72×（0.92–3.08×） | 1.39×（1.13–2.83×） |

同框架比较在大 batch 和 prefill 的一些形状上 INT8 较快，但整体中位数仍稍慢；速度收益随形状变化，不能据此声称省下的读取流量已抵消反量化开销。对 FlashInfer 的差距主要来自不同 attention kernel 实现，不能归因于存储位宽单一因素。FlashInfer 组测的是 FlashInfer attention wrapper；当前环境没有加载 xLLM 原生扩展，因此 BF16 cache 写入使用 standalone Triton writer。full-call 包含 cache 写入与 attention，结果不是 xLLM 已接入后的端到端服务延迟。

64 MiB 预算按页对齐后的单层 KV cache 字节数选取最大 batch。每格列出 `最大 batch / full-call p50 / 处理 token 吞吐`；吞吐列括号为三轮最小至最大值。

| Context | Query tokens | BF16 | INT8 |
| ---: | ---: | ---: | ---: |
| 1024 | 1 | B16 / 0.359 ms / 44,550 tok/s | B31 / 0.609 ms / 50,907 tok/s（50,895–51,168） |
| 1024 | 16 | B16 / 0.343 ms / 745,595 tok/s | B31 / 1.599 ms / 310,175 tok/s（309,787–311,001） |
| 4096 | 1 | B4 / 0.366 ms / 10,930 tok/s | B7 / 0.576 ms / 12,157 tok/s（11,218–12,687） |
| 4096 | 16 | B4 / 0.392 ms / 163,370 tok/s | B7 / 1.636 ms / 68,479 tok/s（65,917–68,980） |

预算下 INT8 可接纳 batch 约为 BF16 的 1.94 倍。decode 的总 token 吞吐在 context 1024 提升约 14%，context 4096 提升约 11%；后者三轮区间较宽，收益不算稳定。16-token prefill 虽然 batch 增大，整体吞吐仍只有 BF16 的约 42%（context 1024）和 40%（context 4096），full-call 延迟也更高。因此目前只看到 decode 场景有初步容量转吞吐信号，尚不足以判定服务能力整体提升。

INT8 相对 BF16 的最大输出误差为 relative L2 0.00981、absolute 0.00293（预算扫描的 absolute 最大值 0.00488）。固定形状运行峰值高于计时前已分配量最多约 0.54 MiB；64 MiB 预算扫描最多约 4.27 MiB，出现在 context 1024、B31 decode 的 INT8 case。这里的 KV 预算不包括输入、workspace、权重或 allocator 开销；增量峰值也不是整机显存需求。

### split-KV 前后

固定形状非预算结果可分别对照 split 前的 `results/20260930T111918Z_637ed32f`（同框架）与 `results/20260930T111953Z_f305c856`（FlashInfer）；探索数据不用于主要结论。按 8 个 shape 中位数比较，同框架 attention INT8/BF16 从 0.98× 变为 1.06×、full-call 从 1.06× 变为 1.11×。FlashInfer 对照的 attention 从 1.66× 变为 1.72×，full-call 从 1.41× 变为 1.39×。形状间变化方向不一致，没有显示稳定的整体吞吐改善。预算 decode 在 context 4096 的吞吐比从初版 1.08× 变为 1.12×，但三轮比值分别为 1.01–1.16×；该幅度接近轮间波动，不能单独作为 split kernel 的确定收益。

### 判断与边界

本轮支持的结论是：INT8 cache 在单层预算下可容纳约 1.94 倍 batch；decode 吞吐出现约 11–14% 的初步提升，短 prefill 明显退化。INT8 同框架执行尚未普遍快于 BF16，FlashInfer BF16 的 attention/full-call 仍有明显优势。是否值得合并仍需在相同完整设备显存预算下接入 xLLM runtime，测模型权重、workspace、调度和多层 cache 后的可接纳请求数、总吞吐及延迟；本轮不能外推为生产 QPS 或服务收益。
