# FlashInfer 原生 FP8 KV 探测报告

日期：2026-10-05。设备：RTX 5060 Ti（SM 12.0），Torch 2.13.0+cu132，CUDA 13.2，FlashInfer 0.6.18.post1。结论：FlashInfer 原生 paged decode/prefill kernel 可调用，独立数值与计时探测通过；但当前服务端全 K/V FP8 E4M3 与 Triton 动态 E4M3 都出现明显生成退化，不能作为 BF16 的可用替代，也没有更改默认配置。

## 原生调用验证

接入新增 `FlashInferFP8Backend`，沿用 C++ 的分页 payload、scale 张量与 backend 生命周期；缓存字节通过零拷贝 FP8 视图交给 FlashInfer，写入由融合 Triton kernel 完成。启用环境变量 `XLLM_QUANTIZED_BACKEND=flashinfer`，配合 FP8 cache dtype 和 eager 模式；benchmark 使用 `--fp8-backend flashinfer`。默认量化路径保持不变，使用方法见根 README。

形状为 Q BF16、KV FP8 E4M3/E5M2、Hq/Hkv=12/2、D=128、page=128、B=1、KV context=256。decode 使用 `BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD", use_tensor_cores=True)`；`plan(indptr, indices, last_page_len, 12, 2, 128, 128, q_data_type=torch.bfloat16, kv_data_type=...)`；`run(q, cache, k_scale=1.0, v_scale=1.0)`。prefill 使用 `BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")`；plan 参数相同并设 `causal=True`，另传 `qo_indptr`；Q 长度 1 和 16 均成功。cache 为 NHD `[pages, 2, page, Hkv, D]`，直接保存 FP8 payload。

相对 BF16 cache 输出的相对 L2：

| 路径 | E4M3 | E5M2 |
|---|---:|---:|
| decode, tensor cores | 0.02739 | 0.05194 |
| FA2 prefill, Q=1 | 0.02454 | 0.05514 |
| FA2 prefill, Q=16 | 0.02473 | 0.05498 |

`k_scale`/`v_scale` 标量参数可用。上述直接 probe 使用 scale=1.0；项目侧包含非单位 scale、长 Q 和乱序/部分页的 FlashInfer 回归已由主线程验证通过。FlashInfer FP8 path 使用固定 K/V scalar scale；它不是旧的 per-token/head 动态 codec。

## 服务 smoke 与正确性边界

本地 Qwen2.5-1.5B-Instruct（Qwen H12/KV2）使用 Python model、eager、graphs 关闭、memory utilization 0.80、max seqs 8。请求通过同一个 `/v1/chat/completions` chat template，temperature=0、top_p=1、max_tokens=24、startup/request seed=20261005。两条串行请求加 8 并发请求均 HTTP 200。含乘法在内的九道算术题，BF16 auto 为 9/9 正确，FlashInfer 全 K/V FP8 E4M3 为 0/9；`27*43` 返回 `28 times 43 equals 96.`。这两臂的算术响应均按 `stop` 结束，配对输入 token 数相同，错误未由输出上限截断。中文请求两边均能返回文本，BF16 中文响应达到长度上限。原 Triton 动态 E4M3 对相同请求也产生残缺或错误的算术输出，其中两条达到长度上限；因此退化不只出现在 FlashInfer 后端。

HF 本地诊断使用与服务一致的 45-token 乘法 prompt；28 层 post-RoPE Q/K/V callback 的 causal future-mask 审计通过。另一个 layer-statistics pass 使用 49-token prompt（末尾多了 “Please reason carefully.”），需与精确服务 prompt 区分。该 pass 的 K/V 最大绝对值为 320/18.625，无 E4M3 448 饱和；fixed scale 1 的每层 QDQ 相对 L2 中位数 K/V 为 2.68%/2.65%，动态 per-token/head 为 2.36%/2.49%。固定 scale 非零变零比例中位数低于 0.18%。

有效的 eager 控制使用相同 `qwen2.eager_attention_forward` 数学路径，仅改变 K/V QDQ；未量化 callback 与原生 eager logits maxabs=0，280 个完整 prefill mask callback 均验证 future token 被屏蔽。用正确答案 `1161` teacher-force 时，K+V fixed E4M3 的 logits 相对 L2 为 0.806，K+V dynamic 为 0.866；K-only fixed 为 0.781，V-only fixed 为 0.043，且 V-only top-1 与该 HF BF16 基线一致。HF BF16 greedy 本身输出 `1131`，所以 top-1 agreement 不是正确率，也不把 HF greedy 和 xLLM server 输出混作绝对答案比较。这组实验支持“此 prompt 对 K 量化更敏感”，但没有证明服务端具体 ABI/layout 错误，也不能据此归咎固定 scale 或 FP8 库。

HF 五臂结果在 [`hf_kv_qdq_eager_control.json`](../results/flashinfer-fp8-smoke-20261005/hf_kv_qdq_eager_control.json)，层统计在 [`hf_kv_qdq.json`](../results/flashinfer-fp8-smoke-20261005/hf_kv_qdq.json)，可复现脚本在 [`source/`](../results/flashinfer-fp8-smoke-20261005/source/)。`hf_kv_qdq_logits.json` 与 `hf_kv_qdq_logits_dense_control.json` 保留为无效的跨模式对照：FP8 arm 使用 FP32 dense attention，而 BF16 eager 使用不同算术路径；文件内已有 `validity` 标记。更多服务原文、逐项 finish reason 和日志路径见结果目录 [README](../results/flashinfer-fp8-smoke-20261005/README.md)。

## 计时

标准扫描 context 1024/4096、batch 1/4、Q=1/16，分别使用 H12/KV2 与 H32/KV8；长上下文扫描 16K/32K、batch 8/16、Q=1/16、H12/KV2。每个配置预热 10 次，采样 100 次，重复 3 轮，合计保存 192 条结果。

以下数据为 context=4096、batch=4、三轮各 100 次采样的 p50 中位数，格式顺序 BF16/E4M3/E5M2。计时由 `perf_counter` 加 `torch.cuda.synchronize()` 完成，包含 Python dispatch；plan 不在 timed call 内，`full-call` 包含 cache writer。数据不是纯 GPU kernel 时间，不能保证 FP8 总体更快。

| 头数 Hq/Hkv | Q | attention p50 (ms) | full-call p50 (ms) | 相对 L2 |
|---|---:|---:|---:|---:|
| 12/2 | 1 | 0.095 / 0.091 / 0.090 | 0.350 / 0.221 / 0.223 | 0 / 0.038 / 0.074 |
| 12/2 | 16 | 0.175 / 0.185 / 0.179 | 0.234 / 0.258 / 0.240 | 0 / 0.038 / 0.076 |
| 32/8 | 1 | 0.410 / 0.123 / 0.164 | 0.556 / 0.256 / 0.370 | 0 / 0.039 / 0.076 |
| 32/8 | 16 | 0.476 / 0.310 / 0.261 | 0.537 / 0.318 / 0.326 | 0 / 0.039 / 0.076 |

长上下文用 Qwen H12/KV2、context=32768、batch=16、Q=16，BF16/E4M3 的 attention p50 为 1.961/1.962 ms，full-call 为 2.039/2.060 ms，E4M3 relative L2=0.039。KV payload 加保留的 scale allocation 后压缩比为 1.94x；FP8 在此长上下文点没有更高吞吐。

| 格式 | KV cache（MiB） | full-call p50（ms） | 吞吐（token/s） |
|---|---:|---:|---:|
| BF16 | 512 | 2.039 | 125.5k |
| FP8 E4M3 | 264 | 2.060 | 124.3k |

每调用处理 256 个 query token。cache 容量包括 payload 与 scale；运行期 peak allocated 约 1469.9/1221.9 MiB，包含输入和 workspace，独立于 cache 指标。allocator reserved 与整机 HBM 指标不可当作 cache 大小。

benchmark JSONL、summary CSV、environment metadata 和与测试时 SHA256 一致的源码快照位于：[`H12/KV2`](../results/flashinfer-fp8-qwen-20261005/20261005T080704Z_1c22af2d)、[`H32/KV8`](../results/flashinfer-fp8-20261005/20261005T081442Z_55562968)、[`long-context`](../results/flashinfer-fp8-long-20261005/20261005T081614Z_460bc622)。

量化注意力和 FlashInfer 相关 GPU 回归通过，包括增加尾页 NaN/255 毒化覆盖后的 12 项测试。默认量化路径保持 Triton，FlashInfer FP8 需显式启用。测试服务均已停止。
