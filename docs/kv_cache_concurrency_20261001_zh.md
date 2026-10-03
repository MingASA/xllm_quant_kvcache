# Qwen2.5-1.5B-Instruct 多请求并发试验

同一组 32 道 GSM8K，Python executor、BF16 权重、greedy、512-token 输出上限与停止条件相同；每组先预热 4 个请求。缓存自动分配，`max_cache_size=0`、`max_memory_utilization=0.87`，关闭 CUDA Graph。吞吐计时为第一个 HTTP 请求发出至最后一个响应返回，不含模型加载及本地 tokenizer 准备。

| 指标 | BF16，客户端并发 8 | INT8，客户端并发 16 |
|---|---:|---:|
| 成功请求 | 32/32 | 32/32 |
| 输出 token 数 | 5,885 | 4,705 |
| 请求窗口耗时 | 23.56 s | 14.34 s |
| 输出吞吐 | 249.8 token/s | 328.2 token/s |
| 请求吞吐 | 1.358 request/s | 2.232 request/s |
| 客户端延迟 p50 / p95 | 3.35 / 7.77 s | 5.55 / 9.98 s |
| 测量窗口显存峰值（外部采样） | 14,134 MiB | 14,263 MiB |
| GSM8K strict | 15/32 | 7/32 |
| GSM8K flexible | 22/32 | 14/32 |

INT8 的输出吞吐高 31.4%，但输出 token 数少 20.0%，请求延迟也更高。两组并发不同，不是量化格式的同负载速度对照。数值答案匹配减少 8 题，属于需继续关注的质量退化信号，而非已证明可接受的服务收益。

补充扫描中，BF16 并发 16 和 32 也均成功，分别达到约 356.0 和 505.0 输出 token/s。因此不能把 INT8 并发 16 相比 BF16 并发 8 的收益解释为只有量化才能支持更多并发。本组短输入也没有证明 BF16 的缓存容量已成为瓶颈。

原始响应、配置、日志与显存采样保存在 `/mnt/e/AI/huggingface/xllm-kv-quality/concurrency-scan-20261001/`，主对照为 `C8/auto/measured/auto.jsonl` 和 `C16/int8/measured/int8.jsonl`。服务及采样进程已停止；不同并发的质量结果仅作探索性描述。
