# xLLM GSM8K 真实 FP8 KV cache 评测（2026-10-05）

## 实验设置

本轮通过 xLLM Python 模型实现运行 Qwen2.5-1.5B-Instruct 与 GSM8K test 全 1,319 题。三个 arm 使用相同题目顺序、官方 5-shot、temperature 0、top_p 1、seed 17、512-token 最大输出、stop strings `Question:`/`</s>`/`<|im_end|>` 和 stop IDs 151645/151643。每臂单独启动并关闭服务，使用相同 tokenizer、权重、eager 执行和自动 cache 预算。

| Arm | `kv_cache_dtype` | backend | 请求并发 / `max_seqs_per_batch` |
| --- | --- | --- | ---: |
| BF16 | `auto` | FlashInfer | 32 / 32 |
| FP8 E4M3 | `fp8_e4m3` | 原有 Triton quantized KV 路径 | 50 / 50 |
| FP8 E5M2 | `fp8_e5m2` | 原有 Triton quantized KV 路径 | 50 / 50 |

两种 FP8 arm 使用原有按 token/head 动态 scale 写入量化 KV cache 的路径。实验未启用 QDQ fake、quality transform、FlashInfer FP8 fixed-scale backend 或静默 fallback。BF16 C32 和 FP8 C50 参照原 INT8 全量阶段的并发设置；所有 arm 都一次连续跑完 1,319 题，不重放历史分段并发。由于并发不同，本报告不比较各 arm 的速度。

各 arm 先在独立 smoke 目录运行同一题集前 4 题以确认 API 与缓存路径，再运行全量评测；答案正确率不是是否继续全量的门槛。smoke 只用于启动健康检查，不并入全量评分。

## GSM8K 全量结果

| Arm | Strict EM | Flexible EM | `finish_reason=length` | HTTP 请求失败/超时 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 567/1319 (42.99%) | 844/1319 (63.99%) | 16/1319 (1.21%) | 0 |
| FP8 E4M3 | 2/1319 (0.15%) | 27/1319 (2.05%) | 513/1319 (38.89%) | 0 |
| FP8 E5M2 | 0/1319 (0.00%) | 1/1319 (0.08%) | 1312/1319 (99.47%) | 0 |

按题与 BF16 配对的正确性状态：

| 对比与指标 | 两臂都正确 | BF16 only | FP8 only | 两臂都错误 |
| --- | ---: | ---: | ---: | ---: |
| E4M3 Strict EM | 2 | 565 | 0 | 752 |
| E4M3 Flexible EM | 23 | 821 | 4 | 471 |
| E5M2 Strict EM | 0 | 567 | 0 | 752 |
| E5M2 Flexible EM | 0 | 844 | 1 | 474 |

逐题差异示例（`_eval_index=6`）：题目问三地羊的总数，Seattle 有 20 只；Charleston 是 Seattle 的 4 倍，即 80 只；Toulouse 是 Charleston 的 2 倍，即 160 只；总数为 260。BF16 回答给出 `#### 260`。同一索引下，E4M3 输出从错误的 `T2 has 20/3...` 开始并重复数字直到长度上限；E5M2 输出重复的 `answer=answer...` 并达到长度上限。注意，E4M3 的 `400 feet` 输出属于另一题 `_eval_index=41`（官方答案 200），不属于本羊题，故不将它与 BF16 的 260 配对引用。代表性样例仅展示实际输出，不代表所有错误模式。配对按 `_eval_index` 对齐，并核验 prompt/record SHA256。E4M3 strict paired 状态为 2 题两臂正确、565 题仅 BF16 正确、0 题仅 E4M3 正确、752 题两臂均错误；flexible 为 23、821、4、471 题。E5M2 strict 为 0、567、0、752 题；flexible 为 0、844、1、474 题。

结论：该模型在本次真实动态 scale FP8 KV-cache 路径下，E4M3 与 E5M2 均大幅退化，且大量生成达到长度上限；因此不能把退化仅归因于此前固定 scale 的 QDQ 诊断，这两种配置目前不适合直接用于该模型。此结论限定于本模型、当前 xLLM 路径与评测设置。

## 产物与复现

全量原始输出、每臂配置、server manifest、日志和配对分数位于 `/mnt/e/AI/xllm-eval-data/xllm-fp8-gsm8k-20261005/`；四题 smoke 单独位于 `smoke/`。本次执行源码的逐文件快照和 SHA256 位于 `source_snapshot/` 与 `source_snapshot_manifest.json`，实验输入和 arm 配置位于 `experiment_manifest.json`。

```bash
source build/cmake.cuda-x86/resume-env.sh
/home/mingasa/venvs/xllm/bin/python tools/run_xllm_fp8_gsm8k.py --phase full
```

`--resume` 仅续跑匹配输入、源文件快照和配置的部分 JSONL journal。任何启动、CUDA/OOM 或请求错误都会保留日志并使 runner 失败；本报告记录全量完成状态和实际错误数。
