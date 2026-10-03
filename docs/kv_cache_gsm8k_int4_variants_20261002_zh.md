<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# GSM8K INT4 KV 质量变换对照（2026-10-02）

## 结论

三臂均在同一 xLLM Python eager / FlashInfer 路径、BF16 权重和 BF16 常驻 KV cache 下完成 GSM8K test 全部 1,319 题。INT4 两臂是写入 BF16 cache 前对新 K/V 做 fake-quantize/dequantize 的质量实验，不是压缩缓存实现；结果不证明 KV 容量或生产服务吞吐收益。

| 臂 | 变换 | Strict EM | Flexible EM | `finish_reason=length` | 输出 token 总数 |
|---|---|---:|---:|---:|---:|
| Fresh BF16 | 无变换（`auto`） | 578/1,319 (43.8%) | 853/1,319 (64.7%) | 15/1,319 (1.14%) | 204,105 |
| V-only INT4 | 只量化 V，G=128，seed 17 | 543/1,319 (41.2%) | 853/1,319 (64.7%) | 18/1,319 (1.36%) | 201,437 |
| RHT INT4 | K/V 均做符号随机化 Walsh–Hadamard 旋转后 INT4，G=32，seed 17 | 0/1,319 (0.0%) | 5/1,319 (0.38%) | 1,318/1,319 (99.92%) | 675,185 |

按 `_eval_index` 配对，三臂数据记录与渲染 prompt 的 SHA256 均逐题一致，覆盖 index 0–1318。相对 fresh BF16：V-only strict 配对翻转为 auto-only 正确 143、V-only-only 正确 108、正确性相同 1,068；flexible 为 103、103、1,113，即共有 206 题 flexible 正误状态翻转。因此 flexible 总分相同不表示逐题无损；strict 净下降 35 题也不应在未检查答案文本和类别后归因于某一个单独格式因素。RHT-G32 strict 为 auto-only 正确 578、RHT-only 正确 0、相同 741；flexible 为 849、1、469。RHT-G32 生成文本出现重复退化，且几乎所有响应耗尽 512-token 上限；其 flexible EM 与 strict EM 不应被解释为通常意义上的正常解题表现。

## 协议与产物

- 模型：Qwen2.5-1.5B-Instruct，BF16 权重；本地 checkpoint SHA256：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`。
- 数据：1,319 条 GSM8K test，原有有序准备/合并输入 `/mnt/e/AI/xllm-eval-data/int4-model-eval-20261002/gsm8k/data.jsonl`；SHA256：`93377ddca91b89cfa35e987104ee7b420a1b17d42b91351b70993d57d0159cac`。五个官方训练 shots 文件为 `/mnt/e/AI/xllm-eval-data/gsm8k-3101c7d5072418e28b9008a6636bde82a006892c/prepared/five_train_examples.json`，SHA256：`4e076bef9962698cf3df4701f093437c86c925dcfa987556ea4fdb0fbd93a3ce`。
- 生成：官方 GSM8K 5-shot prompt，temperature 0、top_p 1、seed 17、`max_tokens=512`，stop strings `Question:`、`</s>`、`<|im_end|>`；未显式传 stop-token IDs，服务沿用 Qwen 默认 EOS 151645。旧基线首 50 题曾额外显式传入 151643，所以不将该子集当作完全相同协议的对照。
- 服务：Python eager、FlashInfer、`max_cache_size=0`、`max_memory_utilization=0.87`、`max_tokens_per_batch=4096`、`max_seqs_per_batch=64`，graph/prefill piecewise graph/Python graph backend 均关闭；client request concurrency 64。模型启动日志记录 BF16 模型、`kv_cache_dtype=auto`；INT4 臂另有 transform 启动日志明确列出 mode、group size、seed，以及 persistent cache BF16 / not compressed。
- 三臂完整结果、配置与 server manifest 位于 `/mnt/e/AI/xllm-eval-data/gsm8k-int4-variants-20261002/{auto,v-only-int4,int4-rht-g32}/full/` 与各臂目录的 `server_manifest.json`。配对汇总 JSON：`/mnt/e/AI/xllm-eval-data/gsm8k-int4-variants-20261002/summary.json`。各 server manifest SHA256：auto `d5c13be8809dbdf0e4a803f80925564713173ab5fd2cc165cbac385735b4fd3c`；V-only `6ce9fe27754d18274a6575a40687e3e710634ecccfea96dc5a401bd87fe406e6`；RHT-G32 `0fa665a88de4b412ccd782241888df4a436a47eb22f6f3f742ea9fe811e08c44`。

## 测时边界

用客户端记录的最早 `request_started_utc` 到最晚 `request_finished_utc` 作为 run wall time；请求速率为 1,319 / wall time，生成 token 速率为所有 API `completion_tokens` / wall time。auto 为 84.91 s、15.53 request/s、2,404 generated token/s；V-only 为 90.33 s、14.60 request/s、2,230 generated token/s；RHT-G32 为 295.57 s、4.46 request/s、2,284 generated token/s。RHT-G32 的 token 速率主要受其长重复输出和 512-token 上限影响，不是有用服务吞吐，也不构成 INT4 或旋转的性能结论。三臂的 KV footprint 都是 BF16；这里不能据此宣称省显存、扩大容量或得到压缩 cache 的实际服务收益。

## 参考对照与验证

- `tools/test_eval_kv_cache_quality_resume.py`：10 项通过，包含两种新 arm 的 manifest 检查；新模式要求 `kv_cache_mode=auto` 且 `quality_transform_mode` 与臂严格匹配。
- `tests/python/test_kv_quality_transform.py`：11 项通过。独立随机 BF16 tensor 对照中，V-only G=128 GPU 结果逐值匹配生产 `KVCacheCodec('int4', 128)` CPU encode/decode 后转 BF16；RHT-G32 同样逐值匹配 seed 17 的 CPU rotation/G32 codec 对照。三种完整评测各先完成 4 题 smoke，seed、5-shot、最大生成上限及 stop 设置与 full run 相同。
- 相邻 `test_flashinfer_attention.py` / `test_model_executor.py` 合跑为 94 项通过、4 项失败；失败均是当前 CUDA 环境不支持 NPU device/backend 的既有 NPU 测试，不涉及本次 transform。对触碰的 evaluator/test 文件 Ruff 检查通过。
- E 盘完整 JSONL 为并发 append 顺序，评分和配对必须先按 `_eval_index` 排序；不要把文件行序当作源顺序。

本轮 V-only G128 与 K/V G32+旋转同时改变了量化对象、粒度与旋转，因此不是单因素“有无旋转”的因果对照。所测 RHT-G32 方案没有 outlier sidecar、centering 或 affine offset；四题 smoke、11 项 transform refs 和全量结果共同显示的是这个具体实现/策略在当前路径中的退化，不能泛化为所有 rotation/outlier 方案必然无效。此结果衡量的是指定实现阶段中的质量变化，不定位历史低精度结果的唯一原因。

前序失败案例与诊断见 [INT4 模型评测](kv_cache_int4_model_eval_20261002_zh.md) 和 [实现与原因诊断](kv_cache_int4_diagnosis_20261002_zh.md)。本次新目录独立保存，不覆盖前序报告或证据。
