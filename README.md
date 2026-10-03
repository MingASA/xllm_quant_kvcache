<!-- Copyright 2022 JD Co.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# xLLM KV Cache 量化实验

本仓库是基于 xLLM 上游审计基线 [`cf4c7732`](https://github.com/xLLM-AI/xllm/tree/cf4c7732d52da3a39cb35b0125a6b08cf2229c69)维护的 KV cache 量化实验 fork，目标是评估低比特 KV 对缓存容量、模型质量和 attention 执行的影响。本文的上游能力对照以该审计基线为准。

## 项目范围

上游已有 MLU 专用 INT8 主 KV 路径。本分支把缓存格式、存储预算、scale 元数据和复制生命周期扩展到统一的 cache 接口，并接入 CUDA Python Qwen2/Qwen2.5 executor。普通 MHA/GQA 的 Python eager 路径支持 INT8、FP8 E4M3/E5M2 和 packed INT4；MLU 保留原有 INT8 路径。该实现用于实验和后续优化验证。

| 方向 | 上游已有基础 | 本分支实验扩展 |
|---|---|---|
| 缓存与预算 | 分页 KV、MLU INT8 专用支持 | 统一格式识别、scale 分配与复制、按 payload 和 scale 计入容量预算；CUDA Python 路径支持 INT8/FP8/INT4 |
| 写入与 attention | 各后端原生 cache/attention 路径 | GPU 量化分页写入；按页读取、即时反量化，并用在线 softmax 合并 attention 输出 |
| 调度接口 | Block table、slot mapping、backend 生命周期 | 沿用现有分页表、slot 和 backend 创建/销毁流程；不生成整份连续反量化 KV 副本 |

## 实现路径

每个新 token 的 K/V 按 token/head 动态计算 scale，随后写入对应物理页；INT4 将两个有符号 4-bit 值打包到一个字节。attention 根据 block table 逐页加载量化值，在计算中即时恢复数值，并在线合并各页 softmax 结果。分页布局使已有页面可保持不变，scale 与 payload 随页面共同管理。当前 CUDA 量化实现限定 Python executor、eager、Qwen2/Qwen2.5 普通 MHA/GQA；MLA、NPU、native CUDA executor、图模式及多种 offload/并行组合不在此路径支持范围内。

主要修改分布在 `xllm/core/framework/kv_cache` 与 `xllm/core/runtime`（C++ cache 预算、分配和 worker 生命周期）、`xllm/python/attention`、`xllm/python/model_executor`、`xllm/python/models`（Python attention/executor/model 接线），以及 `tools/` 和 `tests/`（基准、评测与回归覆盖）。

## 实验结果

单层容量对比来自 per-token/head scale 布局；数字包含 cache payload 与 scale，不代表模型整体显存。INT8 kernel 优化结果比较同一单层 INT8 路径的旧版与新版。

单层实验使用相同输入和分页布局，比较 BF16 FlashInfer、同框架 BF16 与量化路径；记录 attention 输出误差、cache 字节和运行峰值，并分别测量 cache 写入与 attention 的完整阶段时间及 query-token 吞吐。decode Q=1、prefill Q=16 等形状重复采样；INT8 kernel 优化另对 decode Q=1、prefill Q=16 和长上下文 mixed 负载做旧新对照。模型质量实验使用完整 GSM8K 1,319 题、逐题相同 prompt、固定生成参数，并校验结果配对。

| 实验 | 结果 | 解读 |
|---|---|---|
| 单层 cache 容量 | INT8/FP8 约为 BF16 的 1/1.94；packed INT4 约为 1/3.76 | 分别约省 48.4% / 73.4% KV 字节，不等于整模型 HBM 节省比例 |
| INT8 kernel，32K mixed prefill/decode | 新版 70.989 ms，旧版 1,198.355 ms，快 16.88× | 同为 INT8 的单层旧新实现对照；不能据此声称相对 FlashInfer 或整服务加速 |
| GSM8K 全量质量，Qwen2.5-1.5B，RTX 5060 Ti 16GB，C64 | BF16：strict 578/1,319（43.82%），flexible 853/1,319（64.67%） | 基线 |
| 同上：V-only INT4 G128 / K+V RHT INT4 G32 | V-only：543/1,319（41.17%），853/1,319（64.67%）；RHT：0/1,319（0%），5/1,319（0.38%） | 两个 variant 是 GPU 量化再反量化后写入 BF16 cache 的 FlashInfer 质量对照，未压缩常驻 KV；RHT 有 1,318 题达到 512-token 上限 |

真实压缩 cache 的 plain INT4 全量 serving 评测质量严重退化（GSM8K strict 0/1,319、flexible 14/1,319）；FP8 尚无完整模型质量评测。INT4 诊断确认半整数舍入差异和 decode 读取未写页面 tail scale 两项问题，后者可导致 NaN 传播，均待修复和专项回归。仅 V INT4 若进入真实 cache，按当前 per-token/head scale 开销估算理论节省约 36.7% KV 字节，仍需完整服务实现验证。

较早的真实压缩 INT8 serving 评测中，GSM8K strict 从 BF16 43.59% 降至 35.86%，flexible 从 64.90% 降至 59.44%；LongBench 八项各 50 题的混合指标 macro 为 BF16 34.793、INT8 37.509。INT8 服务运行较慢；两次运行并发不同，不能据此作公平速度对比。该组历史服务结果与上表最新的三臂 BF16-cache 质量实验是不同实验。

## 收益与限制

优先收益目标是同等 KV 显存预算容纳更多 token，从而提升可用上下文或并发；实际幅度取决于权重、workspace、请求长度和页面碎片。固定 batch 下的速度提升没有得到证明。新版 INT8 对长上下文混合负载有效，纯 decode 与短 prefill 未见明显提速；完整服务是否受益，需要在相同模型、请求、并发和 HBM 预算下测量端到端吞吐与质量。

INT4 质量结果目前不支持生产采用。V-only 的全量准确率接近 BF16，但逐题存在 206 题 flexible 正误变化；K/V RHT G32 严重退化。完整 GSM8K 三臂使用相同 Qwen2.5-1.5B-Instruct、Python eager / FlashInfer、请求并发 64；RHT 的输出耗尽率说明其分数不能按正常解题吞吐解读。更广泛的模型、长上下文和服务容量验证仍是后续工作。

## 使用方法

按[入门指南](README_zh.md#入门指南)准备可用的 xLLM 环境，并安装本分支构建产物。Blackwell GPU 需使用匹配的 PyTorch/CUDA、Triton 与 FlashInfer。单层 benchmark 支持 CPU/CUDA、多种 cache 格式、context、batch、query 长度、warmup、迭代次数和独立 rounds 扫描：

```bash
python tools/benchmark_kv_cache.py --device cuda \
  --formats bf16 int8 fp8_e4m3 fp8_e5m2 int4 \
  --contexts 1024 4096 --batches 1 4 --query-lengths 1 16 \
  --warmup 10 --iterations 100 --rounds 3 \
  --output-dir results/kv-cache
```

真实模型验证使用本地 Qwen2/Qwen2.5 模型路径；普通 INT8 serving 选择 Python eager，并通过 `--python_model_path` 指向当前仓库。以下配置使用 0.87 显存预算、最多 16 个并发序列和 4096 batch tokens：

```bash
MODEL=/path/to/Qwen2.5-1.5B-Instruct
xllm serve --model="$MODEL" --model_impl=python \
  --python_model_path="$PWD" --kv_cache_dtype=int8 \
  --max_memory_utilization=0.87 --max_seqs_per_batch=16 \
  --max_tokens_per_batch=4096 --enable_graph=false \
  --enable_prefill_piecewise_graph=false --python_graph_backend=off
```

`--kv_cache_dtype=auto` 使用模型 BF16 cache。INT4 为实验选项，可替换上面的 `int8`；当前 plain INT4 模型质量明显退化。GSM8K 质量变换对照使用 `XLLM_KV_QUALITY_MODE=v_only_int4` 或 `int4_rht_g32`，并保持 `--kv_cache_dtype=auto`、关闭图模式；该模式量化后反量化并写回 BF16 cache。完整评测配置、manifest 和逐题结果见下列报告。

## 报告与项目链接

- [实现审计与接入范围](docs/kv_cache_audit_zh.md)
- [实验阶段总结与 INT8 kernel 数据](docs/kv_cache_experiment_summary_20261001_zh.md) · [INT8 优化记录](docs/int8-kernel-optimization-20261001.md)
- [GSM8K INT4 变体全量对照](docs/kv_cache_gsm8k_int4_variants_20261002_zh.md) · [INT4 诊断与已知问题](docs/kv_cache_int4_diagnosis_20261002_zh.md)
- [xLLM 上游项目](https://github.com/xLLM-AI/xllm) · [上游中文介绍](README_zh.md)
