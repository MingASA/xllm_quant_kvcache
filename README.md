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

基于 xLLM 的 KV cache 量化实验分支：INT8、FP8 E4M3/E5M2、packed INT4。已有 CPU 冒烟记录及 RTX 5060 Ti 16GB 上的模型质量实验；实验边界与结论见下文。

## 历史 CPU 冒烟结果（2026-09-26）

2026-09-26，提交 `189beb35`，PyTorch 2.7.1+cpu，单线程；单层 GQA：B=1、T=129、Q=1、Hq/Hkv=32/8、D=128、page=128。随机输入，seed=2026；预热 1 次、采样 3 次，延迟仅作冒烟记录。

| 格式 | 实际 cache（KiB） | 输出相对 L2 误差 | CPU p50（ms） |
|---|---:|---:|---:|
| BF16 连续 SDPA | 516 | 0.166% | 9.907 |
| INT8 | 528 | 0.873% | 2.549 |
| FP8 E4M3 | 528 | 3.452% | 5.925 |
| FP8 E5M2 | 528 | 7.217% | 3.158 |
| INT4 | 272 | 16.119% | 9.276 |

- 量化 cache 含 FP32 scales，分配 256 个槽位；BF16 对照仅存 129 token。**同为 256 槽位时 BF16 为 1024 KiB**，INT8/FP8 节省 **48.44%**，INT4 节省 **73.44%**。尾页未填满时不能直接套用压缩率。
- 误差相对未量化 FP32 attention 输出，不是模型准确率。当前 INT4 误差较大，尚不能证明真实模型质量可接受。
- BF16 计时仅 attention；量化计时为 execute（含写入）。实现不同、样本少，**不能据此报告加速倍数，更不能外推 GPU 性能**。
- 原有量化测试 **58 passed**；五格式 CPU benchmark 完成；无 CUDA 时明确失败，不静默回退。

[原始样本](docs/benchmark_results/cpu_smoke_20260926.jsonl) · [CSV](docs/benchmark_results/cpu_smoke_20260926.csv) · [运行说明与限制](docs/kv_cache_benchmark_zh.md)

## GPU 模型质量实验（2026-10-02）

Qwen2.5-1.5B-Instruct、RTX 5060 Ti 16GB、完整 GSM8K test 集（1,319 题），三臂客户端请求并发均为 64。以下是在 Python eager / FlashInfer 路径上的质量对照，不代表压缩缓存容量或吞吐收益。

| 方案 | KV 变换 | 严格准确率 | 宽松准确率 |
|---|---|---:|---:|
| 全新 BF16 基线 | 无 | 578/1,319 (43.82%) | 853/1,319 (64.67%) |
| 仅 V INT4 | V，分组大小 128，不旋转 | 543/1,319 (41.17%) | 853/1,319 (64.67%) |
| RHT INT4 | K/V，分组大小 32 | 0/1,319 (0.00%) | 5/1,319 (0.38%) |

两个 INT4 变体在 GPU 上量化并反量化新值，之后写入常驻 BF16 cache，通过 FlashInfer 做质量对照；它们没有压缩存储 KV。仅 V INT4 的宽松总分与 BF16 相同，但有 206 题的逐题宽松正误状态发生翻转，因此不等于无损。RHT 有 1,318/1,319 题耗尽 512-token 输出上限。另一项 plain INT4 serving 评测使用真实压缩 cache，观察到严重质量退化；两者是不同路径。旧 INT8 数据仅作历史对照，不代表生产支持结论。

诊断确认了两个 INT4 边界问题：半整数舍入可能不一致；decode 可能读取尚未写入的页面尾部 scale 并传播 NaN。RHT 结果不证明 RoPE 坐标有误：本实验在 RoPE 后旋转，并逆旋转回原坐标。仅 K INT4 在四题 teacher-forced 诊断中也观察到质量退化，并非完整 1,319 题评测；本次具体 RHT 结果不能推广到所有旋转方案。

报告：[GSM8K INT4 变体对照](docs/kv_cache_gsm8k_int4_variants_20261002_zh.md) · [INT4 诊断](docs/kv_cache_int4_diagnosis_20261002_zh.md) · [此前模型评测](docs/kv_cache_int4_model_eval_20261002_zh.md) · [实验摘要](docs/kv_cache_experiment_summary_20261001_zh.md)

## 运行

已有 PyTorch 环境下：

```bash
python tools/benchmark_kv_cache.py --device cpu --contexts 129 --warmup 1 --iterations 3 --output-dir results
# GPU 就绪后（需要支持 Blackwell 的 PyTorch/CUDA 与驱动）
python tools/benchmark_kv_cache.py --device cuda --contexts 1024 4096 --warmup 10 --iterations 100 --output-dir results
```

自动输出环境/源码哈希、JSONL 原始样本及 CSV。当前为 Python eager 分页 reference，非融合 kernel；C++/整服务、真实模型质量及 GPU 性能未验证。

[量化审计](docs/kv_cache_audit_zh.md) · [原项目中文说明](README_zh.md) · [上游 xLLM](https://github.com/xLLM-AI/xllm)
