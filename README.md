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

基于 xLLM 的实验分支：INT8、FP8 E4M3/E5M2、packed INT4。**仅完成 CPU 验证；RTX 5060 Ti 16GB 尚未实测。**

## 当前结果

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

## 运行

已有 PyTorch 环境下：

```bash
python tools/benchmark_kv_cache.py --device cpu --contexts 129 --warmup 1 --iterations 3 --output-dir results
# GPU 就绪后（需要支持 Blackwell 的 PyTorch/CUDA 与驱动）
python tools/benchmark_kv_cache.py --device cuda --contexts 1024 4096 --warmup 10 --iterations 100 --output-dir results
```

自动输出环境/源码哈希、JSONL 原始样本及 CSV。当前为 Python eager 分页 reference，非融合 kernel；C++/整服务、真实模型质量及 GPU 性能未验证。

[量化审计](docs/kv_cache_audit_zh.md) · [原项目中文说明](README_zh.md) · [上游 xLLM](https://github.com/xLLM-AI/xllm)
