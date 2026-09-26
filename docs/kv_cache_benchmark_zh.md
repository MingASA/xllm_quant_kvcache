# RTX 5060 Ti 到货前：KV cache benchmark 准备

## 范围与状态

本工具直接调用 `xllm/python/attention/quantized.py`，不复制另一套量化算法。
覆盖 BF16 对照、INT8、FP8 E4M3/E5M2、packed INT4；支持 CPU/CUDA、随机物理页、GQA、decode 和 chunked prefill。
这是**单层合成输入的实验后端 benchmark**，不是整模型吞吐测试，不需要权重、xLLM C++ 动态库或完整服务构建。
原量化实现已单独提交为 `b9c13693`，benchmark 是后续独立提交。

当前机器仅完成 CPU 验证；CUDA、驱动兼容性、实际显存峰值及性能必须等显卡到位验证。没有把实验实现认定为生产就绪。

## 环境准备

当前已建好 `/home/cjy/work/xllm_doc/.venv_kv_cpu`，使用 Python 3.12、PyTorch 2.7.1 CPU。
到货后另建 GPU 环境，不覆盖 CPU 环境：

```bash
cd /home/cjy/xllm
nvidia-smi
python3 -m venv /home/cjy/work/xllm_doc/.venv_kv_cuda
/home/cjy/work/xllm_doc/.venv_kv_cuda/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
/home/cjy/work/xllm_doc/.venv_kv_cuda/bin/python tools/benchmark_kv_cache.py --device cuda --preflight-only --output-dir /home/cjy/work/xllm_doc/benchmark_results
```

需要支持 RTX 5060 Ti 的 NVIDIA 驱动。这里固定 2.7.1/cu128 是为了与当前 CPU 基线保持同一版本，不代表推荐生产环境采用旧版本。
[PyTorch 2.7 官方说明](https://pytorch.org/blog/pytorch-2-7/)确认该系列引入 Blackwell/CUDA 12.8 支持。
可升级 PyTorch，但升级前后应分开记录结果。无需为这个纯 PyTorch benchmark 单独编译 CUDA Toolkit；未来构建 xLLM/CUDA kernel 另行配置。
preflight 会执行 BF16 CUDA 矩阵乘及所选 codec 的设备端编解码；没有 CUDA 时退出码为 1，绝不悄悄切换 CPU。

## 直接运行

先运行实际后端的小规模 GPU 冒烟，再跑 1K/4K decode。每条命令生成独立结果目录，不覆盖旧数据。

```bash
cd /home/cjy/xllm
/home/cjy/work/xllm_doc/.venv_kv_cuda/bin/python tools/benchmark_kv_cache.py --device cuda --contexts 33 --batch 2 --query-tokens 3 --head-dim 9 --heads 4 --kv-heads 2 --page-size 16 --warmup 1 --iterations 3 --output-dir /home/cjy/work/xllm_doc/benchmark_results
/home/cjy/work/xllm_doc/.venv_kv_cuda/bin/python tools/benchmark_kv_cache.py --device cuda --contexts 1024 4096 --warmup 10 --iterations 100 --output-dir /home/cjy/work/xllm_doc/benchmark_results
```

chunked prefill：上一条增加 `--query-tokens 16`；并发：增加 `--batch 2`。
之后逐级尝试 `--contexts 8192 16384`，不要第一次就扫最大容量。默认 `head_dim=128, heads=32, kv_heads=8, page_size=128` 是合成 GQA 形状，不代表已经运行某个真实模型。
可用 `--formats int8 int4` 缩小范围；用 `--device cpu` 运行 CPU，同一个程序不需要改代码。

当前 CPU 冒烟命令：

```bash
/home/cjy/work/xllm_doc/.venv_kv_cpu/bin/python /home/cjy/xllm/tools/benchmark_kv_cache.py --device cpu --contexts 33 --batch 2 --query-tokens 3 --head-dim 9 --heads 4 --kv-heads 2 --page-size 16 --warmup 1 --iterations 3 --output-dir /home/cjy/work/xllm_doc/benchmark_results
```

## 记录内容与比较方法

| 文件/字段 | 用途与限制 |
|---|---|
| `environment.json` | 命令参数、软件包版本、Git commit/dirty 状态、关键源文件 SHA256、驱动输出、GPU 型号/架构/空闲容量、完成或失败状态 |
| `results.jsonl` | 每个 case 的全部指标和逐次延迟；每完成一个 case 刷盘，后续失败仍保留已完成结果 |
| `summary.csv` | 每个格式/context/stage 的 p50/p95、实际 cache bytes、输出相对误差，适合表格分析 |
| `cache_tensor_bytes` | payload + FP32 scale 的实际张量字节数；量化 cache 包含页尾 padding，不含 allocator/metadata |
| `peak_allocated_including_fixtures_bytes` | CUDA 计时阶段的进程张量分配峰值，**包含输入/参考测试所需 fixture，不是部署 KV 大小** |
| `peak_increment_over_setup_bytes` | 相对计时前已分配张量的增量；不是完整模型 workspace 上界 |
| `quantization_relative_l2` | 相对未量化 FP32 attention 的输出误差，含 BF16 输出舍入；不是困惑度、任务准确率或质量保证 |
| `implementation_max_abs` | 与相同量化缓存解码后的 FP32 attention 比较；断言 rtol=0.02/atol=0.005，用于排除算法/页映射错误 |

延迟是每次调用后 CUDA synchronize 的墙钟时间，包含 Python、kernel launch 和同步，预热不计入。
量化路径分别计时写入、prepare、execute（**包含写入**）、prepare+execute；这些是独立实验，不应机械相加。
反复执行时覆盖同一批尾部 token，测固定形状稳态，不模拟 autoregressive 长度增长。
BF16 对照是 contiguous PyTorch SDPA attention-only，不含 KV 写入/prepare，不是 main 的原生分页后端；只能作为参考，不能据此声称相对 main/vLLM 的加速。
初始化、全量历史 KV 写入、误差参考计算不计入延迟；此工具不能给出 TTFT。

每个 case 使用相同固定种子的正态随机输入。真实模型的 RoPE 后异常值、长上下文误差累积尚未覆盖。
正式比较至少重复三个独立进程，固定形状、PyTorch 版本、CPU 线程数和其他 GPU 负载；先看 raw samples，再解释 p95。三次迭代的冒烟不用于性能结论。
发生异常/OOM 时退出码为 1，记录失败 case，保留已完成行；不要把未跑完的格式当作零耗时。

## 实现检查：已具备与仍需验证

优点：payload 和 scale 一起管理；INT4 实际打包；FP8 存储真实浮点编码；按页反量化结合 online softmax，不展开整个 KV pool。奇数 head_dim、尾页、GQA 等已有 CPU 覆盖。

| 级别 | 位置 | 发现与最小后续路径 |
|---|---|---|
| Important | `quantized.py:60`，`prepare():174`，`_attend_pages():253` | `.item()`/主机元数据搬运及 Python 逐页 dispatch 带来同步与大量 launch；本 benchmark 如实计入。先获得 GPU 分项数据，再决定融合热点，不能仅凭减少字节断言更快 |
| Important | `quantized.py:169`，`worker_impl.cpp` 的量化分支 | 实验路径明确禁止 graph，限制为 CUDA Python eager MHA/GQA；不能据此宣布原生 CUDA、MLA、PD 或 offload 已支持。下一阶段单独验证 C++ 分配、元数据桥接及真实模型入口 |
| Important | FP8 encode/decode 与设备部署 | CPU roundtrip 不能证明 Blackwell CUDA kernel 可执行；用 preflight 加小规模实际后端运行验证，不只检查 cuda.is_available |
| Minor | 本 benchmark 的 BF16 对照 | 连续 SDPA 与量化分页 reference 的算法/计时边界不同。公平的生产性能比较仍需同一服务配置、模型及 native paged BF16 后端 |

本轮未增加大规模测试体系；复跑原有 `test_quantized_attention.py`：58 passed（PyTorch 2.7.1 CPU）。C++/整服务没有编译验证。

## 16GB 设备上的预期与边界

RTX 5060 Ti 是 Blackwell 消费卡，16GB 版本与标称 448GB/s 带宽见 [NVIDIA 官方规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/)。不能直接套用 H100 的 kernel 或测速结论。

默认 D=128、Hkv=8，每个 token 每层 K+V：BF16=4096B；INT8/FP8=2112B（含两个 FP32 scale/head）；INT4=1088B。
所以 KV 理论压缩约 1.94x / 3.76x，而不是精确 2x / 4x。4K context 单层分别 16MiB / 8.25MiB / 4.25MiB；32 层对应 512MiB / 264MiB / 136MiB，未计其他内存。
以 448GB/s 仅计算单层历史 KV 顺序读的理想下界，约 37.4/19.3/9.95 微秒；不是实际 attention 延迟，未计重读、反量化、计算、launch、scale 处理等。本 reference 很可能被 dispatch 而非带宽限制。

本工具只分配一层，不加载模型，因此可以先验证量化与分页逻辑而不耗尽 16GB。
未来整模型实验应从较小模型开始：8B BF16 权重仅按 80 亿参数就约 14.9GiB，额外 KV/activation/runtime 会使 16GB 很紧；KV 量化不能解决所有权重占用问题。
