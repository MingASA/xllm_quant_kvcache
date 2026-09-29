# KV cache benchmark：GPU 单层实验

## 范围与状态

本工具直接调用 `xllm/python/attention/quantized.py`，不复制量化实现。
覆盖 BF16、INT8、FP8 E4M3/E5M2、packed INT4；支持 CPU/CUDA、随机物理页、GQA、decode 和小段 prefill。
这是**单层合成输入的实验后端 benchmark**，不是整模型吞吐测试，不需要权重、xLLM C++ 动态库或完整服务构建。
BF16 与量化格式使用同一组输入、分页映射和 eager online-softmax 分页算法，分开计时写缓存、attention 和完整调用。
这组数据用于回答量化误差、KV 张量实际占用及单层执行速度，不代表模型任务效果或生产服务性能。

## 环境准备

仓库声明 Python `>=3.10`，没有在 `pyproject.toml` 固定 PyTorch 版本。benchmark 需要可用 CUDA 的 PyTorch 环境；本轮使用 `/home/mingasa/venvs/vllm`，Python 3.12.13、PyTorch 2.13.0+cu132，可识别 RTX 5060 Ti（sm_120）。
命令直接调用该环境的 Python，不依赖 shell 激活脚本：

```bash
/home/mingasa/venvs/vllm/bin/python tools/benchmark_kv_cache.py --device cuda --preflight-only --output-dir results
```

需要支持 RTX 5060 Ti 的 NVIDIA 驱动。更换 PyTorch/CUDA 版本后应分开记录结果。无需为这个纯 PyTorch benchmark 单独编译 CUDA Toolkit；未来构建 xLLM/CUDA kernel 另行配置。
preflight 会执行 BF16 CUDA 矩阵乘及所选 codec 的设备端编解码；没有 CUDA 时退出码为 1，绝不悄悄切换 CPU。

## 直接运行

先运行实际后端的小规模 GPU 冒烟，再运行完整矩阵。每条命令生成独立结果目录，不覆盖旧数据。

```bash
/home/mingasa/venvs/vllm/bin/python tools/benchmark_kv_cache.py --device cuda --contexts 33 --batch 2 --query-tokens 3 --head-dim 9 --heads 4 --kv-heads 2 --page-size 16 --warmup 1 --iterations 3 --output-dir /tmp/kv_cache_smoke
/home/mingasa/venvs/vllm/bin/python tools/benchmark_kv_cache.py --device cuda --contexts 1024 4096 --batches 1 4 --query-lengths 1 16 --formats bf16 int8 fp8_e4m3 fp8_e5m2 int4 --warmup 10 --iterations 100 --output-dir results
```

完整矩阵应独立运行三轮；程序会为每轮建立唯一结果子目录。默认 `head_dim=128, heads=32, kv_heads=8, page_size=128` 是合成 GQA 形状，不代表已经运行某个真实模型。
之后可逐级尝试 `--contexts 8192 16384`。保留 `--batch` 和 `--query-tokens` 单值形式可运行旧式单 case 命令；矩阵使用 `--batches` 和 `--query-lengths`。
可用 `--formats int8 int4` 缩小范围；用 `--device cpu` 运行 CPU，同一个程序不需要改代码。

当前 CPU 冒烟命令：

```bash
/home/mingasa/venvs/vllm/bin/python tools/benchmark_kv_cache.py --device cpu --contexts 33 --batch 2 --query-tokens 3 --head-dim 9 --heads 4 --kv-heads 2 --page-size 16 --warmup 1 --iterations 3 --output-dir /tmp/kv_cache_cpu_smoke
```

## 记录内容与比较方法

| 文件/字段 | 用途与限制 |
|---|---|
| `environment.json` | 命令参数、软件包版本、Git commit/dirty 状态、关键源文件 SHA256、驱动输出、GPU 型号/架构/空闲容量、完成或失败状态 |
| `results.jsonl` | 每个 case 的全部指标和逐次延迟；每完成一个 case 刷盘，后续失败仍保留已完成结果 |
| `summary.csv` | 每个格式/context/batch/query/stage 的 p50/p95、tokens/s、缓存字节、峰值分配和相对 BF16 输出误差 |
| `kv_cache_bytes` / `kv_cache_mib` | cache payload 与量化 FP32 scales 的实际张量字节数；包含页尾 padding，不含 allocator/metadata |
| `runtime_peak_allocated_bytes` | CUDA allocator 在计时样本中的峰值分配；绝对值包含输入和 cache fixture |
| `runtime_peak_increment_bytes` | runtime peak 减去计时前已分配字节数，反映运行临时张量增量；不是完整模型 workspace 上界 |
| `output_relative_l2_vs_bf16` / `output_max_abs_vs_bf16` | 与相同分页 BF16 attention 输出相比；不代表困惑度、任务准确率或质量保证 |
| `implementation_max_abs` | 量化后端输出与解码后量化 KV 的独立 dense FP32 attention 参考之间的最大绝对差；断言 rtol=0.02/atol=0.005 |
| `processed_tokens_per_call` / `tokens_per_second_p50` | 每次调用处理的 query token 数及按各次样本延迟计算的吞吐量；是单层合成请求吞吐 |

延迟是每次调用后 CUDA synchronize 的墙钟时间，包含 Python、kernel launch 和同步，预热不计入。各 stage 分别报告 query token 数与吞吐量。
`cache_write` 只写本轮新增 K/V，包含量化编码；`attention` 读取已写入的分页缓存，不重复写；`full_call` 包含分页计划准备、cache write 和 attention。BF16 与量化使用相同输入、物理 page table、有效 context 和算法循环，量化 full call 走真实 experimental backend。
反复执行时覆盖同一批尾部 token，测固定形状稳态，不模拟 autoregressive 长度增长。初始化、全量历史 KV 写入、误差参考计算不计入延迟；此工具不能给出 TTFT。

每个 case 使用相同固定种子的正态随机输入。真实模型的 RoPE 后异常值、长上下文误差累积尚未覆盖。
正式比较至少重复三个独立进程，固定形状、PyTorch 版本、CPU 线程数和其他 GPU 负载；先看 raw samples，再解释 p95。三次迭代的冒烟不用于性能结论。
发生异常/OOM 时退出码为 1，记录失败 case，保留已完成行；不要把未跑完的格式当作零耗时。

## RTX 5060 Ti 首轮数据（2026-09-29）

PyTorch 2.13.0+cu132，RTX 5060 Ti 16 GB；默认 `head_dim=128, heads=32, kv_heads=8, page_size=128`。三个独立进程各完成 40 个 case、每个 stage 100 个计时样本。下表误差为 8 种形状上的最大值；显存与时间为 context 4096、batch 4、query 16 的三轮中位数。

| 格式 | 相对 L2 / 最大绝对误差 | Cache MiB（压缩比） | 峰值 allocated / setup 增量 MiB | 写入 / attention / 完整调用 p50 ms | 完整调用 tokens/s |
|---|---:|---:|---:|---:|---:|
| BF16 | 0 / 0 | 64（1.00×） | 164.14 / 3.26 | 0.592 / 119.932 / 127.225 | 503 |
| INT8 | 0.95% / 0.00293 | 33（1.94×） | 133.39 / 3.51 | 3.147 / 132.387 / 142.107 | 450 |
| FP8 E4M3 | 3.75% / 0.01758 | 33（1.94×） | 133.39 / 3.51 | 2.081 / 128.318 / 140.460 | 456 |
| FP8 E5M2 | 7.43% / 0.03857 | 33（1.94×） | 133.39 / 3.51 | 1.947 / 126.331 / 141.568 | 452 |
| INT4 | 16.96% / 0.05469 | 17（3.76×） | 117.77 / 3.89 | 2.288 / 189.754 / 209.668 | 305 |

BF16 参考缓存为 64 MiB；INT8/FP8 减少 48.4%，INT4 减少 73.4%。Runtime peak 是 PyTorch allocator 的绝对峰值，包含合成输入；`setup 增量` 是运行期超过输入和 cache 已分配基线的峰值增量。
8 种形状逐一比较完整调用，INT8、FP8 E4M3、FP8 E5M2 的量化/BF16 p50 延迟比中位数分别为 1.192×、1.193×、1.195×，INT4 为 1.754×；每种量化格式都在 0/8 种形状上更快。4096/batch4/query16 这个 case 的 full-call p50 在三轮间有约 17%～21% 的极差（相对三轮中位数），因此绝对延迟应看作本轮环境下的估计值。

结果目录：`results/20260929T080019Z_de1782a3`、`results/20260929T080947Z_581e2f88`、`results/20260929T081803Z_c60858e1`。它们记录了完整逐样本延迟、误差、缓存字节和 CUDA allocator 峰值；真实模型任务效果仍待后续阶段验证。
三轮启动时环境没有安装 `pip`，初次 package inventory 命令失败。运行结束后安装了 `pip 26.2.1`，并将安装前的环境包清单补入三份 manifest；清单与当前环境 `pip freeze` 去掉新增 pip 包后的结果一致。benchmark 现从 Python distribution metadata 读取包清单，不依赖 pip。

## 实现检查：已具备与仍需验证

优点：payload 和 scale 一起管理；INT4 实际打包；FP8 存储真实浮点编码；按页反量化结合 online softmax，不展开整个 KV pool。奇数 head_dim、尾页、GQA 等已有 CPU 覆盖。

| 级别 | 位置 | 发现与最小后续路径 |
|---|---|---|
| Important | `quantized.py:60`，`prepare():174`，`_attend_pages():253` | `.item()`/主机元数据搬运及 Python 逐页 dispatch 带来同步与大量 launch；本 benchmark 如实计入。先获得 GPU 分项数据，再决定融合热点，不能仅凭减少字节断言更快 |
| Important | `quantized.py:169`，`worker_impl.cpp` 的量化分支 | 实验路径明确禁止 graph，限制为 CUDA Python eager MHA/GQA；不能据此宣布原生 CUDA、MLA、PD 或 offload 已支持。下一阶段单独验证 C++ 分配、元数据桥接及真实模型入口 |
| Important | FP8 encode/decode 与设备部署 | CPU roundtrip 不能证明 Blackwell CUDA kernel 可执行；用 preflight 加小规模实际后端运行验证，不只检查 cuda.is_available |
| Minor | 本 benchmark 的性能范围 | BF16 与量化都使用单层 Python eager 分页算法，包含相同的逻辑调用边界；结果不能替代同一服务配置、模型及 native backend 的生产比较 |

本轮用 PyTorch 2.13.0+cu132 在 CUDA 上完成 context 33 的五格式小规模冒烟，并完成三轮 1K/4K benchmark（共 120 个 case）；`py_compile` 和 CUDA preflight 通过。没有重跑 pytest 套件，也没有编译 C++/完整服务。

## 16GB 设备上的预期与边界

RTX 5060 Ti 是 Blackwell 消费卡，16GB 版本与标称 448GB/s 带宽见 [NVIDIA 官方规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/)。不能直接套用 H100 的 kernel 或测速结论。

默认 D=128、Hkv=8，每个 token 每层 K+V：BF16=4096B；INT8/FP8=2112B（含两个 FP32 scale/head）；INT4=1088B。
所以 KV 理论压缩约 1.94x / 3.76x，而不是精确 2x / 4x。4K context 单层分别 16MiB / 8.25MiB / 4.25MiB；32 层对应 512MiB / 264MiB / 136MiB，未计其他内存。
以 448GB/s 仅计算单层历史 KV 顺序读的理想下界，约 37.4/19.3/9.95 微秒；不是实际 attention 延迟，未计重读、反量化、计算、launch、scale 处理等。本 reference 很可能被 dispatch 而非带宽限制。

本工具只分配一层，不加载模型，因此可以先验证量化与分页逻辑而不耗尽 16GB。
未来整模型实验应从较小模型开始：8B BF16 权重仅按 80 亿参数就约 14.9GiB，额外 KV/activation/runtime 会使 16GB 很紧；KV 量化不能解决所有权重占用问题。
