<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# xLLM KV Cache 审计与量化实现说明

审计日期：2026-09-19。原项目基线：`cf4c7732d52da3a39cb35b0125a6b08cf2229c69`。
本地实现分支：`feat/kv-cache-quantization`。修改尚未提交或推送。

2026-09-20 后续探索见 [多格式量化实验与实现说明](kv_cache_quantization_exploration_zh.md)：在独立 CPU lab 中增加 INT2～INT8、FP8、NF4、FP4、分组/非对称量化、旋转与高精度尾部，附 69 个合成案例的实际存储/误差结果。新增格式没有接入服务端参数，不扩大本文生产路径的支持范围。

进一步的机制探索见该文第 8 节：以最差 head 与最差 token 向量误差双门控，在真实 tensor 字节成本下选择页面格式，附自适应与固定 INT4/INT8 的对比。它允许为了误差预算主动保留 native 页面，不宣称量化后必然更快或始终达到某个压缩比。

最新 [机制探索报告](kv_cache_mechanisms_zh.md) 增加页中心化、低秩/稀疏误差修正、head 独立选型，并发现和处理整页量化统计的因果边界；此前 lab 的全缓存重构比较不能直接作为在线因果生成验证。此问题针对跨 token 的实验策略，不改变本文原 per-token/head codec 的支持声明。

## 1. 结论与本次交付范围

**原项目不是完全不支持 KV cache 量化，而是已有 MLU INT8 的专用实现，尚未形成跨后端、跨格式的完整支持。** NPU 的 INT8 indexer cache 也已存在，但 indexer 是稀疏注意力的索引缓存，不等于主 K/V 已量化。

主要障碍不是缺少 `int8`/`float8` 类型定义，而是以下链路没有统一：

```text
配置格式 → 容量估算 → 物理布局与分配 → 量化写入
                                       ↓
                    scale 的持有、传递、复制、生命周期
                                       ↓
                           attention 读取与反量化
```

本次在原有存储框架上接通了 **CUDA + Python eager 的实验性量化 attention 路径**，实现 INT8、FP8 E4M3、FP8 E5M2 和实际打包的 INT4；CPU 用于验证同一 Python 实现的数值与分页行为。它不是完整生产支持：本机没有 CUDA/NPU/MLU 设备与完整 xLLM C++ 构建环境，未做服务器端到端、设备内核性能或模型精度验收。尤其不能把本次修改描述成“已实现 NPU KV 量化”或“已达到 vLLM 的融合内核性能”。

## 2. 原项目具体在哪里断开

本节链接固定到审计前的 commit，行号不受本地修改影响。

### 2.1 配置看起来有 dtype，但执行端只识别 INT8

原 [worker_impl.cpp:514](https://github.com/xLLM-AI/xllm/blob/cf4c7732d52da3a39cb35b0125a6b08cf2229c69/xllm/core/runtime/worker_impl.cpp#L514) 的关键判断是：

```cpp
const bool enable_kv_cache_quant = options_.kv_cache_dtype() == "int8";
```

随后在非 MLU 编译中直接报错。`kv_cache.cpp`、`create_quantized_kv_cache_tensors()` 也有仅 MLU 的限制。原 [配置校验](https://github.com/xLLM-AI/xllm/blob/cf4c7732d52da3a39cb35b0125a6b08cf2229c69/xllm/core/framework/config/kv_cache_config.cpp#L149) 只检查 `indexer_cache_dtype`，没有严格校验主 `kv_cache_dtype`。

结果是：增加 FP8/INT4 参数字符串，不能让分配器、写入算子和 attention 自动具备对应能力。

### 2.2 FP8 容量估算与实际分配不一致，是明确的正确性风险

原 [kv_cache_estimation.cpp:38](https://github.com/xLLM-AI/xllm/blob/cf4c7732d52da3a39cb35b0125a6b08cf2229c69/xllm/core/framework/kv_cache/kv_cache_estimation.cpp#L38) 已把 `fp8_e4m3`、`fp8_e5m2` 按每元素 1 字节估算。然而 worker 不会为这两个字符串开启量化，普通分配器继续使用模型 dtype。

例如模型为 BF16，head_dim=128：估算每个 K/V 向量为 `128 + 4` 字节（包括假定的 scale），实际非量化向量仍为 `128 × 2` 字节。实际 payload 与估算的比值约为 `256/132=1.94`。据此计算出的可分配 block 数可能明显过大，存在启动分配 OOM 风险。这里是静态代码推导，未在本机复现设备 OOM。

这是本次必须同时修改估算和分配的原因：仅增加一个 FP8 attention 分支仍不足以保证内存预算正确。

### 2.3 已有 QuantizedKVCacheImpl，物理格式却固定为 INT8

原 [kv_cache_utils.cpp:285](https://github.com/xLLM-AI/xllm/blob/cf4c7732d52da3a39cb35b0125a6b08cf2229c69/xllm/core/framework/kv_cache/kv_cache_utils.cpp#L285) 在量化分配中写死：

```cpp
quantized_options.dtype(torch::kChar);
```

K/V scale 是去掉末尾 head_dim 后的 FP32 张量，即每 token、每 KV head 一个 scale。这对原有 INT8 路径合理，但还缺少：

- FP8 E4M3/E5M2 的编码区分、转换及反量化；
- INT4 的半字节打包、奇数 head_dim 的 padding；
- 逻辑 head_dim 与物理存储维度的区分；
- 与格式一致的容量计算。

把 INT4 数值放进 INT8 张量、却保留原来的元素数量，不能获得 4-bit 存储节省。

### 2.4 MLU 已有完整专用链路，CUDA/NPU 没有对应接线

原 [MLU attention](https://github.com/xLLM-AI/xllm/blob/cf4c7732d52da3a39cb35b0125a6b08cf2229c69/xllm/core/layers/mlu/attention.cpp#L96) 会读取 K/V scale，用 `quant_to_paged_cache` 写入，并把 scale 传给 decode attention。但参数中量化位宽写死为 8；chunked prefill 则先 `dequant_from_paged_cache`，再执行浮点 flash attention。

原 [CUDA FlashInfer 路径](https://github.com/xLLM-AI/xllm/blob/cf4c7732d52da3a39cb35b0125a6b08cf2229c69/xllm/core/layers/cuda/flashinfer_attention.cpp#L150) 使用普通 `reshape_paged_cache`，没有把 K/V scale 接进该路径；plan 还使用输入 key 的 dtype。仅把 cache 分配成 8-bit，会造成读写布局和数值语义不匹配。

原 [NPU FIA 调用](https://github.com/xLLM-AI/xllm/blob/cf4c7732d52da3a39cb35b0125a6b08cf2229c69/xllm/core/kernels/npu/npu_fused_infer_attention.cpp#L607) 将 `antiquant_scale`、`key_antiquant_scale`、`value_antiquant_scale` 等传为 `none_tensor`。这说明仓库没有接入这些反量化输入，不能据此判断 Ascend 硬件完全不支持量化。[Ascend 官方 FIA 文档](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md) 列出了量化/反量化参数与 INT8/INT4 数据类型；具体可用组合仍受芯片、CANN 版本、布局和算子约束影响。

### 2.5 Python executor 没有拿到主 K/V 的 scale

原 [py_executor_impl.cpp:181](https://github.com/xLLM-AI/xllm/blob/cf4c7732d52da3a39cb35b0125a6b08cf2229c69/xllm/core/runtime/py_executor_impl.cpp#L181) 绑定的缓存 tuple 有 K/V、indexer、linear/DSV4 状态与 indexer scale，但没有主 K/V scale。原 `LayerCache` 同样没有这两个字段。

即使 C++ 分配了量化数据，Python attention 也无法按 scale 还原。该缺口与“Python 是否支持 float8 dtype”是两个不同问题。

### 2.6 缓存复制、搬运也属于量化协议

`QuantizedKVCacheImpl::swap_blocks()` 已会复制 K/V 和 scales，这是可复用的基础。但原 CUDA 融合 block-copy 路径只持有 K/V 指针，直接用于量化缓存会漏掉 scale。

此外，原 `get_cache_tensors()` 可以枚举主 K/V scales，但 `QuantizedKVCacheImpl` 没有覆盖 `get_block_type_tensors()`，后者继承的基础实现只返回 payload。这是不同缓存枚举接口之间的契约缺口；不能因为一个接口有 scales，就假设所有 transfer/store 路径都完整。

原项目已经禁止主 KV 量化与 PD 分离、host offload 的部分组合。本次继续限制这些组合。新增 FP8 用 uint8 保存编码后，未来 transfer/store 协议还必须显式携带格式、量化粒度、scale 约定与布局版本；单靠 `uint8 + shape` 无法区分 E4M3 和 E5M2。

## 3. KV cache 效率与算法评价

### 3.1 已有优势（Strengths）

| 机制与代码证据 | 对 KV 效率的收益 | 适用边界 |
| --- | --- | --- |
| [BlockManagerImpl](../xllm/core/framework/block/block_manager_impl.cpp)、逻辑 block table | 按块增长，避免每个请求预留完整最大上下文；空闲 block 可重用 | 仍有尾块内部碎片、元数据与保留 block 开销 |
| [PrefixCache](../xllm/core/framework/prefix_cache/prefix_cache.cpp) 的链式 XXH3-128 hash、哈希表、LRU | 相同完整前缀可共享物理 KV，减少重复 prefill 和存储 | 收益取决于命中率；命中前缀仍需被后续 decode 读取 |
| Block 引用计数、`swap_blocks` | 支持共享和分叉时复制，避免总是深拷贝整段 KV | payload 与 scale 必须共同复制 |
| [KVCacheShape](../xllm/core/framework/kv_cache/kv_cache_shape.cpp) 的局部 KV head、MLA 形状 | GQA/MQA 减少 KV head 数；MLA 保存潜变量，减少缓存维度 | 这是模型结构收益，不代表 xLLM 独有的量化算法优势 |
| [滑窗管理](../xllm/core/framework/block/sliding_window_block_manager.cpp)、linear state、DSV4 分组池 | 可按不同注意力类型管理存储，回收不再需要的窗口状态 | 不能把所有状态都套用同一 K/V 量化布局 |
| [分层缓存传输](../xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp) | 扩展前缀缓存层级，增加复用机会 | PCIe/网络传输有代价，不自动降低单 token 时延 |
| [Zero-eviction 调度](../xllm/core/scheduler/zero_eviction_scheduler.cpp#L200) | 模拟后续 block 需求，尝试减少驱逐后重复 prefill | 依赖 decode 长度估计，保守 admission 可能牺牲并发 |

这套系统在 KV 生命周期管理上已有较完整的工程基础。其 paging、prefix sharing 思路与公认的 [PagedAttention](https://arxiv.org/abs/2309.06180) 属于同类设计。**仅依据源代码，不能证明 xLLM 在相同模型、硬件、SLO 下优于 vLLM**；多平台适配和多种 cache 类型是工程覆盖面，而不是统一的性能排名。

### 3.2 容量收益应该怎样计算

普通 MHA/GQA，假设 K/V 的 head_dim 都为 D，单 rank 持有 L 层、H 个本地 KV head、T 个 token：

```text
FP16/BF16:    M = 2 × L × H × T × (2D)
INT8/FP8:     M = 2 × L × H × T × (D + 4)
本次 INT4:    M = 2 × L × H × T × (ceil(D/2) + 4)
```

括号中的 `4` 是每 token/head 的 FP32 scale；K、V 分别有自己的 scale。不计页尾碎片、allocator 对齐、QKV 激活、attention workspace 和模型权重。

以 L=32、H=8、D=128、T=4096 为例：

| 格式 | 每 token 的全部层 KV | 4096 tokens | 相对 BF16 容量倍数 |
| --- | ---: | ---: | ---: |
| BF16/FP16 | 128 KiB | 512 MiB | 1.00× |
| INT8/FP8 + FP32 scales | 66 KiB | 264 MiB | 1.94× |
| packed INT4 + FP32 scales | 34 KiB | 136 MiB | 3.76× |

这些是公式值；新增 Python 测试也检查了实际 payload + scales 的 tensor 字节数。它们不是吞吐或 GPU 峰值显存测量。MLA、不同 K/V 维度、混合层、TP 复制 KV head 等场景必须重新按实际布局计算。

默认 block_size=128。若请求长度对 block_size 的余数近似均匀，每个活跃序列尾块平均浪费约 `(128-1)/2=63.5` 个 token slot；最坏为 127。长请求下占比较小，短请求高并发时可能明显。小 block 可减少碎片并细化前缀匹配，却增加 block table、hash 和寻址开销，需结合后端支持实测。

### 3.3 带宽节省不等于等比例提速

decode 通常需读取历史 KV。压缩 payload 可减少其读流量，但会增加 scale 读取、解包、类型转换和反量化。实际收益取决于注意力是否受 KV 带宽限制、是否融合反量化、batch/context 长度，以及其他层计算占比。

原 MLU chunked prefill 分配 `total_seqlens × heads × dim` 的反量化 K/V 临时张量，随后再执行 flash attention。因此虽然常驻 KV 已压缩，峰值 workspace 和读写流量仍可能较大。这属于应优化的路径，而不是“INT8 一定更慢/更快”的证据。

本次参考实现按页反量化，每次处理最多 64 个 query。采用在线 softmax 合并各页结果，score 临时张量约为 `Hq × 64 × page_size`，不创建全缓存的 BF16/FP32 副本，也不创建完整 Q×T score 矩阵。写入阶段仍有当前输入 K/V 的 FP32 量化临时张量。

但它包含 Python 页循环、逐层检查、设备到主机的元数据读取和多个 PyTorch 算子启动；长 prefill 会按 query tile 多次扫描 KV。**本次方案是内存布局与正确性基线，预期不能与融合 attention 内核竞争吞吐。** 在线 softmax 的合并思想可参考 [FlashAttention 论文](https://arxiv.org/abs/2205.14135)，本实现并不是 FlashAttention CUDA 内核。

## 4. 对照公认实现：应该借鉴什么

以下资料于审计日期查阅，`latest/main` 会变化，不能将开发分支能力推广到所有历史版本和后端。

| 参考 | 已查证的做法 | 对本项目的意义 |
| --- | --- | --- |
| [vLLM FP8 KV 文档](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) | 区分 per-tensor、per-head scale 与校准方式；存在后端约束 | 应将格式、scale 粒度、后端能力一起建模，不能只保存 dtype 字符串 |
| [vLLM v0.28.0 Triton backend](https://docs.vllm.ai/en/v0.28.0/api/vllm/v1/attention/backends/triton_attn/) | per-token/head 的 scale 占用纳入 cache 布局；INT4 按半字节 payload 计算空间 | 预算必须覆盖量化元数据，写入和 attention 必须遵循同一布局 |
| [vLLM INT4 实现](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/ops/int4_per_token_head/) | 有独立的打包、写入、attention 读取和 RHT 变换，包含零点处理 | INT4 是完整算法路径，不能把 INT8 的 bit 参数改成 4 就结束 |
| [LMDeploy INT4/INT8](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html) | per-head、per-token 的在线非对称量化 | 展示了无需离线 KV 校准的工程路线；其精度/性能结果不能直接移植到 xLLM |
| [KIVI](https://arxiv.org/abs/2402.02750) | 根据分布差异对 key 使用 per-channel、对 value 使用 per-token 量化 | 低位宽时应考虑 K/V 的不同离群值结构；本次简单对称 per-token/head INT4 不是 KIVI 复现 |

本次选择动态 per-token/head scale，是因为它可以随 token 独立写入，已缓存前缀的 scale 不需被后续 token 更新，便于和现有分页与共享前缀结合。代价是每个 K/V 向量额外 4 字节，且 INT4 容易受单个 head 内离群值影响。

FP8 的动态范围更大并不意味着精度一定更好：E4M3 和 E5M2 的有效精度不同；INT8 在合适 scale 下也有优势。低位宽应通过困惑度、长上下文检索和实际任务评测选型，不能只比较随机张量误差。

## 5. 本次如何修改

### 5.1 统一格式解析，修复预算与实际存储的断开

新增 [kv_cache_dtype.h](../xllm/core/framework/kv_cache/kv_cache_dtype.h)：

- `KVCacheDtype` 显式区分 AUTO、INT8、FP8_E4M3、FP8_E5M2、INT4；
- `fp8` 作为 E4M3 的别名；未知字符串直接报错；
- `kv_cache_storage_head_dim()` 统一 INT4 的 `ceil(D/2)` 存储维度。

[KVCacheConfig](../xllm/core/framework/config/kv_cache_config.cpp) 增加主 dtype 校验和帮助说明；[容量估算](../xllm/core/framework/kv_cache/kv_cache_estimation.cpp) 使用同一解析规则和打包维度；[worker](../xllm/core/runtime/worker_impl.cpp) 对所有已识别量化格式启用量化分配，避免 FP8 被当成普通 BF16 分配。

### 5.2 保留逻辑维度，按格式分配物理 payload

[KVCacheCreateOptions](../xllm/core/framework/kv_cache/kv_cache_utils.h) 新增 `quantized_dtype`，旧调用仅设置 `enable_kv_cache_quant(true)` 时仍默认 INT8。

[分配实现](../xllm/core/framework/kv_cache/kv_cache_utils.cpp) 的变化：

| 格式 | 物理 dtype | 最后一个物理维度 | 编码含义 |
| --- | --- | --- | --- |
| INT8 | int8 | D | 对称量化整数 |
| FP8 E4M3 | uint8 | D | 真正 float8_e4m3fn 的位编码 |
| FP8 E5M2 | uint8 | D | 真正 float8_e5m2 的位编码 |
| INT4 | uint8 | ceil(D/2) | 每字节两个 signed nibble，低位保存前一个通道 |

FP8 使用 byte 容器，是为了让缓存 scatter/copy 不依赖某个 PyTorch 版本的 float8 索引算子覆盖面；转换时使用实际 float8 dtype 再 `view(uint8)`，读取时按正确 FP8 格式 reinterpret，**不是把浮点直接转成 uint8 数值**。

### 5.3 接通 C++ → Python 的 scale 契约

[py_executor_impl.cpp](../xllm/core/runtime/py_executor_impl.cpp) 将运行时 `kv_cache_dtype` 传给 Python executor，并在原有 11 个缓存 tuple 字段后追加 K/V scales。

[LayerCache](../xllm/python/attention/backend.py) 增加 `key_scale`、`value_scale`。旧 tuple 字段顺序保持兼容；短 tuple 中缺少的字段仍补为 `None`。量化 backend 在绑定时检查 payload、scale 的形状、dtype、连续性和设备。

### 5.4 实现真正的量化写入与 attention 读取

新增 [quantized.py](../xllm/python/attention/quantized.py)：

1. `KVCacheCodec`：K/V 分别按 token/head 动态 absmax 取 scale；INT8 使用 [-127,127]，INT4 使用 [-7,7]，FP8 按相应格式范围缩放。
2. 对零向量使用 scale=1；对极小 scale 设有限正下界；拒绝 NaN/Inf 输入。
3. INT4 使用补码 nibble 打包；奇数 D 的最后一个高 nibble 补零。
4. `write_quantized_kv()` 同时写 payload 与 scales，忽略 -1 padding slot，拒绝越界或重复写入；K/V 两侧编码完成后才修改缓存。
5. `QuantizedPagedAttentionBackend` 按 block table 读取实际页面，逐页反量化并在线合并 softmax；支持普通 MHA/GQA 的 prefill、chunked prefill 和 decode，使用右对齐因果 mask。
6. 测试覆盖共享完整前缀、独立尾块和滑窗 attention mask。滑窗 mask 测试不代表已支持独立滑窗回收池；分组缓存仍被限制。

[executor.py](../xllm/python/model_executor/executor.py) 在显式指定量化格式时选择该 backend。它不会在 FlashInfer 执行失败后尝试量化或静默改回浮点路径。

### 5.5 修复复制与枚举，保留明确的能力边界

[QuantizedKVCacheImpl](../xllm/core/framework/kv_cache/quantized_kv_cache_impl.cpp) 的 block-type tensor map 现在包含 K/V scales，原有 `swap_blocks()` 的 payload+scale 同步复制继续使用。

[worker](../xllm/core/runtime/worker_impl.cpp) 对量化 cache 跳过只处理 K/V payload 的 CUDA 融合 block-copy，使用已有的包含 scale 的复制实现。

启动检查限制实验路径为 CUDA、Python 模型、eager、普通 MHA/GQA；拒绝 native CUDA、NPU、MLA、indexer、混合 linear/分组缓存、CP/分层切分、speculative decode、PD、host offload/store、XTensor 和 sleep mode 等尚未接通的组合。MLU 继续保留原 INT8 专用路径，不开放 FP8/INT4。

## 6. 使用方式与验证结果

### 6.1 在具备 CUDA 构建环境的机器上尝试

从本分支重新编译 xLLM；Python 包也必须来自本分支。选择仓库已有 Python 实现的普通 GQA 文本模型，例如 Qwen3。以下是需追加到正常启动命令的选项，不是本机已验收的完整服务命令：

```bash
--model_impl=python \
--kv_cache_dtype=int8 \
--enable_graph=false \
--enable_prefill_piecewise_graph=false \
--python_graph_backend=off
```

`kv_cache_dtype` 可替换为 `fp8`、`fp8_e4m3`、`fp8_e5m2`、`int4`。`float8` 不是本分支的参数别名。保留默认关闭的 PD、host offload、XTensor、sleep 和 speculative 功能。若启用了 in-batch prefix 复用，需要额外验证调度生成的 slot 是否存在同批重复写入；参考 backend 会拒绝该情况。

### 6.2 已完成的本地验证

测试环境：Linux CPU、Python 3.12、临时 venv 中的 PyTorch `2.14.0+cpu`；无加速卡。

- 新增 [量化测试](../tests/python/test_quantized_attention.py)：58 项通过，包括三类格式、两种 FP8 编码、奇数维度 INT4、scale 字节开销、分页 scatter、跨页 chunked prefill/decode、共享前缀、GQA/MHA、滑窗、query tiling、异常输入与 tuple 兼容。
- 加上 [executor 测试](../tests/python/test_model_executor.py) 后，可在本机运行的选择集为 **139 passed，4 deselected**。完整尝试中的四个失败是已有 NPU 分派测试依赖 `torch_npu` 或 `torch.device("npu")`，当前 CPU 环境不满足条件；没有把它们算作通过。
- 新增 C++ 回归用例覆盖配置解析、8-bit/INT4 容量估算、真实分配大小以及 payload/scale 的重叠 block 复制。这些 C++ 用例**尚未执行**：本机没有现成 xLLM 构建目录、完整依赖和设备 SDK。
- 修改过的 C++ 文件使用仓库 `.clang-format` 和 clang-format **20.1.6** 格式化；Python 使用 Ruff 检查。

CPU 核心验证命令：

```bash
python -m pytest tests/python/test_quantized_attention.py -q
```

CPU 环境下的 executor 验证选择集：

```bash
python -m pytest tests/python/test_quantized_attention.py tests/python/test_model_executor.py -q \
  -k 'not test_deepseek_v4_creates_dsa_backend and not test_npu_device_creates_npu_backend and not test_prefill_cp_uses_npu_backend_with_dcp_group and not test_decode_cp1_uses_sfa_dcp_backend'
```

CPU 数值测试说明算法与缓存读写自洽，不证明 CUDA 算子组合、C++ 编译和完整服务器启动已通过。随机张量比较也不能替代模型质量评测。

## 7. 审计问题与优先级（Issues）

### Critical：必须修正

- **原 FP8 预算与分配断开**：原 `kv_cache_estimation.cpp:38` 与 `worker_impl.cpp:517` 对 dtype 的理解不一致，可能高估可用 blocks 并 OOM。本分支通过统一解析、物理分配和能力检查处理；仍需 CUDA C++/服务验收。

### Important：应继续完成

- **缺少后端能力闭环**：原 `kv_cache_utils.cpp:285`、CUDA/NPU attention 读写路径、`py_executor_impl.cpp:181`。本分支只接通 CUDA Python 参考路径；NPU、native CUDA 和设备融合内核仍待实现。
- **量化元数据不是所有生命周期接口的一等成员**：原基础 `get_block_type_tensors()` 不枚举 K/V scales，CUDA 融合复制不处理 scales。本分支修复普通 QuantizedKVCacheImpl 的枚举与复制选择；不据此开放 indexed cache、PD/host/store 的量化兼容。
- **原 MLU chunked prefill 的全量反量化临时空间**：`core/layers/mlu/attention.cpp:180`。建议用可消费量化 pages 的 attention 或分块反量化，实际收益须 profiling 验证。
- **缺少端到端质量/性能证据**：仅实现格式不足以称为生产支持。FP8/INT4 尤其需要长上下文检索、困惑度、业务任务和多轮增量生成对比。

### Minor：可按 profiling 结果优化

- [block_hasher.cpp:41](../xllm/core/framework/prefix_cache/block_hasher.cpp#L41) 在非首个块的 hash 计算中分配和释放临时拼接缓冲；预计算 hash 路径能减少重算，但未覆盖路径仍有分配开销。
- [prefix_cache.cpp:237](../xllm/core/framework/prefix_cache/prefix_cache.cpp#L237) 的 LRU 驱逐可能扫描并跳过仍被引用的节点；[ConcurrentBlockManagerImpl](../xllm/core/framework/block/concurrent_block_manager_impl.cpp) 使用锁保护操作。需要测量高并发命中/驱逐时的锁占用与尾延迟后，决定是否优化可驱逐队列。

## 8. 后续建议与合入判断（Recommendations / Assessment）

建议以本次可复现的格式/数值测试作为契约，分后端推进：

1. **CUDA**：先完成 C++ 编译和真实模型端到端测试，再将 per-token/head 量化写入、FP8/INT8 读取和 INT4 解包融合到设备内核，补 graph 与 block-copy 测试。
2. **NPU**：按目标芯片/CANN 的 FIA 合法组合设计 INT8 scale 布局，接入量化写入及 antiquant 参数；NZ/ND 不能混用。FP8 与 INT4 分别评估算子和硬件支持，不能解除一个宏限制就宣告支持。
3. **低位宽精度**：比较本次对称 INT4、非对称 per-token/head、分组量化，以及 KIVI/RHT 等方法；必要时保留敏感层或最近 token 的高精度数据。
4. **跨层级搬运**：为 payload、scales、格式标识和量化参数定义可版本化的共同 schema，然后再开启 offload/store/PD。
5. **公平性能测试**：固定模型、权重 dtype、硬件、SLO、输入/输出长度、batch、前缀命中率、scheduler 与 graph 设置，分别报告常驻/峰值显存、TTFT、TPOT、吞吐、命中率与质量。不能把增加 batch 后的吞吐收益称为固定 batch 下等比例时延下降。

**Ready to merge? 作为生产量化支持：No。作为供进一步设备验证的实验分支：可审阅。** 原因是软件链路和 CPU 参考数值已覆盖，但 C++ 构建、设备端到端执行、质量验收与融合性能路径尚未完成。当前交付用于把原来分散的量化缺口变成明确、可测试的实现与后续工作边界。
