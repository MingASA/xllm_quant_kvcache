<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# KV cache 量化探索：从支持 dtype 到可组合的存储算法

日期：2026-09-20；本地分支：`feat/kv-cache-quantization`。

后续机制、反例与因果性修正见 [机制探索报告](kv_cache_mechanisms_zh.md)。注意：本文原 69/9 个案例是回顾性缓存重构比较；整页校准可能间接使用页内较早 query 之后的数据，不能据此验证在线因果生成。新 `--mechanisms` 将 query chunk 保留在高精度尾部，并启用显式因果检查。

本轮是提前研究，不是生产上线：保留此前的工程接线实验，另外增加独立 CPU 实验模块，比较更多格式、粒度和缓存策略。**下文的 INT2/3/5/6/7、NF4、FP4 等新增能力只属于实验模块，不是 xLLM 服务端已经支持的新参数。** 没有新增生产入口或扩大硬件支持声明。

原工程审计、固定 commit 的代码证据以及此前 CUDA Python eager 接线说明，见 [KV cache 审计与量化实现说明](kv_cache_audit_zh.md)。本文着重解释下一步解决什么问题、怎么实现、有什么代价。

## 1. 原项目为什么难以支持量化

原项目并非完全没有量化：已有 MLU INT8 专用路径。但配置、预算、物理分配、量化写入、scale 传递、attention 读取、缓存复制没有统一契约。典型断点是：估算器认识 FP8，worker 却只把 `int8` 当量化；分配器固定 INT8；Python 缓存 tuple 缺少主 K/V scales；CUDA/NPU attention 没有接通对应 scale 输入。因此，增加一个 dtype 字符串不能形成完整功能，甚至会出现“按一字节预算、实际按两字节分配”的错误。

原有分页、前缀共享、GQA/MLA 布局是良好基础，但不自动解决低位宽精度。此前补上的 per-token/head 对称 INT8/FP8/INT4 只能作为基线：一个离群值会放大整个组的 scale；偏移分布浪费对称区间；K/V 的分布不同；进一步减少位宽后，scale 和零点本身又可能吃掉大量收益。

本轮因此把两个问题分开：**工程链路是否完整**由前一份审计解释；**什么量化算法值得接入**由独立实验回答。没有设备融合内核和真实模型评测前，不把后者直接合进生产调度器。

## 2. 新实现分别负责什么

| 文件 | 职责 | 明确不负责 |
| --- | --- | --- |
| [kv_cache_codec_lab.py](../tools/kv_cache_codec_lab.py) | `QuantSpec` 描述格式、分组、scale、变换；`encode()` 返回自包含 payload 与元数据；`decode()` 还原 FP32 | 生产 cache ABI、设备融合内核 |
| [kv_cache_quantization_lab.py](../tools/kv_cache_quantization_lab.py) | K/V 独立策略、分页封存、高精度尾部/sink、前缀分叉、逐页 attention、字节统计 | 调度、PD 传输、offload、GPU graph |
| [explore_kv_quantization.py](../tools/explore_kv_quantization.py) | 固定种子合成分布，对比实际存储和 K/V、attention 误差 | 困惑度、真实业务质量、GPU 吞吐排名 |

统一输入为 `[tokens, kv_heads, head_dim]`。格式描述不再只有位宽，而是 `(format, group_size, axis, affine, scale_dtype, rotation, outlier_fraction, seed)`；K、V 可以分别选择。解码统一回原坐标系的 FP32，使分页 attention 不必知道各页是 INT3、NF4 还是旋转后的 FP8。

## 3. 格式与算法：扩大覆盖面，但不混淆名称

| 选项 | 实际编码/处理 | 解决的问题与代价 |
| --- | --- | --- |
| INT2～INT8 | 真实 bit packing；INT3/5/6/7 支持跨字节；对称或 affine | 可研究连续的容量/误差权衡；非整字节解包未必适合实际设备 |
| FP8 E4M3/E5M2 | 转为真实 float8，再保存其 uint8 位编码 | 比较不同动态范围与有效精度；不能把浮点直接 `to(uint8)` 当 FP8 |
| NF4 | 非均匀 16 项码本索引，两个索引/字节 | 对近似正态分布更有针对性；不是通用“4-bit 浮点” |
| FP4 E2M1 | 带符号的 `0, 0.5, 1, 1.5, 2, 3, 4, 6`，4-bit 编码 | 研究低精度浮点码本；不是 bitsandbytes 的 FP4 映射，也不代表完整 NVFP4 |
| native | 保留输入 dtype 的独立副本 | 无损基线及高精度页面 |
| token/channel 分组 | token：每个 token/head 沿 D 分组；channel：每个 head/channel 沿 T 分组 | 缩小组内动态范围；组越小，元数据占比越大 |
| float32/float16/pow2 scale | 4/2/1 字节 scale；pow2 保存向上取整的二进制指数 | 降低元数据；FP16 有范围限制，pow2 的量化步长可能增大 |
| RHT | 确定种子符号翻转 + 归一化 Hadamard 变换 | 摊平部分离群通道；增加变换成本，非二次幂 D 需 padding |
| 离群值旁路 | 每组保留 top-k 绝对值，用 FP32 值 + INT32 索引 | 主体不必为极端值让出动态范围；每个旁路元素额外 8 字节 |

NF4 码本参照 [bitsandbytes 的实现](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py)。FP4 仅实现数值编码；[NVIDIA NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) 还定义分块 E4M3 scale 与全局 FP32 scale 等约定。本实验没有实现这套完整布局，也不能把“FP4 + pow2 scale”称为兼容 MXFP4 的硬件实现。

### 3.1 非对称量化不是简单多存一个 zero-point

整数对称量化使用组内 absmax 和有符号对称范围。affine 使用组内 min/max，在非负整数编码区间保存数据，并存浮点 offset：`x_hat = q * stored_scale + offset`。这适合带偏移或明显非对称的分布，但需要额外 offset 字节。它不是与任何外部内核零点 ABI 自动兼容的格式。

关键细节是**先确定实际存下来的 scale，再用它量化**。如果用 FP32 scale 编码，随后独立把 scale 压成 FP16/pow2，解码器看到的步长就不同，误差会额外扩大。全零/常数组、极小 scale、非有限值、padding 都需明确处理；用于分组补齐的 padding 不参与 min/max 校准。数值范围无法满足所选 scale 表示时应显式失败，而非产生悄悄传播的 NaN。

### 3.2 K、V 不必使用相同策略

[KIVI](https://arxiv.org/abs/2402.02750) 根据 K/V 分布差异采用不同量化方向。本实验借鉴其 K-channel/V-token 和高精度 residual 思路，允许例如 K 用 INT8、V 用 INT4，或 K 沿 token 轴成组、V 沿 head_dim 成组；**这不是 KIVI 的完整复现，也没有继承论文的精度或吞吐结论**。

channel 分组会遇到增量生成问题：新 token 若更新旧组 extrema，就必须重新编码已有 token，同时影响共享前缀。解决方式是只对完整旧页做一次编码，scale 随页面冻结；未封页部分保持原精度。代价是组不能跨越页面无限延伸，page_size 与 group_size 会共同决定 padding 和元数据开销。

### 3.3 旋转与旁路的正确性边界

[vLLM 的 INT4 实现](https://docs.vllm.ai/en/v0.27.1/api/vllm/v1/attention/ops/int4_per_token_head/) 包含打包、RHT 和零点处理，说明低位宽支持不只是修改 dtype。这里采用归一化正交 Hadamard：编码做符号翻转和变换，解码先逆变换再恢复符号；非二次幂 head_dim 先补齐，必须在完整维度逆变换后再裁剪。

本实验在 decode 时恢复原始 K/V 坐标，所以 query 不需要旋转；这比融合 attention 内核容易对照，但多了计算，且与 vLLM 的具体布局、归一化和元数据编码不兼容。若启用旁路，离群值保存在变换后的分组空间，先还原旁路值再逆变换，不能把两个坐标空间的值混用。

## 4. 缓存生命周期比单次张量量化更重要

追加数据先进入高精度尾部。设页长 P、最少保留尾长 R；当尾部不少于 `R + P` 时，把最旧 P 个 token 编成新页面。总长度达到 R 后，尾长在 `[R, R+P-1]`，并非恰好 R。即使 `R=0`，不足一页的尾部也保持高精度，避免反复重量化。

`sink_tokens` 用于保留前部高精度数据；策略按完整相交页保留，因此实际保留量可能向上取整到页边界。它是可选容量/精度策略，不意味着所有模型都需要 attention sink。

封存页面不再被 append 修改。分叉共享已封存前缀，复制可变尾部，避免一条分支改写另一条分支的 scale。注意：这是 API 的不可变使用约定，PyTorch Tensor 本身不是强制只读；外部不能对已发布页面张量原地修改。尾部保存独立副本，不以切片 view 留住整段旧的未压缩底层存储。

attention 按页解码并以在线 softmax 合并结果，支持 GQA 和右对齐因果位置；不要求永久解压完整历史 K/V。这里借鉴的是 [FlashAttention 的在线归约/IO 思路](https://arxiv.org/abs/2205.14135)，并非其高效设备内核。Python 循环、bit unpack、FP32 临时张量和每页变换都存在额外成本。

## 5. 算法优越度应如何判断

没有单个格式能只凭位宽胜出。[vLLM FP8 文档](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) 体现了 scale 与后端约束的重要性；[LMDeploy](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html) 的在线 per-head/per-token 非对称 INT4/INT8 提供另一条工程路线。本实验把这些选择拆开比较，而不是复制一个名字就认定性能相同。

以每组 G 个元素、b-bit payload、s 字节 scale、z 字节 offset 估算，忽略 padding 时有效位宽为：

```text
b_effective ≈ b + 8(s + z)/G
```

例如 G=32，INT4 + FP32 scale 是 5 bit/元素；再加 FP32 offset 是 6 bit/元素，相对 BF16 的理想压缩比从 4×降至 3.2×或 2.67×。旁路每个元素再增加 64 bit，且实际按组向上取整选 k：G=32 时即使只指定 1%，也至少选择 1/32，而非恰好 1%。尾部、sink、旋转 padding、分组 padding 都进一步改变最终结果。

所以实验按真正保留的 tensor 统计 payload、scales、offsets、旁路索引/值、高精度数据，报告实际压缩比与有效位宽，不用“INT2 理论八倍”代替测量。该统计不包含 Python 对象、allocator 对齐/缓存、运行时临时张量；不是进程 RSS 或 GPU 峰值显存。分支之间共享页面时，不能把各分支逻辑字节直接相加当作物理总占用。

同时报告三个误差：K 相对 L2、V 相对 L2、attention 输出相对 L2。K 误差会影响 softmax 分配，V 误差影响加权结果，单看重构误差无法决定生成质量。RHT、旁路、affine 也可能在某种分布改善、另一种分布退化，应作为可组合策略而非默认全部开启。

## 6. 如何运行与阅读实验

在安装 PyTorch 的环境，从仓库根目录运行：

```bash
python -m tools.explore_kv_quantization
```

驱动使用固定随机种子，对普通分布、偏移分布和离群值分布比较格式与策略；详细配置以输出为准。使用合成数据是为了快速暴露量化机制的行为，不是为了代表真实模型。

实测环境为 PyTorch `2.6.0+cpu`，单线程；种子 2026，BF16 源数据，T=256、Hkv=2、Hq=4、D=64、Q=8，P=32。共运行 23 种配置 × 3 种分布 = 69 个探索案例。普通基线为完整 BF16 K/V 的 131072 字节；大多数配置 G=32、FP32 scale、R=0，具体差异由名称和 driver 中的配置表给出。追加按 37 token 分块，attention 使用最后 8 个位置的右对齐因果 mask，参考结果为未量化 BF16 K/V 上的 FP32 dense attention。

以下误差均为 **attention 输出相对 L2**，不是任务准确率下降；[全部结果 CSV](kv_cache_quantization_results.csv) 同时保留 K/V 重构误差、resident bytes、有效位宽、residual/sink 配置。运行脚本的完整 JSON 还包含 K/V 的具体 QuantSpec 和逐项字节分解。

| 配置（对应 driver 名称） | 实际压缩比 | 普通分布误差 | 通道离群分布误差 | 偏移分布误差 |
| --- | ---: | ---: | ---: | ---: |
| `int2_sym` | 5.333× | 93.08% | 83.16% | 38.07% |
| `int3_sym` | 4.000× | 30.58% | 56.31% | 3.37% |
| `int4_sym` | 3.200× | 13.27% | 67.33% | 1.36% |
| `int8_sym` | 1.778× | 0.73% | 2.97% | 0.08% |
| `fp8_e4m3` | 1.778× | 3.26% | 3.98% | 0.50% |
| `fp8_e5m2` | 1.778× | 6.66% | 6.36% | 1.06% |
| `nf4` | 3.200× | 12.14% | 37.67% | 1.75% |
| `fp4_e2m1` | 3.200× | 14.18% | 25.75% | 2.26% |
| `int4_affine` | 2.667× | 10.38% | 60.09% | 0.57% |
| `int4_rotated` | 3.200× | 12.82% | 17.40% | 2.04% |
| `int4_outliers` | 2.286× | 10.71% | 68.25% | 1.36% |
| `int4_fp16_scales` | 3.556× | 13.27% | 66.49% | 1.38% |
| `int4_pow2_scales` | 3.765× | 20.68% | 60.94% | 2.41% |
| `k_channel_v_token` | 2.207× | 10.76% | 15.80% | 0.58% |
| `k_int8_v_int4` | 1.969× | 9.26% | 10.66% | 0.87% |
| `int4_residual32` | 2.510× | 12.89% | 66.01% | 1.33% |

这轮实验具体说明：

- **低位宽“能编码”不等于“质量可用”**：INT2 普通分布输出误差达 93.08%，不能只看到 5.33× 容量压缩。INT3 的真实跨字节打包也只是具备研究基础，并非已经满足模型质量要求。
- **离群值需要选择策略，而非统一降位宽**：通道离群分布的 INT4 输出误差从 67.33% 降到旋转方案的 17.40%，旋转本身不增加本例 D=64 的常驻 tensor 字节；K-channel/V-token + affine + R=32 为 15.80%，但压缩比降至 2.21×。后者同时改变多个变量，不能将改善全部归因于 channel 分组。
- **单看重构误差会选错策略**：旁路使通道离群样本的 K 相对误差从 18.03% 降至 9.05%，attention 误差却从 67.33% 变为 68.25%。softmax 对具体误差方向敏感；不能宣称保留离群值必然改善 attention。
- **元数据压缩有可见收益，也有代价**：本例 INT4 的 FP16 scale 将压缩比从 3.20×提高到 3.56×，普通分布误差几乎不变；pow2 scale 提高到 3.76×，但普通分布输出误差增至 20.68%。尚未证明这种取舍适合某个真实模型。
- **分布会改变方法排序**：affine 在偏移样本中降低误差，RHT 却可能退化。偏移样本的 V 含非零均值，输出 norm 更大，因此相对误差不能跨分布直接解释为模型更容易量化。

以上不是 vLLM/LMDeploy 对比 benchmark。格式实现、元数据布局、硬件路径均不同；INT8 在这些小组动态校准样本上误差较小，也不代表它在所有模型上优于 FP8，或设备速度更快。

本轮只做探索运行和必要的静态检查，不新增大型测试套件。没有 CUDA/NPU/MLU 端到端验收，没有吞吐/延迟结论，也没有证明 INT2/INT3 已能保持某个真实模型的质量。

额外的简短手工检查确认：native 的完整 prefill、分块及单 token causal/GQA 输出与 dense 最大绝对差不超过 `3.58e-7`；相同输入一次追加与每 37 token 追加的封页解码结果逐位一致；fork 共享旧页、复制尾部，分支追加不改变原缓存。三个新增 Python 模块通过 Ruff 与语法检查。这些仅确认参考实现的基本自洽性，不替代生产验收。

## 7. 对未来工程接入的具体建议

先用实际模型捕获的逐层 Q/K/V 替换合成数据，选择误差/字节开销合适的少数组合，再验证长上下文检索、生成质量和真实峰值显存。实验接口已经让“哪种格式、哪个方向、多少 residual、是否旋转”成为可单独比较的变量，不必先改动生产调度器。

若决定接入生产，需要把格式、scale/offset 类型、粒度、旋转约定、padding、旁路、版本共同定义成布局契约，并让分配器、预算、attention、复制和 transfer/store 使用同一描述。前缀命中也必须兼容量化策略与布局版本，不能只凭 token 相同就跨格式重用。

最后才为选定硬件实现融合写入/解包/attention，并处理 graph、并发和跨层级搬运。当前实验的价值是让这些选择可观察、可解释；不是在没有模型和硬件证据时宣布新增格式已经适合生产。

## 8. 新机制：双误差门控的页面自适应量化

前一轮暴露的问题是：统一 INT4 遇到离群页面会严重失真；统一 INT8 又为容易量化的页面付出不必要的容量。新增 `PageQuantPolicy` 不再固定格式，而是在封页时为 K、V **分别寻找满足误差约束的最小常驻 tensor 字节方案**。这是本仓库的探索性机制组合，不声称首次提出混合精度；[Adaptive KV Cache Quantization](https://arxiv.org/abs/2502.15075) 已研究 K/V 非对称精度分配。这里的实现不复现其算法或继承其效果。

### 8.1 为什么不是简单比较 MSE

每个候选都真实编码、解码，再相对原始页面计算两个 FP64 评估指标：

1. 每个 KV head 的相对 L2，取最差 head，防止整体均值掩盖某个 head 的失真；零范数分母下限为 `1e-30`。
2. 每个 token/head 向量的绝对 L2 误差，取最差向量，防止少数 token 的错误被页面均值稀释，也防止大幅值页面轻易通过相对误差检查。

同时满足两道门槛后，按实际 `nbytes()` 选择最小者，包括 scales、offsets、padding 和旁路。相同字节数按候选顺序决定，不宣称全局最优：这仅是给定候选集、独立 K/V 约束下的局部最小存储搜索。源数据和候选中间结果不保留在缓存中，只保存获选编码。

`native` 必须显式列入候选集，作为预算过严时的正常选项；这不是捕获异常后的静默降级。非法配置、scale 溢出、codec 异常仍直接传播，append 在完整构造新页面前不提交缓存状态。预算必须有限且非负。实验使用 BF16；若输入为 FP64，native 解码为 FP32 也可能无法满足零误差预算，此时明确失败。

### 8.2 从重构误差到 attention 的条件上界

设全部缓存 token/head 的 K、V 最大向量误差分别为 δK、δV，原 V 的最大向量范数为 M；对某个 query，范数为 Q，标准 attention scale 为 `1/sqrt(D)`。由 Cauchy–Schwarz，logit 最大绝对误差不超过 `ε = Q·δK/sqrt(D)`。在相同 causal mask 下，利用 softmax 的保守 L∞→L1 扰动界，可得单个 attention 输出的绝对向量误差：

```text
||output_quant - output_reference||₂ ≤ 2·min(1, ε)·M + δV
```

这是本说明中的数学推导，不是引用论文的实验结论。推导将输出差拆成注意力权重变化乘原 V、以及量化 V 的凸组合；上界通常很松。它不含 FP32 attention 算术舍入误差，也不是相对误差、整层误差、模型生成质量或长期误差累积保证。未来 query 范数若没有上界，就不能提前给出统一 attention 输出保证。

实现选型只读取当前被封存页面，不使用评估 query、页面以后的 token 或评估 attention 结果，因此不会靠评估 query 挑选格式。但若回顾性评估 query 位于该页面内部，校准仍可能包含其位置之后的 token；后续新增因果检查与高精度 query 尾部处理此边界。driver 仅在选型完成后报告实际 query 对应的条件上界和实际最大输出误差。报告包含每页完整 QuantSpec，方便解释位宽为何变化。

### 8.3 生命周期与成本

同一个缓存允许不同页面格式不同，K/V 也可不同；封页后保持格式和 scale 不变。fork 继承不可变策略并共享旧页，sink 页面仍优先保留 native，tail 策略不变。相同输入页的选择不依赖 append 分块方式。没有新增生产参数或更改既有 cache ABI。

真实代价是每页枚举多个 codec、重复编码解码及误差计算，增加封页延迟和临时内存。本版是离线搜索参考，不是高吞吐在线调度器；也没有全局总字节预算约束。未来可用离线画像裁剪候选、再验证快速决策器，但不能将当前 CPU 搜索直接宣称为设备加速。Python 格式描述符和策略对象仍不计入 tensor 字节统计。

### 8.4 复现实验

```bash
python -m tools.explore_kv_quantization --adaptive
```

与前述固定 INT4/INT8 使用相同 seed、形状和三个分布。候选包含 FP16 scales 的 INT3/4/6/8、旋转 INT4、channel affine INT4、FP8 E4M3、NF4，以及 native；K 的门槛为 `(最差 head 相对 L2≤0.08，最大向量误差≤0.5)`，V 为 `(0.1，1.0)`。这些是显式实验预算，不是通用推荐参数。固定基线仍用 FP32 scales，因此结果同时反映 scale 表示、格式和自适应选择的作用，不能单独归因于混合位宽。

PyTorch `2.6.0+cpu` 实测如下，误差为 attention 输出相对 L2；[完整数值结果](kv_cache_adaptive_results.csv) 另含最大向量误差和条件上界。

| 分布 | 固定 INT4：压缩 / 误差 | 固定 INT8：压缩 / 误差 | 自适应：压缩 / 误差 |
| --- | ---: | ---: | ---: |
| 普通 | 3.200× / 13.27% | 1.778× / 0.73% | 2.909× / 9.35% |
| 通道离群 | 3.200× / 67.33% | 1.778× / 2.97% | 1.476× / 1.43% |
| 偏移 | 3.200× / 1.36% | 1.778× / 0.08% | 2.612× / 0.44% |

普通分布的 8 个 K 页都选 INT6，V 选 7 页 NF4、1 页 token 对称 INT4；通道离群分布的 K 选 3 页 INT8、5 页 native，V 全选 INT8；偏移分布的 K 选 7 页 INT6、1 页 INT8，V 全选 channel affine INT4。这说明选择不仅改变位宽，也会改变分组和数值格式。完整描述以运行产生的 `page_formats` 为准，不仅看 `format` 字段。

自适应不是无条件击败 INT8：普通分布下它更省，但误差也更大；离群分布下为遵守 K 的绝对误差门槛，保留高精度页面，比固定 INT8 更占容量。三种分布的最大 K 向量误差分别为 0.272、0.486、0.484，最大 V 向量误差分别为 0.993、0.470、0.737，均在显式预算内。普通分布实际最大输出向量误差为 0.138，上界为 7.009，体现上界的保守性，不能将上界大小当作预计误差。

本轮仅作 9 个小规模比较与必要手工检查，不新增测试套件：检查追加分块不改变选择/解码结果、fork 共享旧页且追加不改原缓存、零预算选 native、codec 错误传播且不提交部分缓存。未作实际模型与设备验收。
