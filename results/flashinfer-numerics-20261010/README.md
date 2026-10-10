# FlashInfer / Triton：数值差异与性能诊断

结论：本次补测所有 GPU attention 形状均为 FlashInfer 更快。输出差异确实存在，主要证据指向计算精度路径不同，而不是 packed V 读取错误。这里的数值检查只覆盖 4 题、前 32 个固定历史 token，不能代替全量质量测试。

## 固定输入、历史 token 和 Cache 的逐层检查

同一 Qwen2.5-1.5B-Instruct 权重，以 Transformers BF16 模型作为诊断外壳，将全部 28 层 attention 替换为真实 xLLM kernel。四道已有 GSM8K prompt，每题 prefill 加 32 个 teacher-forced decode 位置，两分支喂完全相同的 token IDs/positions，不让模型各自生成后再比较。诊断外壳的线性层/RoPE 与服务实现不同；这些结果定位 attention 差异，不宣称复现完整 xLLM 服务的 logits。

每一层还在相同 Q、K、低精度 V payload 和 scale 下同时调用另一个 backend，隔离此前层误差累积。另设 BF16 shadow reference：把解码后的 V 舍入为 BF16，用 BF16 FlashInfer 计算。shadow 是诊断参考，不在压缩服务里使用。

| V | 下一 token 分歧 | 相同 QKV 下量化 FlashInfer vs BF16 shadow | 参考检查平均 RMS：FlashInfer / Triton |
|---|---:|---:|---:|
| bf16_int8 | 0/132 | 7392 次比较均逐位相同 | 0.001333 / 0.000954 |
| bf16_int4 | 8/132 | 7392 次比较均逐位相同 | 0.001338 / 0.000947 |

FP32 dense reference 在层 0/13/27、step 0/1/8/32 的最后一个 query 上检查，每个格式 96 次。INT4 的下一 token 分歧说明小的 attention 差异可以在后续层、后续 Cache 量化中放大；不能把误差容限通过等同于生成等价。

Triton decode 用 FP32 解码 V 和标量归约；prefill 把 weights/V 拆成 BF16 高低部分，PV 使用三次 dot 以保留更多精度。FlashInfer 将 V 和 attention weights 舍入到 BF16 后用标准 Tensor Core MMA。FlashInfer 的量化路径与 BF16 shadow 逐位相同，支持解码正确、差异来自数值路径的判断；不排除未覆盖形状的问题。

原 C20 服务的同一 backend 三次重跑，也有不同答案：INT8 的 Triton/FlashInfer 分别 51/58 题，INT4 分别 80/85 题。temperature=0 不保证不同动态 batch 下生成逐位相同。因此服务答案差异还包含 batch/调度引起的数值变化，不能全部归因于更换 backend。

## GPU、eager 调用和 prepare 分开测

上下文 768、page128；B1/2/8/20，Q1 decode 或 Q128 chunked prefill；相同随机 Q/K/V，5 次随机 backend 顺序重复，报告中位数。GPU 用已预热 graph 的 CUDA events；eager 另计 50 次调用墙钟。prepare 单独测，发生在全部 graph 测量完成以后。没有与模型诊断或服务同时占用 GPU。

| B | Q | V | Triton GPU us | FlashInfer GPU us | GPU 加速 | Triton eager us | FlashInfer eager us | Triton prepare us | FlashInfer prepare us |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | bf16_int8 | 19.7 | 9.8 | 2.01× | 101.2 | 97.1 | 290.9 | 461.6 |
| 1 | 1 | bf16_int4 | 20.9 | 9.6 | 2.18× | 103.5 | 97.8 | 295.5 | 458.4 |
| 1 | 128 | bf16_int8 | 130.5 | 33.2 | 3.93× | 138.1 | 59.2 | 211.5 | 364.2 |
| 1 | 128 | bf16_int4 | 122.4 | 32.6 | 3.76× | 129.8 | 59.8 | 316.5 | 324.7 |
| 2 | 1 | bf16_int8 | 20.1 | 9.7 | 2.08× | 66.8 | 54.8 | 205.0 | 336.9 |
| 2 | 1 | bf16_int4 | 21.1 | 9.5 | 2.22× | 67.1 | 54.7 | 182.2 | 339.0 |
| 2 | 128 | bf16_int8 | 248.7 | 59.4 | 4.18× | 255.7 | 70.4 | 273.8 | 333.2 |
| 2 | 128 | bf16_int4 | 228.2 | 58.7 | 3.89× | 235.6 | 70.5 | 331.5 | 332.9 |
| 8 | 1 | bf16_int8 | 36.1 | 17.6 | 2.05× | 69.8 | 54.7 | 309.8 | 505.7 |
| 8 | 1 | bf16_int4 | 38.8 | 17.6 | 2.21× | 67.4 | 55.8 | 219.7 | 375.1 |
| 8 | 128 | bf16_int8 | 727.8 | 159.2 | 4.57× | 731.8 | 163.4 | 344.2 | 381.0 |
| 8 | 128 | bf16_int4 | 668.7 | 153.1 | 4.37× | 672.6 | 163.9 | 328.7 | 388.2 |
| 20 | 1 | bf16_int8 | 78.3 | 37.3 | 2.10× | 86.5 | 49.0 | 316.4 | 421.6 |
| 20 | 1 | bf16_int4 | 84.5 | 36.5 | 2.32× | 92.9 | 47.4 | 237.1 | 583.1 |
| 20 | 128 | bf16_int8 | 1693.5 | 335.5 | 5.05× | 1700.6 | 340.9 | 584.9 | 484.2 |
| 20 | 128 | bf16_int4 | 1554.9 | 327.2 | 4.75× | 1577.3 | 336.7 | 579.7 | 585.0 |

FlashInfer 的 GPU decode 约快 2.0–2.3×，Q128 prefill 约快 3.8–5.0×。小 batch eager 调用的收益较小，因为 Python/FFI 和 launch 开销会掩盖 GPU 时间；decode 的 prepare 在本组通常也是 FlashInfer 更慢。但 prepare 每个模型 forward 一次，复用于全部层，不能乘 28。

原 C20/L768 下，按单层 GPU 时间简单乘 28，INT8 attention 约 2.2 ms → 1.0 ms；这只是固定形状估算，不能直接当作服务整步时间。服务还包含 MLP、投影、norm、writer、调度和请求等待，所以整个服务不会获得同样的倍数加速。尚未做整条服务 trace 的算子归因。

第一版补测 `latency.json` 无效：graph 捕获后又 prepare，Triton 会替换 graph 引用的 metadata allocation，造成错误计时。已标记在 `invalid_latency_measurement.json`，修正为两阶段分离后重测；本页只使用 `latency_final.json`。原先已提交的 kernel 对比没有在 graph 测量中重复 prepare，不受这个新增诊断工具问题影响。

当前判断：Triton 不是 GPU attention 更快，而是在这里保留了更多计算精度。FlashInfer 可以继续用于四 arm 实验；必须以同一 FlashInfer BF16 arm 为 baseline，并由全量 benchmark 判断量化与额外舍入后的最终质量。不要把与 Triton 的比较称为严格相同计算精度下的加速。

原始数据：[逐位置 logits/top1 汇总](fixed_history.json)、[逐层统计](numerics_summary.json)、[GPU/eager/prepare 原始重复](latency_final.json)、[CSV](latency.csv)、[环境与源码](provenance.json)。每个样本、后端的完整层级误差记录保存在本目录。
