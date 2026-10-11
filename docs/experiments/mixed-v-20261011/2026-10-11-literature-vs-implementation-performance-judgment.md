# KV量化研究收益与本次实现的判断

结论：研究中已有decode kernel与端到端吞吐收益，不能把当前负面结果解释成“KV量化本来就不加速”。本次有证据支持实现与工作划分造成可避免损失，但尚未证明修好后一定获得端到端收益。

## 研究证据与比较边界

- [KIVI原论文](https://arxiv.org/html/2402.02750v2)报告2.35–3.47×吞吐，来自K/V均量化及更大可运行batch，不能直接推广到本次V-only、固定到达trace的TTFT/TPOT收益。
- [FlashInfer作者的性能报告](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)报告FP8 decode kernel最高约2×加速；是特定硬件/shape下的attention kernel结果，不是所有格式、所有服务负载都2×。作者也讨论了split-KV与架构/工作划分的关系。
- 本次K保持BF16，只压缩V；理想INT8情况下K+V载荷从4降到3字节/元素，约减少25%（不计scale等元数据），只表示该部分显存访问的机会。即便这部分完全带宽受限且开销为零，相关访问时间的理想加速也仅4/3；其他算子、prefill、排队与量化开销决定整服务结果。不能直接套用全KV量化收益。

## 本次直接证据

1. 同形状batch36/ctx12288，INT8实际kernel流量估计减少23.5%，attention快11.2%。收益机制在本机已经成立。
2. batch36→37时，CTA72→74超过36SM×2块的单波槽位。INT8 duration0.946→1.481ms，而DRAM利用率83.2%→54.6%，SM活跃周期均值/最大值97.9%→62.2%。寄存器未造成驻留块数变化，无spill；未发现warp平均等待相称增长。证据支持空间工作不均衡与长CTA尾部是该拐点主因。
3. 保留同kernel，仅在独立诊断更改KV分片，INT8 b37主kernel1.481→1.123ms；无profiler完整eager调用也同方向改善。证明存在可优化实现因素，不是单凭理论推断。
4. 本次为FlashInfer FA2上的自定义K/V mixed-dtype扩展，不等价于已经针对SM120/全部shape调优的官方低精度decode路径。当前低精度V在kernel内恢复BF16参加后续计算；此路径节省全局访问，但引入转换/scale处理，不能自动获得低精度计算吞吐收益。反量化独立时间尚未精确分离。
5. 服务里量化arm常见decode batch更大，也更容易跨过拐点；mixed batch的大prefill仍会阻塞同轮decode输出。分片诊断尚不证明服务TTFT/TPOT改善，prefill不是量化必然能加速的部分。

判断：优先处理量化路径的planner/CTA划分与不同batch形状的尾部效率，而非继续寻找更大并发或任意修改调度预算。优化后必须用相同trace验证服务收益，保留当前正式负面数据；不能承诺优化后必然转正，也不能把当前结果推广为KV量化普遍无效。硬件证据详见[NCU分析](2026-10-11-nsight-compute-mixed-v-wall-analysis.md)。
