# INT8 KV attention kernel 优化复测

## 实现

- INT8 写入改用 `tl.div_rn` 计算归一化值；BF16 输入恰落在半整数时，payload 与 PyTorch codec 的 round-to-even 结果一致。
- Prefill 在一个 CTA 内复用 GQA 组的 K/V tile。对 BF16 query/cache 采用 hi/lo 分量：QK 两次 dot、PV 三次 dot（省略低低项），保留反量化值和 softmax 权重精度。不能把这条路径简化为直接对 BF16 做一次舍入。
- 候选1仅复用 GQA tile，小 batch 回退，已弃用。候选2对 batch≤4 且 block-table 容量≥1024 的 prefill 使用4个 page 分区和 FP32 partial-softmax 合并，其余路径保留 GQA 复用。

测试 `tests/python/test_quantized_attention.py`：143 passed。包括 BF16 context 1025/window=3、INT8 context 17 / table 8×128 / Q3 / window=3 的空分区无 NaN 检查，以及 BF16 round-tie writer 对照。误差阈值未放宽。

## 固定形状结果

RTX 5060 Ti，预热10次、每轮采样100次、3轮，使用三轮 p50 和吞吐中位数。固定形状命令使用 FlashInfer BF16 对照。INT8 full-call p50 和吞吐如下：

| Context / batch / Q | 优化前 | 候选2 | 提速 |
| --- | ---: | ---: | ---: |
| 1024 / 1 / 16 | 0.283 ms / 56.5k tok/s | 0.209 ms / 76.4k tok/s | 1.35× |
| 1024 / 4 / 16 | 0.351 ms / 182.2k tok/s | 0.291 ms / 219.9k tok/s | 1.21× |
| 4096 / 1 / 16 | 0.443 ms / 36.1k tok/s | 0.343 ms / 46.6k tok/s | 1.29× |
| 4096 / 4 / 16 | 0.998 ms / 64.1k tok/s | 0.598 ms / 107.0k tok/s | 1.67× |

对应 attention 提速为 1.03×、1.28×、1.35×、1.73×。候选1在 B1 prefill 明显退化（context 4096 attention 0.379→0.775 ms，full-call 0.443→0.839 ms），因此改为候选2。候选2最大实测提速 1.73×，未触发额外复跑门槛。

重点 shape context 4096 / B4 / Q16：INT8 cache-write 为 0.0680→0.0670 ms（941k→955k tok/s），attention 为 0.9200→0.5308 ms（69.6k→120.6k），full-call 为 0.9982→0.5983 ms（64.1k→107.0k）。BF16 FlashInfer 对照同轮为 cache-write 0.0917→0.0610 ms（698k→1,050k tok/s）、attention 0.3306→0.2973 ms（193.6k→215.3k）、full-call 0.3839→0.3734 ms（166.7k→171.4k）。INT8 候选2相对 FlashInfer BF16 仍慢：该 shape attention p50 为 1.79×、full-call 为 1.60×；吞吐分别为 BF16 的 56% 和 62%。BF16 写入/attention 本身也有轮间移动，不能把完整改变量都归因于 INT8 kernel。

固定 batch 的 Q1 attention 未改计算路径，前后基本相同；full-call 个别小 batch 结果波动，不能归因于 `tl.div_rn` 写入修复。

## 64 MiB 单层 KV 预算

INT8 可用的最大 batch 仍约是 BF16 的 1.94×。Q16 的 full-call 吞吐优化前→候选2为：

| Context | BF16 最大 batch / tok/s | INT8 最大 batch / tok/s |
| ---: | ---: | ---: |
| 1024 | B16 / 695k→750k | B31 / 310k→525k |
| 4096 | B4 / 182k→181k | B7 / 68k→116k |

候选2提升了预算 prefill，但 INT8 总吞吐仍分别只有 BF16 的约70%和64%。预算只限制单层 KV cache，不含输入、临时区、workspace、权重或服务调度，不能外推成生产并发或 QPS。

INT8 输出误差上限为 relative L2 0.00981、absolute 0.00293；预算扫描 absolute 最大值 0.00488。cache 压缩约1.94×（已含 FP32 scales）。新增 partial 输出/LSE 临时区后，固定形状运行峰值增量最大4.53 MiB（context 1024 / B4 / Q16 INT8）；预算扫描最大4.27 MiB。这是相对计时前分配量的增量，不是整机显存需求。

## 可复现结果与边界

优化前 fixed / 64 MiB 目录：`results/20260930T140408Z_966718d1`、`results/20260930T140428Z_11b4e6a3`。候选1目录：`results/20260930T141542Z_6734a786`、`results/20260930T141602Z_44f000e9`（仅作取舍依据）。候选2 fixed / 64 MiB 目录：`results/20260930T142239Z_dfea1f9f`、`results/20260930T142256Z_737e11d6`。

结果支持保留候选2继续验证：部分 prefill shape 有 1.2–1.7× 单层提速，INT8 仍未追平 FlashInfer BF16，Q16 固定预算下吞吐仍较低。本阶段没有真实模型或服务调度结果；生产价值还需纳入完整设备预算、模型权重和多层 cache 后评估。
