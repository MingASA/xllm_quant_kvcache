# INT4 KV cache 诊断记录（2026-10-02）

Copyright 2026 The xLLM Authors.

## Strengths

- 用真实 Qwen2.5-1.5B BF16 eager 前向捕获 RoPE 后 Q/K/V，并在同一张量上分开比较 codec、GPU writer、Triton attention 和 FP32 dense attention。诊断脚本只在 `tools/` 下运行，不修改生产实现。
- 五个代表层 GPU 写入的 scale 与 Torch codec scale 逐项完全相同。生产 Triton attention 与由生产 writer 写入、再解码后交给 dense attention 的结果接近；只对 INT4 writer 增加 `tl.div_rn` 的临时克隆则在该样本中消除了所有 payload 差异。
- 报告区分已复现的边界问题与历史模型质量结果；单样本诊断不被当作 full-suite 质量退化的因果证明。

## Issues

### Critical (Must Fix)

无。

### Important (Should Fix)

1. **INT4 writer 的归一化除法可能改变半整数处的量化码。** [quantized_triton.py](/home/mingasa/projects/xllm_quant_kvcache/xllm/python/attention/quantized_triton.py:106) 仅 INT8 (`FORMAT == 0`) 使用 `tl.div_rn`；INT4 (`FORMAT == 3`) 落入普通 `key_values / key_scale` 除法。真实 BF16 RoPE 后张量中观察到 L0 K 的 `value=-159`、`scale=45.4285736`：Torch FP32 归一化为 `-3.499999832`，Torch nearest-even 量化为 `-3`，生产 writer 写成 `-4`。在本次 145-token 片段中，各层每个 K/V payload 有 1–19 个 packed byte 不同（每个 payload 共 18,560 bytes）；scale 无差异。临时复制 kernel module、只让 INT4 分支也使用 `tl.div_rn` 后，五层 K/V payload 都与 Torch codec 完全匹配。**建议**将该精确除法策略覆盖到 INT4，并增加 BF16 半整数边界回归测试。当前已测到的是稀疏 ±1 code 差异，不能据此认定它单独造成 full-suite 的严重退化。

2. **Decode attention 未把有效 token mask 传入 K/V 解码器，未写页面尾部可能被读取。** [quantized_triton.py](/home/mingasa/projects/xllm_quant_kvcache/xllm/python/attention/quantized_triton.py:271) 和 [quantized_triton.py](/home/mingasa/projects/xllm_quant_kvcache/xllm/python/attention/quantized_triton.py:284) 的 decode 调用未传 `valid_tokens`；同一 kernel 已计算的 `valid` 只用于后续 score masking。对长度 145、page size 128、d=128、Hq=12/Hkv=2 的 INT4 decode（8 splits），把第二页未写位置 17–127 的 scales 设为 NaN 后，输出由全 finite 变为 non-finite；零填充 tail 对照仍全 finite。这证明无效尾部 scale 可通过无效注意力项传播 NaN。**建议**将 token-valid mask 同时传给 K/V 解码，并补充 poisoned-tail 与常规非整页 decode 测试。该实验只证明边界 bug 可触发，未证明历史 1319 GSM 或 LongBench 的输入曾使此条件出现 NaN。

### Minor (Nice to Have)

- `tools/kv_cache_codec_lab.py` 的 `QuantSpec.rotation` 与 `outlier_fraction`（例如 [kv_cache_codec_lab.py](/home/mingasa/projects/xllm_quant_kvcache/tools/kv_cache_codec_lab.py:61)、[kv_cache_codec_lab.py](/home/mingasa/projects/xllm_quant_kvcache/tools/kv_cache_codec_lab.py:293)）属于独立探索 codec，并未接入 serving 的 writer/cache layout/attention 路径。应继续明确标记这种边界，避免把 lab 结果误读成线上实现能力。

## Harness 审计

第一版诊断曾注册自定义 `ALL_ATTENTION_FUNCTIONS` key，却没有同时注册 `ALL_MASK_ATTENTION_FUNCTIONS`。Transformers 因而没有为自定义 backend 构造 causal mask，attention callback 收到 `attention_mask=None`。第一版 `real_qkv_single_prompt.json` 和 `model_e2e_one_prompt.json` 中所有模型/QKV 数值均作废，不用于结论；原文件保留作审计记录。修订后两个 registry 都注册对应 eager mask builder，脚本在多 token prefill 断言 mask 确实屏蔽未来位置。

修订 harness 的 causal 自检通过：755-token prompt 加 60 个 teacher-forced 答案 token 后，815-token prefill mask 为 `[1,1,815,815]` BF16；308 次多 token callback 均验证 future masked。原生 eager 与注册 wrapper 的 logits 最大绝对差为 0；替换 teacher-forced 未来 token 后，前缀 logits 最大绝对差亦为 0。有效结果保存在文件名带 `_masked` 的诊断 JSON 中。

## 本次实验观察

完整评测仍以 [kv_cache_int4_model_eval_20261002_zh.md](/home/mingasa/projects/xllm_quant_kvcache/docs/kv_cache_int4_model_eval_20261002_zh.md) 为准。以下只汇报诊断小样本，不取代正式评测。

- Stage 1 的真实 QKV 捕获输入是 GSM8K 首题原始完整官方 5-shot chat-template prompt（755 tokens），未拼 teacher-forced 答案；HF Qwen2 eager BF16 前向捕获层 0/7/14/21/27 RoPE 后 Q/K/V。前向最多捕获 512 token；层对照仅使用前 145 个 KV token 和最后 64 个 query。该窗口覆盖 128-token page tail，不是整条 prompt 的质量评估。独立 Stage 2 才将 60 个 teacher-forced 答案 token 拼接到 755-token prompt，形成 815-token 模型输入。
- FP32 dense attention 输出回 BF16，与原始 KV dense oracle 比较，INT4 dense attention output relative-L2（层 0/7/14/21/27）为 `1.3096 / 0.5370 / 0.2779 / 0.2461 / 0.2499`；rotation-only 为 `1.0736 / 0.2270 / 0.1618 / 0.1433 / 0.1349`。这仅是五层、145 KV token 的局部误差。
- 首题前 145 KV token 的 codec reconstruction relative-L2（层 0/7/14/21/27）：K 为 `0.0726 / 0.1815 / 0.2054 / 0.2225 / 0.2722`，V 为 `0.2574 / 0.1625 / 0.1481 / 0.1325 / 0.1659`。脚本未记录零码率或 K/V absmax/median 分布，因此不从这份输出补推该统计；codec error 反映本样本该窗口，不代表所有 token/layer。
- 生产 writer 的 K/V payload 在五层分别出现 `15/13, 1/9, 8/10, 7/17, 8/19` 个 byte 与 Torch codec 不同；各 payload 有 18,560 bytes，scale 逐项精确相同。样例差异位于 ±3.5 附近。临时 writer clone 仅给 INT4 加 `tl.div_rn` 后，五层所有 K/V payload 差异归零。
- 生产 writer cache 解码后送入 FP32 dense，生产 Triton **prefill attention**（64 queries）与该 dense 结果的 relative-L2（层 0/7/14/21/27）是 `8.02e-5 / 3.27e-5 / 4.04e-5 / 4.21e-5 / 5.75e-5`；`tl.div_rn` clone prefill Triton 对自身 GPU-written dense cache 的 relative-L2 是 `5.83e-5 / 2.41e-5 / 2.83e-5 / 1.31e-5 / 3.34e-5`。这些误差只验证本窗口的 prefill Triton attention 与各自 writer cache 对应 dense 的一致性，不可泛化为 decode 已验证。单 token decode 的 Triton 输出与 **Torch-codec 量化/反量化 KV 的 dense reference** relative-L2（层同上顺序）为 `0.3774 / 0.0101 / 0.0166 / 0.0217 / 0.0203`；该比较同时包含生产 writer 与 Torch codec 的少数 tie 量化差异，不能单独归因 decode attention 主体，也不能把 L0差异解释为已定位的 decode bug。另行 poison-tail 实验才隔离证明未写 tail scale NaN 可污染 decode 输出。
- 对原始 GSM8K idx 0–3 的答案固定 token teacher forcing 共 218 token，加权 CE/top-1/KL 汇总如下。此处是模型诊断指标，不是数据集准确率。

  | attention 对照 | token 加权 CE | 与 BF16 top-1 一致 | token 加权 KL（BF16→对照） |
  |---|---:|---:|---:|
  | BF16 eager | 0.6094 | 218/218 (100.00%) | 0 |
  | 未量化 BF16 FP32 dense | 0.6189 | 213/218 (97.71%) | 0.0201 |
  | INT8 dense | 0.6203 | 211/218 (96.79%) | 0.0201 |
  | INT4 dense（G128） | 9.3723 | 2/218 (0.92%) | 9.0160 |
  | INT4 rotation（G128） | 7.9763 | 8/218 (3.67%) | 7.7237 |
  | INT4 rotation（G32 lab-like） | 7.5552 | 1/218 (0.46%) | 7.2213 |
  | 仅 K INT4（V 保持 BF16） | 9.0429 | 1/218 (0.46%) | 8.7113 |
  | 仅 V INT4（K 保持 BF16） | 0.5855 | 211/218 (96.79%) | 0.0248 |

  在这四个样本中，只量化 K、保持 V 为 BF16，已足以重现接近双 K/V INT4 的 logits 退化；只量化 V 则与 BF16 接近。这是对当前假量化路径的因果局部对照，不证明所有 1319/GSM 与 LongBench 历史结果的唯一原因。两处 writer/decode 边界问题也仍需独立修复和回归。
- G32 variant 把旋转后 128 维按每 32 维分组量化，与当前 per-token/head G128 不是同一配置。五层 GPU G32 reconstruction 对独立 `tools/kv_cache_codec_lab.py` CPU encode/decode 的 K/V relative-L2 平均为 `1.09e-7`、最大 `1.14e-7`，验证该实验实现高度吻合 lab codec；但四题上 G32 没恢复质量（weighted CE 7.5552、top-1 1/218）。G128 rotation 也仍严重退化，不能假定 rotation/outlier 变体天然解决问题；现有 lab 没有接入生产 serving。
- idx0 greedy 最多生成 32 token：BF16 正常续写推理开头；INT4 生成重复换行；G128 rotation 反复输出 `for/to` 等片段。G32 未跑额外 greedy。

有效诊断 JSON：`/mnt/e/AI/xllm-eval-data/int4-model-eval-20261002/diagnostics/real_qkv_single_prompt_masked_g32.json`、`/mnt/e/AI/xllm-eval-data/int4-model-eval-20261002/diagnostics/model_e2e_samples_0_3_masked.json`。无 mask 的首轮文件为 `real_qkv_single_prompt.json` 与 `model_e2e_one_prompt.json`，保留但已作废。修订版单题文件 `real_qkv_single_prompt_masked.json`、`model_e2e_one_prompt_masked.json` 也保留；四题汇总以最新 `samples_0_3` 文件为准。临时 `tl.div_rn` clone 只在诊断进程运行，不写回生产源文件。诊断已结束，未实施生产修复。

## Recommendations

- 先修复并独立回归上述 writer tie 与 decode tail-mask 两个边界，再跑有限的真实 BF16 query + d128/GQA6 数值测试；修复前不重跑 full model suite，也不把已完成的正式结果改写成新结果。
- 继续扩展少量不同 GSM prompt 的固定 teacher-forced 对照，报告 K-only/V-only/rotation 分解；再据证据评估更改 scale granularity、rotation 或 outlier 策略。当前真实张量单 prompt 只提供机制线索。
- INT4 与 INT8当前 prefill 算法路径并非单纯位宽对照：INT4 使用 FP32/TF32x3 点积且 `grouped_heads=1`，INT8 使用 BF16 hi/lo tensor-core 路径且支持 GQA6。不要把速度差解释成纯 bit-width 收益。

## Assessment

**Ready to merge?** No — the production INT4 path still has the two Important correctness issues above; this report does not implement their fixes.

**Reasoning:** 两个 Important 边界已通过可复现实验确认，代码应由 kernel owner 决策修复并补生产测试。它们是否解释历史质量退化仍未完全确定，特别是少数 writer tie 不足以单独解释全套质量结果。
