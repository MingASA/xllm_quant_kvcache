<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# INT4 实操问题：固定证据与复现索引

保存日期：2026-10-02。生产代码基线为 `c53893b2`；本证据提交不修复生产 kernel，也不重写历史评测结果。

## 报告与结论范围

- [完整模型评测报告](../../docs/kv_cache_int4_model_eval_20261002_zh.md)：GSM8K 1319 题（并发 64）、LongBench 400 题（并发 30），包含分数、吞吐、延迟和比较限制。
- [实现检查与原因诊断](../../docs/kv_cache_int4_diagnosis_20261002_zh.md)：写入半整数舍入差异、decode 未写尾页 NaN 污染，以及真实 KV 和四提示因果对照。
- 现有旋转/outlier codec 属于独立 lab，未接入本轮服务端。四提示、218 个答案 token 的有效诊断表明：当前 K-only INT4 量化足以重现退化；V-only 较接近 BF16；G128 旋转和 lab 风格 G32 旋转都未恢复这些提示的质量。这不是对全部历史样本唯一根因的证明，也不是 INT4 格式整体无价值的证明。
- 两个生产实现缺陷已复现，但尚未修复。相同 GPU-written cache 的低误差 attention 对照仅验证本次 prefill 样本，不能据此宣布所有 decode 路径正确。

## 仓库内有效证据

| 文件 | 范围 | SHA-256 |
|---|---|---|
| `diagnostics/real_qkv_single_prompt_masked_g32.json` | 因果前向捕获真实 Q/K/V，五层 codec/writer/prefill/decode 边界及 G32 lab 对照 | `21f0eb153586be86171c7ec057811afdabd4c0dea6a1d3eac7cf25195694cd70` |
| `diagnostics/model_e2e_samples_0_3_masked.json` | 四提示固定答案 token 的 logits/CE/KL 对照、因果 mask 自检、首题短生成 | `940254ba0de2318c262f34755403bf79b8b822ec877fa832351ff4c8d756f7db` |

这些 JSON 为原文件逐字节副本。原始逐题输出、请求配置、日志与评分仍在 `/mnt/e/AI/xllm-eval-data/int4-model-eval-20261002/`。

**无效数据不要引用：** 首轮 `real_qkv_single_prompt.json` 和 `model_e2e_one_prompt.json` 缺少 causal mask，已作废；它们仅保留用于审计。首次 LongBench NarrativeQA 非 corrected attempt 也被排除。有效正式 LongBench 输出位于各任务的 `int4_corrected/`，评分位于 `paired_score_corrected/`。

## 完整归档

独立压缩包：`/mnt/e/AI/xllm-eval-archives/int4-model-eval-20261002-20261002T121544Z.tar.gz`。

SHA-256：`d98b431457121e82dccd6ab363760b1fdfdefc4ac1a0c9e6211cf18282928b31`。

包内包含整个 INT4 结果目录（有效及作废尝试均保留）、两份报告、诊断/评测/合并脚本、评测回归测试及 GSM8K 五个训练示例。模型权重和外部 BF16/INT8 原始结果未重复打包；其位置和模型 hash 见评测报告。归档与原始结果都在 E 盘，属于同盘副本，不是异地备份。Git 远端尚未确认收到本次证据。

校验示例：

```bash
sha256sum /mnt/e/AI/xllm-eval-archives/int4-model-eval-20261002-20261002T121544Z.tar.gz
tar -tzf /mnt/e/AI/xllm-eval-archives/int4-model-eval-20261002-20261002T121544Z.tar.gz
```

## 诊断复现

在仓库根目录、GPU 空闲时使用以下命令。它运行独立诊断进程，不启动服务、不改生产 kernel；输出选择新的目录，不覆盖固定证据。

```bash
source build/cmake.cuda-x86/resume-env.sh
/home/mingasa/venvs/xllm/bin/python tools/diagnose_int4_kv.py \
  --model /mnt/e/AI/models/Qwen2.5-1.5B-Instruct \
  --gsm8k-data /mnt/e/AI/xllm-eval-data/int4-model-eval-20261002/gsm8k/data.jsonl \
  --gsm8k-shots /mnt/e/AI/xllm-eval-data/gsm8k-3101c7d5072418e28b9008a6636bde82a006892c/prepared/five_train_examples.json \
  --output /mnt/e/AI/xllm-eval-data/int4-diagnosis-replay/real_qkv.json \
  --model-e2e-output /mnt/e/AI/xllm-eval-data/int4-diagnosis-replay/model_e2e.json \
  --model-e2e-samples 4
```

原生产源码指纹：

| 文件 | SHA-256 |
|---|---|
| `xllm/python/attention/quantized.py` | `b748801dfe65dee29216ccd90f4677d83a752413fcc2a4eb5de7bc25408bae3d` |
| `xllm/python/attention/quantized_triton.py` | `d8a92f29915d202e74c6efd5954c359fe575eb220a9612b091cdba062a6429e6` |
| `xllm/python/models/qwen2.py` | `6de9eb37955111b9c23747715ed5b6fd2b81d1efbab3dd2ea20d5402bfe22ca2` |
| `tools/diagnose_int4_kv.py` | `242e074ad409017bde5f7e1bc202c812946470184047cd56c04c5626c97912df` |

后续修复请新增结果目录并与本快照对照，不覆盖本次失败案例或无效尝试的审计记录。
