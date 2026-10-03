<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# GSM8K：仅 V INT4 与旋转 G32 的完整质量对照

[实验报告](../../docs/kv_cache_gsm8k_int4_variants_20261002_zh.md)；本目录 `summary.json` 是 E 盘汇总的逐字节副本，SHA-256 为 `42f5984335d684b61b886bb2c79e2bff4cfcddf2b441e9e2df9ba8bd291115ff`。

三臂各 1319 题、并发 64、同一 5-shot prompt 和生成设置。BF16、仅 V INT4 G128、K/V RHT INT4 G32 的 strict/flexible 正确数分别为 `578/853`、`543/853`、`0/5`。详细逐题比较、长度限制与计时边界见报告和汇总。

**实验边界：** 常驻 cache 均为 BF16，变体对新 K/V 在 GPU 上量化、反量化，再交给原 FlashInfer 写入和执行。这是质量对照，不证明压缩显存、容量收益或生产 kernel 性能。仅 V 的宽松总分与 BF16 相同，但 206 题正误状态翻转，不是逐题无损。RHT G32 不包含 outlier sidecar、centering 或 affine offset。

## 原始结果与独立归档

- 原始逐题 JSONL、配置、日志和服务证明：`/mnt/e/AI/xllm-eval-data/gsm8k-int4-variants-20261002/`。
- 完整归档：`/mnt/e/AI/xllm-eval-archives/gsm8k-int4-variants-20261002-20261002T133348Z.tar.gz`。
- 归档 SHA-256：`aad4f45d5b81e0204fe4a03daaeb556de5faffb89202eb6f8f25979f0da42c6f`。

归档包含三臂结果目录、报告、此次运行的 GPU 变换/FlashInfer/factory/codec 源码及测试、评测脚本、1319 条输入与五个训练示例；不重复保存模型权重。基础仓库提交为 `7d823304`，实验改动源码已收入归档。原始结果与归档在同一 E 盘，不是异地备份；这些新结果尚未推送到 Git 远端。

## 复现注意事项

- 原 xLLM CUDA 环境：`source build/cmake.cuda-x86/resume-env.sh`；本地模型 `Qwen2.5-1.5B-Instruct`。完整协议与模型 SHA-256 见报告。
- 服务端常驻 cache 选择 `--kv_cache_dtype=auto`，Python model implementation、graph 全关闭。启动前显式设置 `XLLM_KV_QUALITY_MODE=none`、`v_only_int4` 或 `int4_rht_g32`；后两者启动日志必须显示 `persistent_cache=BF16 (not compressed)`。
- evaluator 新臂名为 `auto`、`v-only-int4`、`int4-rht-g32`。变体 manifest 必须同时证明 `kv_cache_mode=auto` 和正确的 `quality_transform_mode`。新复现应生成自己的启动证明和新输出目录，不能把旧 manifest 当新服务的证明。
- loopback 服务本轮使用 `127.0.1.1:18994`，请求需绕过代理；`NO_PROXY` 包含 `127.0.1.1,localhost`。
- JSONL 按并发完成顺序落盘；评分和配对须按 `_eval_index` 排序，不可按文件行序配对。
- 不覆盖本目录的固定汇总、原始 JSONL、前序 INT4 失败报告或旧归档。

## 运行源码指纹

| 文件 | SHA-256 |
|---|---|
| `xllm/python/attention/kv_quality_transform.py` | `16eb2b231e04887842dad39f0afdfcfd9dfb85f98011331c23f96da55771a408` |
| `xllm/python/attention/flashinfer.py` | `dcf6adb555bca557f0fbe752c0b110edbc1acefae76e698875459daf4a277bf4` |
| `xllm/python/model_executor/executor.py` | `010108cfd08a5e77b8fb073e80b4069b60d9929addc668a477d50469338af37a` |
