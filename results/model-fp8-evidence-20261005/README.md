# FP8 / KV 量化评测精简证据（2026-10-05）

此目录由 `tools/export_fp8_experiment_evidence.py` 从 E 盘原始结果只读导出。每条压缩 JSONL 仅保留题目 ID、评测索引、prompt/source record SHA256、生成文本、token 计数、finish reason 和错误标记；不包含 LongBench 输入 passage、完整 prompt、few-shot 文本或模型权重。配对分数只保留 item scores/flip。原始 E 盘数据与日志未改动；完整大日志以源路径和 SHA256 记录在 `source_log_index.json`，另摘录有限 backend/scale 行供查证。

## 结果计数摘要

- 单侧 FP8 QDQ GSM8K：`{"auto": 1319, "k-only-fp8": 1319, "v-only-fp8": 1319}`。配对准确率：`{"k-only-fp8": {"auto_strict_em": 0.4351781652767248, "auto_flexible_em": 0.6474601971190296, "k-only-fp8_strict_em": 0.009855951478392721, "k-only-fp8_flexible_em": 0.01819560272934041}, "v-only-fp8": {"auto_strict_em": 0.4351781652767248, "auto_flexible_em": 0.6474601971190296, "v-only-fp8_strict_em": 0.4336618650492798, "v-only-fp8_flexible_em": 0.645185746777862}}`。
- 单侧 FP8 QDQ LongBench：每任务每臂条数见 `qdq-single-sided-fp8/summary.json`；八任务 × 50 条 × 3 臂。任务配对分数亦在该文件。
- xLLM 真实动态 scale FP8 GSM8K：`{"auto": 1319, "fp8-e4m3": 1319, "fp8-e5m2": 1319}`；逐题配对分数见其 `summary.json`。
- vLLM BF16/E4M3 GSM8K：`{"bf16": {"n": 64, "strict_correct": 27.0, "flexible_correct": 38.0, "length": 0}, "fp8-e4m3": {"n": 64, "strict_correct": 1.0, "flexible_correct": 1.0, "length": 12}}`；这是 64 题参考结果。vLLM 文件中的 `run_vllm_fp8_reference.py` 是复现源码，不是 worker 启动时冻结的 runtime snapshot；边界详见 vLLM 报告。

## 来源与复核

详细实验结论和配置见仓库 `docs/kv_cache_fp8_kv_ablation_20261005_zh.md`、`docs/xllm_fp8_gsm8k_20261005_zh.md`、`docs/vllm_fp8_reference_20261005_zh.md`、`docs/flashinfer-fp8-integration-20261005.md`。各实验原始 E 盘路径、文件 SHA256、导出 artifact SHA256、条数校验及导出脚本 hash 位于 `export_index.json`；`SHA256SUMS.txt` 对导出目录文件逐项校验（不含该 checksums 文件自身）。

QDQ 两 benchmark 共用冻结 runtime 源快照，原 tar 原样保存在 `qdq-single-sided-fp8/source/source_snapshot.tar.gz`；xLLM 真实 FP8 源码以目录形式及源 manifest 归档。vLLM reproduction source 有效性边界记录在本目录 `vllm-reference/metadata/run_config.json` 与对应报告。
