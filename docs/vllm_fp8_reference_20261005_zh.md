# vLLM 原生 BF16/FP8 KV cache 参考实验

日期：2026-10-05。模型为本机 `Qwen2.5-1.5B-Instruct`。这是 vLLM 原生离线推理对照：权重两组都为 BF16，唯一有意改变的推理精度是 KV cache（`auto` 对 `fp8`，即 E4M3）。每个 arm 在独立子进程中运行，避免前一组的 CUDA 分配影响后一组。

| 数据 | BF16 cache | FP8 E4M3 cache | 观测 |
|---|---:|---:|---|
| 9 道短算术，24 token 上限 | 9/9 正确 | 0/9 正确 | FP8 有 8 条 `stop`、1 条 `length`；BF16 全部 `stop` |
| GSM8K 前 64 题，5-shot，seed 17 | strict EM 27/64（42.19%）；flexible EM 38/64（59.38%） | strict EM 1/64（1.56%）；flexible EM 1/64（1.56%） | BF16 全部 `stop`；FP8 有 52 条 `stop`、12 条达到 512-token 上限 |

算术题沿用 `results/flashinfer-fp8-smoke-20261005/README.md` 的乘法和八道加法原题。用于表格的 BF16 arm 是在 FP8 arm 之后以 `generation_config=vllm`、`repetition_penalty=1.0` 重新运行的校正结果，输出为 `1161` 及正确的八个和；FP8 九题均未得到正确的数字答案，有一题生成到 token 上限。早期未显式固定 generation config 的 BF16 初跑单独保留为 `arithmetic-bf16-initial-default-generation.json`，不计入表格，避免和 FP8 结果混用。GSM8K 的 prompt 由 `tools/eval_kv_cache_quality.py` 的 `_gsm_prompt` 和五条原 few-shot 示例生成，分数使用同一文件的 `_gsm_score`。GSM 输出在 `_gsm_score` 所定义的严格匹配与宽松匹配口径下都只有 1/64。

两种 cache 配置的 GSM prompt token IDs 完全相同，SHA-256 均为 `aea3d5d5b534091301cb03efceb2cbbf47a197319594c62caff3698f719f7d85`。算术 prompt token IDs SHA-256 均为 `a720d966aafcc6ad6ecb8633924f45da759abe96ab70d7e8f8359ed02dc82829`。GSM 输入集 SHA-256 为 `93377ddca91b89cfa35e987104ee7b420a1b17d42b91351b70993d57d0159cac`；5-shot 文件 SHA-256 为 `4e076bef9962698cf3df4701f093437c86c925dcfa987556ea4fdb0fbd93a3ce`。

运行环境为 vLLM 0.30.0、PyTorch 2.13.0+cu132、FlashInfer 0.6.18.post1、Transformers 5.17.0，NVIDIA GeForce RTX 5060 Ti（sm120）。两组都设置 `dtype=bfloat16`、`enforce_eager=True`、`enable_prefix_caching=False`、`gpu_memory_utilization=0.80`、`max_model_len=4096`、`generation_config=vllm`、`repetition_penalty=1.0`。算术用 greedy、24 token、seed 20261005，不设 stop strings；GSM 用 greedy、512 token、seed 17，并使用 evaluator 原有 stop strings `Question:`、`</s>`、`<|im_end|>`。

日志确认两组都实际选择 `AttentionBackendEnum.FLASHINFER`。FP8 arm 的 FlashInfer 日志确认 prefill/decode query dtype 均为 BF16，decode backend 为 `xqa`，KV dtype 为 `torch.float8_e4m3fn`。vLLM 对未校准的 FP8 KV cache 有默认 scale=1.0 的处理路径；安装版本的实现还会在缺少 q scale 时令其使用 k scale。日志含有 vLLM 关于 FP8 scale 的警告。实验没有做校准，也没有在框架外覆写 scale，因此结果代表本机该模型 checkpoint 加当前原生 vLLM 默认 scale 的路径。

此参考实验表明，这个严重退化可以在独立的原生 vLLM 路径中复现，且不是 xLLM 独有现象；它不证明 xLLM 的原生 KV 写入/读取、kernel 或调度路径没有额外问题。vLLM 0.30.0 的 `CacheDType` 配置不含“仅 K”或“仅 V”开关；本机对 `CacheConfig(cache_dtype="fp8_k")` 与 `CacheConfig(cache_dtype="fp8_v")` 的构造也都触发 literal validation error。因此本文没有把非原生补丁包装成 vLLM 单侧 KV 结果。独立的 K-only/V-only ablation 仍需在支持该能力的实现中进行。

复现入口：[tools/run_vllm_fp8_reference.py](../tools/run_vllm_fp8_reference.py)。完整参数、版本、输入哈希、prompt/output token IDs、生成文本、finish reason 及启动日志保存在 `E:\AI\xllm-eval-data\vllm-fp8-reference-20261005\`，包含 `run_config.json`、四个 arm 的 JSON、各启动日志和 runner 源码快照。运行时没有为每个 worker 保存逐字节源码快照；因此元数据中的 `reproduction_source_sha256` 标识当前可复现版本，不能作为历史 worker 启动时源码的溯源证明。初跑文件保留其旧日志；该初跑在元数据 finalize 时写入的源码 SHA 也不是有效的运行时源码证明。
