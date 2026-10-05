# FP8 K-only / V-only KV 质量消融（2026-10-05）

## 范围与结论

本实验比较 BF16 基线、仅 K 的 FP8 E4M3 QDQ、仅 V 的 FP8 E4M3 QDQ 对输出质量的影响。测试了 Qwen2.5-1.5B-Instruct：GSM8K test 全部 1,319 题，以及 LongBench 八任务各 50 题，共 400 题/臂。采样为 greedy、seed 17；GSM8K 使用官方 5-shot，LongBench 复用 seed 17 的固定抽样和 context-cap 数据。

这里的 FP8 是写入缓存前对选定侧执行 scale=1 的 E4M3 quantize/dequantize（QDQ）精度消融。K/V 最终仍存于 BF16 cache，并由 BF16 query 和原版 FlashInfer attention 消费；不测原生 FP8 cache 存储、显存节省或压缩服务吞吐。FlashInfer native `DTypeKV` 对 K/V 共用 dtype，当前不支持原生 mixed-dtype K/V。

本轮 QDQ 与原生 FP8 的注意力算术路径不完全相同：初始 prefill 使用 ragged BF16 attention，QDQ 后写入 BF16 cache；chunked prefill/decode 使用 paged BF16 attention。原生 FP8 路径会让 prefill 也使用 paged FP8 attention。因此本实验隔离选定 K/V 数值扰动对模型质量的影响，不能完全代表原生 FP8 attention 的输出。

GSM8K 显示 K 对本模型与此路径很敏感：仅 K QDQ 后 strict EM 从 43.5% 降至 1.0%，262/1,319 个输出达到 512-token 上限；仅 V QDQ 的 strict EM 为 43.4%，接近基线。该结果描述这个模型、量化尺度和执行路径，不证明其他模型或 FP8 策略必然退化。

## GSM8K

三臂都覆盖相同的 1,319 条、使用同一 prompt、tokenizer 和生成上限。HTTP 请求并发和 `max_seqs_per_batch` 均为 64；模型为 Python eager、FlashInfer backend、`kv_cache_dtype=auto`、graph 关闭、自动 cache 预算，最大输出 512 token。算术 smoke 单独记录在各结果根目录的 `arms/*/arithmetic_smoke.json`，不并入 GSM8K 分数。

| Arm | Strict EM | Flexible EM | `finish_reason=length` | 生成 token 总数 |
| --- | ---: | ---: | ---: | ---: |
| BF16 auto | 574/1,319 (43.5%) | 854/1,319 (64.7%) | 12 (0.9%) | 203,825 |
| K-only FP8 QDQ | 13/1,319 (1.0%) | 24/1,319 (1.8%) | 262 (19.9%) | 274,591 |
| V-only FP8 QDQ | 572/1,319 (43.4%) | 851/1,319 (64.5%) | 19 (1.4%) | 201,705 |

按题配对的正确性翻转：

| 指标 | Arm | 两臂都正确 | BF16 only | FP8 only | 两臂都错误 |
| --- | --- | ---: | ---: | ---: | ---: |
| Strict EM | K-only | 5 | 569 | 8 | 737 |
| Strict EM | V-only | 525 | 49 | 47 | 698 |
| Flexible EM | K-only | 16 | 838 | 8 | 457 |
| Flexible EM | V-only | 804 | 50 | 47 | 418 |

以上配对状态由相同 `_eval_index` 逐题统计。K-only smoke 中 9 道算术题均未答对；V-only 与 BF16 smoke 的 9 道算术题均答对。烟测只作诊断，不计入 benchmark 分数。

结果、manifest 与配置：`/mnt/e/AI/xllm-eval-data/fp8-kv-quality-ablation-gsm8k-20261005/`。逐题输出位于 `runs/gsm8k/{auto,k-only-fp8,v-only-fp8}.jsonl`，配对评分位于 `runs/gsm8k/paired-{k-only-fp8,v-only-fp8}/paired_scores.json`。

## LongBench

每个任务使用固定的 50 条输入和官方输出长度。context-cap prepared manifest 为 `/mnt/e/AI/xllm-eval-data/longbench-5e628be450b7e67fb7ae6e201bd6d8f7056f7672/prepared/seed17_n50_contextcap32704/manifest.json`（SHA256 `2ae099ff6607c64cec7bb1026d19f1ee787be5f9d9a9edf99b32430f6fea3695`）。NarrativeQA 的 23/50 条 context 经 head/tail 截断以满足预算；其余任务未截断。三臂的请求并发和 `max_seqs_per_batch` 均为 8。

| 任务（50 条） | 指标 | BF16 | K-only FP8 QDQ | V-only FP8 QDQ |
| --- | --- | ---: | ---: | ---: |
| narrativeqa | QA F1 | 0.18912 | 0.00237 | 0.19420 |
| qasper | QA F1 | 0.36416 | 0.03722 | 0.35906 |
| multifieldqa_en | QA F1 | 0.43939 | 0.05417 | 0.46546 |
| 2wikimqa | QA F1 | 0.32767 | 0.00640 | 0.30944 |
| hotpotqa | QA F1 | 0.40089 | 0.01534 | 0.40342 |
| musique | QA F1 | 0.15853 | 0.00857 | 0.16541 |
| lcc | 代码相似度 | 0.44760 | 0.33500 | 0.45240 |
| repobench-p | 代码相似度 | 0.45860 | 0.29780 | 0.45560 |
| 八任务等权 macro | 混合任务指标 | **0.34824** | **0.09461** | **0.35063** |

该 macro 是八项抽样任务均分后的等权平均，不是 LongBench 全套官方总分。每臂八个结果文件均为 50 条，配对 scorer 核对了相同 `_eval_index`、原始记录和 prompt SHA256；HTTP 失败/超时为 0。达到官方每任务输出上限的请求数：BF16 139/400 (34.8%)、K-only 374/400 (93.5%)、V-only 130/400 (32.5%)。K-only 长度上限比例与极低各任务分数相符，故其 macro 不应脱离输出退化单独解读。HTTP task-window 跨度求和分别为 BF16 412.6 秒、K-only 469.3 秒、V-only 409.8 秒；不同输出长度下只作本次运行描述，不作格式性能结论。

结果目录为 `/mnt/e/AI/xllm-eval-data/fp8-kv-quality-ablation-longbench-20261005/`。逐题输出、逐任务配对分数、运行配置、server manifest 与日志均保存在 `runs/<task>/` 和 `arms/<arm>/`。

## 分析与后续取舍

两项 benchmark 都显示：该模型在 E4M3、scale=1 下对 K 的精度变化非常敏感，仅 V 的总体分数接近 BF16。V-only 的 GSM8K strict 正确性仍有 49 道由对变错、47 道由错变对；LongBench 各任务也有升降，不能称为逐题无损或效果提升。

本轮使用 BF16 存储和原版 FlashInfer attention，绕开了原生 FP8 缓存读取，K-only 仍重现明显退化。这支持 K 的量化数值误差是一个足以引发退化的因素，不能将此前全部损伤归因于 FlashInfer FP8 kernel；原生 FP8 路径的布局和数值行为仍需独立验证。K 的误差会改变 attention 分数及 softmax 权重，V 的误差作用于加权输出，这提供了敏感性差异的机制解释，但没有定位到具体敏感层或通道。

后续优先候选为 BF16 K + FP8 V 的真实混合缓存，以及已经验证过的 INT8。当前 FlashInfer 分页接口共用 K/V dtype，混合存储需要自定义 reader 或扩展 kernel。真正落地后还需测常驻缓存大小、吞吐和延迟；本轮不能证明任何显存或速度收益。K 的 FP8 若继续探索，应单独评估校准、敏感层保留高精度等策略。

## 复现与审计

从仓库根目录运行（LongBench 将 `gsm8k` 参数改为 `longbench`，并使用单独输出目录）：

```bash
source build/cmake.cuda-x86/resume-env.sh
/home/mingasa/venvs/xllm/bin/python tools/run_kv_fp8_quality_ablation.py gsm8k \
  --output-root /mnt/e/AI/xllm-eval-data/fp8-kv-quality-ablation-gsm8k-20261005
```

runner 每次依次启动并关闭一个 arm 的服务；`--resume` 会要求输入、配置和逐题 journal 均匹配后续跑。完整源码快照为 `/mnt/e/AI/xllm-eval-data/fp8-kv-quality-ablation-longbench-20261005/source_snapshot.tar.gz`，SHA256 `1b3c3111ddc0bfaf147e0c7f83638510077a3407d5d06a0f67f0136660d0a821`；本轮 Git HEAD 为 `2c00f80fc9c56834404345e9afc477a716616f9a`。首次 smoke 与两次未进入 GSM 推理的准备失败尝试均保留在独立目录，不纳入上述结果；smoke3 算术输出另行保存，不计入 benchmark 分数。实验输入哈希、runner/evaluator 源码 SHA256、模型 provenance 和服务 manifest 均保存在各实验根目录的 `experiment_manifest.json` 与 `arms/*/server_manifest.json`。
