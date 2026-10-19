# Qwen2.5 1.5B / 7B / 14B KV 实验复现（H800）

先完成 [Hopper README](../hopper-mixed-v/README.md) 的依赖、pinned FlashInfer 源码、CUDA 构建和环境设置。这里替换其 1.5B 模型与实验启动部分；不复制原 GPU 的编译产物、gates 或模型 provenance。H200 同样使用 SM90 路径，结果单独报告。

## 参数与实现审计

| Model | hidden | Q heads | KV heads | GQA | head dim | layers |
|---|---:|---:|---:|---:|---:|---:|
| 1.5B | 1536 | 12 | 2 | 6 | 128 | 28 |
| 7B | 3584 | 28 | 4 | 7 | 128 | 28 |
| 14B | 5120 | 40 | 8 | 5 | 128 | 48 |

来源：[7B 官方 config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)、[14B 官方 config](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/config.json)。两个新模型均不共享 embedding/lm_head，默认上下文 32768。本协议使用单卡 TP=1，不启用 YaRN 或滑动窗口。

* `flashinfer_mixed.py` 的构造检查、planner 参数和 layer 检查现在读取模型形状。
* CUDA wrapper 的 paged-KV heads、Q stride、GQA fast-div、grid.z、split-KV merge heads 同步参数化。head_dim=128 保持为本轮支持边界；没有仅移除 assert。
* 底层 FA2 kernel 按 runtime GQA 映射 head，未修改 tile、warp、SM90/SM120 架构策略、量化 scale 或舍入算法。mixed 路径包括 decode 在内均使用 FA2 BatchPrefill planner；不经过 CUDA-core decode 的离散 GQA dispatch。
* Qwen2 模型配置、QKV 投影、layer 数、KV allocation 本来来自 ModelArgs；7B/14B 的独立 lm_head 和 sharded safetensors 已有加载路径，无需复制模型实现。新增对实际 layer 数、KV shape 的服务启动检查。
* runner 不再强行注入 SM120 的 stock FlashInfer 编译缓存/架构。分片权重逐文件流式 SHA256，记录聚合 identity；不会把 1.5B 的单文件 hash 当作新模型身份。
* 冻结 trace 导入保留题目、prompt、顺序、max_tokens，仅按目标 tokenizer 重新核算 token 数。超出上下文直接失败，不截断或重新选题。历史结果不修改。

## 两类实验不要混淆

1. **K/V 精度敏感性**：复现原 `auto / k-only-fp8 / v-only-fp8` 质量协议，固定 scale=1、E4M3 QDQ，常驻 BF16 cache，stock FlashInfer。用于比较 K 与 V 量化的质量回退，不用它声称压缩服务加速。
2. **真实压缩服务**：四组 BF16、BF16 K + INT8/FP8/packed INT4 V，统一 `flashinfer_mixed`，动态 token/head scale。记录正常 EOS 的质量与吞吐、TTFT/TPOT/E2E、GPU/缓存指标。它与上述固定 scale QDQ 是不同量化协议，各自对比自己的 BF16 baseline。

本次没有实现真实低精度 K 常驻的 FlashInfer kernel。

## 数据与环境

沿用 Hopper README 的 `$BUNDLE/bench`（1319 GSM8K、400 LongBench 冻结请求）与 `$BUNDLE/govreport`（原 200 条长输出协议）。质量消融还需把原机器 `/mnt/e/AI/xllm-eval-data` 下这三个目录原样复制到目标机 `$EVAL_DATA`：

* `int4-model-eval-20261002/gsm8k/data.jsonl`
* `gsm8k-3101c7d5072418e28b9008a6636bde82a006892c/prepared/five_train_examples.json`
* `longbench-5e628be450b7e67fb7ae6e201bd6d8f7056f7672/prepared/seed17_n50_contextcap32704/`

同时保留 prepared 目录内的 manifest.json 等协议文件；目录不能缺失，也不能重新抽样替代。

```bash
export BUNDLE=/data/hopper-reproduction-inputs
export RUN_ROOT=/data/xllm-h800
export XLLM_EVAL_DATA_ROOT=/data/xllm-eval-data
export XLLM_LONGBENCH_REPO="$BUNDLE/longbench-scorer"
export XLLM_SERVER_BINARY="$PWD/build/cmake.cuda-x86/xllm/xllm"
export XLLM_KV_QUALITY_MODE=none
export XLLM_FLASHINFER_MIXED_KERNEL_VARIANT=original
# 清除原机器的 stock FlashInfer 缓存覆盖，按当前 GPU 编译。
unset FLASHINFER_OPS_PATH FLASHINFER_CUDA_ARCH_LIST
```

模型提前放在目标机本地目录，包含完整 config/tokenizer/generation_config 与所有 safetensors 分片及 index；本次开发不下载这些权重。

## 正确性 → 最小端到端 → 全量

```bash
python -B tools/check_flashinfer_mixed_v.py --output "$RUN_ROOT/gates-qwen25"
export GATES="$RUN_ROOT/gates-qwen25/correctness_gates.json"
```

门禁要求三种形状 × 四精度的 108 个 attention 测试、Compute Sanitizer 和 allocator/codec gates；旧的 36-case gates 不能用于新模型。每台目标 GPU 各跑一次。

先做各新模型的最小质量 smoke，生成独立 provenance（不计正式结果）：

```bash
export XLLM_MODEL_PATH=/data/models/Qwen2.5-7B-Instruct
export XLLM_MODEL_ID=Qwen2.5-7B-Instruct
export XLLM_MODEL_PROVENANCE="$RUN_ROOT/7b-smoke-provenance.json"
python - <<'PY'
import json, os
from pathlib import Path
from tools.qwen25_experiment_model import checkpoint_provenance
p = Path(os.environ['XLLM_MODEL_PROVENANCE'])
with p.open('x') as f:
    json.dump(checkpoint_provenance(Path(os.environ['XLLM_MODEL_PATH'])), f, indent=2)
PY
python -B tools/run_kv_fp8_quality_ablation.py all --smoke-only \
  --output-root "$RUN_ROOT/7b-quality-smoke"
```

14B 同样操作，将以上路径和 ID 的 `7B/7b` 改为 `14B/14b`。真实 mixed-V 服务最小测试沿用 Hopper README 的 smoke 命令，但导入时必须加 `--model "$XLLM_MODEL_PATH" --model-provenance "$XLLM_MODEL_PROVENANCE"`，使用这里的新 gates；不能复用旧 checkpoint manifest。

最小真实压缩服务测试（沿用上面的模型环境变量，四种格式各一题）：

```bash
python -B tools/prepare_mixed_v_reproduction.py --source "$BUNDLE/bench" \
  --output "$RUN_ROOT/$XLLM_MODEL_ID-service-smoke" --smoke-samples 1 \
  --model "$XLLM_MODEL_PATH" --model-provenance "$XLLM_MODEL_PROVENANCE"
python -B tools/run_mixed_v_cache_experiment.py run \
  --output "$RUN_ROOT/$XLLM_MODEL_ID-service-smoke" --gates "$GATES" \
  --attention-backend flashinfer_mixed --workloads gsm8k --no-joint-quality \
  --concurrency 1 --repetitions 1 --replay-cycles 1 --warmup 1 \
  --use-default-token-budget --max-seqs-per-batch 256 --max-concurrent-requests 512 \
  --max-memory-utilization 0.93
```

通过后，每个模型一条命令运行两轮质量消融，然后运行四组真实压缩服务（每个负载两次重放，不额外重复三次）：

```bash
python -B tools/run_qwen25_kv_experiments.py \
  --model /data/models/Qwen2.5-7B-Instruct --output "$RUN_ROOT/qwen25-7b" \
  --trace-source "$BUNDLE/bench" --govreport-source "$BUNDLE/govreport" --gates "$GATES"
python -B tools/run_qwen25_kv_experiments.py \
  --model /data/models/Qwen2.5-14B-Instruct --output "$RUN_ROOT/qwen25-14b" \
  --trace-source "$BUNDLE/bench" --govreport-source "$BUNDLE/govreport" --gates "$GATES"
```

两条命令顺序执行，不与其他 GPU 任务混跑。输出目录必须不存在。只跑一阶段可指定 `--stage quality` 或 `--stage performance`，仍用独立输出目录。启动命令保存在 `commands.json`，每次服务保留环境、原始结果与模型身份。

服务配置复现历史两条独立协议，不混成一组比较：GSM8K 高并发 128/160/208/256、GovReport 8/16/32/64/128 使用 xLLM 默认 token budget、max_seqs_per_batch=256、admission=512；原 LongBench 40 组使用原 token budget=4096、max_seqs_per_batch=128、admission=200、并发 8/16/32/64/128。均保持 max_memory_utilization=0.93，所有格式配置相同、正常 EOS、原 max_tokens。模型增大可能使同一请求集更难接纳，这是结果，不自动缩小上下文或负载。

kernel benchmark 的 workload/repetitions 保持原工具默认，只指定模型 heads：

```bash
python -B tools/benchmark_flashinfer_mixed_v.py --gates "$GATES" \
  --num-qo-heads 28 --num-kv-heads 4 --output "$RUN_ROOT/kernel-7b.json"
python -B tools/benchmark_flashinfer_mixed_v.py --gates "$GATES" \
  --num-qo-heads 40 --num-kv-heads 8 --output "$RUN_ROOT/kernel-14b.json"
```

## 已验证与未验证

本地 SM120：三模型 head 形状的 attention tests **108 passed**；Compute Sanitizer memcheck **108 passed，0 errors**；checkpoint profile/分片身份 CPU 测试 **3 passed**。stock FlashInfer `BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True, backend="fa2")` 的 GQA5/GQA7 tiny CUDA 测试均通过 dense reference（最大绝对误差分别 0.0074501 / 0.00649309）；实际经 FA2 prefill module 完成 JIT 编译与 decode。这些是 synthetic attention 验证，不等于大模型端到端质量或速度验证。开发机没有 H800/H200，也未下载 7B/14B 权重，因此目标 GPU 编译运行、分片模型实际加载、全量质量和服务吞吐仍须按上述流程验证；没有生成或虚构大模型性能结论。
