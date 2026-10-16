# H800 / H200 mixed-V 复现

本分支支持 SM90（H800、H200）和原 SM120（RTX 5060 Ti）Qwen2.5-1.5B-Instruct Python eager 路径。**SM90 已交叉编译，当前开发机只有 SM120；H800/H200 的运行正确性、速度、质量尚未实测。** 下列 gates 必须在各目标机器运行成功后再开始服务实验，不能复制 5060 Ti 的 gates 或 CUDA 二进制冒充验证。

## 架构审计与实现边界

| 项目 | 原 SM120 | 新 SM90 |
|---|---|---|
| 编译目标 | `compute_120 / sm_120`，原编译选项 | `compute_90 / sm_90`，独立 `XLLM_MIXED_V_SM90` 分支 |
| shared memory/SM 校验 | 100 KiB | 228 KiB，另验证 opt-in block 上限 |
| Q tile 16/64/128 的 NUM_MMA_KV | 1/4/2，原样保留 | 2/8/4，按 pinned FlashInfer BF16 FA2 预算 |
| 默认 Q/KV warps | decode 1/4；prefill 4/1 | 相同 |
| planner | 本机 stock FlashInfer FA2 BF16 planner | 同一 planner，在本机获取 SM 数和资源；无固定 SM 数 |
| attention | BF16 MMA、FP32 accumulation、causal、split-KV/merge | 相同算法；tile 增大可能改变浮点归约顺序 |
| 量化 | K BF16；V INT8 / E4M3 / packed signed INT4；FP32 token/head scale | 完全复用 codec、writer、block copy、量化/舍入语义 |

`cp.async`、`ldmatrix`、BF16 `mma.sync`、shared-memory swizzle 和寄存器解码均为可用于 Hopper 的实现；不依赖 SM120 独有指令。此路径未迁移到 FA3/WGMMA/TMA，也不宣称达到 Hopper 最优速度。SM90 的 tile 来自 BF16 Q 占用 `Q_tile * 128 * 2` 字节、K/V 每 MMA tile 占用 `(128+128)*16*KV_warps*2` 字节、每 SM 目标 2 CTA 和寄存器上限 `8/NUM_MMA_Q`，向下取受支持的 1/2/4/8。四 arm 共用架构策略，避免按精度单独放宽 planner 预算。

`original` 默认不变；`packed_decode` 及其余已有实验策略仍需显式环境变量选择。packed/register/scales/vector/matrix/balanced 策略是算法消融，非自动按显卡型号切换的最优策略。此次只交叉编译 `original`、`packed_decode`；其他变体不能算已通过 SM90 验证。SM120 的热路径、tile、warp、精度与编译优化选项未改变，两个大 CUDA header 未改动。

资源依据：[NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/archive/13.0.2/hopper-tuning-guide/index.html)、[NVIDIA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus)。H200 与 H800 使用同一 SM90 FA2 适配，不硬编码 SM 数、显存容量或显存带宽。不同 GPU 分开报告；使用“GPU 显存 / VRAM、显存带宽”，5060 Ti 的 GDDR7 不称作 HBM。

## 1. 准备输入（原实验机器执行一次）

以下命令从仓库根目录运行，激活能导入原实验 `torch`、`flashinfer` 的 Python 环境。原始数据/模型不随 Git 上传；必须传输冻结 trace 和官方 scorer，不能重新随机选题。`BUNDLE` 放在有空间的磁盘。

```bash
export BUNDLE=/mnt/e/AI/hopper-reproduction-inputs
mkdir -p "$BUNDLE/bench" "$BUNDLE/govreport"
cp results/flashinfer-mixed-v2-20261010/formal-two-passes/trace_*.json* "$BUNDLE/bench/"
cp results/flashinfer-govreport-seq256-20261011/formal-two-passes/trace_*.json* "$BUNDLE/govreport/"
cp /mnt/e/AI/xllm-eval-data/model_provenance.json "$BUNDLE/"
cp -a /mnt/e/AI/xllm-eval-tools "$BUNDLE/longbench-scorer"
cp -a /mnt/e/AI/models/Qwen2.5-1.5B-Instruct "$BUNDLE/model"
python -m pip freeze > "$BUNDLE/original-python-freeze.txt"
python - <<'PY'
import flashinfer, pathlib, subprocess, os
package = pathlib.Path(flashinfer.__file__).resolve().parent
subprocess.run(['tar', '--exclude=__pycache__', '--exclude=*.so', '-czf',
                os.environ['BUNDLE'] + '/flashinfer-pinned-source.tgz',
                '-C', str(package.parent), package.name], check=True)
PY
```

将整个 `$BUNDLE` 传输到目标机器。trace 导入器会检查 SHA256、数量、ID 和每条 prompt 的 hash。正式数据：GSM8K 1319；LongBench 原 8-task 共 400；GovReport 200。GovReport 保留原 trace 的 head/tail 截断审计与 512/1024 输出协议，**不重新根据新卡的 BF16 输出选择上限**。

## 2. 目标机环境与构建

在 H800、H200 上分别执行。前提为已具备本仓库 CUDA 构建依赖的 Linux x86_64 xLLM 开发环境（CMake/Ninja、C++ 编译器、vcpkg/native 依赖、CUDA Toolkit、compute-sanitizer）。本地验证组合是 Python 3.12、PyTorch `2.13.0+cu132`、CUDA Toolkit 13.2、FlashInfer `0.6.18.post1`；这是验证组合，不是声称所有较低 CUDA/PyTorch 版本均受支持。使用其他依赖组合必须重新编译并通过 gates。PyTorch 的 CUDA runtime 和 nvcc 应使用同一 CUDA 主版本，驱动须支持该 runtime。

激活目标机 CUDA Python 环境后安装项目所需依赖（已有 torch 应保留；不要混用另一个环境构建 C++）：

```bash
python -m pip install 'flashinfer-python==0.6.18.post1' pytest aiohttp numpy \
  transformers safetensors sentencepiece rouge jieba fuzzywuzzy matplotlib psutil ninja
export BUNDLE=/data/hopper-reproduction-inputs   # 改成实际传输目录
export RUN_ROOT=/data/xllm-h800                  # H200 改为 /data/xllm-h200
mkdir -p "$RUN_ROOT" "$RUN_ROOT/cache" "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
export FLASHINFER_WORKSPACE_BASE="$RUN_ROOT/cache/flashinfer"
export XLLM_FLASHINFER_MIXED_BUILD_DIR="$RUN_ROOT/cache/mixed-v"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export XLLM_MODEL_PATH="$BUNDLE/model"
export XLLM_LONGBENCH_REPO="$BUNDLE/longbench-scorer"
export XLLM_KV_QUALITY_MODE=none
export XLLM_FLASHINFER_MIXED_KERNEL_VARIANT=original
export XLLM_FLASHINFER_MIXED_NVCC="$(command -v nvcc)"
```

原环境的 FlashInfer 是以 `upstream.json` 的 header hash 为准的 pinned 源码。仅版本号相同不足以保证一致。下面把原实验源码安装到**目标机专用虚拟环境**，不传输旧 CUDA `.so`；stock planner 和 mixed extension 会在目标 GPU 上重新构建：

```bash
SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
tar -xzf "$BUNDLE/flashinfer-pinned-source.tgz" -C "$SITE"
nvidia-smi --query-gpu=index,uuid,name --format=csv
# 选一张空闲卡的实际 UUID，H800/H200 每台分别选择；UUID 避免 CUDA/NVML 编号顺序差异。
export CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-ACTUAL-UUID
export XLLM_TELEMETRY_GPU="$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import torch, flashinfer
p = torch.cuda.get_device_properties(0)
assert (p.major, p.minor) == (9, 0), p
assert p.shared_memory_per_multiprocessor == 233472, p
print(torch.__version__, torch.version.cuda, flashinfer.__version__, p)
PY
export TORCH_CUDA_ARCH_LIST=9.0a
export CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=90a'
python setup.py build --device cuda --generate-so false
cmake --build build/cmake.cuda-x86 --target kv_cache_estimation_test kv_cache_test -j 4
export XLLM_SERVER_BINARY="$PWD/build/cmake.cuda-x86/xllm/xllm"
ldd "$XLLM_SERVER_BINARY"
```

`ldd` 不应出现 `not found`；native 依赖由目标开发环境提供，不能照搬原 WSL 的 `LD_LIBRARY_PATH`。目标机尚未实测整套 native 构建，若出现平台依赖错误必须先解决，不能使用 SM120 server binary。原 `third_party/xllm_ops` 工作区本来有未提交改动，本次未包含或覆盖；目标构建以 Git 中的 submodule revision 为准。

验证 checkpoint：

```bash
python - <<'PY'
import hashlib, json, os, pathlib
p = json.loads((pathlib.Path(os.environ['BUNDLE'])/'model_provenance.json').read_text())
h = hashlib.sha256()
with (pathlib.Path(os.environ['XLLM_MODEL_PATH'])/p['weights']['file']).open('rb') as f:
    for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
assert h.hexdigest() == p['weights']['sha256']
PY
```

## 3. Correctness → 最小服务验证 → kernel benchmark

不传 `--allocator-gates` 会在当前机器重新运行 C++ allocator/copy/容量测试、215 项 Python codec/writer/attention 测试，再执行 mixed-V dense/reference tests 和 Compute Sanitizer。输出目录必须全新；失败不得继续正式实验。

```bash
python tools/check_flashinfer_mixed_v.py --output "$RUN_ROOT/gates"
export GATES="$RUN_ROOT/gates/correctness_gates.json"
python tools/prepare_mixed_v_reproduction.py --source "$BUNDLE/bench" \
  --output "$RUN_ROOT/smoke" --smoke-samples 1
python tools/run_mixed_v_cache_experiment.py run --output "$RUN_ROOT/smoke" \
  --gates "$GATES" --attention-backend flashinfer_mixed --workloads gsm8k \
  --no-joint-quality --concurrency 1 --repetitions 1 --replay-cycles 1 --warmup 1 \
  --use-default-token-budget --max-seqs-per-batch 256 --max-concurrent-requests 512 \
  --max-memory-utilization 0.93
python tools/benchmark_flashinfer_mixed_v.py --gates "$GATES" \
  --output "$RUN_ROOT/kernel-original.json" --repetitions 5
```

smoke 默认遍历四 arm，保留正常 EOS/原 max_tokens；它只是完整服务链路检查，不是质量或性能结论。kernel benchmark 内的 BF16 与 stock FA2 逐位比较也必须通过；输出 eager wall 与 CUDA-event graph timings，graph 数字不是 graph-off 服务延迟。

需要测已优化的 `packed_decode` 时，使用独立 gates、独立输出目录，四 arm 都使用同一 variant：

```bash
export XLLM_FLASHINFER_MIXED_KERNEL_VARIANT=packed_decode
python tools/check_flashinfer_mixed_v.py --output "$RUN_ROOT/gates-packed" \
  --allocator-gates "$RUN_ROOT/gates/allocator_gates.json"
python tools/benchmark_flashinfer_mixed_v.py \
  --gates "$RUN_ROOT/gates-packed/correctness_gates.json" \
  --output "$RUN_ROOT/kernel-packed.json" --repetitions 5
export XLLM_FLASHINFER_MIXED_KERNEL_VARIANT=original
```

## 4. 正式实验（默认不自动执行）

以下每组均四 arm、同 checkpoint、同 trace/arrival order、正常 EOS、同 sampling。`--repetitions 1 --replay-cycles 2` 表示每个 benchmark 两次全量回放，不是再重复三遍。runner 默认两次预热参数，实际每个并发档预热至少一个满并发批次；预热不计入测量。使用本机相同 `0.93` 显存比例，**不把 5060 Ti 的 15 GiB 硬编码到 H800/H200**。

### A. 原 40 组协议：GSM8K + LongBench，8/16/32/64/128

保留历史 4096 token budget、128 batch limit；明确作为历史协议，不与后来的 default-token 协议混合。

```bash
python tools/prepare_mixed_v_reproduction.py --source "$BUNDLE/bench" --output "$RUN_ROOT/original40"
python tools/run_mixed_v_cache_experiment.py run --output "$RUN_ROOT/original40" \
  --gates "$GATES" --attention-backend flashinfer_mixed --workloads gsm8k longbench \
  --concurrency 8 16 32 64 128 --repetitions 1 --replay-cycles 2 \
  --max-seqs-per-batch 128 --max-concurrent-requests 200 --max-memory-utilization 0.93
python tools/report_mixed_v_reproduction.py --output "$RUN_ROOT/original40"
```

### B. GSM8K 高并发：128/160/208/256，xLLM 默认 token budget

```bash
python tools/prepare_mixed_v_reproduction.py --source "$BUNDLE/bench" --output "$RUN_ROOT/gsm-high"
python tools/run_mixed_v_cache_experiment.py run --output "$RUN_ROOT/gsm-high" \
  --gates "$GATES" --attention-backend flashinfer_mixed --workloads gsm8k \
  --concurrency 128 160 208 256 --repetitions 1 --replay-cycles 2 \
  --use-default-token-budget --max-seqs-per-batch 256 --max-concurrent-requests 512 \
  --max-memory-utilization 0.93
python tools/report_mixed_v_reproduction.py --output "$RUN_ROOT/gsm-high"
```

历史最后一次 GSM8K 主分析到 208；这里 256 为可选扩展。严格复现已有点时从命令中去掉 `256`。默认 token budget 在本 commit 为 10240，runner 不传覆盖值，并校验服务日志；prefill chunk 使用 xLLM 默认解析。

### C. GovReport：8/16/32/64/128，原冻结长上下文长输出 trace

```bash
python tools/prepare_mixed_v_reproduction.py --source "$BUNDLE/govreport" --output "$RUN_ROOT/govreport"
python tools/run_mixed_v_cache_experiment.py run --output "$RUN_ROOT/govreport" \
  --gates "$GATES" --attention-backend flashinfer_mixed --workloads longbench \
  --concurrency 8 16 32 64 128 --repetitions 1 --replay-cycles 2 \
  --use-default-token-budget --max-seqs-per-batch 256 --max-concurrent-requests 512 \
  --max-memory-utilization 0.93
python tools/report_mixed_v_reproduction.py --output "$RUN_ROOT/govreport"
```

历史 GovReport 复用了 batch limit 128 的 BF16，量化组记录为 256，client concurrency 均不超过 128。本次 H800/H200 **统一全部 arm 为 256**，避免保留配置不一致；此差异必须在跨实验报告中披露。GovReport 用官方 ROUGE-L scorer，不用 QA F1；分数、paired flip、length exhaustion 按 load/repetition/cycle 分开保存。

## 5. 结果与审计

每个新实验目录保留 server command、有效 scheduler 参数、checkpoint/trace provenance、commit/diff、环境/软件版本、native build manifest、allocation、逐请求 responses、服务 metrics 与每秒 GPU 采样。`report/` 生成 throughput/latency/显存图、质量图和 JSON/CSV；`reproduction_quality.json` 包含 GSM8K Strict/Flexible EM、各 LongBench task、8-task macro、GovReport ROUGE-L、paired flip 与 length-exhaustion。只对完整回放评分，缺少的 arm 不补造数据。

GPU memory utilization 是显存活动时间比例，不等于实际 GB/s；真实 bandwidth/traffic 需要单独 Nsight Compute 分析。不同 GPU 的绝对速度不可当作量化收益，收益只能在同一卡、同配置的 BF16 与 mixed-V 之间比较。原历史结果不修改；所有输出路径使用新的 `$RUN_ROOT`。H800 与 H200 各自运行 fresh gates、smoke、kernel/服务实验并分别保存结果，不能共享服务测量结果或 correctness 声明。

## 本次适配验证记录（2026-10-16）

- SM90 `original` 和 `packed_decode` 均完成交叉编译，生成独立 `.so`；编译日志只有 FlashInfer 未使用变量 warning。此项是交叉编译检查，不代表在 SM90 GPU 上执行过。
- 本机仅有 RTX 5060 Ti（SM120）。此前 fresh gates 通过：allocator estimation 25 项、KV cache 26 项、codec 215 项；mixed-V correctness 36 项，Compute Sanitizer memcheck 零错误。gates 中的 native build architecture 为 `sm_120`。
- 四臂最小 GSM8K smoke（auto、BF16/INT8、BF16/E4M3、BF16/INT4）各完成 1 条请求；四个 native build manifest 均为 `sm_120`。随后 reproduction report 入口成功生成性能与 paired-quality 报告和图。单题分数仅确认报告链路可运行，不作为质量结论。
- H800/H200 尚无可用设备；SM90 上的运行正确性、性能和模型质量仍待分别实测。
