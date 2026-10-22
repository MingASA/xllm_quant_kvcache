# 原生 FlashInfer BF16 / FP8 Joint KV 交付

开发基线：`e36f04f2f0b41471c28f2133d8acc653a28a4edb`；分支：`feat/native-flashinfer-joint-kv`。这里只在独立 5060 开发机修改源码，**未登录 H800，未验证 H800 14B 模型质量、速度、torch 2.10.0+cu130 环境或完整模型 CUDA Graph**。外部 `aiperf_h800_v4_native.py` 不属于本次改动。

## 路径与证据

| 责任 | 源码位置 |
|---|---|
| BF16 / FP8 选择 | `executor.py::_create_attention_backend()`：auto 且无 QUANTIZED_BACKEND → `FlashInferBackend`；FP8 + flashinfer → `FlashInferFP8Backend` |
| FP8 写入 | `flashinfer_fp8.py::write_cache()` → 复用已有 `quantized_triton.write_quantized_kv(fixed_scales=...)`，只量化/scatter，不用 Triton Attention |
| paged prefill / chunked prefill | native pair profile 的 BF16 `FlashInferBackend.prepare/execute_attention()`；FP8 subclass 的同名方法 → stock `BatchPrefillWithPagedKVCacheWrapper` FA2 |
| paged decode | 两臂共用 `_prepare_decode()`：stock `BatchDecodeWithPagedKVCacheWrapper`，Qwen14B GQA5 使用 tensor cores / FA2 |
| graph replay | parent plan/fast_decode_plan 显式传 KV dtype；FP8 使用 `_active_decode_wrapper`；静态 metadata buffers 与 FP8 view 持续存活，cache storage 换址会拒绝 |
| allocation | `FlashInferBackend.bind_kv_caches()` → `native_kv.cache_allocation()`，从实际 LayerCache tensors 记录 dtype/shape/bytes |
| 运行时 marker | `ModelExecutor.bind_kv_caches()`，在绑定后每 Worker 一次 `NATIVE_KV_BACKEND {...}`；重复 bind 不再输出 |

`class` 来自实际对象，`kv_cache_dtype` 来自对象的 cache codec，`attention_impl=flashinfer` 对应该对象；`graph_enabled` 来自真实 decode graph runner 是否创建，另记录 runner class。它表示启用了 decode graph 路线，不承诺每个 shape 都捕获/重放（prefill 和 bucket 外 decode 仍按 runner 的原有规则 eager）。marker 还含实际 FlashInfer version、Q/KV dtype、scale、FP8 view alias 检查等。外部脚本应同时检查 marker 与所有 48 层 `KV_ALLOCATION`，不能只检查启动参数或 HTTP 200。

配对配置只增加一个显式开关：`XLLM_FLASHINFER_NATIVE_KV=1`，两臂都使用 stock **FA2 + paged 初始 prefill**。它保留原 BF16 默认 auto/ragged 路线供其他用途；不要把 FA2 配对实验与 BF16 auto/FA3 最佳配置混为同一个 baseline。该 profile 锁定 FlashInfer `0.6.18.post1`，不支持就失败，不换成自定义 Attention。

## 缩放与真实显存

本次是**整个模型 K、V 各一个固定标量**，两者可不同；不是旧 Triton 每 token/head 动态 scale。

```
K8 = E4M3_RN(clamp(K / sK, -448, 448))
V8 = E4M3_RN(clamp(V / sV, -448, 448))
logits = Q @ K8.T * attention_scale * sK
output = softmax(logits) @ V8 * sV
```

复用的 writer 把标量写进 allocator 已有 FP32 token/head scale tensor，page copy 连同 payload/scale 一起复制；FlashInfer读取的是不复制的 float8 views，读时使用标量 `k_scale/v_scale`，不逐元素读取 scale tensor。这个冗余 scale 开销被计入实际 resident bytes。没有 BF16 cache shadow，也不在读取前展开整个上下文缓存。writer 的新 token BF16 投影输入和 FlashInfer 内部计算不属于 BF16 常驻 shadow。

Qwen14B 每层每 token：BF16 K/V 实际 `8*128*2*2=4096` bytes；FP8 payload `8*128*2=2048` bytes，两个 FP32 scale `8*2*4=64` bytes，总计 **2112 bytes**（48 层为 196608 / 101376 bytes/token）。最终容量和 page 数以每个目标服务的 allocation 日志为准；backend workspace 另占显存，两臂共用同样的 wrapper/workspace 策略。C++ allocator/copy 实现已有这些存储，本次没有修改 C++，不需要为本改动单独编译 C++；目标 binary 必须已有该 allocator/Python ABI，若不满足，由用户在 H800 **独立构建新 binary**，不替换旧实验 binary。

**当前不给未测的 14B scale 数值，1.0 也不算已校准。** 新服务检查入口默认在 BF16 eager 的首个固定语义请求上采样：每层第一次真实 prefill 和第一次 decode 的 post-RoPE K / V absmax，只保留统计量，不保存激活；padding slot 排除。其余两题用于语义对比，不扩充首轮 scale 样本。建议值为：

```
sK = 2**ceil(log2(1.1 * observed_K_absmax / 448))
sV = 2**ceil(log2(1.1 * observed_V_absmax / 448))
```

尺度还要求能表示为正常 FP32 数（建议公式以 FP32 最小 normal 值兜底，防止 scale 写入变成零）。10% 是对已观测幅度的余量，向上取二次幂使缩放稳定、可复用，不代表覆盖未来所有请求。E4M3 的有限范围 448；大于 `448*s` 会饱和，过大的 scale 又会损失小幅值（最小 subnormal 对应 `s/512`）。全局 outlier、长位置 K、未见任务仍有质量风险。检查输出包含实测 max、覆盖范围、96 条 layer/phase 记录和建议数值；保留完整质量评估后才能下结论。诊断采样有 CPU 同步，**不得放进 AIPerf timed server**；开启诊断时 graph 明确拒绝。

API 依据为本机安装的 0.6.18.post1 真实执行及 [FlashInfer v0.6.18 prefill source](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/prefill.py)、[decode source](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/decode.py)：plan 显式接受 `kv_data_type`；paged run 的 `k_scale` 合入 softmax scale、`v_scale` 乘输出。没有把 ragged FP8、FA3 或新版其他 API 的行为当作本路线证据。

## 5060 → H800 同步

修改/新增的文件清单（共同基线上的一次源码提交）：

```
xllm/python/attention/flashinfer.py
xllm/python/attention/flashinfer_fp8.py
xllm/python/attention/native_kv.py
xllm/python/model_executor/executor.py
tools/check_native_flashinfer_service.py
tests/python/test_flashinfer_attention.py
tests/python/test_flashinfer_native_kv.py
tests/python/test_native_kv_evidence.py
docs/native_flashinfer_joint_kv_h800.md
```

新工具还复用基线中的 `qwen25_experiment_model.py`、`eval_kv_cache_quality.py`、`run_qwen25_joint_kv_quality.py` 的 provenance、token 计数和服务/缓存验证函数；这些文件未修改。同步完整 Git 版本，或核对基线与全部依赖，不只手动拷贝一个 backend 文件。

用户在 H800 手动执行下面步骤。已有 `$BASE/xllm_quant_kvcache-joint14b` 可能带手工补丁，**只读查看，保留它**。从可访问的 GitHub origin fetch 本分支，新增 candidate worktree；不用 Raw GitHub，不强制更新旧 worktree：

```bash
BASE=/export/home/yinjinrun/.borrow/chenjunyu/xllm-h800
REPO="$BASE/xllm_quant_kvcache-joint14b"
git -C "$REPO" status --short
git -C "$REPO" fetch origin feat/native-flashinfer-joint-kv
REF=origin/feat/native-flashinfer-joint-kv
git -C "$REPO" rev-parse "$REF"  # 必须等于交付 commit SHA
CANDIDATE="$BASE/xllm_quant_kvcache-native-kv-$(git -C "$REPO" rev-parse --short "$REF")"
git -C "$REPO" worktree add --detach "$CANDIDATE" "$REF"
```

如果 H800 无法 fetch，交付还提供带版本号的增量 Git bundle（基线 e36f04f2 为 prerequisite）和源码 SHA256 清单；由**用户自己** SCP 到 H800。先 `git bundle verify <bundle>`，通过后从本地 bundle fetch 该分支，再按上面的 worktree 方式部署。验证不通过代表基线缺失，不能用手工覆盖旧文件绕过。Codex 不会尝试 SSH/SCP。

用户沿用已有 H800 独立单 GPU Docker 镜像/依赖，在新隔离容器将 **candidate** 只读挂载到 `/workspace/xllm-native-kv`，模型只读挂到 `/models/Qwen2.5-14B-Instruct`；新结果目录挂到 `/workspace/experiments/native-kv/<run-id>`。不要覆盖旧 worktree、共享 venv、运行 binary 或升级 torch/FlashInfer/Triton。目标环境保持 FlashInfer 0.6.18.post1 / torch 2.10.0+cu130 / Triton 3.6.0。在容器内核对 HEAD 和清单：

```bash
cd /workspace/xllm-native-kv
git rev-parse HEAD  # worktree 的 .git 指向宿主路径时可在宿主机核对，勿“修复”旧仓库
sha256sum -c /path/to/delivered-source.sha256
```

若容器只挂载了 worktree 而没有宿主 Git 元数据，请同时把相应 Git 元数据**只读**按原绝对路径挂载；新检查入口需读取 commit。也可以使用完整 checkout 部署 candidate。清单路径与 metadata mount 由用户按现有 Docker 配置填写，不假设镜像名或 binary 路径。

## 一条真实服务验收命令（用户在 H800 容器执行）

```bash
cd /workspace/xllm-native-kv
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -B tools/check_native_flashinfer_service.py \
  --model /models/Qwen2.5-14B-Instruct \
  --server-binary /path/to/independently-verified-h800-xllm \
  --gpu 0 --output /workspace/experiments/native-kv/RUN-ID/eager
```

binary 路径和 RUN-ID 待用户填写。只显露一张空闲 GPU，两臂顺序运行。此命令会 BF16 eager 采样建议 scales → 真正 FP8 服务运行相同三题，检查 48 层实际 pages、唯一 marker、SSE role/content/finish/usage/[DONE]、至少一次多 token decode、本地/API prompt token 一致性，并记录答案、重新分词后的首个差异位置及相似度、length exhaustion。三题覆盖多步算术、状态跟踪、代码语义；任一答案不符合预定语义条件会保留报告并失败。它只是一小组有意义的验收，**不代表 GSM8K/LongBench 质量或服务性能已通过**。初次权重 SHA256 计算可能较慢，不占用模型 GPU。

默认检查的缩放统计保存到 `eager/calibration.json`，实际建议数值在 `eager/report.json`。如使用已有统计，`--fp8-k-scale <实测值> --fp8-v-scale <实测值>` 显式设定；或 `--scales-file .../eager/calibration.json`（会核对同目录模型 provenance）。

代码现已适配 **decode CUDA Graph**。本机 tiny kernel graph 通过，但 H800 完整 14B Graph 仍待目标机验收。要一次验证两臂 graph，在新输出目录运行同一命令并添加：

```bash
  --graph cudagraphs --scales-file /workspace/experiments/native-kv/RUN-ID/eager/calibration.json
```

两个 graph server 都禁用在线采样，并使用相同调度/预算/bucket策略；不捕获 prefill，未命中 bucket 仍按原生 runner 规则执行。若 H800 graph 失败，保留日志、明确报告“目标机 Graph 尚未对齐”，先用两臂相同 `--graph off` 做 **production-relevant native eager comparison**；不自动 retry 或隐藏回退，不称其为最佳生产基线。

## 外部 AIPerf 0.13.0 的服务配置契约

原有 xLLM `/v1/chat/completions` SSE 服务接口未改动，[AIPerf chat endpoint](https://github.com/ai-dynamo/aiperf/tree/v0.13.0/src/aiperf/endpoints) 可使用 stream=true 的 role/content chunks、finish_reason、可选 usage 和 `[DONE]`。这里只验证 HTTP/SSE 服务契约，未在 H800 执行 AIPerf，也不改外部输入或脚本。

共同：`XLLM_FLASHINFER_NATIVE_KV=1`、`XLLM_KV_QUALITY_MODE=none`，prefix cache 两臂都禁用；clear `XLLM_FP8_CALIBRATION_OUTPUT`。group=`14b-native`，arm=`bf16` / `fp8_e4m3`。模型、同一固定输入、EOS、生成上限、batch/scheduler/显存预算完全相同：

* BF16：`--kv_cache_dtype=auto`；**unset XLLM_QUANTIZED_BACKEND**。
* FP8：`--kv_cache_dtype=fp8_e4m3`；`XLLM_QUANTIZED_BACKEND=flashinfer`，显式传递实测 `XLLM_FP8_K_SCALE` / `XLLM_FP8_V_SCALE`。
* eager 配对：两臂 `--enable_graph=false --enable_prefill_piecewise_graph=false --python_graph_backend=off`。
* 经 H800 验收后 graph 配对：两臂 `--enable_graph=true --enable_prefill_piecewise_graph=false --python_graph_backend=cudagraphs`。

Prefix Cache 的 payload/copy机制已有支持，固定 scales 与 byte views 不改变 page copy；本次仍统一关闭以避免引入共享前缀影响。没有新开发 Prefix Cache、TP/DP 或 INT8/INT4 Attention。

## 已有本机证据

当前 RTX 5060 Ti / SM120（不是 H800）：GQA5 FP8 prefill、decode、graph replay 与 dense/eager 对齐，BF16 同 FA2/paged profile 的 prefill→graph decode 对齐；metadata 换内容但地址稳定、FP8 view 别名、负 slot 均已测试。CPU 验证 48 层实际 tensor 字节、采样 scale 公式与 marker/dispatch。具体通过计数与日志随交付记录；这些结果不能代替 SM90、torch 2.10 和真实 14B 的验证。没有给出伪造的 14B scale、质量或加速数字。

验证记录：原生 GQA5 FP8 3 项、BF16 配对路径 1 项、FP8 empty/copy/reuse 1 项通过；原有 FlashInfer 回归 12 项、CPU evidence/SSE 12 项通过。executor 全文件为 89 passed / 4 failed，4 个 NPU 用例因本机不支持 NPU device/模块未通过，未算作通过。日志保存在 5060 的 `/mnt/e/AI/xllm-storage/reports/native-flashinfer-20261022/`。

验证边界补充：BF16 GPU 单测在 pytest 的 runtime 隔离 stub 下使用已有 BF16 scatter writer 替身；FlashInfer planner/attention/graph 是实际 GPU 调用，但不等于已验证原 C++ writer 的真实服务。page copy 用例直接复制 payload 与 scale 来验证 view/索引复用；原 C++ copy 实现未改，本轮未重新编译或执行 C++ copy 测试。真实 writer、48 层 allocator 和完整服务须由上面的 H800 入口最终验证。
