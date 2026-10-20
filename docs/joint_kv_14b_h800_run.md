# H800：14B 联合 KV 量化质量实验（手动部署）

开发与离线测试只在 5060 完成，尚未在 H800 验证。四臂 `auto/int8/fp8_e4m3/int4` 均为单 GPU Python eager / Triton，关闭 graph、QDQ 和 prefix cache。此入口只测质量，不替代之前的 FlashInfer mixed-V 性能实验。

## 1. 宿主机创建新 worktree

以下第一步可直接复制；不会覆盖现有仓库。目标目录必须尚不存在。

```bash
export BASE=/export/home/yinjinrun/.borrow/chenjunyu/xllm-h800
git -C "$BASE/xllm_quant_kvcache-qwen25" fetch origin feat/joint-kv-14b-quality
git -C "$BASE/xllm_quant_kvcache-qwen25" worktree add --detach \
  "$BASE/xllm_quant_kvcache-joint14b" origin/feat/joint-kv-14b-quality
```

## 2. 选择已有镜像、空闲单卡与真实数据路径

沿用现有 H800 隔离容器使用的 **CUDA/PyTorch/xLLM 已构建镜像及运行环境**；本任务未获知镜像名和 binary 的容器内路径，不猜测它们。只新建一个容器，不停止、重启或修改其他容器。下面 `待填写` 必须换成实际值；native binary 必须在该镜像中可执行，必要时按现有部署方法只读挂载其构建产物及依赖。新 runner 用 `--python_model_path` 加载新 worktree 中的 Python 代码。

```bash
export IMAGE='待填写：现有已验证镜像名或镜像ID'
export SERVER_BINARY_IN_IMAGE='待填写：镜像内现有xllm可执行文件绝对路径'
export GPU_UUID='待填写：空闲H800的GPU-UUID'
export GSM8K_HOST='待填写：单轮GSM8K原始JSONL绝对路径'
export SHOTS_HOST='待填写：five-shot JSON绝对路径'
export LONGBENCH_HOST='待填写：八任务prepared数据目录绝对路径'
export LB_REPO_HOST='待填写：包含LongBench/config和metrics.py的官方仓库根目录'
export RUN_ID="joint14b-$(date -u +%Y%m%dT%H%M%SZ)"
export CNAME="xllm-$RUN_ID"
mkdir -p "$BASE/experiments/joint-kv-14b/$RUN_ID"
# 数据/模型只读；结果与编译缓存只写入本次新目录；仅暴露一张 GPU。
docker run -d --name "$CNAME" --restart=no --gpus "device=$GPU_UUID" \
  --ipc=private --shm-size=8g \
  --mount "type=bind,src=$BASE/xllm_quant_kvcache-joint14b,dst=/workspace/xllm_quant_kvcache-joint14b,readonly" \
  --mount "type=bind,src=$BASE/experiments/joint-kv-14b/$RUN_ID,dst=/workspace/experiments/joint-kv-14b/$RUN_ID" \
  --mount type=bind,src=/mnt/cfs/9n-das-admin/llm_models/Qwen2.5-14B-Instruct,dst=/models/Qwen2.5-14B-Instruct,readonly \
  --mount "type=bind,src=$GSM8K_HOST,dst=/inputs/gsm8k.jsonl,readonly" \
  --mount "type=bind,src=$SHOTS_HOST,dst=/inputs/five-shot.json,readonly" \
  --mount "type=bind,src=$LONGBENCH_HOST,dst=/inputs/longbench,readonly" \
  --mount "type=bind,src=$LB_REPO_HOST,dst=/inputs/longbench-official,readonly" \
  -e RUN_ID -e XLLM_SERVER_BINARY="$SERVER_BINARY_IN_IMAGE" \
  -e HF_HOME="/workspace/experiments/joint-kv-14b/$RUN_ID/cache/huggingface" \
  -e XDG_CACHE_HOME="/workspace/experiments/joint-kv-14b/$RUN_ID/cache" \
  -e TRITON_CACHE_DIR="/workspace/experiments/joint-kv-14b/$RUN_ID/cache/triton" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --workdir /workspace/xllm_quant_kvcache-joint14b --entrypoint /bin/bash "$IMAGE" -lc 'exec sleep infinity'
docker exec -it "$CNAME" bash
```

容器内 `python` 应是原先可导入 torch/triton/transformers 及官方 LongBench scorer 依赖的环境；沿用现有镜像所需的 `LD_LIBRARY_PATH`，不要搬入 5060 的 CUDA `.so`。若原部署使用虚拟环境，先激活它。无需安装 FlashInfer；不下载权重或数据。没有公布服务端口，评测只走容器内部 loopback。

## 3. 容器内先审计，再单卡 smoke

```bash
cd /workspace/xllm_quant_kvcache-joint14b
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RUN="/workspace/experiments/joint-kv-14b/$RUN_ID"
COMMON=(--model /models/Qwen2.5-14B-Instruct --server-binary "$XLLM_SERVER_BINARY"
        --gsm8k /inputs/gsm8k.jsonl --gsm8k-shots /inputs/five-shot.json
        --longbench-data /inputs/longbench --longbench-repo /inputs/longbench-official
        --gpu 0 --port 18996)
python -B tools/run_qwen25_joint_kv_quality.py "${COMMON[@]}" \
  --output "$RUN/formal-round1" --dry-run > "$RUN/plan.json"
python -m json.tool "$RUN/plan.json"
```

核对 `data_audit` 中各文件 SHA256、`rows == unique_questions`、`selected_rows` 与实际 **每轮** 规模。`data_audit=null` 代表输入未就绪，不能开始正式实验。八任务文件名为 narrativeqa/qasper/multifieldqa_en/2wikimqa/hotpotqa/musique/lcc/repobench-p.jsonl。不要把旧 QDQ 的两个重复结果合并为双倍样本，也不要传预测结果 JSONL 充当源数据。重复问题或 ID 会直接拒绝；`--limit` 是每任务固定前缀，仅标为 smoke。

```bash
# 四臂依次启动，各运行一条 GSM8K，真实分配检查在正式评测请求前完成。
python -B tools/run_qwen25_joint_kv_quality.py "${COMMON[@]}" \
  --tasks gsm8k --limit 1 --output "$RUN/smoke" \
  > "$RUN/smoke.console.log" 2>&1
```

查看每臂 `arms/<arm>/attempt-0001/{server.log,allocation.json,server_manifest.json,eval-gsm8k.log}` 与 `arms/<arm>/integrity.json`；FP8 payload 是 `torch.uint8` 中的 E4M3 bits，INT4 是 packed `uint8`，两者均有独立 FP32 key_scale/value_scale。auto 必须是 BF16 K/V 且无 scale。任一 dtype/shape/bytes 或严格 prompt token 校验不符立即停止。首个 bootstrap 不计评分。完整通过后才手动执行下一步。

## 4. 正式入口与恢复

```bash
python -B tools/run_qwen25_joint_kv_quality.py "${COMMON[@]}" \
  --approved-data-manifest "$RUN/plan.json" --output "$RUN/formal-round1" \
  > "$RUN/formal-round1.console.log" 2>&1
# 如需复现第二次重复：完全相同数据和参数，独立输出，独立统计，不能扩大样本量。
python -B tools/run_qwen25_joint_kv_quality.py "${COMMON[@]}" \
  --approved-data-manifest "$RUN/plan.json" --output "$RUN/formal-round2" \
  > "$RUN/formal-round2.console.log" 2>&1
```

默认资源参数来自当前 xLLM：max_memory_utilization=0.8、max_tokens_per_batch=10240、max_seqs_per_batch=200、admission=200、block_size=128、max_cache_size=0。质量请求 concurrency=1；需要匹配已确认的旧质量协议时，显式传 `--concurrency` 及相同资源预算，四臂共用，先更新 dry-run。不要为了容纳一个 arm 单独改配置。生成设置复用评分器：seed=17、temperature=0、top_p=1、GSM8K max_tokens=512、LongBench 官方 maxlen 与正常 EOS/stop；超过 32768 的 prompt+generation 直接失败，不重新截断。

失败后重复 **原命令** 加 `--resume`，控制台日志用新文件或 `>>` 追加。程序只终止自身创建的服务进程组，释放这张卡；新 attempt 目录保留旧日志。恢复核对源码/模型/数据/参数和历史 allocation，验证非顺序逐题 journal，跳过已完成题和完整 arm。缺失/损坏的最后一行、重复索引或记录不匹配会保留文件并拒绝恢复；不会静默截掉坏记录。更改 arm/task/limit/seed/预算需要新输出目录。

宿主机可另开终端只读查看：

```bash
tail -f "$BASE/experiments/joint-kv-14b/$RUN_ID/formal-round1.console.log"
find "$BASE/experiments/joint-kv-14b/$RUN_ID/formal-round1/arms" -name server.log
```

逐题结果在 `runs/<task>/<auto|int8|fp8-e4m3|int4>.jsonl`；原评分器生成 `paired/<task>/<arm>/paired_scores.json`，包含逐题分数与 flip；`summary.json/csv` 含 Strict/Flexible EM、length-exhaustion、各任务分数及完整八任务 macro（0–1）。`experiment_manifest.json` 记录源码 commit/hash、模型分片 hash、数据 hash、请求协议、GPU 与依赖；`complete.json` 和每臂 integrity 记录完整性摘要。更改源码后不能继续旧 run。保留这些原始文件交给 GPT 审查；暂不据此宣称 H800 性能或质量结果。
