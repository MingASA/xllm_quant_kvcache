# 本地 xLLM KV Cache 效果评估

此工具比较本地 xLLM OpenAI 兼容 API 的两个配置，保存两边的逐题原始响应、API token 用量、配置和配对分数。它不会下载数据、启动服务或修改模型。首轮请先用固定小样本 smoke；完整评测由操作者在确认服务和数据就绪后手动启动。

## 评测约定

- 主对比使用相同 checkpoint、`model_impl=python`、相同请求路径和 API 参数，只把 KV cache 从 `auto` 改为 `int8`。每次只运行一份服务：先跑 auto，停止服务后跑 int8，以适配单卡显存。原生 BF16 只做单独接入正确性验证，不进入 KV 效果配对。
- LongBench 使用官方 prompt、每个任务官方 `max_new_tokens` 和官方 `metrics.py` scorer。脚本按官方格式对每个参考答案打分并取最大值；QA 任务报告 token F1，LCC/RepoBench-P 报告 code similarity。主任务为 narrativeqa、qasper、multifieldqa_en、2wikimqa、hotpotqa、musique、lcc、repobench-p。按官方 `pred.py`，lcc/repobench-p 不套 chat 模板并走 `/v1/completions`；其余任务走 `/v1/chat/completions`，由相同模型服务应用 tokenizer 的 chat template。
- GSM8K 使用官方 lm-evaluation-harness 主配置的 `Question: ...\n Answer:` 格式、5-shot、greedy/temperature 0，并按官方 `until=['Question:', '</s>', '<|im_end|>']` 发送 stop 字符串。逐题结果分别报告 strict 与 flexible extraction exact match，保留 strict `scores`/`flip` 和独立的 `flexible_scores`/`flexible_flip`，两种指标不混成一个分数或 flip。此 harness 固定 `max_tokens=512`；官方 YAML 未在 task 文件中固定此值，因此它是本实验的显式生成上限。五个训练示例需由操作者预先准备成 JSON 文件；该脚本不随机抽样，也不下载数据。
- 每个请求统一 `temperature=0`、`top_p=1`、固定 `seed`，两边相同 `max_tokens` 和 stop token 配置。`--stop-token-ids` 是可选的数值数组参数；不传时沿用服务/模型默认 stop IDs，工具不会硬编码 Qwen 默认值。Qwen2.5-1.5B-Instruct 本轮命令显式传 `--stop-token-ids 151645 151643`，且 auto/int8 两次必须完全一致。配置会记录完整 generation 参数，配对评分会拒绝 stop IDs 不一致。本地 tokenizer 对 chat template 后输入计数，并要求输入 token 数 + 输出上限不超过 32,768；不截断长样本。API `usage.prompt_tokens` 与本地计数都会保存，差异会警告；传 `--strict-token-count` 会在差异或 API 未返回 token 用量时中止。当前 smoke 对超限样本做显式排除并记录；这不能代表完整 LongBench 分数。完整评测不能静默丢弃长样本，需先实现 BF16/INT8 共用且可审计的截断策略。
- Qwen2.5 官方 blog 的 GSM8K 73.2 是另一套评测协议的模型参考值；PolarQuant 论文报告的 LongBench 八任务均值 38.88 使用其自身实现/脚本（包含 128K 模型配置）。两者只作为背景，不是本地结果目标，也不应声称复现。

## 准备输入

脚本只读本地 JSONL。LongBench 行沿用官方数据格式，至少要有 `context`、`input`、`answers`，可保留 `_id`；`answers` 是字符串数组。数据应来自官方 `THUDM/LongBench` test split。GSM8K JSONL 每行至少含 `question`、`answer`（官方答案含 `####` 数值标记）；`--gsm8k-shots` 指向恰好五个 `{ "question": ..., "answer": ... }` 对象组成的 JSON 数组。

当前固定来源版本、原始文件 SHA256、样本 ID、Qwen tokenizer 预检都在 `/mnt/e/AI/xllm-eval-data/smoke_manifest.json`。LongBench 八项各准备官方前五行；其中 narrativeqa 第五行在服务端 chat-template 计数下为 37,097 prompt tokens，超过 32K，因此原始五行文件保留，`prepared/within_32768/` 中的可运行 smoke 共 39 条（narrativeqa 4 条，其余七项各 5 条）。GSM8K 使用官方原始 train/test 文件，test 前 10 条与 train 前 5 条作为固定 shots。`auto_server_manifest.json`/`int8_server_manifest.json` 记录已有成功 smoke 配置；每次新 run 仍须独立确认 endpoint 就绪。

每个 run 需要 `--server-manifest`。JSON 必须来自该次实际启动并已就绪的服务；只有服务真正启动、端点能返回模型响应后才设 `server_started=true`。字段含 `model_impl`（auto/int8 都应为 `python`）、`graph_mode`（必须 `off`/false）、`kv_cache_mode`（分别 `auto`/`int8`）、`checkpoint_path`、`checkpoint_sha256`（经核实的权重快照指纹）、`model_config_sha256`、`chat_template_sha256`、`model`。SHA 字段必须是 64 位小写十六进制。脚本核对 checkpoint 路径、本地 `config.json` hash 和 `tokenizer_config.json` 中 chat template hash；离线评分还要求两份 manifest 的 checkpoint 和 template 字段相同。图模式强制关闭，避免两个服务经过不同图执行路径。

结构示意（hash 值须替换为实际值；`server_started=true` 只能在服务端点已经就绪后记录）：

```json
{
  "server_started": true,
  "model_impl": "python",
  "graph_mode": "off",
  "kv_cache_mode": "auto",
  "checkpoint_path": "/mnt/e/AI/models/Qwen2.5-1.5B-Instruct",
  "checkpoint_sha256": "<verified 64-character weight SHA256>",
  "model_config_sha256": "<local config.json SHA256>",
  "chat_template_sha256": "<local chat template SHA256>",
  "model": "Qwen2.5-1.5B-Instruct"
}
```

`tools/test_fixtures/server_not_started_valid_hashes.json` 是离线拒绝用 fixture：三个 SHA 字段均是格式有效的哨兵值，但 `server_started=false`，loader 必须在访问 checkpoint 前拒绝它。

可用 `sha256sum /mnt/e/AI/models/Qwen2.5-1.5B-Instruct/config.json` 取得模型配置 hash。chat template hash 按下式计算（与脚本的规范化方式一致）：

```bash
python -c 'import hashlib,json,pathlib; p=pathlib.Path("/mnt/e/AI/models/Qwen2.5-1.5B-Instruct/tokenizer_config.json"); d=json.loads(p.read_text()); print(hashlib.sha256(json.dumps(d.get("chat_template"),ensure_ascii=False,sort_keys=True).encode()).hexdigest())'
```

评分依赖安装在独立的 `~/venvs/xllm`，不要更改已有 vLLM 环境。运行时需要 Transformers/本地 checkpoint tokenizer；LongBench scorer 的导入还需要其官方 `LongBench/metrics.py` 依赖（`fuzzywuzzy`、`rouge`、`jieba`）。API 请求使用 Python 标准库，不要求 OpenAI SDK。当前环境可从只读 vLLM site-packages 复用 Transformers：

```bash
source ~/venvs/xllm/bin/activate
export HF_HOME=/mnt/e/AI/huggingface
export PYTHONPATH=/home/mingasa/venvs/vllm/lib/python3.12/site-packages
```

```bash
source ~/venvs/xllm/bin/activate
export HF_HOME=/mnt/e/AI/huggingface
python tools/eval_kv_cache_quality.py \
  --task narrativeqa \
  --data /mnt/e/AI/xllm-eval-data/longbench-5e628be450b7e67fb7ae6e201bd6d8f7056f7672/prepared/within_32768/narrativeqa.jsonl \
  --longbench-repo /mnt/e/AI/xllm-eval-tools \
  --tokenizer /mnt/e/AI/models/Qwen2.5-1.5B-Instruct \
  --model Qwen2.5-1.5B-Instruct \
  --server-manifest /path/to/verified-auto-server.json \
  --arm auto --url http://127.0.0.1:8000/v1 \
  --limit 3 --seed 17 --stop-token-ids 151645 151643 --output /path/to/results
```

`--limit` 固定使用文件前 N 行，适合先做 smoke；完整评测去掉它。服务关闭后，用同一命令把 `--arm auto` 改为 `--arm int8`，把 URL 指向 int8 服务、manifest 换成实际 int8 配置，并为两次运行指定同一个输入数据、seed、stop-token-ids、tokenizer 和输出目录。分别生成 `auto.jsonl`/`auto.config.json` 与 `int8.jsonl`/`int8.config.json`。最后离线配对评分，不需要服务或 tokenizer：

```bash
python tools/eval_kv_cache_quality.py \
  --task narrativeqa --longbench-repo /path/to/LongBench \
  --auto-results /path/to/results/auto.jsonl \
  --int8-results /path/to/results/int8.jsonl \
  --output /path/to/results
```

评分会核对每行原始数据和渲染 prompt 的 SHA256，避免错误配对。GSM8K 命令将 `--task` 改为 `gsm8k`，提供 `--gsm8k-shots /mnt/e/AI/xllm-eval-data/gsm8k-3101c7d5072418e28b9008a6636bde82a006892c/prepared/five_train_examples.json`，使用 `/mnt/e/AI/xllm-eval-data/gsm8k-3101c7d5072418e28b9008a6636bde82a006892c/prepared/test_first10.jsonl`，并移除 `--longbench-repo`。工具会自动发送官方 GSM8K stop strings 并记录在 generation config；两臂 stop 配置不同会被 paired scorer 拒绝。默认产物位于 `$HF_HOME/xllm-kv-quality/<UTC 时间>/`；可用 `--output` 指定位置。GSM8K 评分同样可在推理完成后离线进行。主任务八个子分数的宏平均应在汇总时对任务等权计算，不要将不同任务题数混成 micro average。

原生 BF16 接入检查执行独立的一条请求，不传入离线配对评分。当前一次 smoke 已返回 `xLLM smoke test passed.`，产物为 `build/cmake.cuda-x86/native-bf16-smoke-final.log` 和 `native-bf16-smoke-final-response.json`。这只能验证该次 native eager BF16 接入，不验证 graph 路径或其他依赖组合；native 与 Python auto/INT8 的 KV 质量结果隔离。

## 官方来源与解释边界

## FlashInfer ABI 与验证边界

- 本地构建使用 FlashInfer `0.6.18.post1` FA2 AOT ABI 时，需显式设置 `-DFLASHINFER_FA2_0_6_18_ABI=ON`。该宏默认 `OFF`，只为四个 CUDA 源文件选择新调用签名；无运行时 ABI 自动探测或 fallback。`Dockerfile.cuda` 默认源码/Python 包版本仍为 `0.6.2`/`0.6.14`，旧 AOT 继续用默认旧 ABI。FA3 paged、CUDA Graph capture/replay 及旧/新依赖的运行兼容性尚未验证。

- [LongBench README / 数据格式](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md)、[官方 prompt](https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2prompt.json)、[官方输出长度](https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2maxlen.json)、[官方 scorer](https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py)、[官方推理逻辑](https://github.com/THUDM/LongBench/blob/main/LongBench/pred.py)。每个 arm 的 `<arm>.config.json` 记录本地 LongBench checkout git revision 与评分依赖版本。离线自测使用官方 checkout `2e00731f8d0bff23dc4325161044d0ed8af94c1e`；`metrics.py` SHA256 为 `e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323`。
- [lm-evaluation-harness GSM8K 配置](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml)定义 5-shot、greedy、`until` stop strings、exact-match 忽略规则和 strict/flexible 提取（task metadata version 3.0）；此工具复用这些 prompt/评分规则，但 API 服务端 chat template、本地固定 shots 和生成长度仍应与官方报告的具体设置区分。`<arm>.config.json` 记录 task YAML 版本与 Transformers 版本；这里使用轻量本地 exact-match 实现，没有安装整套 lm-evaluation-harness。
- [Qwen2.5 官方 blog](https://qwenlm.github.io/blog/qwen2.5-llm/)报告 Qwen2.5-1.5B-Instruct GSM8K 73.2。[PolarQuant NeurIPS 2025 论文](https://papers.nips.cc/paper_files/paper/2025/file/48aa1c882f99f5cdd3c76f7c0ede44a7-Paper-Conference.pdf)表格报告同名模型 LongBench 八项平均 38.88；其评测脚本/模板和 120K 截断协议与此处 32K 上限不同，因此不可直接横向宣称复现。
