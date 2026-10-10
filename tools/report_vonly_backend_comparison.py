# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Report the paired C20/100-question V-only backend comparison."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger
from tools import eval_kv_cache_quality as evaluator
from tools.report_mixed_v_cache_experiment import _telemetry

_BACKENDS = ("triton", "flashinfer_mixed")
_ARMS = ("bf16_int8", "bf16_int4")


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def _collect(root: Path) -> list[dict]:
    trace = evaluator._jsonl(root / "trace_gsm8k.jsonl")
    if len(trace) != 100:
        raise ValueError("Expected exactly 100 GSM8K questions")
    records = []
    for backend in _BACKENDS:
        for arm in _ARMS:
            for repetition in range(3):
                run = root / backend / arm / "gsm8k" / f"c20_r{repetition}"
                summary = json.loads((run / "summary.json").read_text())
                config = json.loads((run / "run_config.json").read_text())
                rows = evaluator._jsonl(run / "responses.jsonl")
                indexed = {row["index"]: row for row in rows}
                if len(rows) != 100 or set(indexed) != set(range(100)):
                    raise ValueError(f"Incomplete/duplicate responses: {run}")
                if (
                    config.get("attention_backend") != backend
                    or config["concurrency"] != 20
                    or config["replay_cycles"] != 1
                ):
                    raise ValueError(f"Wrong measured backend/load: {run}")
                if (
                    summary["requests"] != 100
                    or summary["peak_client_inflight"] != 20
                    or summary["attention_compiles_during_measurement"]
                ):
                    raise ValueError(f"Invalid measurement: {run}")
                strict, flexible = [], []
                for index, expected in enumerate(trace):
                    row = indexed[index]
                    if row["id"] != expected["id"] or row["prompt_sha256"] != expected["prompt_sha256"]:
                        raise ValueError(f"Request trace mismatch: {run}/{index}")
                    if row["usage"]["prompt_tokens"] != expected["prompt_tokens"]:
                        raise ValueError(f"Prompt token count mismatch: {run}/{index}")
                    scores = evaluator._gsm_score(row["prediction"], expected["row"]["answer"])
                    strict.append(scores[0])
                    flexible.append(scores[1])
                records.append(
                    dict(
                        backend=backend,
                        arm=arm,
                        repetition=repetition,
                        elapsed_s=summary["elapsed_s"],
                        request_throughput=summary["request_throughput"],
                        token_throughput=summary["token_throughput"],
                        generated_tokens=summary["generated_tokens"],
                        mean_output_tokens=summary["generated_tokens"] / 100,
                        length_exhaustion=summary["length_exhaustion"],
                        strict_em=statistics.mean(strict),
                        flexible_em=statistics.mean(flexible),
                        peak_client_inflight=summary["peak_client_inflight"],
                        event_loop_lag_max_s=summary["event_loop_lag_max_s"],
                        **{
                            f"{metric}_{p}": summary[f"{metric}_s"][p]
                            for metric in ("ttft", "tpot", "e2e")
                            for p in ("p50", "p95", "p99")
                        },
                        gpu=_telemetry(run / "telemetry.jsonl"),
                    )
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    records = _collect(root)
    aggregate = {}
    for backend in _BACKENDS:
        aggregate[backend] = {}
        for arm in _ARMS:
            rows = [row for row in records if row["backend"] == backend and row["arm"] == arm]
            numeric = [key for key, value in rows[0].items() if isinstance(value, (float, int)) and key != "repetition"]
            aggregate[backend][arm] = {metric: _mean_std([row[metric] for row in rows]) for metric in numeric}
    (root / "service_summary.json").write_text(
        json.dumps(
            dict(protocol="3 repeats, sample SD; 100 questions each", records=records, aggregate=aggregate), indent=2
        )
        + "\n"
    )
    fields = [key for key in records[0] if key != "gpu"]
    with (root / "service.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    kernels = json.loads((root / "kernels.json").read_text())
    kernel_rows = []
    for case in kernels["cases"]:
        for arm in _ARMS:
            triton = case["median"][f"triton/{arm}"]
            native = case["median"][f"flashinfer_mixed/{arm}"]
            kernel_rows.append(
                dict(
                    arm=arm,
                    batch=20,
                    query_tokens=case["query_tokens_per_request"],
                    context_tokens=case["context_tokens"],
                    triton_gpu_us=triton["gpu_us"],
                    flashinfer_gpu_us=native["gpu_us"],
                    speedup=triton["gpu_us"] / native["gpu_us"],
                    triton_eager_us=triton["eager_us"],
                    flashinfer_eager_us=native["eager_us"],
                    shared_writer_gpu_us=case["median"][f"shared_writer/{arm}"]["gpu_us"],
                )
            )
    with (root / "kernels.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(kernel_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(kernel_rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes, ("e2e_p95", "tpot_p95"), ("p95 E2E (s)", "p95 TPOT (s)")):
        x = np.arange(2)
        for i, backend in enumerate(_BACKENDS):
            values = [aggregate[backend][arm][metric]["mean"] for arm in _ARMS]
            errors = [aggregate[backend][arm][metric]["std"] for arm in _ARMS]
            ax.bar(x + (i - 0.5) * 0.35, values, 0.35, yerr=errors, capsize=4, label=backend)
        ax.set_xticks(x, ["V-INT8", "V-INT4"])
        ax.set_ylabel(title)
        ax.legend()
    fig.suptitle("GSM8K 100 questions, concurrency 20, mean ± sample SD over 3 runs")
    fig.tight_layout()
    fig.savefig(root / "service_latency.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for arm in _ARMS:
        rows = [row for row in kernel_rows if row["arm"] == arm]
        ax.plot(range(len(rows)), [row["speedup"] for row in rows], marker="o", label=arm)
    ax.set_xticks(range(len(rows)), [f"Q{row['query_tokens']}/L{row['context_tokens']}" for row in rows])
    ax.set_ylabel("Triton / FlashInfer GPU attention latency (higher is faster)")
    ax.set_title("V-only attention, batch 20, same inputs and paging")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "kernel_speedup.png", dpi=180)
    plt.close(fig)

    def _cell(backend: str, arm: str, metric: str, factor: float = 1.0) -> str:
        value = aggregate[backend][arm][metric]
        return f"{value['mean'] * factor:.2f} ± {value['std'] * factor:.2f}"

    lines = [
        "# V-only INT8/INT4：Triton 与 FlashInfer 对比",
        "",
        "使用同一个 Qwen2.5-1.5B-Instruct checkpoint，K 保持 BF16，V 常驻 INT8 或 packed INT4，FP32 token/head scale。FlashInfer 是本次独立 FA2 扩展；两者共用现有 Triton writer/codec，没有全局 BF16 V Cache。",
        "",
        "服务协议：现有固定 benchmark 顺序的前 100 道 GSM8K，prompt 715–819 tokens；并发 20；每组预热 20 请求后测 100 请求，重复 3 次。temperature=0、top_p=1、seed=17、正常 EOS、max_tokens=512。Python eager/graph-off，相同 scheduler（4096 tokens/batch、128 seqs/batch）、prefix cache off、max_memory_utilization=0.93。两后端串行运行，避免 GPU 资源竞争。",
        "",
        "结果为 3 次指标的均值 ± 样本标准差；p95 是每次 100 请求 p95 的平均，不把重复的问题当独立质量样本。",
        "",
        "| V | Backend | 总时间 s | req/s | generated tok/s | p95 TTFT ms | p95 TPOT ms | p95 E2E s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in _ARMS:
        for backend in _BACKENDS:
            lines.append(
                f"| {arm} | {backend} | {_cell(backend, arm, 'elapsed_s')} | {_cell(backend, arm, 'request_throughput')} | {_cell(backend, arm, 'token_throughput')} | {_cell(backend, arm, 'ttft_p95', 1000)} | {_cell(backend, arm, 'tpot_p95', 1000)} | {_cell(backend, arm, 'e2e_p95')} |"
            )
    lines += [
        "",
        "![服务延迟](service_latency.png)",
        "",
        "| V | 平均 output tokens/request | Strict EM % | Flexible EM % | length exhaustion/100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in _ARMS:
        for backend in _BACKENDS:
            lines.append(
                f"| {backend}/{arm} | {_cell(backend, arm, 'mean_output_tokens')} | {_cell(backend, arm, 'strict_em', 100)} | {_cell(backend, arm, 'flexible_em', 100)} | {_cell(backend, arm, 'length_exhaustion')} |"
            )
    lines += [
        "",
        "输出长度不强制相同。质量分数只描述这 100 题，用来检查输出行为；不能作为全量 GSM8K 或量化相对 BF16 的质量结论。",
        "两个 backend 的 attention 数值路径不同：FlashInfer 在 shared memory 反量化为 BF16 后做 Tensor Core MMA；Triton 保留其原有运算路径。因此即使低精度 payload 相同，输出也不保证逐位一致。本次单算子逐项对照通过 atol=0.015、rtol=0.02，正常 EOS 服务结果仍需要如实保留长度和答案差异。",
        "",
        "| V | Q tokens/request | Context tokens | Triton GPU us | FlashInfer GPU us | GPU speedup | 共享 writer GPU us |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in kernel_rows:
        lines.append(
            f"| {row['arm']} | {row['query_tokens']} | {row['context_tokens']} | {row['triton_gpu_us']:.2f} | {row['flashinfer_gpu_us']:.2f} | {row['speedup']:.2f}× | {row['shared_writer_gpu_us']:.2f} |"
        )
    lines += [
        "",
        "![单算子](kernel_speedup.png)",
        "",
        "单算子均 B20、page128，相同 Q/K/V 和 shuffled pages，5 次重复、随机 backend/arm 顺序。CUDA event 测预热后的 graph batch 16×10；另存 50 次 eager 墙钟。Graph 只是 kernel 诊断；服务仍 graph-off。attention-only 包括其 split/merge，不计 planner 和 writer。writer 单独测量，两个后端用的是同一个实现。Q768×20 的完整 prefill 是独立 kernel 形状；服务实际会受 4096-token batch budget 限制而分批。",
        "",
        "原始文件：[服务 CSV](service.csv)、[服务汇总 JSON](service_summary.json)、[单算子 CSV](kernels.csv)、[单算子 JSON](kernels.json)、[协议与 trace 哈希](protocol.json)。各 backend 子目录保存 startup 参数、环境/源码哈希、实际 resident allocation、100 条逐请求响应和低开销 GPU/server 采样。",
        "",
        "所有 12 次服务 replay 都必须完成 100 请求、并发峰值 20、prompt/token 计数匹配，且没有 attention/writer JIT，才生成本页。TTFT 是客户端首个非空文本的时间；TPOT=(E2E−TTFT)/(API generated tokens−1)。延迟从 HTTP 提交起，包含 server 等待，排除客户端 semaphore 排队。memory utilization 是 GDDR7 显存控制器活跃时间，不等于峰值显存带宽利用率。只测 C20，不据此宣称饱和吞吐或容量上限。",
        "",
    ]
    conclusions = []
    for arm in _ARMS:
        baseline = aggregate["triton"][arm]
        native = aggregate["flashinfer_mixed"][arm]
        throughput_gain = (native["token_throughput"]["mean"] / baseline["token_throughput"]["mean"] - 1) * 100
        tpot_drop = (1 - native["tpot_p95"]["mean"] / baseline["tpot_p95"]["mean"]) * 100
        e2e_change = (native["e2e_p95"]["mean"] / baseline["e2e_p95"]["mean"] - 1) * 100
        conclusions.append(
            f"{arm}：generated tok/s 变化 {throughput_gain:+.1f}%，p95 TPOT 降低 {tpot_drop:.1f}%，p95 E2E 变化 {e2e_change:+.1f}%。"
        )
    lines[2:2] = [
        "本次 C20 服务实测，相对 Triton：" + " ".join(conclusions),
        "",
        "新 kernel 的 decode 单步更快，但本组回答平均变长，p95 整体响应时间反而上升。INT8 完成 100 请求略快，INT4 略慢；不能把单算子加速倍数直接当作服务加速倍数。",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    logger.info("Comparison report: %s", root / "README.md")


if __name__ == "__main__":
    main()
