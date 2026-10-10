# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Summarize durable mixed-V runs, paired quality, and measured GPU telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger
from tools import eval_kv_cache_quality as evaluator
from tools import run_kv_fp8_quality_ablation as assets
from tools.run_mixed_v_cache_experiment import ARMS


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _telemetry(path: Path) -> dict[str, Any]:
    import numpy as np

    fields = ("vram_mib", "gpu_util_pct", "memory_util_pct", "power_w")
    values: dict[str, list[float]] = defaultdict(list)
    for record in evaluator._jsonl(path):
        for name, raw in zip(fields, record.get("gpu_raw", "").split(",")):
            try:
                values[name].append(float(raw.strip()))
            except ValueError:
                pass
        for line in (record.get("metrics") or "").splitlines():
            if line.startswith("#"):
                continue
            match = re.search(
                r"(?:^|_)(num_running_requests|num_waiting_requests|kv_cache_utilization_perc|num_used_blocks|num_free_blocks)(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
                line,
            )
            if match:
                values[match[1]].append(float(match[2]))
    return {name: {"mean": float(np.mean(v)), "peak": max(v), "samples": len(v)} for name, v in values.items() if v}


def _full_cycle_responses(run: Path, trace: list[dict[str, Any]], cycle: int) -> dict[str, Any]:
    if not (run / "summary.json").exists():
        return {}
    config = json.loads((run / "run_config.json").read_text())
    cycles = config["replay_cycles"]
    if cycle >= cycles:
        return {}
    responses = evaluator._jsonl(run / "responses.jsonl")
    indexed = {row["index"]: row for row in responses}
    if len(responses) != len(trace) * cycles or set(indexed) != set(range(len(trace) * cycles)):
        raise ValueError(f"Completed joint run has incomplete or duplicate response indices: {run}")
    for index, row in indexed.items():
        expected = trace[index % len(trace)]
        if row["id"] != expected["id"] or row["prompt_sha256"] != expected["prompt_sha256"]:
            raise ValueError(f"Joint response does not match immutable trace: {run}/{index}")
    return {indexed[i]["id"]: indexed[i] for i in range(cycle * len(trace), (cycle + 1) * len(trace))}


def _quality_repetition(
    root: Path, report: Path, repetition: int, concurrency: int | None = None, cycle: int = 0
) -> dict[str, Any]:
    traces = evaluator._jsonl(root / "trace_quality.jsonl")
    _, _, metrics, _ = evaluator._longbench_assets(assets.LONG_BENCH_REPO)
    predictions: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        if concurrency is not None:
            outputs = {}
            for workload in ("gsm8k", "longbench"):
                run = root / arm / workload / f"c{concurrency}_r{repetition}"
                trace = evaluator._jsonl(root / f"trace_{workload}.jsonl")
                outputs.update(_full_cycle_responses(run, trace, cycle))
            if set(outputs) == {row["id"] for row in traces}:
                predictions[arm] = outputs
            continue
        files = list((root / arm / "quality").glob(f"c*_r{repetition}/responses.jsonl"))
        files = [path for path in files if (path.parent / "summary.json").exists()]
        if files:
            responses = evaluator._jsonl(files[0])
            if len(responses) == len(traces):
                predictions[arm] = {row["id"]: row for row in responses}
    if not predictions:
        return {}
    result: dict[str, Any] = {"tasks": {}, "arms": list(predictions), "paired": {}}
    for task in ("gsm8k", *assets.TASKS):
        task_rows = [row for row in traces if row["task"] == task]
        summary, pairs = evaluator._score_pair(
            task,
            [r["row"] for r in task_rows],
            {arm: [outputs[row["id"]]["prediction"] for row in task_rows] for arm, outputs in predictions.items()},
            metrics,
        )
        result["tasks"][task] = summary
        suffix = f"r{repetition}" if concurrency is None else f"c{concurrency}_r{repetition}_cycle{cycle}"
        _write_json(report / f"paired_{task}_{suffix}.json", pairs)
        result["paired"][task] = {}
        if "auto" in predictions:
            for arm in predictions:
                if arm == "auto":
                    continue
                changes = [row["scores"][arm] - row["scores"]["auto"] for row in pairs]
                result["paired"][task][arm] = {
                    "improved": sum(d > 0 for d in changes),
                    "regressed": sum(d < 0 for d in changes),
                    "equal": sum(d == 0 for d in changes),
                }
    result["longbench_macro"] = {
        arm: sum(result["tasks"][task][arm] for task in assets.TASKS) / len(assets.TASKS) for arm in predictions
    }
    result["length_exhaustion"] = {
        arm: sum(r["finish_reason"] == "length" for r in outputs.values() if r["task"] == "gsm8k")
        for arm, outputs in predictions.items()
    }
    return result


def _aggregate_quality(results: dict[str, Any]) -> dict[str, Any]:
    from statistics import mean, pstdev

    if not results:
        return {}
    arms = [arm for arm in ARMS if any(arm in result["arms"] for result in results.values())]
    aggregate: dict[str, Any] = {"arms": arms, "tasks": {}, "longbench_macro": {}, "length_exhaustion": {}}
    stats: dict[str, Any] = {}
    for task in ("gsm8k", *assets.TASKS):
        aggregate["tasks"][task] = {}
        stats[task] = {}
        for arm in arms:
            keys = [f"{arm}_strict_em", f"{arm}_flexible_em"] if task == "gsm8k" else [arm]
            for key in keys:
                values = [result["tasks"][task][key] for result in results.values() if key in result["tasks"][task]]
                aggregate["tasks"][task][key] = mean(values)
                stats[task][key] = {"mean": mean(values), "std": pstdev(values), "observations": len(values)}
    for arm in arms:
        selected = [result for result in results.values() if arm in result["arms"]]
        aggregate["longbench_macro"][arm] = mean(result["longbench_macro"][arm] for result in selected)
        aggregate["length_exhaustion"][arm] = [result["length_exhaustion"][arm] for result in selected]
        stats.setdefault("longbench_macro", {})[arm] = {
            "mean": aggregate["longbench_macro"][arm],
            "std": pstdev(result["longbench_macro"][arm] for result in selected),
            "observations": len(selected),
        }
    aggregate["observations"] = results
    aggregate["statistics"] = stats
    return aggregate


def _quality(root: Path, report: Path) -> dict[str, Any]:
    joint_cells = set()
    for arm in ARMS:
        for path in (root / arm).glob("*/c*_r*/run_config.json"):
            config = json.loads(path.read_text())
            if config.get("quality_protocol") == "joint_full_trace":
                for cycle in range(config["replay_cycles"]):
                    joint_cells.add((config["concurrency"], config["repetition"], cycle))
    if joint_cells:
        by_concurrency: dict[str, Any] = {}
        for concurrency in sorted({cell[0] for cell in joint_cells}):
            results = {}
            for c, repetition, cycle in sorted(joint_cells):
                if c == concurrency:
                    result = _quality_repetition(root, report, repetition, c, cycle)
                    if result:
                        results[f"r{repetition}_cycle{cycle}"] = result
            if results:
                by_concurrency[str(concurrency)] = _aggregate_quality(results)
        if not by_concurrency:
            return {}
        reference = min(map(int, by_concurrency))
        aggregate = dict(by_concurrency[str(reference)])
        aggregate.update(
            protocol="joint_full_trace",
            reference_concurrency=reference,
            by_concurrency=by_concurrency,
            observation_definition="one full benchmark cycle at a fixed concurrency and repetition",
        )
    else:
        repetitions = sorted(
            {
                int(path.parent.name.split("_r")[1])
                for arm in ARMS
                for path in (root / arm / "quality").glob("c*_r*/summary.json")
            }
        )
        results = {str(r): result for r in repetitions if (result := _quality_repetition(root, report, r))}
        aggregate = _aggregate_quality(results)
        if not aggregate:
            return {}
        aggregate["protocol"] = "separate"
    _write_json(report / "quality.json", aggregate)
    return aggregate


def _scan_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    from statistics import mean

    output: dict[str, Any] = {}
    for workload in ("gsm8k", "longbench"):
        output[workload] = {}
        for arm in ARMS:
            rows = [row for row in runs if row["arm"] == arm and row["workload"] == workload]
            levels = sorted({row["concurrency"] for row in rows})
            if not levels:
                continue
            points = []
            for c in levels:
                repeats = [row for row in rows if row["concurrency"] == c]
                ttfts = [row["ttft_s"]["p95"] for row in repeats if row.get("ttft_s")]
                points.append(
                    {
                        "concurrency": c,
                        "repetitions": len(repeats),
                        "request_throughput": mean(row["request_throughput"] for row in repeats),
                        "token_throughput": mean(row["token_throughput"] for row in repeats),
                        "ttft_p95_s": mean(ttfts) if ttfts else None,
                        "waiting_peak": max(
                            (row["gpu"].get("num_waiting_requests", {}).get("peak", 0) for row in repeats), default=0
                        ),
                    }
                )
            peak = max(point["request_throughput"] for point in points)
            candidate = next(point for point in points if point["request_throughput"] >= peak * 0.95)
            beyond = [point for point in points if point["concurrency"] > candidate["concurrency"]]
            plateau = any(
                abs(point["request_throughput"] - candidate["request_throughput"]) <= 0.05 * peak for point in beyond
            )
            queue = any(point["waiting_peak"] > 0 for point in points)
            low_ttft, high_ttft = points[0]["ttft_p95_s"], points[-1]["ttft_p95_s"]
            tail_ratio = high_ttft / max(low_ttft, 1e-12) if low_ttft is not None and high_ttft is not None else None
            output[workload][arm] = {
                "points": points,
                "saturation_candidate": candidate["concurrency"],
                "confirmed": plateau and queue and tail_ratio is not None and tail_ratio >= 2,
                "queue_observed": queue,
                "high_to_low_ttft_ratio": tail_ratio,
                "criteria": "95% of peak mean req/s, plateau within 5% at a higher load, waiting>0, tail ratio>=2",
            }
    return output


def _report(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = root / "report"
    output.mkdir(exist_ok=True)
    allocations = {
        arm: json.loads((root / arm / "allocation.json").read_text())
        for arm in ARMS
        if (root / arm / "allocation.json").exists()
    }
    quality = _quality(root, output)
    runs: list[dict[str, Any]] = []
    for arm in ARMS:
        for path in (root / arm).glob("*/c*_r*/summary.json"):
            if path.parent.parent.name == "quality":
                continue
            summary = json.loads(path.read_text())
            if summary.get("measurement_version") != 2:
                raise ValueError(f"Outdated measurement protocol; keep archived and rerun in a new directory: {path}")
            gpu = _telemetry(path.parent / "telemetry.jsonl")
            runs.append(
                {
                    "arm": arm,
                    "workload": path.parent.parent.name,
                    "repetition": int(path.parent.name.split("_r")[1]),
                    **summary,
                    "gpu": gpu,
                }
            )
    _write_json(output / "performance.json", runs)
    scan = _scan_summary(runs)
    _write_json(output / "saturation.json", scan)
    if runs:
        fields = (
            "arm",
            "workload",
            "concurrency",
            "repetition",
            "requests",
            "elapsed_s",
            "request_throughput",
            "token_throughput",
            "generated_tokens",
            "measurement_version",
            "peak_client_inflight",
            "client_cpu_fraction_one_core",
            "event_loop_lag_p95_s",
            "event_loop_lag_max_s",
            "ttft_p95_s",
            "tpot_p95_s",
            "e2e_p95_s",
            "vram_peak_mib",
            "gpu_util_pct",
            "memory_util_pct",
        )
        with (output / "performance.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in runs:
                record = {name: row[name] for name in fields if name in row}
                record["event_loop_lag_p95_s"] = (row.get("event_loop_lag_s") or {}).get("p95")
                for metric in ("ttft", "tpot", "e2e"):
                    record[f"{metric}_p95_s"] = (row.get(f"{metric}_s") or {}).get("p95")
                for field, source, agg in (
                    ("vram_peak_mib", "vram_mib", "peak"),
                    ("gpu_util_pct", "gpu_util_pct", "mean"),
                    ("memory_util_pct", "memory_util_pct", "mean"),
                ):
                    record[field] = row["gpu"].get(source, {}).get(agg)
                writer.writerow(record)
    if allocations:
        arms = list(allocations)
        per_layer = [allocations[arm]["layers"][0]["bytes_per_token"] for arm in arms]
        capacity = [allocations[arm]["token_capacity"] for arm in arms]
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].bar(arms, per_layer)
        axs[0].set_ylabel("KV bytes/token/layer (including scales)")
        axs[1].bar(arms, capacity)
        axs[1].set_ylabel("Allocated KV token capacity")
        measured = [
            max((row["gpu"].get("vram_mib", {}).get("peak", 0) for row in runs if row["arm"] == arm), default=0)
            for arm in arms
        ]
        axs[2].bar(arms, measured)
        axs[2].set_ylabel("Measured GPU VRAM peak (MiB)")
        for ax in axs:
            ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output / "memory.png", dpi=180)
        plt.close(fig)

    def _plot_curves(metrics: tuple[str, ...], filename: str, tail: bool) -> None:
        fig, axs = plt.subplots(2, len(metrics), figsize=(6 * len(metrics), 8), squeeze=False)
        for i, workload in enumerate(("gsm8k", "longbench")):
            for j, metric in enumerate(metrics):
                for arm in ARMS:
                    grouped: dict[int, list[float]] = defaultdict(list)
                    for row in runs:
                        if row["workload"] == workload and row["arm"] == arm:
                            value = (row.get(metric) or {}).get("p95") if tail else row.get(metric)
                            if value is not None:
                                grouped[row["concurrency"]].append(value)
                    levels = sorted(grouped)
                    if levels:
                        axs[i, j].errorbar(
                            levels,
                            [np.mean(grouped[c]) for c in levels],
                            yerr=[np.std(grouped[c]) for c in levels],
                            marker="o",
                            label=arm,
                        )
                axs[i, j].set_xscale("log", base=2)
                axs[i, j].set_xlabel("Closed-loop concurrency")
                axs[i, j].set_ylabel(("p95 " if tail else "") + metric)
                axs[i, j].set_title(workload)
                axs[i, j].legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    if runs:
        _plot_curves(("request_throughput", "token_throughput"), "throughput.png", False)
        _plot_curves(("ttft_s", "tpot_s"), "tail_latency.png", True)
    if quality:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        arms = quality["arms"]
        for i, (title, scores) in enumerate(
            (
                ("GSM8K Flexible EM", {arm: quality["tasks"]["gsm8k"][f"{arm}_flexible_em"] for arm in arms}),
                ("LongBench 8-task macro", quality["longbench_macro"]),
            )
        ):
            score_stats = quality["statistics"]["gsm8k" if i == 0 else "longbench_macro"]
            axs[i].bar(
                arms,
                [100 * scores[arm] for arm in arms],
                yerr=[100 * score_stats[f"{arm}_flexible_em" if i == 0 else arm]["std"] for arm in arms],
                capsize=4,
            )
            axs[i].set_title(title)
            axs[i].set_ylabel("Score (%)")
            axs[i].tick_params(axis="x", rotation=25)
        if quality.get("protocol") == "joint_full_trace":
            fig.suptitle(f"Quality from timed full traces, concurrency={quality['reference_concurrency']}")
        fig.tight_layout()
        fig.savefig(output / "quality.png", dpi=180)
        plt.close(fig)
    lines = [
        "# Mixed-precision V cache experiment",
        "",
        "Fair baseline: all arms use Python eager, graph off, "
        "the same Triton paged-attention algorithm, scheduler and memory budget (server_command.json). Prefix caching is off.",
        "",
        "GPU: RTX 5060 Ti 16GB GDDR7 VRAM. NVIDIA memory utilization is sampled memory-controller activity; "
        "it is not a measured percentage of peak VRAM bandwidth. No memory-traffic or dequant-cost claim is made "
        "without a profiler trace.",
        "",
        "Token counts are API generated tokens; normal EOS and original generation "
        "caps apply. TTFT measures the first nonempty streamed text. TPOT uses (E2E - TTFT) divided by "
        "generated token intervals. Text buffering can affect both. Mean curves and error bars summarize repetitions.",
        "Joint runs score these same timed responses at every concurrency, separately for each full replay cycle; "
        "replayed IDs are never collapsed. Quality standard deviations describe cycle/repetition variation.",
        "",
        "| Arm | KV bytes/token/layer | Relative bytes | KV token capacity | Resident KV bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline = allocations.get("auto", {}).get("layers", [{}])[0].get("bytes_per_token")
    for arm, alloc in allocations.items():
        count = alloc["layers"][0]["bytes_per_token"]
        relative = f"{count / baseline:.6f}" if baseline else "pending"
        lines.append(f"| {arm} | {count} | {relative} | {alloc['token_capacity']} | {alloc['resident_bytes']} |")
    lines += [
        "",
        "With a fixed VRAM allocation budget, compression increases token capacity; total resident cache bytes "
        "can remain similar. These values come from the actual server tensors.",
        "",
        "| Arm | Workload | Max tested concurrency | Peak mean req/s | Peak mean tok/s | p95 TTFT at highest load (s) | "
        "p95 TPOT at highest load (s) | GSM8K Flexible EM | LongBench macro |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for workload in ("gsm8k", "longbench"):
            rows = [row for row in runs if row["arm"] == arm and row["workload"] == workload]
            if not rows:
                continue
            c = max(r["concurrency"] for r in rows)
            tail = [r for r in rows if r["concurrency"] == c]

            def _avg_tail(name: str, tail_rows: list[dict[str, Any]] = tail) -> str:
                vals = [r[name]["p95"] for r in tail_rows if r.get(name)]
                return f"{np.mean(vals):.4f}" if vals else "unavailable"

            load_quality = quality["by_concurrency"].get(str(c), {}) if "by_concurrency" in quality else quality
            gsm = load_quality.get("tasks", {}).get("gsm8k", {}).get(f"{arm}_flexible_em", "pending")
            long = load_quality.get("longbench_macro", {}).get(arm, "pending")
            lines.append(
                f"| {arm} | {workload} | {c} | {max(p['request_throughput'] for p in scan[workload][arm]['points']):.3f} | "
                f"{max(p['token_throughput'] for p in scan[workload][arm]['points']):.3f} | {_avg_tail('ttft_s')} | "
                f"{_avg_tail('tpot_s')} | {gsm} | {long} |"
            )
    if quality:
        lines += [
            "",
            "## Full-trace quality (mean across completed cycles/repetitions)",
            "",
            "| Arm | GSM8K strict EM | GSM8K flexible EM | GSM8K length exhaustion | LongBench macro |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm in quality["arms"]:
            gsm_scores = quality["tasks"]["gsm8k"]
            lines.append(
                f"| {arm} | {gsm_scores[f'{arm}_strict_em']:.6f} | "
                f"{gsm_scores[f'{arm}_flexible_em']:.6f} | {quality['length_exhaustion'][arm]} | "
                f"{quality['longbench_macro'][arm]:.6f} |"
            )
        lines += ["", "| Task | " + " | ".join(quality["arms"]) + " |", "|---|" + "---:|" * len(quality["arms"])]
        for task in assets.TASKS:
            lines.append(
                "| " + task + " | " + " | ".join(f"{quality['tasks'][task][arm]:.6f}" for arm in quality["arms"]) + " |"
            )
        if quality.get("protocol") == "joint_full_trace":
            lines += [
                "",
                f"The quality figure above uses concurrency {quality['reference_concurrency']}. "
                "The following table retains each load; one observation is one full replay cycle.",
                "",
                "| Concurrency | Arm | Full cycles scored | GSM8K strict EM | GSM8K flexible EM | LongBench macro |",
                "|---:|---|---:|---:|---:|---:|",
            ]
            for c, scores in quality["by_concurrency"].items():
                for arm in scores["arms"]:
                    gsm = scores["tasks"]["gsm8k"]
                    count = scores["statistics"]["longbench_macro"][arm]["observations"]
                    lines.append(
                        f"| {c} | {arm} | {count} | {gsm[f'{arm}_strict_em']:.6f} | "
                        f"{gsm[f'{arm}_flexible_em']:.6f} | {scores['longbench_macro'][arm]:.6f} |"
                    )
    lines += [
        "",
        "Max tested concurrency is not a certified stability limit. Saturation requires a plateau in "
        "throughput, scheduler waiting evidence and rising tail latency. If the permitted scan does not bracket it, "
        "report saturation as unconfirmed; this protocol is capped at concurrency 128.",
        "",
        "Joint quality is scored from complete timed full-benchmark cycles at matching load/repetition/cycle. "
        "Per-task scores, paired improvement/regression counts and observation counts are in quality.json "
        "and paired_*.json. Length exhaustion lists retain each cycle/repetition; missing cells remain pending.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    logger.info("Report written to %s (%d performance runs)", output, len(runs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    _report(parser.parse_args().output)


if __name__ == "__main__":
    main()
