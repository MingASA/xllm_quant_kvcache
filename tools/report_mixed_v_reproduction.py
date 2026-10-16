# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Report service measurements and score complete timed benchmark replay cycles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger
from tools import eval_kv_cache_quality as evaluator
from tools import report_mixed_v_cache_experiment as report
from tools import run_kv_fp8_quality_ablation as assets


def _score(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "trace_manifest.json").read_text())
    _, _, metrics, _ = evaluator._longbench_assets(assets.LONG_BENCH_REPO)
    results = {}
    for workload in ("gsm8k", "longbench"):
        if workload not in manifest["traces"]:
            continue
        path = root / f"trace_{workload}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["traces"][workload]["sha256"]:
            raise ValueError(f"Trace changed: {path}")
        trace = evaluator._jsonl(path)
        cells = set()
        for arm in report.ARMS:
            for path in (root / arm / workload).glob("c*_r*/run_config.json"):
                config = json.loads(path.read_text())
                if config["trace_sha256"] != manifest["traces"][workload]["sha256"]:
                    raise ValueError(f"Run uses another trace: {path}")
                for cycle in range(config["replay_cycles"]):
                    cells.add((config["concurrency"], config["repetition"], cycle))
        for concurrency, repetition, cycle in sorted(cells):
            predictions = {
                arm: report._full_cycle_responses(root / arm / workload / f"c{concurrency}_r{repetition}", trace, cycle)
                for arm in report.ARMS
            }
            predictions = {arm: rows for arm, rows in predictions.items() if rows}
            if not predictions:
                continue
            cell: dict[str, Any] = {
                "workload": workload,
                "concurrency": concurrency,
                "repetition": repetition,
                "cycle": cycle,
                "tasks": {},
                "paired": {},
                "length_exhaustion": {
                    a: sum(r["finish_reason"] == "length" for r in rows.values()) for a, rows in predictions.items()
                },
            }
            for task in sorted({r["task"] for r in trace}):
                rows = [r for r in trace if r["task"] == task]
                scores, pairs = evaluator._score_pair(
                    task,
                    [r["row"] for r in rows],
                    {a: [outputs[r["id"]]["prediction"] for r in rows] for a, outputs in predictions.items()},
                    metrics,
                )
                cell["tasks"][task] = scores
                cell["paired"][task] = {}
                if "auto" in predictions:
                    for arm in predictions:
                        if arm == "auto":
                            continue
                        kinds = ("scores", "flexible_scores") if task == "gsm8k" else ("scores",)
                        cell["paired"][task][arm] = {}
                        for kind in kinds:
                            delta = [r[kind][arm] - r[kind]["auto"] for r in pairs]
                            cell["paired"][task][arm][kind] = {
                                "improved": sum(v > 1e-12 for v in delta),
                                "regressed": sum(v < -1e-12 for v in delta),
                                "equal": sum(abs(v) <= 1e-12 for v in delta),
                                "mean_delta": mean(delta),
                            }
                item_path = root / "report" / f"paired_{task}_c{concurrency}_r{repetition}_cycle{cycle}.json"
                item_path.write_text(json.dumps(pairs, indent=2) + "\n")
            if set(cell["tasks"]) == set(assets.TASKS):
                cell["longbench_macro"] = {
                    a: mean(cell["tasks"][task][a] for task in assets.TASKS) for a in predictions
                }
            results[f"{workload}_c{concurrency}_r{repetition}_cycle{cycle}"] = cell
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output
    report._report(root, performance_only=True)
    result = _score(root)
    if not result:
        raise ValueError("No complete replay cycle available for quality scoring")
    (root / "report/reproduction_quality.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib.pyplot as plt

    metrics = [
        ("gsm8k", "strict_em"),
        ("gsm8k", "flexible_em"),
        ("gov_report", "rouge_l"),
        ("longbench_macro", "macro"),
    ]
    for task, metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        present = False
        for arm in report.ARMS:
            by_load: dict[int, list[float]] = {}
            for cell in result.values():
                scores = cell.get("longbench_macro", {}) if task == "longbench_macro" else cell["tasks"].get(task, {})
                key = f"{arm}_{metric}" if task == "gsm8k" else arm
                if key in scores:
                    by_load.setdefault(cell["concurrency"], []).append(scores[key])
            if by_load:
                xs = sorted(by_load)
                ax.plot(xs, [100 * mean(by_load[x]) for x in xs], marker="o", label=arm)
                present = True
        if present:
            ax.set(xlabel="Client concurrency", ylabel=f"{metric} (%)", title=task)
            ax.legend()
            fig.tight_layout()
            fig.savefig(root / "report" / f"quality_{task}_{metric}.png", dpi=180)
        plt.close(fig)
    logger.info("Scored %d complete timed replay cycles", len(result))


if __name__ == "__main__":
    main()
