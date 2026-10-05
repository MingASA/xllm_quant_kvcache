#!/usr/bin/env python3
"""Compare native vLLM BF16 and FP8 KV cache on fixed local prompts.

Each arm runs in a separate child process so CUDA allocations are released
between configurations. This deliberately uses vLLM's public API only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_kv_cache_quality import _gsm_prompt, _gsm_score  # noqa: E402

MODEL = Path("/mnt/e/AI/models/Qwen2.5-1.5B-Instruct")
DATA = Path("/mnt/e/AI/xllm-eval-data/int4-model-eval-20261002/gsm8k/data.jsonl")
SHOTS = Path(
    "/mnt/e/AI/xllm-eval-data/gsm8k-3101c7d5072418e28b9008a6636bde82a006892c/prepared/five_train_examples.json"
)
OUT = Path("/mnt/e/AI/xllm-eval-data/vllm-fp8-reference-20261005")
SMOKE = ROOT / "results/flashinfer-fp8-smoke-20261005/README.md"
ARITHMETIC = [
    "What is 27 times 43? Answer with the number only.",
    *[f"What is {a} + {b}? Answer with the number only." for a, b in zip(range(11, 19), range(33, 55, 3))],
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prompts(task: str) -> tuple[list[str], list[dict[str, Any]]]:
    if task == "arithmetic":
        return ARITHMETIC, [{"question": q} for q in ARITHMETIC]
    rows = _jsonl(DATA)[:64]
    shots = json.loads(SHOTS.read_text(encoding="utf-8"))
    return [_gsm_prompt(row["question"], shots) for row in rows], rows


def _worker(task: str, arm: str) -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    prompts_text, rows = _prompts(task)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    token_ids = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True)
        for p in prompts_text
    ]
    token_ids = [x.get("input_ids") if hasattr(x, "get") else x for x in token_ids]
    token_ids = [ids.tolist() if hasattr(ids, "tolist") else ids for ids in token_ids]
    token_ids = [list(ids[0] if len(ids) > 0 and isinstance(ids[0], list) else ids) for ids in token_ids]
    config = {
        "arm": arm,
        "task": task,
        "model": str(MODEL),
        "model_config_sha256": _sha256(MODEL / "config.json"),
        "tokenizer_config_sha256": _sha256(MODEL / "tokenizer_config.json"),
        "prompt_token_ids_sha256": hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest(),
        "prompt_count": len(prompts_text),
        "prompt_token_counts": [len(ids) for ids in token_ids],
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "device": torch.cuda.get_device_name(0),
        "seed": 17 if task == "gsm8k" else 20261005,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512 if task == "gsm8k" else 24},
        "engine_flags": {
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto" if arm == "bf16" else "fp8",
            "attention_backend": "FLASHINFER",
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "trust_remote_code": True,
            "seed": 17 if task == "gsm8k" else 20261005,
            "max_model_len": 4096,
            "generation_config": "vllm",
            "repetition_penalty": 1.0,
            "stop": ["Question:", "</s>", "<|im_end|>"] if task == "gsm8k" else None,
        },
    }
    llm = LLM(
        model=str(MODEL),
        dtype="bfloat16",
        kv_cache_dtype=config["engine_flags"]["kv_cache_dtype"],
        attention_backend="FLASHINFER",
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
        seed=config["seed"],
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        generation_config="vllm",
    )
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=config["sampling"]["max_tokens"],
        seed=config["seed"],
        repetition_penalty=1.0,
        stop=config["engine_flags"]["stop"],
    )
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in token_ids], params, use_tqdm=True)
    records = []
    for i, output in enumerate(outputs):
        candidate = output.outputs[0]
        rec = {
            "index": i,
            "prompt": prompts_text[i],
            "prompt_token_ids": token_ids[i],
            "prompt_tokens": len(token_ids[i]),
            "output": candidate.text,
            "output_token_ids": candidate.token_ids,
            "output_tokens": len(candidate.token_ids),
            "finish_reason": candidate.finish_reason,
            "stop_reason": candidate.stop_reason,
        }
        if task == "gsm8k":
            rec["answer"] = rows[i]["answer"]
            rec["strict_em"], rec["flexible_em"] = _gsm_score(candidate.text, rows[i]["answer"])
        records.append(rec)
    (OUT / f"{task}-{arm}.json").write_text(
        json.dumps({"config": config, "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--task", choices=("arithmetic", "gsm8k"))
    parser.add_argument("--arm", choices=("bf16", "fp8"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        config_path = OUT / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["reproduction_source_sha256"] = _sha256(Path(__file__))
        config["runtime_versions"] = {
            package: importlib.metadata.version(package)
            for package in ("vllm", "torch", "flashinfer-python", "transformers")
        }
        config["engine_flags"] = {
            "dtype": "bfloat16",
            "kv_cache_dtype": {"bf16": "auto", "fp8": "fp8"},
            "attention_backend": "FLASHINFER",
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 4096,
            "generation_config": "vllm",
            "repetition_penalty": 1.0,
            "gsm8k_stop": ["Question:", "</s>", "<|im_end|>"],
            "arithmetic_stop": None,
        }
        config.pop("source_sha256", None)
        config["source_sha256_status"] = (
            "not_valid_as_runtime_provenance; only the reproducible current source is hashed"
        )
        (OUT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        for task in ("arithmetic", "gsm8k"):
            for arm in ("bf16", "fp8"):
                result_path = OUT / f"{task}-{arm}.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["config"].pop("source_sha256", None)
                result["config"]["reproduction_source_sha256"] = config["reproduction_source_sha256"]
                result["config"]["source_sha256_status"] = config["source_sha256_status"]
                result["config"]["dataset_sha256"] = _sha256(DATA)
                result["config"]["shots_sha256"] = _sha256(SHOTS) if task == "gsm8k" else None
                result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    if args.worker:
        assert args.task and args.arm
        _worker(args.task, args.arm)
        return
    (OUT / "run_config.json").write_text(
        json.dumps(
            {
                "model": str(MODEL),
                "dataset": str(DATA),
                "dataset_sha256": _sha256(DATA),
                "shots": str(SHOTS),
                "shots_sha256": _sha256(SHOTS),
                "source_sha256": _sha256(Path(__file__)),
                "prompt_source": str(SMOKE),
                "prompt_source_sha256": _sha256(SMOKE),
                "flags": [
                    "dtype=bfloat16",
                    "kv_cache_dtype=auto/fp8",
                    "attention_backend=FLASHINFER",
                    "enforce_eager=True",
                    "enable_prefix_caching=False",
                    "gpu_memory_utilization=0.80",
                    "max_model_len=4096",
                ],
                "python": sys.executable,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for task in ("arithmetic", "gsm8k"):
        for arm in ("bf16", "fp8"):
            log = OUT / f"{task}-{arm}.log"
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", "--task", task, "--arm", arm],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env=os.environ.copy(),
                )
            if completed.returncode:
                raise SystemExit(f"{task}/{arm} failed with status {completed.returncode}; see {log}")


if __name__ == "__main__":
    main()
