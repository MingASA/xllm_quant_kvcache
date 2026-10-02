# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Run one local xLLM API arm or score saved paired KV cache results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.logger import logger  # noqa: E402

LONG_BENCH_TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "2wikimqa",
    "hotpotqa",
    "musique",
    "lcc",
    "repobench-p",
)
MAX_CONTEXT_TOKENS = 32768
GSM_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
GSM_FLEXIBLE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
GSM_IGNORE = (",", r"\$", r"(?s).*#### ", r"\.$")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


_VOLATILE_MANIFEST_FIELDS = frozenset(
    {
        "server_attestation",
        "verified_smoke_log",
        "verified_smoke_response",
        "verified_api_port",
        "smoke_response_status",
        "smoke_response_text",
        "smoke_api_prompt_tokens",
        "verified_at_utc",
        "attestation_timestamp",
        "readiness_timestamp",
    }
)


def _assert_resume_config_compatible(existing: dict[str, Any], current: dict[str, Any]) -> None:
    """Require exact run equivalence except explicitly volatile server attestation evidence."""
    existing_normalized = dict(existing)
    current_normalized = dict(current)
    for config in (existing_normalized, current_normalized):
        manifest = dict(config.get("server_manifest", {}))
        for key in _VOLATILE_MANIFEST_FIELDS:
            manifest.pop(key, None)
        config["server_manifest"] = manifest
        # The digest covers the allowlisted attestation evidence as well as semantic manifest data.
        config.pop("server_manifest_sha256", None)
    if existing_normalized != current_normalized:
        keys = sorted(set(existing_normalized) | set(current_normalized))
        mismatch = next((key for key in keys if existing_normalized.get(key) != current_normalized.get(key)), "unknown")
        raise ValueError(f"resume run config differs for {mismatch}; existing output is preserved")


def _result_record(
    index: int,
    row: dict[str, Any],
    prediction: str,
    usage: dict[str, Any],
    response: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    record = dict(row)
    record.update(
        {
            "_eval_index": index,
            "prediction": prediction,
            "usage": usage,
            "api_response": response,
            "_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "_record_sha256": hashlib.sha256(
                json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
    )
    return record


def _append_result_record(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _validate_resume_prefix(
    path: Path,
    rows: list[dict[str, Any]],
    prompts: list[str],
    local_prompt_tokens: list[int],
    use_chat: bool,
) -> set[int]:
    """Validate the durable index-keyed journal and return its completed row indices."""
    completed: set[int] = set()
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            if not line.endswith(b"\n"):
                raise ValueError(f"resume JSONL has an incomplete final line at row {index}; file preserved")
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"resume JSONL is corrupt at row {index}; file preserved") from error
            row_index = record.get("_eval_index")
            if not isinstance(row_index, int) or row_index < 0 or row_index >= len(rows):
                raise ValueError(f"resume JSONL row index is out of range at row {index}; file preserved")
            if row_index in completed:
                raise ValueError(f"resume JSONL contains duplicate evaluation index {row_index}; file preserved")
            expected_row = rows[row_index]
            expected_hash = hashlib.sha256(
                json.dumps(expected_row, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            expected_prompt_hash = hashlib.sha256(prompts[row_index].encode("utf-8")).hexdigest()
            if record.get("_record_sha256") != expected_hash or record.get("_prompt_sha256") != expected_prompt_hash:
                raise ValueError(f"resume JSONL data/prompt hash mismatch at row {index}; file preserved")
            if not isinstance(record.get("prediction"), str) or not isinstance(record.get("api_response"), dict):
                raise ValueError(f"resume JSONL response is incomplete at row {index}; file preserved")
            usage = record.get("usage")
            if not isinstance(usage, dict) or usage.get("local_prompt_tokens") != local_prompt_tokens[row_index]:
                raise ValueError(f"resume JSONL usage is incomplete at row {index}; file preserved")
            api_usage = record["api_response"].get("usage", {})
            api_prompt_tokens = usage.get("prompt_tokens")
            if (
                not isinstance(api_usage, dict)
                or api_prompt_tokens != local_prompt_tokens[row_index]
                or api_usage.get("prompt_tokens") != api_prompt_tokens
            ):
                raise ValueError(f"resume JSONL prompt token usage mismatch at row {index}; file preserved")
            choices = record["api_response"].get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"resume JSONL API response has no choices at row {index}; file preserved")
            choice = choices[0]
            expected_prediction = (
                (choice.get("message") or {}).get("content") or "" if use_chat else choice.get("text") or ""
            )
            if record["prediction"] != expected_prediction:
                raise ValueError(f"resume JSONL prediction differs from raw response at row {index}; file preserved")
            completed.add(row_index)
    return completed


def _initialize_run_output(
    output_dir: Path,
    arm: str,
    config: dict[str, Any],
    resume: bool,
    rows: list[dict[str, Any]],
    prompts: list[str],
    local_prompt_tokens: list[int],
    use_chat: bool,
) -> tuple[Path, set[int]]:
    """Create a new exclusive output or verify a saved prefix for explicit resume."""
    config_path = output_dir / f"{arm}.config.json"
    result_path = output_dir / f"{arm}.jsonl"
    config_exists, result_exists = config_path.exists(), result_path.exists()
    if not resume:
        if config_exists or result_exists:
            raise FileExistsError(
                f"refusing to overwrite existing {arm} output; choose a new directory or pass --resume"
            )
        _write_new_json(config_path, config)
        with result_path.open("x", encoding="utf-8"):
            pass
        return result_path, set()
    if not config_exists or not result_exists:
        raise FileNotFoundError("--resume requires both the existing arm config and result JSONL; files are preserved")
    try:
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("existing run config is unreadable; output is preserved") from error
    _assert_resume_config_compatible(existing_config, config)
    completed = _validate_resume_prefix(result_path, rows, prompts, local_prompt_tokens, use_chat)
    return result_path, completed


def _run_uncompleted(
    completed: set[int] | int,
    row_count: int,
    process_row: Callable[[int], None],
    concurrency: int = 1,
) -> None:
    """Run missing indices with a bounded worker pool; exceptions propagate without retries."""
    if concurrency < 1:
        raise ValueError("request concurrency must be positive")
    completed_indices = set(range(completed)) if isinstance(completed, int) else completed
    pending = [index for index in range(row_count) if index not in completed_indices]
    if concurrency == 1:
        for index in pending:
            process_row(index)
        return
    pending_indices = iter(pending)
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="kv-quality") as executor:
        active = {
            executor.submit(process_row, index)
            for index in [next(pending_indices, None) for _ in range(concurrency)]
            if index is not None
        }
        while active:
            completed_futures, active = wait(active, return_when=FIRST_COMPLETED)
            for future in completed_futures:
                future.result()
            for _ in completed_futures:
                next_index = next(pending_indices, None)
                if next_index is not None:
                    active.add(executor.submit(process_row, next_index))


def _longbench_assets(repo: Path) -> tuple[dict[str, str], dict[str, int], Any, str]:
    config = repo / "LongBench" / "config"
    prompts = json.loads((config / "dataset2prompt.json").read_text(encoding="utf-8"))
    max_lengths = json.loads((config / "dataset2maxlen.json").read_text(encoding="utf-8"))
    metrics_path = repo / "LongBench" / "metrics.py"
    spec = importlib.util.spec_from_file_location("longbench_official_metrics", metrics_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official LongBench metrics at {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown (source checkout has no readable git revision)"
    return prompts, max_lengths, module, revision


def _api_call(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    timeout: int,
    use_chat: bool,
    stop_token_ids: list[int] | None = None,
    stop_strings: list[str] | None = None,
) -> dict[str, Any]:
    body_data: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if stop_token_ids is not None:
        body_data["stop_token_ids"] = stop_token_ids
    if stop_strings:
        body_data["stop"] = stop_strings
    if use_chat:
        body_data["messages"] = [{"role": "user", "content": prompt}]
        endpoint = "/chat/completions"
    else:
        body_data["prompt"] = prompt
        endpoint = "/completions"
    body = json.dumps(body_data).encode("utf-8")
    url = base_url.rstrip("/") + endpoint
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"API request failed at {url}: {error}") from error
    return result


def _gsm_prompt(question: str, shots: list[dict[str, str]]) -> str:
    examples = "\n\n".join(f"Question: {item['question']}\n Answer: {item['answer']}" for item in shots)
    return f"{examples}\n\nQuestion: {question}\n Answer:"


def _gsm_score(prediction: str, answer: str) -> tuple[float, float]:
    target = answer.split("#### ")[-1].rstrip()

    def normalize(value: str) -> str:
        for pattern in GSM_IGNORE:
            value = re.sub(pattern, "", value)
        return value.lower()

    strict_match = GSM_STRICT.findall(prediction)
    flexible_match = GSM_FLEXIBLE.findall(prediction)
    strict_value = strict_match[0] if strict_match else ""
    flexible_value = next((left or right for left, right in reversed(flexible_match)), "")
    return float(normalize(strict_value) == normalize(target)), float(normalize(flexible_value) == normalize(target))


def _score_pair(
    task: str, records: list[dict[str, Any]], predictions: dict[str, list[str]], longbench_metrics: Any
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    item_rows: list[dict[str, Any]] = []
    scores: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(records):
        per_arm: dict[str, float] = {}
        flexible_per_arm: dict[str, float] | None = None
        if task == "gsm8k":
            flexible_per_arm = {}
            for arm, outputs in predictions.items():
                strict, flexible = _gsm_score(outputs[index], row["answer"])
                scores[f"{arm}_strict_em"].append(strict)
                scores[f"{arm}_flexible_em"].append(flexible)
                per_arm[arm] = strict
                flexible_per_arm[arm] = flexible
        else:
            answers = row["answers"]
            if not isinstance(answers, list):
                answers = [answers]
            scorer = (
                longbench_metrics.code_sim_score if task in {"lcc", "repobench-p"} else longbench_metrics.qa_f1_score
            )
            for arm, outputs in predictions.items():
                value = max(scorer(outputs[index], answer) for answer in answers)
                scores[arm].append(value)
                per_arm[arm] = value
        arm_names = list(predictions)
        item = {
            "index": index,
            "id": row.get("_id", row.get("id", str(index))),
            "scores": per_arm,
            "flip": _score_flip(per_arm, arm_names),
        }
        if flexible_per_arm is not None:
            item["flexible_scores"] = flexible_per_arm
            item["flexible_flip"] = _score_flip(flexible_per_arm, arm_names)
        item_rows.append(item)
    return {key: sum(values) / len(values) for key, values in scores.items()}, item_rows


def _score_flip(scores: dict[str, float], arm_names: list[str]) -> str:
    if scores[arm_names[1]] > scores[arm_names[0]]:
        return "up"
    if scores[arm_names[1]] < scores[arm_names[0]]:
        return "down"
    return "tie"


def _token_count(tokenizer: Any, prompt: str, use_chat: bool) -> int:
    if use_chat:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
        if hasattr(tokens, "get") and tokens.get("input_ids") is not None:
            tokens = tokens["input_ids"]
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        return len(tokens)
    return len(tokenizer(prompt, add_special_tokens=True)["input_ids"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_versions(task: str) -> dict[str, str]:
    packages = ["transformers"]
    if task != "gsm8k":
        packages.extend(("fuzzywuzzy", "jieba", "numpy", "rouge"))
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    if task == "gsm8k":
        versions["gsm8k_reference"] = "lm-evaluation-harness gsm8k.yaml metadata.version=3.0"
    return versions


def _load_server_manifest(path: Path, arm: str, model: str, tokenizer_path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("server_started") is not True:
        raise ValueError(
            "server manifest server_started must be true after successful startup and readiness verification"
        )
    expected_impl = "native" if arm == "native-bf16" else "python"
    expected_kv = "bf16" if arm == "native-bf16" else arm
    required = (
        "model_impl",
        "graph_mode",
        "kv_cache_mode",
        "checkpoint_path",
        "checkpoint_sha256",
        "model_config_sha256",
        "chat_template_sha256",
        "model",
    )
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError(f"server manifest is missing required fields: {', '.join(missing)}")
    hash_keys = ("checkpoint_sha256", "model_config_sha256", "chat_template_sha256")
    invalid_hashes = [key for key in hash_keys if not re.fullmatch(r"[0-9a-f]{64}", manifest[key])]
    if invalid_hashes:
        raise ValueError(f"server manifest fields must be lowercase SHA256 hex: {', '.join(invalid_hashes)}")
    if manifest["model_impl"] != expected_impl:
        raise ValueError(f"{arm} manifest model_impl must be {expected_impl!r}")
    if manifest["graph_mode"] not in (False, "off", "disabled"):
        raise ValueError("quality comparison requires graph_mode off")
    if manifest["kv_cache_mode"] != expected_kv:
        raise ValueError(f"{arm} manifest kv_cache_mode must be {expected_kv!r}")
    if Path(manifest["checkpoint_path"]).resolve() != tokenizer_path.resolve():
        raise ValueError("server manifest checkpoint_path must resolve to the local tokenizer/checkpoint path")
    if manifest["model"] != model:
        raise ValueError("server manifest model does not match --model")
    checkpoint_config = tokenizer_path / "config.json"
    tokenizer_config = tokenizer_path / "tokenizer_config.json"
    if _sha256(checkpoint_config) != manifest["model_config_sha256"]:
        raise ValueError("server manifest model_config_sha256 does not match local config.json")
    tokenizer_data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    template = tokenizer_data.get("chat_template")
    template_bytes = json.dumps(template, ensure_ascii=False, sort_keys=True).encode("utf-8")
    actual_template_hash = hashlib.sha256(template_bytes).hexdigest()
    if actual_template_hash != manifest["chat_template_sha256"]:
        raise ValueError("server manifest chat_template_sha256 does not match local tokenizer_config.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*LONG_BENCH_TASKS, "gsm8k"), required=True)
    parser.add_argument("--data", type=Path, help="Prepared task JSONL; the tool never downloads data.")
    parser.add_argument(
        "--longbench-repo", type=Path, help="Official THUDM/LongBench checkout; required for LongBench."
    )
    parser.add_argument(
        "--gsm8k-shots", type=Path, help="JSON array of five official GSM8K train examples; required for GSM8K."
    )
    parser.add_argument("--tokenizer", type=Path, help="Local checkpoint path used for context-length preflight.")
    parser.add_argument("--server-manifest", type=Path, help="Auditable JSON for the selected server configuration.")
    parser.add_argument("--model", help="Identical served model id for both API endpoints.")
    parser.add_argument("--arm", choices=("auto", "int8", "native-bf16"), help="Run one endpoint then stop its server.")
    parser.add_argument("--url", help="The selected arm's local xLLM API base URL, ending in /v1.")
    parser.add_argument("--auto-results", type=Path, help="Saved auto JSONL for offline paired scoring.")
    parser.add_argument("--int8-results", type=Path, help="Saved INT8 JSONL for offline paired scoring.")
    parser.add_argument(
        "--output", type=Path, help="Output directory; defaults to $HF_HOME/xllm-kv-quality/<UTC timestamp>."
    )
    parser.add_argument("--limit", type=int, help="Fixed prefix subset size, for the initial smoke run.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Maximum in-flight API requests; server must support this value."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume only an exactly matching, validated partial arm in --output."
    )
    parser.add_argument(
        "--stop-token-ids",
        type=int,
        nargs="+",
        help="Optional explicit xLLM stop token IDs; omitted means use the server/model default.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--strict-token-count",
        action="store_true",
        help="Fail on API/local or paired-arm prompt token-count differences.",
    )
    args = parser.parse_args()

    scoring_only = args.auto_results is not None or args.int8_results is not None
    if scoring_only and args.resume:
        parser.error("--resume is only valid in API run mode")
    if args.resume and args.output is None:
        parser.error("--resume requires an explicit --output directory")
    repo_assets = None
    if args.task != "gsm8k":
        if args.longbench_repo is None:
            parser.error("--longbench-repo is required for LongBench tasks")
        repo_assets = _longbench_assets(args.longbench_repo)
    elif not scoring_only and args.gsm8k_shots is None:
        parser.error("--gsm8k-shots is required for GSM8K")

    if scoring_only:
        if args.auto_results is None or args.int8_results is None or args.output is None:
            parser.error("offline scoring requires --auto-results, --int8-results, and --output")
        _score_saved(args, repo_assets[2] if repo_assets is not None else None)
        return
    if (
        args.arm is None
        or args.url is None
        or args.data is None
        or args.tokenizer is None
        or args.model is None
        or args.server_manifest is None
    ):
        parser.error("run mode requires --arm, --url, --data, --tokenizer, --model, and --server-manifest")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    server_manifest = _load_server_manifest(args.server_manifest, args.arm, args.model, args.tokenizer)
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    server_max_seqs = server_manifest.get("runtime_configuration", {}).get("max_seqs_per_batch", 1)
    if not isinstance(server_max_seqs, int) or server_max_seqs < args.concurrency:
        parser.error("server manifest runtime_configuration.max_seqs_per_batch must cover --concurrency")

    rows = _jsonl(args.data)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        parser.error("Input dataset is empty")

    shots: list[dict[str, str]] = []
    if args.task == "gsm8k":
        shots = json.loads(args.gsm8k_shots.read_text(encoding="utf-8"))
        if len(shots) != 5:
            parser.error("Official lm-evaluation-harness GSM8K config uses exactly five few-shot examples")

    prompts: dict[str, str] = {}
    max_tokens: int
    if args.task == "gsm8k":
        max_tokens = 512
        prompts = {str(i): _gsm_prompt(row["question"], shots) for i, row in enumerate(rows)}
    else:
        prompt_templates, task_max_lengths, _, _ = repo_assets
        max_tokens = int(task_max_lengths[args.task])
        template = prompt_templates[args.task]
        prompts = {str(i): template.format(context=row["context"], input=row["input"]) for i, row in enumerate(rows)}

    use_chat = args.task not in {"lcc", "repobench-p"}
    local_prompt_tokens_by_index = {
        index: _token_count(tokenizer, prompt, use_chat) for index, prompt in prompts.items()
    }
    for index, prompt in prompts.items():
        if local_prompt_tokens_by_index[index] + max_tokens > MAX_CONTEXT_TOKENS:
            raise ValueError(f"row {index} exceeds 32,768 prompt+generation tokens; no truncation is applied")

    output_dir = args.output
    if output_dir is None:
        hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        output_dir = hf_home / "xllm-kv-quality" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    arm = args.arm
    predictions: list[str] = []
    usage: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    config = {
        "task": args.task,
        "data": str(args.data.resolve()),
        "input_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "model": args.model,
        "arm": arm,
        "endpoint": args.url,
        "model_impl": "python" if arm in {"auto", "int8"} else "native",
        "tokenizer": str(args.tokenizer.resolve()),
        "sampling": {"temperature": 0, "top_p": 1, "seed": args.seed},
        "generation": {
            "temperature": 0,
            "top_p": 1,
            "seed": args.seed,
            "max_tokens": max_tokens,
            "stop_token_ids": args.stop_token_ids,
            "stop_strings": ["Question:", "</s>", "<|im_end|>"] if args.task == "gsm8k" else [],
        },
        "max_tokens": max_tokens,
        "max_prompt_plus_generation_tokens": MAX_CONTEXT_TOKENS,
        "examples": len(rows),
        "api_mode": "chat-completions" if use_chat else "completions",
        "execution_protocol": {
            "request_concurrency": args.concurrency,
            "max_seqs_per_batch": server_max_seqs,
        },
        "runtime_versions": _runtime_versions(args.task),
        "server_manifest": server_manifest,
        "server_manifest_sha256": _sha256(args.server_manifest),
    }
    if repo_assets is not None:
        config["longbench_revision"] = repo_assets[3]
    if args.task == "gsm8k":
        config["gsm8k_shots"] = str(args.gsm8k_shots.resolve())
        config["gsm8k_shots_sha256"] = hashlib.sha256(args.gsm8k_shots.read_bytes()).hexdigest()
    ordered_prompts = list(prompts.values())
    ordered_prompt_tokens = [local_prompt_tokens_by_index[str(index)] for index in range(len(rows))]
    result_path, completed_indices = _initialize_run_output(
        output_dir,
        arm,
        config,
        args.resume,
        rows,
        ordered_prompts,
        ordered_prompt_tokens,
        use_chat,
    )
    if len(completed_indices) == len(rows):
        logger.info("%s already has all %d verified rows; no API requests issued", arm, len(completed_indices))
        return
    logger.info("Running %s on %d %s examples with concurrency=%d", arm, len(rows), args.task, args.concurrency)
    write_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed_count = len(completed_indices)
    with result_path.open("a", encoding="utf-8") as result_stream:

        def process_row(index: int) -> None:
            prompt = ordered_prompts[index]
            request_started = time.perf_counter()
            request_started_utc = datetime.now(timezone.utc).isoformat()
            response = _api_call(
                args.url,
                args.model,
                prompt,
                max_tokens,
                args.seed,
                args.timeout,
                use_chat,
                args.stop_token_ids,
                config["generation"]["stop_strings"],
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"{arm} API response has no choices at row {index}")
            choice = choices[0]
            prediction = (choice.get("message") or {}).get("content") or "" if use_chat else choice.get("text") or ""
            if not isinstance(prediction, str):
                raise ValueError(f"{arm} API response prediction is not text at row {index}")
            token_usage = response.get("usage", {})
            if not isinstance(token_usage, dict):
                token_usage = {}
            expected_tokens = ordered_prompt_tokens[index]
            saved_usage = {**token_usage, "local_prompt_tokens": expected_tokens}
            record = _result_record(index, rows[index], prediction, saved_usage, response, prompt)
            record["client_latency_seconds"] = time.perf_counter() - request_started
            record["request_started_utc"] = request_started_utc
            record["request_finished_utc"] = datetime.now(timezone.utc).isoformat()
            nonlocal completed_count
            with write_lock:
                _append_result_record(result_stream, record)
            with progress_lock:
                completed_count += 1
            api_prompt_tokens = token_usage.get("prompt_tokens")
            if api_prompt_tokens is None and args.strict_token_count:
                raise ValueError(f"{arm} API response omitted usage.prompt_tokens at row {index}; response was saved")
            if api_prompt_tokens is None:
                logger.warning("%s API response omitted usage.prompt_tokens at row %d", arm, index)
            elif api_prompt_tokens != expected_tokens:
                message = (
                    f"{arm} prompt token count differs at row {index}: local={expected_tokens}, "
                    f"API={api_prompt_tokens}; response was saved, review server tokenizer/template configuration"
                )
                if args.strict_token_count:
                    raise ValueError(message)
                logger.warning(message)
            if completed_count % 10 == 0 or completed_count == len(rows):
                logger.info("%s: %d/%d durable rows", arm, completed_count, len(rows))

        _run_uncompleted(completed_indices, len(rows), process_row, args.concurrency)
    logger.info("Run complete: %s", output_dir)


def _score_saved(args: argparse.Namespace, longbench_metrics: Any) -> None:
    auto_rows = _jsonl(args.auto_results)
    int8_rows = _jsonl(args.int8_results)
    auto_config = json.loads(args.auto_results.with_name("auto.config.json").read_text(encoding="utf-8"))
    int8_config = json.loads(args.int8_results.with_name("int8.config.json").read_text(encoding="utf-8"))
    comparable_keys = (
        "task",
        "data",
        "input_sha256",
        "model",
        "tokenizer",
        "sampling",
        "generation",
        "max_tokens",
        "max_prompt_plus_generation_tokens",
        "examples",
        "api_mode",
        "longbench_revision",
        "gsm8k_shots_sha256",
        "model_impl",
    )
    for key in comparable_keys:
        if auto_config.get(key) != int8_config.get(key):
            raise ValueError(f"auto and int8 run configs differ for {key}")
    auto_protocol = auto_config.get("execution_protocol", {})
    int8_protocol = int8_config.get("execution_protocol", {})
    protocol_keys = set(auto_protocol) | set(int8_protocol)
    allowed_protocol_differences = {"request_concurrency", "max_seqs_per_batch"}
    for key in protocol_keys - allowed_protocol_differences:
        if auto_protocol.get(key) != int8_protocol.get(key):
            raise ValueError(f"auto and int8 run configs differ for execution_protocol.{key}")
    if auto_config.get("arm") != "auto" or int8_config.get("arm") != "int8":
        raise ValueError("paired scoring requires auto and int8 run configs")
    auto_manifest = auto_config["server_manifest"]
    int8_manifest = int8_config["server_manifest"]
    for key in ("checkpoint_path", "checkpoint_sha256", "model_config_sha256", "chat_template_sha256", "model"):
        if auto_manifest.get(key) != int8_manifest.get(key):
            raise ValueError(f"auto and int8 server manifests differ for {key}")
    if len(auto_rows) != len(int8_rows):
        raise ValueError("paired result files have different row counts")
    expected_examples = auto_config.get("examples")
    if not isinstance(expected_examples, int) or len(auto_rows) != expected_examples:
        raise ValueError(f"paired results are incomplete: expected {expected_examples} rows, found {len(auto_rows)}")
    for arm_rows in (auto_rows, int8_rows):
        indices = [row.get("_eval_index") for row in arm_rows]
        if sorted(indices) != list(range(expected_examples)):
            raise ValueError("paired results contain missing or duplicate evaluation indices")
        arm_rows.sort(key=lambda row: row["_eval_index"])
    token_comparisons = []
    for index, (auto_row, int8_row) in enumerate(zip(auto_rows, int8_rows)):
        for key in ("_id", "id", "_prompt_sha256", "_record_sha256"):
            if auto_row.get(key) != int8_row.get(key):
                raise ValueError(f"paired result mismatch at row {index}: {key}")
        auto_tokens = auto_row.get("usage", {}).get("prompt_tokens")
        int8_tokens = int8_row.get("usage", {}).get("prompt_tokens")
        if auto_tokens is None or int8_tokens is None:
            message = f"API prompt token count unavailable for paired row {index}"
            if args.strict_token_count:
                raise ValueError(message)
            logger.warning(message)
        elif auto_tokens != int8_tokens:
            message = (
                f"API prompt token counts differ between arms at row {index}: auto={auto_tokens}, int8={int8_tokens}"
            )
            if args.strict_token_count:
                raise ValueError(message)
            logger.warning(message)
        token_comparisons.append(
            {
                "auto_api_prompt_tokens": auto_tokens,
                "int8_api_prompt_tokens": int8_tokens,
                "auto_local_prompt_tokens": auto_row.get("usage", {}).get("local_prompt_tokens"),
                "int8_local_prompt_tokens": int8_row.get("usage", {}).get("local_prompt_tokens"),
            }
        )
    records = [
        {
            key: value
            for key, value in row.items()
            if key not in {"prediction", "usage", "api_response", "_prompt_sha256", "_record_sha256"}
        }
        for row in auto_rows
    ]
    predictions = {"auto": [row["prediction"] for row in auto_rows], "int8": [row["prediction"] for row in int8_rows]}
    summary, item_rows = _score_pair(args.task, records, predictions, longbench_metrics)
    for item, token_data in zip(item_rows, token_comparisons):
        item["prompt_token_counts"] = token_data
    _write_json(
        args.output / "paired_scores.json",
        {"scores": summary, "items": item_rows, "n": len(records)},
    )
    logger.info("Paired scores written to %s", args.output / "paired_scores.json")


if __name__ == "__main__":
    main()
