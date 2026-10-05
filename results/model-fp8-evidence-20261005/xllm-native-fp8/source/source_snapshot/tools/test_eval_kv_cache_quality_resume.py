# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Offline fixtures for durable per-row evaluation resume behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from tools import eval_kv_cache_quality as evaluator


def _fixture_config(stop_strings: list[str], attestation: str = "old") -> dict[str, Any]:
    manifest = {
        "server_started": True,
        "server_attestation": attestation,
        "verified_smoke_log": f"/tmp/{attestation}.log",
        "model_impl": "python",
        "graph_mode": "off",
        "kv_cache_mode": "auto",
        "checkpoint_path": "/model",
        "checkpoint_sha256": "a" * 64,
        "model_config_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "model": "fixture",
        "runtime_configuration": {"max_seqs_per_batch": 1},
    }
    return {
        "task": "gsm8k",
        "data": "/data/shard.jsonl",
        "input_sha256": "d" * 64,
        "model": "fixture",
        "arm": "auto",
        "endpoint": "http://127.0.0.1:18994/v1",
        "model_impl": "python",
        "tokenizer": "/model",
        "sampling": {"temperature": 0, "top_p": 1, "seed": 17},
        "generation": {
            "temperature": 0,
            "top_p": 1,
            "seed": 17,
            "max_tokens": 512,
            "stop_token_ids": [151645, 151643],
            "stop_strings": stop_strings,
        },
        "max_tokens": 512,
        "max_prompt_plus_generation_tokens": 32768,
        "examples": 2,
        "api_mode": "chat-completions",
        "execution_protocol": {"request_concurrency": 1, "max_seqs_per_batch": 1},
        "gsm8k_shots_sha256": "e" * 64,
        "runtime_versions": {"python": "3.12"},
        "server_manifest": manifest,
        "server_manifest_sha256": "f" * 64 if attestation == "old" else "0" * 64,
    }


def _fixture_rows() -> tuple[list[dict[str, Any]], list[str], list[int], list[dict[str, Any]]]:
    rows = [{"question": "one", "answer": "#### 1"}, {"question": "two", "answer": "#### 2"}]
    prompts = ["prompt one", "prompt two"]
    local_tokens = [1, 1]
    records = []
    for index, row in enumerate(rows):
        response = {
            "choices": [{"message": {"content": f"answer {index}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        records.append(
            evaluator._result_record(
                index,
                row,
                f"answer {index}",
                {"prompt_tokens": 1, "completion_tokens": 1, "local_prompt_tokens": 1},
                response,
                prompts[index],
            )
        )
    return rows, prompts, local_tokens, records


class ResumeTests(unittest.TestCase):
    def test_int4_manifest_is_python_eager_and_int4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model = Path(temporary_dir)
            config_bytes = b'{"model_type":"fixture"}\n'
            tokenizer_data = {"chat_template": "fixture template"}
            (model / "config.json").write_bytes(config_bytes)
            (model / "tokenizer_config.json").write_text(json.dumps(tokenizer_data), encoding="utf-8")
            chat_template_hash = hashlib.sha256(
                json.dumps(tokenizer_data["chat_template"], ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            manifest = {
                "server_started": True,
                "server_attestation": "int4 test fixture",
                "model_impl": "python",
                "graph_mode": "off",
                "kv_cache_mode": "int4",
                "checkpoint_path": str(model),
                "checkpoint_sha256": "a" * 64,
                "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "chat_template_sha256": chat_template_hash,
                "model": "fixture",
                "runtime_configuration": {"max_seqs_per_batch": 64},
            }
            manifest_path = model / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = evaluator._load_server_manifest(manifest_path, "int4", "fixture", model)
            self.assertEqual(loaded["kv_cache_mode"], "int4")
            self.assertEqual(loaded["model_impl"], "python")

    def test_fp8_dtype_manifests_require_matching_storage_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model = Path(temporary_dir)
            config_bytes = b'{"model_type":"qwen2"}\n'
            tokenizer_data = {"chat_template": "fixture template"}
            (model / "config.json").write_bytes(config_bytes)
            (model / "tokenizer_config.json").write_text(json.dumps(tokenizer_data), encoding="utf-8")
            template_hash = hashlib.sha256(
                json.dumps(tokenizer_data["chat_template"], ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            base_manifest = {
                "server_started": True,
                "model_impl": "python",
                "graph_mode": "off",
                "checkpoint_path": str(model),
                "checkpoint_sha256": "a" * 64,
                "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "chat_template_sha256": template_hash,
                "model": "fixture",
                "runtime_configuration": {"max_seqs_per_batch": 50},
            }
            manifest_path = model / "manifest.json"
            for arm, dtype in (("fp8-e4m3", "fp8_e4m3"), ("fp8-e5m2", "fp8_e5m2")):
                manifest_path.write_text(
                    json.dumps({**base_manifest, "kv_cache_mode": dtype, "quantized_backend": "triton"}),
                    encoding="utf-8",
                )
                loaded = evaluator._load_server_manifest(manifest_path, arm, "fixture", model)
                self.assertEqual(loaded["kv_cache_mode"], dtype)
                manifest_path.write_text(
                    json.dumps({**base_manifest, "kv_cache_mode": "auto", "quantized_backend": "triton"}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "kv_cache_mode"):
                    evaluator._load_server_manifest(manifest_path, arm, "fixture", model)

    def test_fp8_paired_result_flags_score_against_auto(self) -> None:
        rows, prompts, local_tokens, records = _fixture_rows()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            for arm, dtype in (("auto", "auto"), ("fp8-e4m3", "fp8_e4m3"), ("fp8-e5m2", "fp8_e5m2")):
                config = _fixture_config(["Question:"])
                config.update(arm=arm, examples=len(rows), quality_transform_mode="none")
                config["server_manifest"].update(kv_cache_mode=dtype, quality_transform_mode="none")
                (output / f"{arm}.config.json").write_text(json.dumps(config), encoding="utf-8")
                result_path = output / f"{arm}.jsonl"
                with result_path.open("w", encoding="utf-8") as stream:
                    for index, (row, prompt) in enumerate(zip(rows, prompts)):
                        prediction = f"#### {index + (arm == 'auto')}"
                        response = {
                            "choices": [{"message": {"content": prediction}}],
                            "usage": {"prompt_tokens": 1},
                        }
                        record = evaluator._result_record(
                            index,
                            row,
                            prediction,
                            {"prompt_tokens": 1, "local_prompt_tokens": local_tokens[index]},
                            response,
                            prompt,
                        )
                        stream.write(json.dumps(record) + "\n")
            for arm in ("fp8-e4m3", "fp8-e5m2"):
                args = argparse.Namespace(
                    task="gsm8k",
                    auto_results=output / "auto.jsonl",
                    int8_results=None,
                    int4_results=None,
                    k_only_fp8_results=None,
                    v_only_fp8_results=None,
                    fp8_e4m3_results=output / "fp8-e4m3.jsonl" if arm == "fp8-e4m3" else None,
                    fp8_e5m2_results=output / "fp8-e5m2.jsonl" if arm == "fp8-e5m2" else None,
                    strict_token_count=True,
                    output=output / f"paired-{arm}",
                )
                evaluator._score_saved(args, None)
                scores = json.loads((args.output / "paired_scores.json").read_text(encoding="utf-8"))
                self.assertEqual(scores["n"], len(rows))

    def test_quality_transform_manifests_keep_bf16_cache_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model = Path(temporary_dir)
            config_bytes = b'{"model_type":"qwen2"}\n'
            tokenizer_data = {"chat_template": "fixture template"}
            (model / "config.json").write_bytes(config_bytes)
            (model / "tokenizer_config.json").write_text(json.dumps(tokenizer_data), encoding="utf-8")
            chat_template_hash = hashlib.sha256(
                json.dumps(tokenizer_data["chat_template"], ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            base_manifest = {
                "server_started": True,
                "server_attestation": "quality transform test fixture",
                "model_impl": "python",
                "graph_mode": "off",
                "kv_cache_mode": "auto",
                "checkpoint_path": str(model),
                "checkpoint_sha256": "a" * 64,
                "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "chat_template_sha256": chat_template_hash,
                "model": "fixture",
                "runtime_configuration": {"max_seqs_per_batch": 64},
            }
            manifest_path = model / "manifest.json"
            for arm, mode in (
                ("v-only-int4", "v_only_int4"),
                ("int4-rht-g32", "int4_rht_g32"),
                ("k-only-fp8", "k_only_fp8"),
                ("v-only-fp8", "v_only_fp8"),
            ):
                manifest = {**base_manifest, "quality_transform_mode": mode}
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                loaded = evaluator._load_server_manifest(manifest_path, arm, "fixture", model)
                self.assertEqual(loaded["kv_cache_mode"], "auto")
                self.assertEqual(loaded["quality_transform_mode"], mode)
            manifest_path.write_text(json.dumps({**base_manifest, "quality_transform_mode": "v_only_int4"}))
            with self.assertRaisesRegex(ValueError, "quality_transform_mode"):
                evaluator._load_server_manifest(manifest_path, "int4-rht-g32", "fixture", model)

    def test_resume_rejects_quality_transform_mode_change(self) -> None:
        rows, prompts, local_tokens, _ = _fixture_rows()
        original = _fixture_config(["Question:"])
        original.update(arm="k-only-fp8", quality_transform_mode="k_only_fp8")
        original["server_manifest"]["quality_transform_mode"] = "k_only_fp8"
        changed = json.loads(json.dumps(original))
        changed["quality_transform_mode"] = "v_only_fp8"
        changed["server_manifest"]["quality_transform_mode"] = "v_only_fp8"
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            evaluator._initialize_run_output(output, "k-only-fp8", original, False, rows, prompts, local_tokens, True)
            with self.assertRaisesRegex(ValueError, "quality_transform_mode"):
                evaluator._initialize_run_output(output, "k-only-fp8", changed, True, rows, prompts, local_tokens, True)

    def test_resume_prefix_and_complete_run_issue_no_requests(self) -> None:
        rows, prompts, local_tokens, records = _fixture_rows()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            path, start = evaluator._initialize_run_output(
                output,
                "auto",
                _fixture_config(["Question:", "</s>", "<|im_end|>"]),
                False,
                rows,
                prompts,
                local_tokens,
                True,
            )
            self.assertEqual(start, set())
            with path.open("a", encoding="utf-8") as stream:
                evaluator._append_result_record(stream, records[0])
            resumed_config = _fixture_config(["Question:", "</s>", "<|im_end|>"], attestation="new")
            path, start = evaluator._initialize_run_output(
                output, "auto", resumed_config, True, rows, prompts, local_tokens, True
            )
            self.assertEqual(start, {0})
            pending: list[int] = []
            evaluator._run_uncompleted(start, len(rows), pending.append)
            self.assertEqual(pending, [1])
            with path.open("a", encoding="utf-8") as stream:
                evaluator._append_result_record(stream, records[1])
            path, start = evaluator._initialize_run_output(
                output, "auto", resumed_config, True, rows, prompts, local_tokens, True
            )
            self.assertEqual(start, {0, 1})
            api_requests: list[int] = []
            evaluator._run_uncompleted(start, len(rows), api_requests.append)
            self.assertEqual(api_requests, [])
            saved = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(
                {"_eval_index", "prediction", "usage", "api_response", "_prompt_sha256", "_record_sha256"}
                <= saved.keys()
            )

    def test_resume_rejects_generation_stop_protocol_mismatch(self) -> None:
        rows, prompts, local_tokens, _ = _fixture_rows()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            evaluator._initialize_run_output(
                output, "auto", _fixture_config([]), False, rows, prompts, local_tokens, True
            )
            with self.assertRaisesRegex(ValueError, "generation"):
                evaluator._initialize_run_output(
                    output,
                    "auto",
                    _fixture_config(["Question:", "</s>", "<|im_end|>"]),
                    True,
                    rows,
                    prompts,
                    local_tokens,
                    True,
                )

    def test_resume_rejects_incomplete_final_line_without_repair(self) -> None:
        rows, prompts, local_tokens, records = _fixture_rows()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            path, _ = evaluator._initialize_run_output(
                output, "auto", _fixture_config(["Question:"]), False, rows, prompts, local_tokens, True
            )
            with path.open("a", encoding="utf-8") as stream:
                evaluator._append_result_record(stream, records[0])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "incomplete final line"):
                evaluator._initialize_run_output(
                    output, "auto", _fixture_config(["Question:"]), True, rows, prompts, local_tokens, True
                )
            self.assertFalse(path.read_bytes().endswith(b"\n"))

    def test_new_run_refuses_existing_result(self) -> None:
        rows, prompts, local_tokens, _ = _fixture_rows()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            evaluator._initialize_run_output(
                output, "auto", _fixture_config(["Question:"]), False, rows, prompts, local_tokens, True
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                evaluator._initialize_run_output(
                    output, "auto", _fixture_config(["Question:"]), False, rows, prompts, local_tokens, True
                )

    def test_paired_scorer_rejects_partial_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            row = {"id": "one", "_prompt_sha256": "p", "_record_sha256": "r", "prediction": "1", "usage": {}}
            for arm in ("auto", "int8"):
                results = output / f"{arm}.jsonl"
                results.write_text(json.dumps(row) + "\n", encoding="utf-8")
                config = _fixture_config(["Question:"])
                config.update(arm=arm, examples=2)
                results.with_name(f"{arm}.config.json").write_text(json.dumps(config), encoding="utf-8")
            args = argparse.Namespace(
                auto_results=output / "auto.jsonl",
                int8_results=output / "int8.jsonl",
                strict_token_count=False,
                output=output,
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                evaluator._score_saved(args, None)

    def test_resume_accepts_out_of_order_journal_and_rejects_duplicate_index(self) -> None:
        rows, prompts, local_tokens, records = _fixture_rows()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            path, _ = evaluator._initialize_run_output(
                output, "auto", _fixture_config(["Question:"]), False, rows, prompts, local_tokens, True
            )
            with path.open("a", encoding="utf-8") as stream:
                evaluator._append_result_record(stream, records[1])
                evaluator._append_result_record(stream, records[0])
            self.assertEqual(evaluator._validate_resume_prefix(path, rows, prompts, local_tokens, True), {0, 1})
            with path.open("a", encoding="utf-8") as stream:
                evaluator._append_result_record(stream, records[1])
            with self.assertRaisesRegex(ValueError, "duplicate evaluation index"):
                evaluator._validate_resume_prefix(path, rows, prompts, local_tokens, True)

    def test_bounded_concurrency_runs_all_indices_and_propagates_errors(self) -> None:
        observed: list[int] = []
        lock = threading.Lock()

        def record(index: int) -> None:
            with lock:
                observed.append(index)

        evaluator._run_uncompleted(set(), 8, record, concurrency=4)
        self.assertEqual(sorted(observed), list(range(8)))

        failed: list[int] = []

        def fail(index: int) -> None:
            failed.append(index)
            raise RuntimeError("fixture HTTP failure")

        with self.assertRaisesRegex(RuntimeError, "fixture HTTP failure"):
            evaluator._run_uncompleted(set(), 5, fail, concurrency=4)
        self.assertLessEqual(len(failed), 4)

    def test_paired_scorer_restores_original_order_from_concurrent_journal(self) -> None:
        rows, prompts, local_tokens, _ = _fixture_rows()
        rows[0]["id"] = "first"
        rows[1]["id"] = "second"
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            for arm in ("auto", "int8"):
                config = _fixture_config(["Question:"])
                config.update(arm=arm, examples=2)
                (output / f"{arm}.config.json").write_text(json.dumps(config), encoding="utf-8")
                response_rows = []
                for index, row in enumerate(rows):
                    prediction = f"#### {index + (arm == 'auto')}"
                    response = {
                        "choices": [{"message": {"content": prediction}}],
                        "usage": {"prompt_tokens": 1},
                    }
                    response_rows.append(
                        evaluator._result_record(
                            index,
                            row,
                            prediction,
                            {"prompt_tokens": 1, "local_prompt_tokens": 1},
                            response,
                            prompts[index],
                        )
                    )
                with (output / f"{arm}.jsonl").open("w", encoding="utf-8") as stream:
                    for record in reversed(response_rows):
                        stream.write(json.dumps(record) + "\n")
            args = argparse.Namespace(
                task="gsm8k",
                auto_results=output / "auto.jsonl",
                int8_results=output / "int8.jsonl",
                strict_token_count=True,
                output=output,
            )
            evaluator._score_saved(args, None)
            scored = json.loads((output / "paired_scores.json").read_text(encoding="utf-8"))
            self.assertEqual([row["id"] for row in scored["items"]], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
