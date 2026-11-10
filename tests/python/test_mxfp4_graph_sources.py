# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Execute the emitted host mask/binding blocks without Torch or CUDA."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

_PATH = Path(__file__).resolve().parents[2] / "xllm/python/attention/mxfp4_graph_sources.py"
_SPEC = importlib.util.spec_from_file_location("mxfp4_graph_sources", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
sources = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sources)


def _block(text: str, start: int) -> str:
    first = text.index("{", start)
    depth = 1
    for end in range(first + 1, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end + 1]
    raise AssertionError("Unbalanced emitted C++ block")


def _compile_and_run(program: str, directory: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        raise unittest.SkipTest("A host C++17 compiler is required for the emitted-block test")
    executable = directory / "mask_contract"
    compiled = subprocess.run([compiler, "-std=c++17", "-O0", "-x", "c++", "-o", str(executable), "-"],
                              input=program, text=True, capture_output=True, timeout=30)
    if compiled.returncode:
        raise AssertionError(compiled.stderr)
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=5)


class GraphSourcesTest(unittest.TestCase):
    def setUp(self) -> None:
        location = os.environ.get("MXFP4_FLASHINFER_SOURCE_ROOT")
        if location is None:
            self.skipTest("Set MXFP4_FLASHINFER_SOURCE_ROOT to pinned FlashInfer package source")
        self.source_package = Path(location)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_emitted_mask_overwrites_dirty_padding(self) -> None:
        source_package, tmp_path = self.source_package, self.tmp_path
        scheduler, _ = sources.read_graph_sources(source_package, source_package / "data/csrc")
        body = scheduler[scheduler.index("inline cudaError_t PrefillPlanImpl("):]
        split = _block(body, body.index("if (split_kv)"))
        assert "block_valid_mask" not in split
        marker = body.index("// Graph launches include padding")
        graph = _block(body, body.index("if (enable_cuda_graph)", marker))
        program = r"""
    #include <algorithm>
    #include <array>
    #include <cassert>
    #include <cstddef>
    #include <cstdint>
    struct PlanInfo { size_t block_valid_mask_offset = 0; };
    struct Allocator {
      int32_t calls = 0;
      size_t aligned_alloc_offset(size_t bytes, size_t alignment, const char*) {
        assert(bytes == 66 && alignment == 16);
        ++calls;
        return 1024;
      }
    };
    template <typename T> T* GetPtrFromBaseOffset(void* base, size_t offset) {
      return reinterpret_cast<T*>(static_cast<unsigned char*>(base) + offset);
    }
    template <bool MATERIALIZE>
    void check(bool enable_cuda_graph, uint32_t new_batch_size) {
      alignas(16) std::array<unsigned char, 2048> memory;
      memory.fill(0xa5);
      void* page_locked_int_buffer = memory.data();
      uint32_t padded_batch_size = 66;
      PlanInfo plan_info;
      Allocator int_allocator;
    """ + graph + r"""
      assert(int_allocator.calls == (enable_cuda_graph ? 1 : 0));
      assert(plan_info.block_valid_mask_offset == (enable_cuda_graph ? 1024 : 0));
      for (uint32_t index = 0; index < 66; ++index) {
        const unsigned char expected = enable_cuda_graph && MATERIALIZE
            ? static_cast<unsigned char>(index < new_batch_size) : 0xa5;
        assert(memory[1024 + index] == expected);
      }
    }
    int main() {
      for (uint32_t real : {0u, 1u, 2u, 4u, 66u}) {
        check<true>(false, real);
        check<true>(true, real);
        check<false>(false, real);
        check<false>(true, real);
      }
    }
    """
        _compile_and_run(program, tmp_path)


    def test_emitted_run_binds_mask_without_split(self) -> None:
        source_package, tmp_path = self.source_package, self.tmp_path
        _, binding = sources.read_graph_sources(source_package, source_package / "data/csrc")
        blocks = []
        position = 0
        for _ in range(2):
            start = binding.index("if (plan_info.split_kv)", position)
            split = _block(binding, start)
            assert "block_valid_mask" not in split
            position = start + len(split)
            graph_start = binding.index("if (plan_info.enable_cuda_graph)", position)
            blocks.append(_block(binding, graph_start))
        assert binding.count("params.block_valid_mask =\n") == 2
        program = r"""
    #include <array>
    #include <cassert>
    #include <cstddef>
    #include <cstdint>
    struct PlanInfo {
      bool enable_cuda_graph;
      size_t block_valid_mask_offset = 1024;
      size_t total_num_rows_offset = 800;
    };
    struct Params { bool* block_valid_mask = nullptr; uint32_t* total_num_rows = nullptr; };
    template <typename T> T* GetPtrFromBaseOffset(void* base, size_t offset) {
      return reinterpret_cast<T*>(static_cast<unsigned char*>(base) + offset);
    }
    int main() {
      alignas(16) std::array<unsigned char, 2048> memory;
      void* int_buffer_ptr = memory.data();
      for (bool enabled : {false, true}) {
        PlanInfo plan_info{enabled};
    """
        for block in blocks:
            program += "{ Params params;\n" + block + r"""
      assert(params.block_valid_mask == (enabled ? GetPtrFromBaseOffset<bool>(int_buffer_ptr, 1024) : nullptr));
      assert(params.total_num_rows == (enabled ? GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, 800) : nullptr));
    }
    """
        _compile_and_run(program + "\n}}\n", tmp_path)


    def test_generated_run_source_is_replaced_not_installed(self) -> None:
        source_package, tmp_path = self.source_package, self.tmp_path
        scheduler, binding = sources.read_graph_sources(source_package, source_package / "data/csrc")
        original = tmp_path / "batch_prefill.cu"
        original.write_text("original generated run")
        jit_binding = tmp_path / "batch_prefill_jit_binding.cu"
        spec = SimpleNamespace(sources=[original, jit_binding], extra_cuda_cflags=["-O3"])
        sources.configure_graph_sources(spec, tmp_path, scheduler, binding)
        assert spec.sources == [tmp_path / "mxfp4_batch_prefill.cu", jit_binding]
        assert spec.sources[0].read_text() == binding
        assert original.read_text() == "original generated run"
        assert spec.extra_cuda_cflags == ["-O3", "-include", str(tmp_path / "mxfp4_scheduler.cuh")]
        assert "#define FLASHINFER_ATTENTION_SCHEDULER_CUH_" in scheduler
        assert '#include "../' not in scheduler and '#include "heap.h"' not in scheduler


    def test_changed_upstream_input_is_rejected(self) -> None:
        source_package, tmp_path = self.source_package, self.tmp_path
        for index, name in enumerate(sources._SOURCE_SHA256):
            with self.subTest(source=name):
                package = tmp_path / str(index) / "flashinfer"
                shutil.copytree(source_package, package)
                target = package / ("data/" + name if name.startswith(("csrc/", "include/")) else name)
                target.write_bytes(target.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "pinned FlashInfer source"):
                    sources.read_graph_sources(package, package / "data/csrc")

    def test_unknown_generator_source_list_is_rejected(self) -> None:
        source_package, tmp_path = self.source_package, self.tmp_path
        spec = SimpleNamespace(sources=[tmp_path / "unrelated.cu"], extra_cuda_cflags=[])
        with self.assertRaisesRegex(ValueError, "expected run source"):
            sources.configure_graph_sources(spec, tmp_path, "scheduler", "binding")
        assert not (tmp_path / "mxfp4_scheduler.cuh").exists()


    def test_existing_private_source_drift_is_rejected(self) -> None:
        source_package, tmp_path = self.source_package, self.tmp_path
        (tmp_path / "mxfp4_scheduler.cuh").write_text("different")
        spec = SimpleNamespace(sources=[tmp_path / "batch_prefill.cu"], extra_cuda_cflags=[])
        with self.assertRaisesRegex(ValueError, "private source changed"):
            sources.configure_graph_sources(spec, tmp_path, "scheduler", "binding")

    def test_fingerprint_covers_templates_parameters_and_all_patches(self) -> None:
        patches = ["prefill", "scheduler", "binding"]
        parameters = {"dtype_kv": "torch.uint8", "head_dim_qk": 256}
        baseline = sources.graph_source_fingerprint(*patches, parameters)
        for index in range(3):
            changed = patches.copy()
            changed[index] += "\n"
            self.assertNotEqual(baseline, sources.graph_source_fingerprint(*changed, parameters))
        self.assertNotEqual(baseline, sources.graph_source_fingerprint(
            *patches, {**parameters, "head_dim_qk": 128}))
        for name, digest in tuple(sources._SOURCE_SHA256.items()):
            with self.subTest(input=name):
                sources._SOURCE_SHA256[name] = "changed"
                changed = sources.graph_source_fingerprint(*patches, parameters)
                sources._SOURCE_SHA256[name] = digest
                self.assertNotEqual(baseline, changed)

    def test_manifest_hash_is_independently_reconstructible(self) -> None:
        manifest = sources.graph_source_manifest("prefill", "scheduler", "binding", {"dtype": "uint8"})
        fingerprint = manifest.pop("fingerprint")
        module_name = manifest.pop("module_name")
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(), fingerprint)
        self.assertEqual(module_name, "xllm_mxfp4_bf16_d256_graph_" + fingerprint)
        self.assertIn("include/flashinfer/attention/prefill.cuh", manifest["inputs"])
