/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <string_view>

#include "core/framework/kv_cache/kv_cache_dtype.h"

namespace xllm {

struct CudaKVCacheExecutionConfig {
  KVCacheDtype kv_cache_dtype = KVCacheDtype::AUTO;
  bool python_model = false;
  bool enable_graph = false;
  bool enable_prefill_piecewise_graph = false;
  std::string_view python_graph_backend = "off";
  std::string_view native_kv;
  std::string_view quantized_backend;
};

inline bool is_native_flashinfer_fp8_decode_graph(
    const CudaKVCacheExecutionConfig& config) {
  return config.python_model &&
         config.kv_cache_dtype == KVCacheDtype::FP8_E4M3 &&
         config.native_kv == "1" && config.quantized_backend == "flashinfer" &&
         config.python_graph_backend == "cudagraphs" && config.enable_graph &&
         !config.enable_prefill_piecewise_graph;
}

// This policy only admits an execution mode. WorkerImpl must still enforce
// ordinary MHA/GQA, CP=1 and the existing offload/PD/speculation restrictions.
// AUTO bypasses this quantized-cache policy, preserving BF16 admission.
inline bool is_cuda_kv_cache_execution_mode_allowed(
    const CudaKVCacheExecutionConfig& config) {
  if (config.kv_cache_dtype == KVCacheDtype::AUTO) {
    return true;
  }
  const bool eager = !config.enable_graph &&
                     !config.enable_prefill_piecewise_graph &&
                     config.python_graph_backend == "off";
  return config.python_model &&
         (eager || is_native_flashinfer_fp8_decode_graph(config));
}

}  // namespace xllm
