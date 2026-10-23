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

#include "core/runtime/cuda_kv_cache_admission.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

CudaKVCacheExecutionConfig native_fp8_graph_config() {
  return {KVCacheDtype::FP8_E4M3,
          true,
          true,
          false,
          "cudagraphs",
          "1",
          "flashinfer"};
}

TEST(CudaKVCacheAdmissionTest, AllowsNativeFlashInferFp8DecodeGraph) {
  const CudaKVCacheExecutionConfig config = native_fp8_graph_config();
  EXPECT_TRUE(is_native_flashinfer_fp8_decode_graph(config));
  EXPECT_TRUE(is_cuda_kv_cache_execution_mode_allowed(config));
}

TEST(CudaKVCacheAdmissionTest, RejectsTritonAndMixedBackendsWithFp8Graph) {
  for (const std::string_view backend : {"triton", "flashinfer_mixed", ""}) {
    CudaKVCacheExecutionConfig config = native_fp8_graph_config();
    config.quantized_backend = backend;
    EXPECT_FALSE(is_cuda_kv_cache_execution_mode_allowed(config)) << backend;
  }
}

TEST(CudaKVCacheAdmissionTest, RequiresExplicitNativeKvOptIn) {
  for (const std::string_view native_kv : {"", "0", "true", "01"}) {
    CudaKVCacheExecutionConfig config = native_fp8_graph_config();
    config.native_kv = native_kv;
    EXPECT_FALSE(is_cuda_kv_cache_execution_mode_allowed(config)) << native_kv;
  }
}

TEST(CudaKVCacheAdmissionTest, RejectsOtherQuantizedDtypesWithGraph) {
  for (const KVCacheDtype dtype : {KVCacheDtype::INT8,
                                   KVCacheDtype::INT4,
                                   KVCacheDtype::FP8_E5M2,
                                   KVCacheDtype::BF16_INT8,
                                   KVCacheDtype::BF16_FP8_E4M3,
                                   KVCacheDtype::BF16_INT4}) {
    CudaKVCacheExecutionConfig config = native_fp8_graph_config();
    config.kv_cache_dtype = dtype;
    EXPECT_FALSE(is_cuda_kv_cache_execution_mode_allowed(config));
  }
}

TEST(CudaKVCacheAdmissionTest, RejectsInvalidNativeGraphModes) {
  CudaKVCacheExecutionConfig config = native_fp8_graph_config();
  config.enable_graph = false;
  EXPECT_FALSE(is_cuda_kv_cache_execution_mode_allowed(config));

  config = native_fp8_graph_config();
  config.enable_prefill_piecewise_graph = true;
  EXPECT_FALSE(is_cuda_kv_cache_execution_mode_allowed(config));

  for (const std::string_view backend : {"off", "torch_compile", ""}) {
    config = native_fp8_graph_config();
    config.python_graph_backend = backend;
    EXPECT_FALSE(is_cuda_kv_cache_execution_mode_allowed(config)) << backend;
  }

  config = native_fp8_graph_config();
  config.python_model = false;
  EXPECT_FALSE(is_cuda_kv_cache_execution_mode_allowed(config));
}

TEST(CudaKVCacheAdmissionTest, PreservesEagerQuantizedExecution) {
  for (const KVCacheDtype dtype : {KVCacheDtype::INT8,
                                   KVCacheDtype::FP8_E4M3,
                                   KVCacheDtype::FP8_E5M2,
                                   KVCacheDtype::INT4,
                                   KVCacheDtype::BF16_INT8,
                                   KVCacheDtype::BF16_FP8_E4M3,
                                   KVCacheDtype::BF16_INT4}) {
    CudaKVCacheExecutionConfig config;
    config.kv_cache_dtype = dtype;
    config.python_model = true;
    EXPECT_TRUE(is_cuda_kv_cache_execution_mode_allowed(config));
    EXPECT_FALSE(is_native_flashinfer_fp8_decode_graph(config));

    config.python_model = false;
    EXPECT_FALSE(is_cuda_kv_cache_execution_mode_allowed(config));
  }
}

TEST(CudaKVCacheAdmissionTest, PreservesAutoDtypeExecutionModes) {
  for (const bool enable_graph : {false, true}) {
    for (const bool enable_prefill_graph : {false, true}) {
      CudaKVCacheExecutionConfig config;
      config.enable_graph = enable_graph;
      config.enable_prefill_piecewise_graph = enable_prefill_graph;
      config.python_graph_backend = "cudagraphs";
      EXPECT_TRUE(is_cuda_kv_cache_execution_mode_allowed(config));
      EXPECT_FALSE(is_native_flashinfer_fp8_decode_graph(config));
    }
  }
}

}  // namespace
}  // namespace xllm
