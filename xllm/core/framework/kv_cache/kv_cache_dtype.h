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

#include <glog/logging.h>

#include <cstdint>
#include <string_view>

namespace xllm {

enum class KVCacheDtype : int8_t { AUTO, INT8, FP8_E4M3, FP8_E5M2, INT4 };

inline KVCacheDtype parse_kv_cache_dtype(std::string_view dtype) {
  if (dtype == "auto") {
    return KVCacheDtype::AUTO;
  }
  if (dtype == "int8") {
    return KVCacheDtype::INT8;
  }
  if (dtype == "fp8" || dtype == "fp8_e4m3") {
    return KVCacheDtype::FP8_E4M3;
  }
  if (dtype == "fp8_e5m2") {
    return KVCacheDtype::FP8_E5M2;
  }
  if (dtype == "int4") {
    return KVCacheDtype::INT4;
  }
  LOG(FATAL) << "Invalid kv_cache_dtype=" << dtype
             << ". Expected auto, int8, fp8, fp8_e4m3, fp8_e5m2 or int4.";
  return KVCacheDtype::AUTO;
}

// Quantized formats use one byte per stored element. INT4 packs adjacent
// channels into the low/high nibble, padding the last channel if D is odd.
inline int64_t kv_cache_storage_head_dim(KVCacheDtype dtype, int64_t head_dim) {
  CHECK_GT(head_dim, 0);
  return dtype == KVCacheDtype::INT4 ? head_dim / 2 + head_dim % 2 : head_dim;
}

}  // namespace xllm
