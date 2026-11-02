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

inline bool is_native_kv_cache_format(std::string_view dtype) {
  return dtype == "fp8_e4m3" || dtype == "nvfp4";
}

inline int64_t native_kv_cache_storage_head_dim(std::string_view dtype,
                                                int64_t head_dim) {
  if (dtype == "nvfp4") {
    CHECK_GT(head_dim, 0);
    CHECK_EQ(head_dim % 16, 0);
    // Packed payload and E4M3 scales share one owning byte page. Python
    // aliases two planes; page copies/VMM preserve both without extra roles.
    return head_dim / 2 + head_dim / 16;
  }
  return head_dim;
}

}  // namespace xllm
