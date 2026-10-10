/* Copyright 2026 The xLLM Authors.
 * Licensed under Apache-2.0. Independent FlashInfer-derived mixed-V prefill.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
// Isolate every CUDA symbol from the installed, unmodified FlashInfer.
#define flashinfer xllm_flashinfer_mixed_prefill
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>

namespace mixed_v {
using BasePrefill = flashinfer::BatchPrefillPagedParams<__nv_bfloat16,
                                                        __nv_bfloat16,
                                                        __nv_bfloat16,
                                                        int32_t>;
template <int32_t Mode>
class PrefillParams : public BasePrefill {
 public:
  static constexpr bool kMixedV = Mode != 0;
  static constexpr int32_t kVMode = Mode;
  const uint8_t* v_bytes = nullptr;
  const float* v_token_scales = nullptr;
};

template <int32_t TileQ, int32_t Mode>
cudaError_t launch_prefill(PrefillParams<Mode> params,
                           __nv_bfloat16* temporary,
                           float* temporary_lse,
                           cudaStream_t stream) {
  using Variant = flashinfer::DefaultAttention<false, false, false, false>;
  constexpr uint32_t kWarpsQ = flashinfer::get_num_warps_q(TileQ);
  constexpr uint32_t kWarpsKV = flashinfer::get_num_warps_kv(TileQ);
  // Exact BF16/head128 heuristic on this SM120 100-KiB shared-memory device.
  constexpr uint32_t kMmaKV = TileQ == 128 ? 2 : TileQ == 64 ? 4 : 1;
  using Traits = flashinfer::KernelTraits<flashinfer::MaskMode::kCausal,
                                          TileQ,
                                          flashinfer::get_num_mma_q(TileQ),
                                          kMmaKV,
                                          8,
                                          8,
                                          kWarpsQ,
                                          kWarpsKV,
                                          flashinfer::PosEncodingMode::kNone,
                                          __nv_bfloat16,
                                          __nv_bfloat16,
                                          __nv_bfloat16,
                                          float,
                                          int32_t,
                                          Variant>;
  static_assert(!Traits::IsInvalid());
  auto kernel =
      flashinfer::BatchPrefillWithPagedKVCacheKernel<Traits,
                                                     PrefillParams<Mode>>;
  cudaError_t status =
      cudaFuncSetAttribute(kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           sizeof(typename Traits::SharedStoragePaged));
  if (status != cudaSuccess) {
    return status;
  }
  __nv_bfloat16* output = params.o;
  if (temporary != nullptr) {
    params.partition_kv = true;
    params.o = temporary;
    params.lse = temporary_lse;
  }
  void* arguments[] = {&params};
  status = cudaLaunchKernel(reinterpret_cast<const void*>(kernel),
                            dim3(params.padded_batch_size, 1, 2),
                            dim3(32, kWarpsQ, kWarpsKV),
                            arguments,
                            sizeof(typename Traits::SharedStoragePaged),
                            stream);
  if (status != cudaSuccess || temporary == nullptr) {
    return status;
  }
  return flashinfer::VariableLengthMergeStates(temporary,
                                               temporary_lse,
                                               params.merge_indptr,
                                               output,
                                               /*lse=*/nullptr,
                                               params.max_total_num_rows,
                                               params.total_num_rows,
                                               /*num_heads=*/12,
                                               /*head_dim=*/128,
                                               /*enable_pdl=*/false,
                                               stream);
}

template <int32_t Mode>
cudaError_t prefill(void* query,
                    void* key,
                    void* value,
                    void* scales,
                    void* output,
                    void* q_indptr,
                    void* kv_indptr,
                    void* indices,
                    void* tails,
                    int32_t batch,
                    int32_t page_size,
                    float sm_scale,
                    cudaStream_t stream,
                    void* float_workspace,
                    void* int_workspace,
                    const int64_t* plan) {
  PrefillParams<Mode> params;
  params.q = static_cast<__nv_bfloat16*>(query);
  params.o = static_cast<__nv_bfloat16*>(output);
  params.paged_kv = flashinfer::paged_kv_t<__nv_bfloat16, int32_t>(
      /*num_heads=*/2,
      page_size,
      /*head_dim=*/128,
      batch,
      flashinfer::QKVLayout::kNHD,
      static_cast<__nv_bfloat16*>(key),
      Mode == 0 ? static_cast<__nv_bfloat16*>(value) : nullptr,
      static_cast<int32_t*>(indices),
      static_cast<int32_t*>(kv_indptr),
      static_cast<int32_t*>(tails));
  params.v_bytes = static_cast<const uint8_t*>(value);
  params.v_token_scales = static_cast<const float*>(scales);
  params.q_indptr = static_cast<int32_t*>(q_indptr);
  params.num_qo_heads = 12;
  params.group_size = flashinfer::uint_fastdiv(6);
  params.q_stride_n = 12 * 128;
  params.q_stride_h = 128;
  params.window_left = -1;
  params.sm_scale = sm_scale;
  auto* integers = static_cast<uint8_t*>(int_workspace);
  auto* floats = static_cast<uint8_t*>(float_workspace);
  params.padded_batch_size = plan[0];
  params.max_total_num_rows = plan[1];
  params.request_indices = reinterpret_cast<int32_t*>(integers + plan[4]);
  params.qo_tile_indices = reinterpret_cast<int32_t*>(integers + plan[5]);
  params.kv_tile_indices = reinterpret_cast<int32_t*>(integers + plan[6]);
  params.o_indptr = reinterpret_cast<int32_t*>(integers + plan[8]);
  params.kv_chunk_size_ptr = reinterpret_cast<int32_t*>(integers + plan[9]);
  __nv_bfloat16* temporary = nullptr;
  float* temporary_lse = nullptr;
  if (plan[14]) {
    params.merge_indptr = reinterpret_cast<int32_t*>(integers + plan[7]);
    temporary = reinterpret_cast<__nv_bfloat16*>(floats + plan[10]);
    temporary_lse = reinterpret_cast<float*>(floats + plan[11]);
  }
  if (plan[13]) {
    return cudaErrorNotSupported;
  }
  switch (plan[3]) {
#define LAUNCH_TILE(TILE) \
  case TILE:              \
    return launch_prefill<TILE>(params, temporary, temporary_lse, stream)
    LAUNCH_TILE(16);
    LAUNCH_TILE(64);
    LAUNCH_TILE(128);
#undef LAUNCH_TILE
    default:
      return cudaErrorInvalidValue;
  }
}
}  // namespace mixed_v

extern "C" int32_t mixed_prefill(void* query,
                                 void* key,
                                 void* value,
                                 void* scales,
                                 void* output,
                                 void* q_indptr,
                                 void* kv_indptr,
                                 void* indices,
                                 void* tails,
                                 int32_t batch,
                                 int32_t page_size,
                                 float sm_scale,
                                 int32_t mode,
                                 void* stream,
                                 void* float_workspace,
                                 void* int_workspace,
                                 const int64_t* plan) {
  if (batch <= 0 || page_size <= 0 || plan == nullptr) {
    return cudaErrorInvalidValue;
  }
  switch (mode) {
#define LAUNCH_MODE(MODE)                                            \
  case MODE:                                                         \
    return mixed_v::prefill<MODE>(query,                             \
                                  key,                               \
                                  value,                             \
                                  scales,                            \
                                  output,                            \
                                  q_indptr,                          \
                                  kv_indptr,                         \
                                  indices,                           \
                                  tails,                             \
                                  batch,                             \
                                  page_size,                         \
                                  sm_scale,                          \
                                  static_cast<cudaStream_t>(stream), \
                                  float_workspace,                   \
                                  int_workspace,                     \
                                  plan)
    LAUNCH_MODE(0);
    LAUNCH_MODE(1);
    LAUNCH_MODE(2);
    LAUNCH_MODE(3);
#undef LAUNCH_MODE
    default:
      return cudaErrorInvalidValue;
  }
}
