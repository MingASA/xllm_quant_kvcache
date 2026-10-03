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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/platform.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/mtp_worker_impl.h"
#include "core/runtime/options.h"

namespace xllm {
namespace {

class DecodeMetadataTestWorker final : public MTPWorkerImpl {
 public:
  DecodeMetadataTestWorker(const ParallelArgs& parallel_args,
                           const torch::Device& device,
                           const runtime::Options& options)
      : MTPWorkerImpl(parallel_args, device, options, WorkerType::LLM) {
    context_.set_model_impl("python");
    target_spec_verify_mode_ =
        mtp_async::TargetSpecVerifyMode::DEEPSEEK_V32_EXPANDED_VERIFY;
  }

  void resolve_decode_context(ForwardInput& input) const {
    update_decode_step_input(input,
                             std::vector<EmbeddingCache::DecodeState>(1));
  }

  ForwardInput build_verify(const ForwardInput& input, bool adaptive) {
    ForwardInput verify_input;
    if (adaptive) {
      prepare_validate_inputs(input, verify_input, std::vector<int32_t>{2});
    } else {
      prepare_validate_inputs(input, verify_input);
    }
    CHECK_EQ(prepare_stream_->synchronize(), 0);
    return verify_input;
  }

  ForwardInput build_verify(const ForwardInput& input,
                            const std::vector<int32_t>& verify_widths) {
    ForwardInput verify_input;
    prepare_validate_inputs(input, verify_input, verify_widths);
    CHECK_EQ(prepare_stream_->synchronize(), 0);
    return verify_input;
  }
};

class MtpDecodeMetadataTest : public ::testing::TestWithParam<int32_t> {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "An NPU is required for worker metadata preparation.";
    }
  }

  ParallelArgs parallel_args() const {
    ParallelArgs args(/*rank=*/0, /*world_size=*/2, /*process_group=*/nullptr);
    args.cp_size(1).kv_split_size(GetParam());
    return args;
  }

  runtime::Options options() const {
    runtime::Options options;
    options.model_id("glm_dcp_metadata_test")
        .block_size(128)
        .num_speculative_tokens(1)
        .max_seqs_per_batch(2)
        .world_size(2)
        .dp_size(1)
        .cp_size(1);
    return options;
  }

  ForwardInput make_input(int32_t position,
                          const torch::Tensor& block_tables) const {
    ForwardInput input;
    input.token_ids_host = torch::tensor({42}, torch::kInt);
    input.positions_host = torch::tensor({position}, torch::kInt);
    input.token_ids = input.token_ids_host.to(torch::Device("npu:0"));
    input.positions = input.positions_host.to(torch::Device("npu:0"));
    input.input_params.meta.num_sequences = 1;
    input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    input.input_params.attention.host.q_seq_lens = {1};
    input.input_params.attention.host.kv_seq_lens = {position + 1};
    input.input_params.attention.host.block_tables = block_tables;
    return input;
  }

  void check_verify_across_logical_page(bool adaptive) {
    DecodeMetadataTestWorker worker(
        parallel_args(), torch::Device("npu:0"), options());
    const int32_t page_size = GetParam() == 1 ? 128 : 256;
    ForwardInput input =
        make_input(page_size - 1, torch::tensor({{10, 11}}, torch::kInt));
    worker.resolve_decode_context(input);
    ASSERT_EQ(input.positions_host.item<int32_t>(), page_size - 1);
    const ForwardInput verify_input = worker.build_verify(input, adaptive);
    const auto& params = verify_input.input_params;

    EXPECT_TRUE(
        torch::equal(verify_input.positions_host,
                     torch::tensor({page_size - 1, page_size}, torch::kInt)));
    EXPECT_TRUE(torch::equal(
        params.attention.device.new_cache_slots.cpu(),
        torch::tensor({11 * page_size - 1, 11 * page_size}, torch::kInt)));
    EXPECT_EQ(params.graph.expanded_kv_seq_lens_vec,
              (std::vector<int32_t>{page_size, page_size + 1}));
    EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_indptr.cpu(),
                             torch::tensor({0, 1, 3}, torch::kInt)));
    EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_indices.cpu(),
                             torch::tensor({10, 10, 11}, torch::kInt)));
    EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_last_page_len.cpu(),
                             torch::tensor({page_size, 1}, torch::kInt)));
  }

  void check_linear_state_rows(bool adaptive) {
    DecodeMetadataTestWorker worker(
        parallel_args(), torch::Device("npu:0"), options());
    ForwardInput input;
    input.token_ids_host = torch::tensor({42, 43}, torch::kInt);
    input.positions_host = torch::tensor({5, 9}, torch::kInt);
    input.token_ids = input.token_ids_host.to(torch::Device("npu:0"));
    input.positions = input.positions_host.to(torch::Device("npu:0"));
    input.input_params.meta.num_sequences = 2;
    input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    input.input_params.attention.host.q_seq_lens = {1, 1};
    input.input_params.attention.host.kv_seq_lens = {6, 10};
    input.input_params.attention.host.block_tables =
        torch::tensor({{10, 11}, {20, 21}}, torch::kInt);
    // Include the sentinel used by models without linear-attention layers.
    input.input_params.embedding.linear_state_ids = {7, -1};
    input.input_params.embedding.linear_state_indices =
        torch::tensor({7, -1}, torch::kInt).to(torch::Device("npu:0"));

    const ForwardInput verify_input =
        adaptive ? worker.build_verify(input, std::vector<int32_t>{1, 2})
                 : worker.build_verify(input, /*adaptive=*/false);
    const torch::Tensor expected =
        adaptive ? torch::tensor({7, -1, -1}, torch::kInt)
                 : torch::tensor({7, 7, -1, -1}, torch::kInt);
    const auto& embedding = verify_input.input_params.embedding;
    ASSERT_TRUE(embedding.linear_state_indices.defined());
    EXPECT_TRUE(torch::equal(embedding.linear_state_indices.cpu(), expected));
    EXPECT_EQ(embedding.linear_state_indices.numel(),
              verify_input.token_ids.numel());
    EXPECT_EQ(embedding.linear_state_ids, (std::vector<int32_t>{7, -1}));
    EXPECT_TRUE(
        torch::equal(input.input_params.embedding.linear_state_indices.cpu(),
                     torch::tensor({7, -1}, torch::kInt)));
  }
};

TEST_P(MtpDecodeMetadataTest, Keeps305TokenContextWithinAllocatedPages) {
  DecodeMetadataTestWorker worker(
      parallel_args(), torch::Device("npu:0"), options());
  const torch::Tensor block_tables =
      GetParam() == 1 ? torch::tensor({{10, 11, 12}}, torch::kInt)
                      : torch::tensor({{10, 11}}, torch::kInt);
  ForwardInput input = make_input(/*position=*/305, block_tables);

  worker.resolve_decode_context(input);

  EXPECT_EQ(input.positions_host.item<int32_t>(), 305);
  EXPECT_EQ(input.input_params.attention.host.kv_seq_lens,
            (std::vector<int32_t>{306}));
}

TEST_P(MtpDecodeMetadataTest, BuildsFixedVerifyAcrossLogicalPage) {
  check_verify_across_logical_page(/*adaptive=*/false);
}

TEST_P(MtpDecodeMetadataTest, BuildsAdaptiveVerifyAcrossLogicalPage) {
  check_verify_across_logical_page(/*adaptive=*/true);
}

TEST_P(MtpDecodeMetadataTest, ExpandsFixedVerifyLinearStateRows) {
  check_linear_state_rows(/*adaptive=*/false);
}

TEST_P(MtpDecodeMetadataTest, ExpandsVariableVerifyLinearStateRows) {
  check_linear_state_rows(/*adaptive=*/true);
}

INSTANTIATE_TEST_SUITE_P(KvSplit,
                         MtpDecodeMetadataTest,
                         ::testing::Values(1, 2));

}  // namespace
}  // namespace xllm
