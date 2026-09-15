// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/reconstruction.h"
#include "codec/vardct_frame.h"
#include "codestream/bit_writer.h"
#include "codestream/encoder.h"
#include "codestream/dc_context_tree_internal.h"
#include "codestream/encoder_internal.h"
#include "codestream/headers.h"
#include "core/ac_strategy.h"
#include "core/frame_geometry.h"
#include "core/image.h"
#include "core/quantizer.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"

namespace {

template <typename T>
gjxl::PlaneView<const T> View(
  const std::vector<T>& values, gjxl::Extent2D extent) {
  return {values.data(), extent, extent.width};
}

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D extent) : extent(extent) {
    for (std::vector<float>& values : plane) {
      values.resize(extent.width * extent.height);
    }
  }

  gjxl::ConstImage3FView ConstView() const {
    return gjxl::ConstImage3FView{{
      gjxl::ConstPlaneF32View{plane[0].data(), extent, extent.width},
      gjxl::ConstPlaneF32View{plane[1].data(), extent, extent.width},
      gjxl::ConstPlaneF32View{plane[2].data(), extent, extent.width},
    }};
  }

  gjxl::Extent2D extent;
  std::array<std::vector<float>, 3> plane;
};

gjxl::Status MakeFrame(
  size_t width, size_t height, gjxl::QuantizerParams quantizer_params,
  gjxl::SimpleVarDctCodestreamProfile profile,
  gjxl::VarDctEncoderFrame* frame, bool flat = false) {

  gjxl::FrameGeometry geometry;
  gjxl::Status status =
    gjxl::FrameGeometry::Create(width, height, &geometry);
  if (!status.ok()) {
    return status;
  }

  ImageStorage opsin(geometry.padded_frame());
  for (size_t y = 0; y < opsin.extent.height; ++y) {
    for (size_t x = 0; x < opsin.extent.width; ++x) {
      const size_t index = y * opsin.extent.width + x;
      const float value =
        flat ? 0.0f : static_cast<float>((x * 17 + y * 11 + 5) % 97) * 0.002f;
      opsin.plane[0][index] = value - 0.01f;
      opsin.plane[1][index] = value + 0.02f;
      opsin.plane[2][index] = 1.2f * value - 0.03f;
    }
  }

  const gjxl::Extent2D blocks = geometry.block_grid().blocks;
  size_t block_count = 0;
  if (!blocks.try_area(&block_count)) {
    return gjxl::Status::InvalidArgument("Test block count overflow");
  }
  gjxl::AcStrategyGrid strategies;
  gjxl::Quantizer quantizer;
  gjxl::ColorCorrelationMap color_correlation;
  if (!(status = gjxl::AcStrategyGrid::Create(blocks, &strategies)).ok()) {
    return status;
  }
  strategies.fill_dct8();
  if (!(status = gjxl::Quantizer::Create(quantizer_params, &quantizer)).ok()) {
    return status;
  }
  if (!(status = gjxl::ComputeInitialColorCorrelationMap(
          opsin.ConstView(), &color_correlation))
         .ok()) {
    return status;
  }
  const std::vector<int32_t> raw_quant(block_count, 29);
  std::vector<uint8_t> sharpness(block_count);
  for (size_t index = 0; index < block_count; ++index) {
    sharpness[index] = static_cast<uint8_t>(index % 8);
  }
  return gjxl::ComputeQuantizedCoefficients(
    opsin.ConstView(),
    {
      .geometry = geometry,
      .strategies = &strategies,
      .raw_quant_field = View(raw_quant, blocks),
      .quantizer = &quantizer,
      .color_correlation = &color_correlation,
      .epf_sharpness = View(sharpness, blocks),
    },
    profile, frame);
}

template <size_t Size>
bool HasBytes(
  const gjxl::BitWriter& writer,
  const std::array<uint8_t, Size>& expected) {
  return std::ranges::equal(writer.padded_bytes(), expected);
}

template <size_t Size>
bool CheckCodestreamHeader(
  gjxl::Extent2D extent,
  const std::array<uint8_t, Size>& expected) {

  gjxl::BitWriter writer;
  const gjxl::Status status =
    gjxl::WriteSimpleCodestreamHeader(extent, &writer);
  if (!status.ok() || !writer.byte_aligned() || !HasBytes(writer, expected)) {
    std::cerr << "Codestream-header fixture failed for "
              << extent.width << 'x' << extent.height << '\n';
    return false;
  }
  return true;
}

bool CheckCodestreamAndFrameHeaders() {
  if (!CheckCodestreamHeader(
        {1, 1},
        std::array<uint8_t, 10>{
          0xFF, 0x0A, 0x00, 0x00, 0x00, 0x90, 0x43, 0x28, 0x5A, 0x04}) ||
      !CheckCodestreamHeader(
        {513, 8193},
        std::array<uint8_t, 11>{
          0xFF, 0x0A, 0x04, 0x00, 0x01, 0x01, 0x08, 0x72, 0x08, 0x45, 0x8B}) ||
      !CheckCodestreamHeader(
        {0x3FFFFFFFu, 0x3FFFFFFFu},
        std::array<uint8_t, 15>{
          0xFF, 0x0A, 0xF6, 0xFF, 0xFF, 0xFF, 0xB1, 0xFF,
          0xFF, 0xFF, 0x4F, 0x0E, 0xA1, 0x68, 0x11})) {
    return false;
  }

  gjxl::BitWriter frame;
  if (!gjxl::WriteSimpleFrameHeader({}, &frame).ok() ||
      frame.bits_written() != 33 ||
      !HasBytes(frame, std::array<uint8_t, 5>{0xE0, 0x1B, 0x12, 0x48, 0x00})) {
    std::cerr << "Default frame-header fixture failed\n";
    return false;
  }

  gjxl::SimpleVarDctCodestreamProfile scaled_profile;
  scaled_profile.x_qm_scale = 3;
  scaled_profile.b_qm_scale = 5;
  gjxl::BitWriter scaled_frame;
  if (!gjxl::WriteSimpleFrameHeader(scaled_profile, &scaled_frame).ok() ||
      scaled_frame.bits_written() != 33 ||
      !HasBytes(
        scaled_frame,
        std::array<uint8_t, 5>{0xE0, 0x1B, 0x2B, 0x48, 0x00})) {
    std::cerr << "Scaled frame-header fixture failed\n";
    return false;
  }

  gjxl::BitWriter atomic;
  if (!atomic.WriteBits(3, 5).ok()) {
    return false;
  }
  gjxl::SimpleVarDctCodestreamProfile unsupported;
  unsupported.quantization_matrix_mode =
    gjxl::QuantizationMatrixMode::kCustom;
  if (gjxl::WriteSimpleCodestreamHeader({0, 1}, &atomic).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      gjxl::WriteSimpleFrameHeader(unsupported, &atomic).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      atomic.bits_written() != 3 ||
      !HasBytes(atomic, std::array<uint8_t, 1>{5})) {
    std::cerr << "Rejected header changed the destination\n";
    return false;
  }
  return true;
}

template <size_t Size>
bool CheckQuantizer(
  gjxl::QuantizerParams params, size_t expected_bits,
  const std::array<uint8_t, Size>& expected) {

  gjxl::BitWriter writer;
  if (!gjxl::WriteSimpleQuantizer(params, &writer).ok() ||
      writer.bits_written() != expected_bits || !HasBytes(writer, expected)) {
    std::cerr << "Quantizer fixture failed for " << params.global_scale
              << ", " << params.quant_dc << '\n';
    return false;
  }
  return true;
}

bool CheckQuantizerSelectors() {
  if (!CheckQuantizer({1, 1}, 20, std::array<uint8_t, 3>{0x00, 0x20, 0x00}) ||
      !CheckQuantizer({2048, 16}, 15,
                      std::array<uint8_t, 2>{0xFC, 0x1F}) ||
      !CheckQuantizer({2049, 32}, 20,
                      std::array<uint8_t, 3>{0x01, 0xA0, 0x0F}) ||
      !CheckQuantizer({4096, 33}, 23,
                      std::array<uint8_t, 3>{0xFD, 0x5F, 0x10}) ||
      !CheckQuantizer({4097, 256}, 24,
                      std::array<uint8_t, 3>{0x02, 0x80, 0xFF}) ||
      !CheckQuantizer({8192, 257}, 32,
                      std::array<uint8_t, 4>{0xFE, 0xFF, 0x00, 0x01}) ||
      !CheckQuantizer({8193, 65536}, 36,
                      std::array<uint8_t, 5>{0x03, 0x00, 0xFC, 0xFF, 0x0F}) ||
      !CheckQuantizer({32768, 10}, 25,
                      std::array<uint8_t, 4>{0xFF, 0x7F, 0x95, 0x00})) {
    return false;
  }

  gjxl::BitWriter atomic;
  if (!atomic.WriteBits(3, 5).ok() ||
      gjxl::WriteSimpleQuantizer({0, 10}, &atomic).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      gjxl::WriteSimpleQuantizer({32769, 10}, &atomic).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      atomic.bits_written() != 3 ||
      !HasBytes(atomic, std::array<uint8_t, 1>{5})) {
    std::cerr << "Rejected quantizer changed the destination\n";
    return false;
  }
  return true;
}

uint64_t Fnv1a64(std::span<const uint8_t> bytes) {
  uint64_t hash = 1469598103934665603ull;
  for (uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

bool CheckEncodedFrame(
  size_t width, size_t height, size_t expected_size,
  uint64_t expected_hash, uint16_t expected_order_mask,
  bool expect_custom_candidate,
  bool expect_dc_ans,
  bool expect_ac_ans,
  bool expect_order_ans) {

  // Frozen byte fixtures use the original gradient/full-tree DC policy and
  // the native context-map search. Decoded pixels match the earlier fixtures.
  gjxl::codestream_internal::ScopedDcTreePolicyForTesting legacy(
    gjxl::codestream_internal::DcTreePolicy::kLegacy);
  gjxl::VarDctEncoderFrame frame;
  gjxl::Status status = MakeFrame(width, height, {3541, 10}, {}, &frame);
  std::vector<uint8_t> first;
  std::vector<uint8_t> second;
  std::vector<uint8_t> profiled;
  gjxl::codestream_internal::VarDctCodestreamProfile profile;
  if (status.ok()) {
    status = gjxl::EncodeVarDctCodestream(
      frame,
      {.entropy_behavior =
         gjxl::VarDctEntropyBehavior::kMaximumCompression,
       .dc_prediction = gjxl::VarDctDcPrediction::kGradient},
      &first);
  }
  if (status.ok()) {
    status = gjxl::EncodeVarDctCodestream(
      frame,
      {.entropy_behavior =
         gjxl::VarDctEntropyBehavior::kMaximumCompression,
       .dc_prediction = gjxl::VarDctDcPrediction::kGradient},
      &second);
  }
  if (status.ok()) {
    status = gjxl::codestream_internal::EncodeVarDctCodestreamProfiled(
      frame,
      {.entropy_behavior =
         gjxl::VarDctEntropyBehavior::kMaximumCompression,
       .dc_prediction = gjxl::VarDctDcPrediction::kGradient},
      &profiled, &profile);
  }
  const uint64_t profile_stage_total =
    profile.validation_nanoseconds + profile.dc_tokenization_nanoseconds +
    profile.ac_tokenization_nanoseconds +
    profile.entropy_optimization_nanoseconds +
    profile.section_writing_nanoseconds + profile.assembly_nanoseconds;
  const uint64_t entropy_substage_work =
    profile.entropy_work.prefix_histogram_build_nanoseconds +
    profile.entropy_work.prefix_histogram_cost_nanoseconds +
    profile.entropy_work.prefix_clustering_nanoseconds +
    profile.entropy_work.prefix_code_build_nanoseconds +
    profile.entropy_work.prefix_value_collection_nanoseconds +
    profile.entropy_work.prefix_config_search_nanoseconds +
    profile.entropy_work.prefix_exact_measurement_nanoseconds +
    profile.entropy_work.ans_prefix_validation_nanoseconds +
    profile.entropy_work.ans_value_collection_nanoseconds +
    profile.entropy_work.ans_value_aggregation_nanoseconds +
    profile.entropy_work.ans_prepared_value_validation_nanoseconds +
    profile.entropy_work.ans_uint_config_nanoseconds +
    profile.entropy_work.ans_histogram_build_nanoseconds +
    profile.entropy_work.ans_model_build_nanoseconds +
    profile.entropy_work.ans_token_cost_nanoseconds +
    profile.entropy_work.selection_nanoseconds;
  const uint64_t section_substage_work =
    profile.section_writing_work.model_and_header_nanoseconds +
    profile.section_writing_work.token_write_nanoseconds +
    profile.section_writing_work.candidate_measure_nanoseconds;
  const uint64_t assembly_substage_total =
    profile.assembly.candidate_selection_nanoseconds +
    profile.assembly.section_size_nanoseconds +
    profile.assembly.frame_header_nanoseconds +
    profile.assembly.toc_and_sections_nanoseconds +
    profile.assembly.output_copy_nanoseconds;
  const uint64_t hash = Fnv1a64(first);
  if (!status.ok() || first.size() != expected_size ||
      hash != expected_hash || first != second || first != profiled ||
      profile_stage_total == 0 ||
      profile.block_context_map_work_nanoseconds == 0 ||
      profile.coefficient_order_work_nanoseconds == 0 ||
      profile.coefficient_tokenization_work_nanoseconds == 0 ||
      profile.coefficient_context_materialization_work_nanoseconds == 0 ||
      profile.coefficient_tokenization_pass_count !=
        (expect_custom_candidate ? 2 : 1) ||
      profile.coefficient_token_count == 0 ||
      profile.coefficient_context_materialization_count !=
        profile.coefficient_tokenization_pass_count ||
      profile.coefficient_materialized_token_count !=
        profile.coefficient_token_count ||
      profile.entropy_work.ans_value_collection_nanoseconds != 0 ||
      profile.entropy_work.ans_value_aggregation_nanoseconds != 0 ||
      profile.entropy_work.ans_prepared_value_validation_nanoseconds == 0 ||
      entropy_substage_work == 0 || section_substage_work == 0 ||
      assembly_substage_total == 0 ||
      profile.assembly_nanoseconds < assembly_substage_total ||
      profile.entropy_model_bits == 0 || profile.entropy_token_bits == 0 ||
      profile.dc_entropy_clusters == 0 ||
      profile.ac_entropy_clusters == 0 ||
      profile.dc_entropy_is_ans != expect_dc_ans ||
      profile.ac_entropy_is_ans != expect_ac_ans ||
      profile.coefficient_order_entropy_is_ans != expect_order_ans ||
      profile.natural_candidate_bytes == 0 ||
      profile.block_context_candidate_count != 1 ||
      profile.compact_block_context_candidate_bytes != first.size() ||
      profile.selected_block_context_candidate_index != 0 ||
      profile.selected_block_context_count != 4 ||
      profile.selected_block_context_qf_threshold_count != 0 ||
      profile.selected_coefficient_order_mask != expected_order_mask ||
      (profile.custom_order_candidate_bytes != 0) != expect_custom_candidate ||
      (expected_order_mask == 0 && expect_custom_candidate &&
       profile.natural_candidate_bytes >=
         profile.custom_order_candidate_bytes) ||
      (expected_order_mask != 0 &&
       (profile.custom_order_candidate_bytes != first.size() ||
        profile.natural_candidate_bytes <=
          profile.custom_order_candidate_bytes)) ||
      profile.total_nanoseconds < profile_stage_total || first.size() < 2 ||
      first[0] != 0xFF || first[1] != 0x0A) {
    std::cerr << "Encoded " << width << 'x' << height
              << " fixture failed: " << status.message()
              << ", size=" << first.size() << ", hash=" << hash
              << ", entropy=" << profile.dc_entropy_is_ans << '/'
              << profile.ac_entropy_is_ans << '/'
              << profile.coefficient_order_entropy_is_ans
              << ", candidates=" << profile.natural_candidate_bytes << '/'
              << profile.custom_order_candidate_bytes << '\n';
    return false;
  }
  return true;
}

bool CheckAdaptiveBlockContextSelection() {
  gjxl::VarDctEncoderFrame frame;
  gjxl::Status status = MakeFrame(256, 256, {3541, 10}, {}, &frame);
  std::vector<uint8_t> output;
  gjxl::codestream_internal::VarDctCodestreamProfile profile;
  if (status.ok()) {
    status = gjxl::codestream_internal::EncodeVarDctCodestreamProfiled(
      frame,
      {.entropy_behavior =
         gjxl::VarDctEntropyBehavior::kMaximumCompression},
      &output, &profile);
  }
  if (!status.ok() || profile.block_context_candidate_count != 5 ||
      profile.compact_block_context_candidate_bytes == 0 ||
      output.size() > profile.compact_block_context_candidate_bytes ||
      profile.coefficient_tokenization_pass_count == 0 ||
      profile.coefficient_tokenization_pass_count > 2 ||
      profile.coefficient_context_materialization_count !=
        profile.coefficient_tokenization_pass_count *
          profile.block_context_candidate_count ||
      profile.coefficient_materialized_token_count !=
        profile.coefficient_token_count *
          profile.block_context_candidate_count ||
      profile.selected_block_context_candidate_index >=
        profile.block_context_candidate_count ||
      profile.selected_block_context_count == 0 ||
      profile.selected_block_context_count > 16 ||
      profile.selected_block_context_qf_threshold_count != 0 ||
      !profile.ac_entropy_is_ans) {
    std::cerr << "Adaptive block-context selection failed: "
              << status.message() << ", bytes=" << output.size()
              << ", compact="
              << profile.compact_block_context_candidate_bytes
              << ", selected="
              << profile.selected_block_context_candidate_index << '\n';
    return false;
  }

  std::vector<uint8_t> balanced_output;
  gjxl::codestream_internal::VarDctCodestreamProfile balanced_profile;
  status = gjxl::codestream_internal::EncodeVarDctCodestreamProfiled(
    frame,
    {.entropy_behavior = gjxl::VarDctEntropyBehavior::kBalanced},
    &balanced_output, &balanced_profile);
  if (!status.ok() || balanced_output.empty() ||
      balanced_profile.block_context_candidate_count != 1 ||
      balanced_profile.compact_block_context_candidate_bytes != 0 ||
      balanced_profile.selected_block_context_candidate_index != 0 ||
      balanced_profile.selected_block_context_count == 0 ||
      balanced_profile.coefficient_tokenization_pass_count != 1 ||
      balanced_profile.coefficient_context_materialization_count != 0 ||
      balanced_profile.coefficient_materialized_token_count != 0) {
    std::cerr << "Single block-context selection failed: "
              << status.message() << ", bytes=" << balanced_output.size()
              << ", contexts="
              << balanced_profile.selected_block_context_count << '\n';
    return false;
  }
  return true;
}

bool CheckAssemblyAndDeterminism() {
  // Values are pinned after independent header fixtures and section-layout
  // checks establish the constituent bit encodings.
  return CheckEncodedFrame(
           8, 8, 198, 8908325733842535201ull, 0, false,
           false, false, false) &&
         CheckEncodedFrame(
           64, 9, 1101, 10867806195538497218ull, 0, true,
           false, true, false) &&
         CheckEncodedFrame(
           257, 9, 3677, 7386469112535575388ull, 1, true,
           false, true, false);
}

bool CheckEntropyBehaviorPlumbing() {
  gjxl::VarDctEncoderFrame frame;
  gjxl::Status status = MakeFrame(64, 9, {3541, 10}, {}, &frame);
  for (const gjxl::VarDctEntropyBehavior behavior : {
         gjxl::VarDctEntropyBehavior::kBalanced,
         gjxl::VarDctEntropyBehavior::kHighDensity,
         gjxl::VarDctEntropyBehavior::kRateOptimized,
         gjxl::VarDctEntropyBehavior::kMaximumCompression}) {
    std::vector<uint8_t> output;
    gjxl::codestream_internal::VarDctCodestreamProfile profile;
    status = gjxl::codestream_internal::EncodeVarDctCodestreamProfiled(
      frame, {.entropy_behavior = behavior}, &output, &profile);
    const bool exhaustive =
      behavior == gjxl::VarDctEntropyBehavior::kMaximumCompression;
    if (!status.ok() || output.empty() ||
        profile.entropy_behavior != behavior ||
        profile.coefficient_order_behavior !=
          gjxl::VarDctCoefficientOrderBehavior::kFull ||
        profile.entropy_model_bits == 0 || profile.entropy_token_bits == 0 ||
        (!exhaustive &&
         profile.section_writing_work.candidate_measure_nanoseconds != 0) ||
        profile.coefficient_context_materialization_count !=
          (exhaustive ? profile.coefficient_tokenization_pass_count : 0)) {
      std::cerr << "Entropy behavior plumbing failed: "
                << status.message() << '\n';
      return false;
    }
  }

  const std::vector<uint8_t> sentinel = {9, 8, 7};
  std::vector<uint8_t> output = sentinel;
  if (gjxl::EncodeVarDctCodestream(
        frame,
        {.entropy_behavior =
           static_cast<gjxl::VarDctEntropyBehavior>(99)},
        &output).code() != gjxl::StatusCode::kInvalidArgument ||
      output != sentinel) {
    std::cerr << "Invalid entropy behavior was not rejected atomically\n";
    return false;
  }
  if (gjxl::EncodeVarDctCodestream(
        frame,
        {.coefficient_order_behavior =
           static_cast<gjxl::VarDctCoefficientOrderBehavior>(99)},
        &output).code() != gjxl::StatusCode::kInvalidArgument ||
      output != sentinel) {
    std::cerr << "Invalid coefficient-order behavior was not rejected "
                 "atomically\n";
    return false;
  }

  std::vector<uint8_t> full_order;
  std::vector<uint8_t> sampled_order_first;
  std::vector<uint8_t> sampled_order_second;
  gjxl::codestream_internal::VarDctCodestreamProfile sampled_profile;
  if (!gjxl::EncodeVarDctCodestream(frame, {}, &full_order).ok() ||
      !gjxl::codestream_internal::EncodeVarDctCodestreamProfiled(
        frame,
        {.coefficient_order_behavior =
           gjxl::VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled},
        &sampled_order_first, &sampled_profile).ok() ||
      !gjxl::EncodeVarDctCodestream(
        frame,
        {.coefficient_order_behavior =
           gjxl::VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled},
        &sampled_order_second).ok() ||
      sampled_order_first != sampled_order_second ||
      sampled_order_first == full_order ||
      sampled_profile.coefficient_order_behavior !=
        gjxl::VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled) {
    std::cerr << "Effort-7 coefficient-order sampling failed\n";
    return false;
  }

  std::vector<uint8_t> maximum_full;
  std::vector<uint8_t> maximum_sampled;
  gjxl::codestream_internal::VarDctCodestreamProfile maximum_profile;
  if (!gjxl::EncodeVarDctCodestream(
        frame,
        {.entropy_behavior =
           gjxl::VarDctEntropyBehavior::kMaximumCompression},
        &maximum_full).ok() ||
      !gjxl::codestream_internal::EncodeVarDctCodestreamProfiled(
        frame,
        {.entropy_behavior =
           gjxl::VarDctEntropyBehavior::kMaximumCompression,
         .coefficient_order_behavior =
           gjxl::VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled},
        &maximum_sampled, &maximum_profile).ok() ||
      maximum_sampled != maximum_full ||
      maximum_profile.coefficient_order_behavior !=
        gjxl::VarDctCoefficientOrderBehavior::kFull) {
    std::cerr << "Maximum compression did not force full coefficient orders\n";
    return false;
  }
  return true;
}

bool CheckAtomicRejections() {
  const std::vector<uint8_t> sentinel = {9, 8, 7};
  std::vector<uint8_t> output = sentinel;
  gjxl::VarDctEncoderFrame empty;
  gjxl::codestream_internal::VarDctCodestreamProfile timing_profile;
  timing_profile.total_nanoseconds = 123;
  const auto original_timing_profile = timing_profile;
  if (gjxl::EncodeVarDctCodestream(empty, &output).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      output != sentinel ||
      gjxl::codestream_internal::EncodeVarDctCodestreamProfiled(
        empty, &output, &timing_profile).code() !=
          gjxl::StatusCode::kInvalidArgument ||
      output != sentinel || timing_profile != original_timing_profile) {
    std::cerr << "Rejected empty frame changed the byte output\n";
    return false;
  }

  gjxl::SimpleVarDctCodestreamProfile profile;
  profile.loop_filter.epf_options.iterations = 1;
  gjxl::VarDctEncoderFrame unsupported;
  gjxl::Status status =
    MakeFrame(8, 8, {3541, 10}, profile, &unsupported);
  if (!status.ok() ||
      gjxl::EncodeVarDctCodestream(unsupported, &output).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      output != sentinel ||
      gjxl::EncodeVarDctCodestream(unsupported, nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Rejected profile changed the byte output\n";
    return false;
  }
  return true;
}

bool CheckDeferredCandidatePrimitives() {
  using gjxl::codestream_internal::CandidateSelectionKey;
  using gjxl::codestream_internal::PhysicalSectionSizesFromBitCounts;
  using gjxl::codestream_internal::PreferAllPrefixCandidate;
  using gjxl::codestream_internal::PreferEncodingCandidate;

  if (!PreferAllPrefixCandidate(100, 100) ||
      !PreferAllPrefixCandidate(101, 100) ||
      PreferAllPrefixCandidate(99, 100) ||
      !PreferEncodingCandidate({99, true, 4}, {100, false, 0}) ||
      !PreferEncodingCandidate({100, false, 7}, {100, true, 0}) ||
      !PreferEncodingCandidate({100, false, 2}, {100, false, 3}) ||
      PreferEncodingCandidate({100, true, 0}, {100, false, 7}) ||
      PreferEncodingCandidate({100, false, 3}, {100, false, 2})) {
    std::cerr << "Deferred candidate tie policy changed\n";
    return false;
  }

  std::vector<size_t> sizes;
  const std::array<uint64_t, 2> single_common = {3, 5};
  const std::array<uint64_t, 2> single_ac = {7, 9};
  if (!PhysicalSectionSizesFromBitCounts(
         single_common, single_ac, 1, &sizes).ok() ||
      sizes != std::vector<size_t>{3}) {
    std::cerr << "Single-group deferred padding is incorrect\n";
    return false;
  }

  const std::array<uint64_t, 3> multi_common = {0, 1, 8};
  const std::array<uint64_t, 3> multi_ac = {9, 16, 17};
  if (!PhysicalSectionSizesFromBitCounts(
         multi_common, multi_ac, 2, &sizes).ok() ||
      sizes != std::vector<size_t>({0, 1, 1, 2, 2, 3})) {
    std::cerr << "Multi-group deferred padding is incorrect\n";
    return false;
  }

  const std::vector<size_t> sentinel = {9, 8, 7};
  sizes = sentinel;
  if (PhysicalSectionSizesFromBitCounts(
        single_common, single_ac, 2, &sizes).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      sizes != sentinel) {
    std::cerr << "Rejected deferred dimensions changed their output\n";
    return false;
  }
  const std::array<uint64_t, 2> overflowing_common = {
    std::numeric_limits<uint64_t>::max(), 1};
  sizes = sentinel;
  if (PhysicalSectionSizesFromBitCounts(
        overflowing_common, single_ac, 1, &sizes).code() !=
        gjxl::StatusCode::kOutOfMemory ||
      sizes != sentinel) {
    std::cerr << "Overflowing deferred size changed its output\n";
    return false;
  }
  return true;
}

bool CheckRateOptimizedFallback() {
  using namespace gjxl;
  using namespace gjxl::codestream_internal;
  bool saw_fallback = false, saw_expanded = false;
  for (auto extent : {Extent2D{1, 1}, {8, 8}, {64, 9}, {129, 97}, {257, 259}}) {
    VarDctEncoderFrame frame;
    if (!MakeFrame(extent.width, extent.height, {3541, 10}, {}, &frame,
                   extent.width == 1).ok())
      return false;
    for (auto order : {VarDctCoefficientOrderBehavior::kFull,
                       VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled}) {
      for (auto prediction : {VarDctDcPrediction::kGradient,
                              VarDctDcPrediction::kWeighted}) {
        const VarDctCodestreamOptions baseline_options{
          .coefficient_order_behavior = order, .dc_prediction = prediction};
        auto rate_options = baseline_options;
        rate_options.entropy_behavior = VarDctEntropyBehavior::kRateOptimized;
        std::vector<uint8_t> baseline, candidate, bare;
        VarDctCodestreamProfile baseline_profile, profile;
        if (!EncodeVarDctCodestreamProfiled(frame, baseline_options, &baseline, &baseline_profile).ok() ||
            !EncodeVarDctCodestreamProfiled(frame, rate_options,
              &candidate, &profile).ok() ||
            !EncodeVarDctCodestream(frame, rate_options, &bare).ok() ||
            candidate != bare || profile.balanced_candidate_bytes != baseline.size() ||
            candidate.size() != std::min(profile.balanced_candidate_bytes,
                                         profile.rate_candidate_bytes) ||
            profile.selected_balanced_fallback !=
              (profile.balanced_candidate_bytes <= profile.rate_candidate_bytes) ||
            profile.coefficient_tokenization_pass_count !=
              baseline_profile.coefficient_tokenization_pass_count ||
            profile.coefficient_token_count != baseline_profile.coefficient_token_count ||
            profile.dc_sample_count != baseline_profile.dc_sample_count ||
            profile.dc_leaf_count != baseline_profile.dc_leaf_count ||
            profile.coefficient_order_behavior != order) {
          std::cerr << "Rate search lost whole-stream fallback or work accounting\n";
          return false;
        }
        if (profile.selected_balanced_fallback) {
          saw_fallback = true;
          if (candidate != baseline ||
              profile.entropy_model_bits != baseline_profile.entropy_model_bits ||
              profile.entropy_token_bits != baseline_profile.entropy_token_bits ||
              profile.ac_entropy_clusters != baseline_profile.ac_entropy_clusters)
            return false;
        } else {
          saw_expanded = true;
        }
      }
    }
  }
  if (!saw_fallback || !saw_expanded)
    std::cerr << "Rate fallback fixture did not exercise both selections\n";
  return saw_fallback && saw_expanded;
}

bool CheckManagedSerializerStorage() {
  using namespace gjxl;
  using namespace resource_budget_internal;
  VarDctEncoderFrame frame;
  const auto prepared = MakeFrame(257, 257, {3541, 10}, {}, &frame);
  if (!prepared.ok()) return false;
  for (auto behavior : {VarDctEntropyBehavior::kBalanced,
                        VarDctEntropyBehavior::kHighDensity,
                        VarDctEntropyBehavior::kRateOptimized,
                        VarDctEntropyBehavior::kMaximumCompression}) {
    std::vector<uint8_t> expected;
    {
      thread_budget_internal::EncodeScope threads(1);
      if (!EncodeVarDctCodestream(frame, {.entropy_behavior = behavior}, &expected).ok())
        return false;
    }
    const auto fallback_peak = DefaultResourceBudget().snapshot().peak_backing_bytes;
    // This exercises writers, token/model storage, and joined worker context.
    // The generous test envelope is not a complete serializer memory plan.
    ResourceBudget budget(64 * 1024 * 1024);
    ResourceReservation job;
    if (!budget.Reserve(64 * 1024 * 1024, &job).ok()) return false;
    std::vector<uint8_t> encoded;
    thread_budget_internal::CpuParticipantTracker tracker;
    {
      ResourceContextScope resources({&job, ResourceClass::kSerializer});
      thread_budget_internal::EncodeScope threads(4, &tracker);
      const auto status = EncodeVarDctCodestream(frame, {.entropy_behavior = behavior}, &encoded);
      if (!status.ok()) { std::cerr << status.message() << '\n'; return false; }
    }
    job.Reset();
    if (encoded != expected || tracker.peak() < 2 || budget.snapshot().peak_backing_bytes == 0 ||
        budget.snapshot().committed_bytes() != 0 ||
        DefaultResourceBudget().snapshot().peak_backing_bytes != fallback_peak) {
      std::cerr << "Managed parallel serialization changed bytes or leaked ownership\n";
      return false;
    }
    ResourceBudget tiny(1);
    ResourceReservation rejected;
    if (!tiny.Reserve(1, &rejected).ok()) return false;
    {
      ResourceContextScope resources({&rejected, ResourceClass::kSerializer});
      const std::vector<uint8_t> sentinel{1, 2, 3};
      encoded = sentinel;
      const auto status = EncodeVarDctCodestream(frame, {.entropy_behavior = behavior}, &encoded);
      if (status.code() != StatusCode::kOutOfMemory || encoded != sentinel ||
          tiny.snapshot().total.backing_count != 0 || tiny.snapshot().total.pending_count != 0) {
        std::cerr << "Managed serializer rejection did not preserve caller output\n";
        return false;
      }
    }
    rejected.Reset();
    if (tiny.snapshot().committed_bytes() != 0) return false;
    if (!EncodeVarDctCodestream(frame, {.entropy_behavior = behavior}, &encoded).ok() ||
        encoded != expected) return false;
  }
  return true;
}

bool CheckAdmittedRateFallback() {
  using namespace gjxl;
  using namespace gjxl::codestream_internal;
  using namespace gjxl::thread_budget_internal;
  const VarDctCodestreamOptions options{
    .entropy_behavior = VarDctEntropyBehavior::kRateOptimized};
  bool saw_balanced = false, saw_expanded = false;
  size_t checked = 0;
  for (const auto extent : {Extent2D{1, 1}, {8, 8}, {64, 9},
                            {129, 97}, {257, 259}}) {
    VarDctEncoderFrame frame;
    if (!MakeFrame(extent.width, extent.height, {3541, 10}, {}, &frame,
                   extent.width == 1).ok()) return false;
    std::vector<uint8_t> expected;
    VarDctCodestreamProfile expected_profile;
    {
      EncodeScope serial(1);
      if (!EncodeVarDctCodestreamProfiled(
            frame, options, &expected, &expected_profile).ok()) return false;
    }
    saw_balanced |= expected_profile.selected_balanced_fallback;
    saw_expanded |= !expected_profile.selected_balanced_fallback;
    for (size_t limit : {1, 2, 4, 8}) {
      for (size_t requested : {0, 1, 2, 8}) {
        std::shared_ptr<const ExecutionDomain> domain;
        if (!ExecutionDomain::Create({0, limit}, &domain).ok()) return false;
        CpuParticipantTracker tracker;
        std::vector<uint8_t> actual;
        VarDctCodestreamProfile profile;
        {
          CpuExecutionScope execution;
          if (!execution.Start(domain, requested).ok()) return false;
          EncodeScope threads(requested, &tracker);
          if (!EncodeVarDctCodestreamProfiled(
                frame, options, &actual, &profile).ok()) return false;
        }
        const auto snapshot = domain->snapshot();
        const size_t cap = requested == 0 ? limit : std::min(limit, requested);
        if (actual != expected || tracker.peak() == 0 || tracker.peak() > cap ||
            tracker.active() != 0 || snapshot.active_cpu_participants != 0 ||
            snapshot.reserved_cpu_workers != 0 ||
            snapshot.suspended_cpu_workers != 0 ||
            snapshot.waiting_cpu_callers != 0 ||
            snapshot.peak_cpu_protected_slots > cap ||
            profile.balanced_candidate_bytes != expected_profile.balanced_candidate_bytes ||
            profile.rate_candidate_bytes != expected_profile.rate_candidate_bytes ||
            profile.selected_balanced_fallback != expected_profile.selected_balanced_fallback ||
            profile.entropy_model_bits != expected_profile.entropy_model_bits ||
            profile.entropy_token_bits != expected_profile.entropy_token_bits ||
            profile.coefficient_token_count != expected_profile.coefficient_token_count ||
            profile.coefficient_tokenization_pass_count != 1 ||
            profile.entropy_optimization_nanoseconds + profile.section_writing_nanoseconds +
              profile.assembly_nanoseconds > profile.total_nanoseconds) {
          std::cerr << "Admitted rate search changed bytes, accounting or CPU limits\n";
          return false;
        }
        ++checked;
      }
    }
  }
  // A one-block frame has no parallel tokenization. Its first serializer
  // worker is the new policy dispatcher; failing that launch must be atomic.
  VarDctEncoderFrame frame;
  if (!MakeFrame(1, 1, {3541, 10}, {}, &frame, true).ok()) return false;
  for (auto kind : {WorkerLaunchFailureKind::kSystemError,
                    WorkerLaunchFailureKind::kBadAlloc}) {
    std::shared_ptr<const ExecutionDomain> domain;
    if (!ExecutionDomain::Create({0, 4}, &domain).ok()) return false;
    const std::vector<uint8_t> sentinel{17, 19};
    auto output = sentinel;
    VarDctCodestreamProfile profile;
    profile.total_nanoseconds = 123;
    profile.balanced_candidate_bytes = 456;
    profile.rate_candidate_bytes = 789;
    WorkerLaunchFaultForTesting fault{
      WorkerLaunchSite::kSerializerSections, 0, kind};
    Status status;
    {
      CpuExecutionScope execution;
      if (!execution.Start(domain, 4).ok()) return false;
      EncodeScope threads(4);
      WorkerLaunchFaultScopeForTesting failure(&fault);
      status = EncodeVarDctCodestreamProfiled(frame, options, &output, &profile);
    }
    const auto snapshot = domain->snapshot();
    const auto expected_code = kind == WorkerLaunchFailureKind::kBadAlloc
      ? StatusCode::kOutOfMemory : StatusCode::kInternal;
    if (!fault.triggered || status.code() != expected_code || output != sentinel ||
        profile.total_nanoseconds != 123 || profile.balanced_candidate_bytes != 456 ||
        profile.rate_candidate_bytes != 789 || snapshot.active_cpu_participants != 0 ||
        snapshot.reserved_cpu_workers != 0 || snapshot.suspended_cpu_workers != 0) {
      std::cerr << "Rate policy launch failure changed output or leaked CPU capacity\n";
      return false;
    }
  }
  if (!saw_balanced || !saw_expanded) {
    std::cerr << "Admitted rate fixtures must exercise both whole-file selections\n";
    return false;
  }
  std::cout << "Admitted rate fallback cases: " << checked
            << "; atomic policy launch failures: 2\n";
  return true;
}

bool CheckDefaultDcPolicy() {
  using namespace gjxl;
  using namespace gjxl::codestream_internal;
  if (VarDctCodestreamOptions{}.dc_prediction != VarDctDcPrediction::kWeighted ||
      CurrentDcTreePolicy() != DcTreePolicy::kAdaptive) {
    std::cerr << "Default DC policy is not weighted/adaptive\n";
    return false;
  }
  const std::array<Extent2D, 4> extents{{{8, 8}, {129, 97}, {384, 384}, {768, 512}}};
  const std::array<uint32_t, 4> expected_leaves{{1, 2, 4, 34}};
  for (size_t i = 0; i < extents.size(); ++i) {
    VarDctEncoderFrame frame;
    if (!MakeFrame(extents[i].width, extents[i].height, {3541, 10}, {}, &frame).ok())
      return false;
    std::vector<uint8_t> defaults, explicit_weighted, profiled;
    if (!EncodeVarDctCodestream(frame, &defaults).ok()) return false;
    {
      ScopedDcTreePolicyForTesting adaptive(DcTreePolicy::kAdaptive);
      if (!EncodeVarDctCodestream(
            frame, {.dc_prediction = VarDctDcPrediction::kWeighted},
            &explicit_weighted).ok()) return false;
    }
    VarDctCodestreamProfile profile;
    if (!EncodeVarDctCodestreamProfiled(frame, {}, &profiled, &profile).ok() ||
        defaults != explicit_weighted || defaults != profiled ||
        profile.dc_leaf_count != expected_leaves[i] ||
        profile.dc_context_count != expected_leaves[i] + 11) {
      std::cerr << "Default DC encoding differs from explicit weighted/adaptive\n";
      return false;
    }
  }
  return true;
}

bool CheckAdaptiveDcTrees() {
  using namespace gjxl;
  using namespace gjxl::codestream_internal;
  for (auto extent : {Extent2D{8,8}, {129,97}, {257,193}, {384,384},
                      {768,512}, {2049,9}, {2049,257}}) {
    VarDctEncoderFrame frame;
    if (!MakeFrame(extent.width, extent.height, {3541,10}, {}, &frame).ok()) return false;
    const auto before_dc = frame.quantized_dc();
    std::array<std::vector<int32_t>, 3> original_dc;
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < before_dc.plane[c].extent.height; ++y)
        original_dc[c].insert(original_dc[c].end(), before_dc.plane[c].Row(y),
            before_dc.plane[c].Row(y) + before_dc.plane[c].extent.width);
    for (auto prediction : {VarDctDcPrediction::kGradient, VarDctDcPrediction::kWeighted})
      for (auto behavior : {VarDctEntropyBehavior::kBalanced, VarDctEntropyBehavior::kHighDensity,
                            VarDctEntropyBehavior::kRateOptimized,
                            VarDctEntropyBehavior::kMaximumCompression}) {
        VarDctCodestreamOptions options{.entropy_behavior = behavior, .dc_prediction = prediction};
        std::vector<uint8_t> legacy, adaptive, repeated, profiled;
        { ScopedDcTreePolicyForTesting choice(DcTreePolicy::kLegacy);
          if (!EncodeVarDctCodestream(frame, options, &legacy).ok()) return false; }
        VarDctCodestreamProfile profile;
        { ScopedDcTreePolicyForTesting choice(DcTreePolicy::kAdaptive);
          if (!EncodeVarDctCodestream(frame, options, &adaptive).ok() ||
              !EncodeVarDctCodestream(frame, options, &repeated).ok() ||
              !EncodeVarDctCodestreamProfiled(frame, options, &profiled, &profile).ok()) return false; }
        size_t samples;
        const DcContextTreeLayout* layout;
        if (!ComputeDcSampleCount(extent.ceil_div(8), &samples).ok() ||
            !SelectDcContextTreeLayout(samples, prediction, DcTreePolicy::kAdaptive, &layout).ok() ||
            profile.dc_sample_count != samples || profile.dc_leaf_count != layout->dc_leaf_count ||
            profile.dc_context_count != layout->context_count || adaptive != repeated || adaptive != profiled ||
            (layout->full_tree() && legacy != adaptive)) {
          std::cerr << "Adaptive DC serialization mismatch " << extent.width << 'x' << extent.height << '\n';
          return false;
        }
      }
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < before_dc.plane[c].extent.height; ++y)
        if (!std::equal(before_dc.plane[c].Row(y), before_dc.plane[c].Row(y) + before_dc.plane[c].extent.width,
            original_dc[c].data() + y * before_dc.plane[c].extent.width)) return false;
  }
  ScopedDcTreePolicyForTesting choice(DcTreePolicy::kAdaptive);
  return CheckManagedSerializerStorage() && CheckAtomicRejections();
}

}  // namespace

int main() {
  if (!CheckCodestreamAndFrameHeaders() || !CheckQuantizerSelectors() ||
      !CheckAssemblyAndDeterminism() || !CheckAdaptiveBlockContextSelection() ||
      !CheckEntropyBehaviorPlumbing() || !CheckAtomicRejections() ||
      !CheckRateOptimizedFallback() || !CheckDeferredCandidatePrimitives() ||
      !CheckManagedSerializerStorage() || !CheckAdmittedRateFallback() ||
      !CheckDefaultDcPolicy() || !CheckAdaptiveDcTrees()) {
    return EXIT_FAILURE;
  }
  std::cout << "All codestream encoder tests passed.\n";
  return EXIT_SUCCESS;
}
