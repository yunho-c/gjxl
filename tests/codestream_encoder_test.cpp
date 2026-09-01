// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/reconstruction.h"
#include "codec/vardct_frame.h"
#include "codestream/bit_writer.h"
#include "codestream/encoder.h"
#include "codestream/encoder_internal.h"
#include "codestream/headers.h"
#include "core/ac_strategy.h"
#include "core/frame_geometry.h"
#include "core/image.h"
#include "core/quantizer.h"

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
  gjxl::VarDctEncoderFrame* frame) {

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
        static_cast<float>((x * 17 + y * 11 + 5) % 97) * 0.002f;
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

  gjxl::VarDctEncoderFrame frame;
  gjxl::Status status = MakeFrame(width, height, {3541, 10}, {}, &frame);
  std::vector<uint8_t> first;
  std::vector<uint8_t> second;
  std::vector<uint8_t> profiled;
  gjxl::codestream_internal::VarDctCodestreamProfile profile;
  if (status.ok()) {
    status = gjxl::EncodeVarDctCodestream(frame, &first);
  }
  if (status.ok()) {
    status = gjxl::EncodeVarDctCodestream(frame, &second);
  }
  if (status.ok()) {
    status = gjxl::codestream_internal::EncodeVarDctCodestreamProfiled(
      frame, &profiled, &profile);
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
      frame, &output, &profile);
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
  return true;
}

bool CheckAssemblyAndDeterminism() {
  // Values are pinned after independent header fixtures and section-layout
  // checks establish the constituent bit encodings.
  return CheckEncodedFrame(
           8, 8, 203, 7880082076206412069ull, 0, false,
           false, false, false) &&
         CheckEncodedFrame(
           64, 9, 1107, 17820242185032511216ull, 0, true,
           false, true, false) &&
         CheckEncodedFrame(
           257, 9, 3851, 18124942738510227601ull, 1, true,
           false, true, false);
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
  profile.loop_filter.gaborish = false;
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

}  // namespace

int main() {
  if (!CheckCodestreamAndFrameHeaders() || !CheckQuantizerSelectors() ||
      !CheckAssemblyAndDeterminism() || !CheckAdaptiveBlockContextSelection() ||
      !CheckAtomicRejections() || !CheckDeferredCandidatePrimitives()) {
    return EXIT_FAILURE;
  }
  std::cout << "All codestream encoder tests passed.\n";
  return EXIT_SUCCESS;
}
