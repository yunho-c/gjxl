// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates the encoder-facing VarDCT frame and fixed AC-group layout.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/reconstruction.h"
#include "codec/vardct_frame_internal.h"

namespace {

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D extent) : extent(extent) {
    for (std::vector<float>& values : plane) {
      values.resize(extent.width * extent.height);
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return gjxl::ConstImage3FView{{
      gjxl::ConstPlaneF32View{plane[0].data(), extent, extent.width},
      gjxl::ConstPlaneF32View{plane[1].data(), extent, extent.width},
      gjxl::ConstPlaneF32View{plane[2].data(), extent, extent.width},
    }};
  }

  gjxl::Extent2D extent;
  std::array<std::vector<float>, 3> plane;
};

void FillSignal(ImageStorage* image) {
  for (size_t y = 0; y < image->extent.height; ++y) {
    for (size_t x = 0; x < image->extent.width; ++x) {
      const float value =
        0.2f * std::sin(0.031f * static_cast<float>(x + 3 * y)) +
        0.1f * std::cos(0.047f * static_cast<float>(2 * x + y));
      const size_t index = y * image->extent.width + x;
      image->plane[0][index] = 0.7f * value + 0.01f;
      image->plane[1][index] = value;
      image->plane[2][index] = 1.2f * value - 0.02f;
    }
  }
}

gjxl::Status Encode(
  gjxl::Extent2D original_extent,
  gjxl::AcStrategyGrid* strategies,
  gjxl::VarDctEncoderFrame* frame,
  gjxl::SimpleVarDctCodestreamProfile profile = {},
  bool mutate_inputs_afterward = false) {

  gjxl::FrameGeometry geometry;
  gjxl::Status status = gjxl::FrameGeometry::Create(
    original_extent, &geometry);
  if (!status.ok()) {
    return status;
  }
  ImageStorage opsin(geometry.padded_frame());
  FillSignal(&opsin);

  const gjxl::Extent2D blocks = geometry.block_grid().blocks;
  if (!strategies->valid()) {
    status = gjxl::AcStrategyGrid::Create(blocks, strategies);
    if (!status.ok()) {
      return status;
    }
    strategies->fill_dct8();
  }

  size_t block_count = 0;
  if (!blocks.try_area(&block_count)) {
    return gjxl::Status::InvalidArgument("Test block grid is too large");
  }
  std::vector<int32_t> raw_quant(block_count, 29);
  std::vector<uint8_t> epf_sharpness(block_count, 4);
  gjxl::Quantizer quantizer;
  gjxl::ColorCorrelationMap color_correlation;
  status = gjxl::Quantizer::Create({3541, 10}, &quantizer);
  if (!status.ok()) {
    return status;
  }
  status = gjxl::ComputeInitialColorCorrelationMap(
    opsin.ConstView(), &color_correlation);
  if (!status.ok()) {
    return status;
  }

  status = gjxl::ComputeQuantizedCoefficients(
    opsin.ConstView(),
    {
      .geometry = geometry,
      .strategies = strategies,
      .raw_quant_field = {raw_quant.data(), blocks, blocks.width},
      .quantizer = &quantizer,
      .color_correlation = &color_correlation,
      .epf_sharpness = {epf_sharpness.data(), blocks, blocks.width},
    },
    profile,
    frame);
  if (!status.ok() || !mutate_inputs_afterward) {
    return status;
  }

  strategies->clear();
  std::ranges::fill(raw_quant, 0);
  std::ranges::fill(epf_sharpness, 0);
  quantizer = {};
  color_correlation = {};
  return gjxl::Status::Ok();
}

bool CheckGroup(
  const gjxl::VarDctEncoderFrame& frame,
  size_t group_index,
  size_t block_x,
  size_t block_y,
  gjxl::Extent2D block_extent,
  size_t used) {

  gjxl::VarDctAcGroupView group;
  if (!frame.GetAcGroup(group_index, &group).ok() ||
      group.block_x != block_x ||
      group.block_y != block_y ||
      group.block_extent != block_extent ||
      group.used_coefficient_count != used) {
    return false;
  }
  for (std::span<const int32_t> channel : group.coefficients) {
    if (channel.size() != gjxl::kVarDctAcGroupCoefficientCapacity ||
        !std::ranges::all_of(
          channel.subspan(used),
          [](int32_t value) { return value == 0; })) {
      return false;
    }
  }
  return true;
}

bool CheckOneBlockAndOwnership() {
  gjxl::AcStrategyGrid strategies;
  gjxl::VarDctEncoderFrame frame;
  const gjxl::Status status = Encode({8, 8}, &strategies, &frame, {}, true);
  if (!status.ok() ||
      !frame.valid() ||
      frame.geometry().frame() != gjxl::Extent2D{8, 8} ||
      frame.strategies().complete() == false ||
      frame.raw_quant_field().Row(0)[0] != 29 ||
      frame.epf_sharpness().Row(0)[0] != 4 ||
      frame.profile() != gjxl::SimpleVarDctCodestreamProfile{} ||
      !frame.quantized_dc().valid() ||
      frame.ac_group_extent() != gjxl::Extent2D{1, 1} ||
      !CheckGroup(frame, 0, 0, 0, {1, 1}, 64)) {
    std::cerr << "One-block frame or deep ownership is invalid: "
              << status.message() << '\n';
    return false;
  }
  return true;
}

bool CheckExactGroup() {
  gjxl::AcStrategyGrid strategies;
  gjxl::VarDctEncoderFrame frame;
  const gjxl::Status status = Encode({256, 256}, &strategies, &frame);
  if (!status.ok() ||
      !frame.valid() ||
      frame.ac_group_count() != 1 ||
      !CheckGroup(
        frame, 0, 0, 0, {32, 32},
        gjxl::kVarDctAcGroupCoefficientCapacity)) {
    std::cerr << "Exact 256x256 group is invalid: "
              << status.message() << '\n';
    return false;
  }
  return true;
}

bool CheckEdgeGroups() {
  gjxl::AcStrategyGrid strategies;
  gjxl::VarDctEncoderFrame frame;
  const gjxl::Status status = Encode({257, 257}, &strategies, &frame);
  if (!status.ok() ||
      !frame.valid() ||
      frame.geometry().padded_frame() != gjxl::Extent2D{264, 264} ||
      frame.ac_group_extent() != gjxl::Extent2D{2, 2} ||
      frame.ac_group_count() != 4 ||
      !CheckGroup(frame, 0, 0, 0, {32, 32}, 65536) ||
      !CheckGroup(frame, 1, 32, 0, {1, 32}, 2048) ||
      !CheckGroup(frame, 2, 0, 32, {32, 1}, 2048) ||
      !CheckGroup(frame, 3, 32, 32, {1, 1}, 64)) {
    std::cerr << "Edge-group layout is invalid: "
              << status.message() << '\n';
    return false;
  }
  return true;
}

bool CheckCrossingStrategyIsRejectedAtomically() {
  gjxl::AcStrategyGrid crossing;
  if (!gjxl::AcStrategyGrid::Create({33, 1}, &crossing).ok() ||
      !crossing.Set(31, 0, gjxl::AcStrategyType::kDct8x16).ok()) {
    return false;
  }
  crossing.fill_empty_dct8();

  gjxl::AcStrategyGrid valid;
  gjxl::VarDctEncoderFrame frame;
  if (!Encode({8, 8}, &valid, &frame).ok()) {
    return false;
  }
  const gjxl::Status status = Encode({264, 8}, &crossing, &frame);
  if (status.ok() ||
      !frame.valid() ||
      frame.geometry().frame() != gjxl::Extent2D{8, 8} ||
      frame.raw_quant_field().Row(0)[0] != 29) {
    std::cerr << "Group-crossing strategy was accepted or changed output\n";
    return false;
  }
  return true;
}

bool CheckMatrixScaleBoundsAndAtomicRejection() {
  gjxl::VarDctEncoderFrame frame;
  gjxl::AcStrategyGrid strategies;
  gjxl::SimpleVarDctCodestreamProfile profile;
  profile.x_qm_scale = 0;
  profile.b_qm_scale = 7;
  if (!Encode({8, 8}, &strategies, &frame, profile).ok() ||
      !frame.valid() || frame.profile() != profile ||
      !gjxl::ValidateSimpleCodestreamFrame(frame).ok()) {
    std::cerr << "Representable matrix-scale bounds were not retained\n";
    return false;
  }

  const gjxl::VarDctEncoderFrame sentinel = frame;
  profile.x_qm_scale = 8;
  if (Encode({8, 8}, &strategies, &frame, profile).ok() ||
      !frame.valid() || frame.profile() != sentinel.profile() ||
      frame.profile().x_qm_scale != 0) {
    std::cerr << "Out-of-range matrix scale changed the prior frame\n";
    return false;
  }
  return true;
}

bool CheckQuantizedAssemblyValidationAndAtomicity() {
  gjxl::FrameGeometry geometry;
  gjxl::AcStrategyGrid strategies;
  gjxl::Quantizer quantizer;
  if (!gjxl::FrameGeometry::Create({8, 8}, &geometry).ok() ||
      !gjxl::AcStrategyGrid::Create({1, 1}, &strategies).ok() ||
      !gjxl::Quantizer::Create({3541, 10}, &quantizer).ok()) {
    return false;
  }
  strategies.fill_dct8();
  std::array<int32_t, 1> raw_quant{29};
  std::array<uint8_t, 1> epf_sharpness{4};
  std::array<int8_t, 1> y_to_x{0};
  std::array<int8_t, 1> y_to_b{0};
  std::array<int32_t, 3> quantized_dc{1, 2, 3};
  std::vector<int32_t> quantized_ac(3 * 64);
  for (size_t index = 0; index < quantized_ac.size(); ++index) {
    quantized_ac[index] = static_cast<int32_t>(index) - 73;
  }
  const std::array<
    gjxl::vardct_frame_internal::QuantizedAcTransformLayout, 1> transforms{{{
      .block_x = 0,
      .block_y = 0,
      .strategy = gjxl::AcStrategyType::kDct8,
      .coefficient_count = 64,
      .coefficient_offsets = {0, 64, 128},
  }}};
  const auto assemble = [&](gjxl::VarDctEncoderFrame* frame) {
    return gjxl::vardct_frame_internal::AssembleVarDctEncoderFrame(
      {
        .geometry = geometry,
        .strategies = &strategies,
        .raw_quant_field = {raw_quant.data(), {1, 1}, 1},
        .quantizer = &quantizer,
        .y_to_x = {y_to_x.data(), {1, 1}, 1},
        .y_to_b = {y_to_b.data(), {1, 1}, 1},
        .epf_sharpness = {epf_sharpness.data(), {1, 1}, 1},
        .profile = {},
        .quantized_dc = {{{
          {quantized_dc.data(), {1, 1}, 1},
          {quantized_dc.data() + 1, {1, 1}, 1},
          {quantized_dc.data() + 2, {1, 1}, 1},
        }}},
        .quantized_ac = quantized_ac,
        .transforms = transforms,
        .reject_unwritten_coefficients = true,
      },
      frame);
  };

  gjxl::VarDctEncoderFrame frame;
  if (!assemble(&frame).ok() || !frame.valid()) {
    std::cerr << "Checked quantized assembly rejected valid input\n";
    return false;
  }
  gjxl::VarDctAcGroupView group;
  if (!frame.GetAcGroup(0, &group).ok() ||
      group.coefficients[0][17] != quantized_ac[17]) {
    std::cerr << "Checked quantized assembly changed coefficient order\n";
    return false;
  }
  const int32_t retained = group.coefficients[0][17];

  quantized_ac[17] =
    gjxl::vardct_frame_internal::kUnwrittenQuantizedCoefficient;
  if (assemble(&frame).ok() || !frame.valid() ||
      !frame.GetAcGroup(0, &group).ok() ||
      group.coefficients[0][17] != retained) {
    std::cerr << "Unwritten AC did not preserve the prior frame atomically\n";
    return false;
  }
  quantized_ac[17] = retained;
  quantized_dc[1] =
    gjxl::vardct_frame_internal::kUnwrittenQuantizedCoefficient;
  if (assemble(&frame).ok() || !frame.valid() ||
      frame.quantized_dc().plane[1].Row(0)[0] != 2) {
    std::cerr << "Unwritten DC did not preserve the prior frame atomically\n";
    return false;
  }
  quantized_dc[1] = 2;
  raw_quant[0] = 0;
  if (assemble(&frame).ok() || !frame.valid() ||
      frame.raw_quant_field().Row(0)[0] != 29) {
    std::cerr << "Invalid raw quant changed the prior frame\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckOneBlockAndOwnership() ||
      !CheckExactGroup() ||
      !CheckEdgeGroups() ||
      !CheckCrossingStrategyIsRejectedAtomically() ||
      !CheckMatrixScaleBoundsAndAtomicRejection() ||
      !CheckQuantizedAssemblyValidationAndAtomicity()) {
    return EXIT_FAILURE;
  }
  std::cout << "All VarDCT encoder-frame tests passed.\n";
  return EXIT_SUCCESS;
}
