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
    {.x_matrix_multiplier = 1.25f, .b_matrix_multiplier = 0.75f},
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
  const gjxl::Status status = Encode({8, 8}, &strategies, &frame, true);
  if (!status.ok() ||
      !frame.valid() ||
      frame.geometry().frame() != gjxl::Extent2D{8, 8} ||
      frame.strategies().complete() == false ||
      frame.raw_quant_field().Row(0)[0] != 29 ||
      frame.epf_sharpness().Row(0)[0] != 4 ||
      frame.coding_options().x_matrix_multiplier != 1.25f ||
      frame.coding_options().b_matrix_multiplier != 0.75f ||
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

}  // namespace

int main() {
  if (!CheckOneBlockAndOwnership() ||
      !CheckExactGroup() ||
      !CheckEdgeGroups() ||
      !CheckCrossingStrategyIsRejectedAtomically()) {
    return EXIT_FAILURE;
  }
  std::cout << "All VarDCT encoder-frame tests passed.\n";
  return EXIT_SUCCESS;
}
