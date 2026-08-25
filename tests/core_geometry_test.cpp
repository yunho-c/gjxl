// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "core/ac_strategy.h"
#include "core/frame_geometry.h"
#include "core/image.h"

namespace {

bool CheckFrameGeometry() {
  gjxl::FrameGeometry geometry;
  const gjxl::Status status =
    gjxl::FrameGeometry::Create(17, 9, &geometry);

  if (!status.ok()) {
    std::cerr
      << "FrameGeometry::Create failed: "
      << status.message()
      << '\n';
    return false;
  }

  if (geometry.frame != gjxl::Extent2D{17, 9} ||
      geometry.padded_frame != gjxl::Extent2D{24, 16} ||
      geometry.block_grid.blocks != gjxl::Extent2D{3, 2}) {

    std::cerr << "FrameGeometry produced incorrect extents\n";
    return false;
  }

  gjxl::FrameGeometry unused;

  if (gjxl::FrameGeometry::Create(0, 9, &unused).ok()) {
    std::cerr << "FrameGeometry accepted an empty frame\n";
    return false;
  }

  if (gjxl::FrameGeometry::Create(
        std::numeric_limits<size_t>::max(),
        1,
        &unused).ok()) {

    std::cerr << "FrameGeometry accepted an overflowing frame\n";
    return false;
  }

  return true;
}

bool CheckImageView() {
  std::array<std::array<float, 8>, 3> storage{};
  const gjxl::PlaneF32View plane{
    .data = storage[0].data(),
    .extent = {3, 2},
    .stride = 4,
  };

  if (!plane.valid() || plane.Row(1) != storage[0].data() + 4) {
    std::cerr << "PlaneView geometry or stride is incorrect\n";
    return false;
  }

  const gjxl::Image3FView image{{
    plane,
    {.data = storage[1].data(), .extent = {3, 2}, .stride = 4},
    {.data = storage[2].data(), .extent = {3, 2}, .stride = 4},
  }};

  if (!image.valid() || image.extent() != gjxl::Extent2D{3, 2}) {
    std::cerr << "Image3View did not compose matching plane extents\n";
    return false;
  }

  gjxl::Image3FView mismatched = image;
  mismatched.plane[2].extent.width = 2;

  if (mismatched.valid()) {
    std::cerr << "Image3View accepted mismatched plane extents\n";
    return false;
  }

  return true;
}

bool CheckAcStrategies() {
  for (size_t i = 0; i < gjxl::kAcStrategyInfos.size(); ++i) {
    const gjxl::AcStrategyInfo& info = gjxl::kAcStrategyInfos[i];

    if (static_cast<size_t>(info.type) != i ||
        info.name.empty() ||
        info.covered_blocks.empty() ||
        info.pixel_extent().empty() ||
        info.coefficient_count() !=
          info.pixel_extent().width * info.pixel_extent().height) {

      std::cerr << "Invalid AC strategy metadata at index " << i << '\n';
      return false;
    }
  }

  const gjxl::AcStrategyInfo* dct16x8 =
    gjxl::GetAcStrategyInfo(gjxl::AcStrategyType::kDct16x8);
  const gjxl::AcStrategyInfo* dct8 =
    gjxl::GetAcStrategyInfo(gjxl::AcStrategyType::kDct8);
  const gjxl::AcStrategyInfo* identity =
    gjxl::GetAcStrategyInfo(gjxl::AcStrategyType::kIdentity);

  if (dct16x8 == nullptr ||
      dct16x8->covered_blocks != gjxl::Extent2D{1, 2} ||
      dct16x8->pixel_extent() != gjxl::Extent2D{8, 16} ||
      dct16x8->coefficient_count() != 128 ||
      dct8 == nullptr ||
      identity == nullptr ||
      dct8->pixel_extent() != identity->pixel_extent() ||
      dct8->type == identity->type) {

    std::cerr << "AC strategy identity or geometry is incorrect\n";
    return false;
  }

  if (gjxl::GetAcStrategyInfo(
        static_cast<gjxl::AcStrategyType>(255)) != nullptr) {

    std::cerr << "Invalid AC strategy was accepted\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!CheckFrameGeometry() ||
      !CheckImageView() ||
      !CheckAcStrategies()) {

    return EXIT_FAILURE;
  }

  std::cout << "All core geometry tests passed.\n";
  return EXIT_SUCCESS;
}
