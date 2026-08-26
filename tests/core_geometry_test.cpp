// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/ac_strategy.h"
#include "core/frame_geometry.h"
#include "core/image.h"
#include "core/image_buffer.h"
#include "core/image_ops.h"

namespace {

bool CheckExtentArea() {
  size_t area = 777;
  if (!gjxl::Extent2D{3, 2}.try_area(&area) || area != 6) {
    std::cerr << "Extent2D produced an incorrect area\n";
    return false;
  }

  area = 777;
  if (!gjxl::Extent2D{
        std::numeric_limits<size_t>::max(),
        0}.try_area(&area) || area != 0) {
    std::cerr << "Extent2D did not handle an empty extent\n";
    return false;
  }

  area = 777;
  if (gjxl::Extent2D{
        std::numeric_limits<size_t>::max(),
        2}.try_area(&area) ||
      area != 777 ||
      gjxl::Extent2D{3, 2}.try_area(nullptr)) {
    std::cerr << "Extent2D accepted an invalid area calculation\n";
    return false;
  }

  const gjxl::Extent2D divided = gjxl::Extent2D{
    std::numeric_limits<size_t>::max(), 65}.ceil_div(64);
  if (divided != gjxl::Extent2D{
        std::numeric_limits<size_t>::max() / 64 + 1, 2}) {
    std::cerr << "Extent2D ceiling division is incorrect\n";
    return false;
  }

  return true;
}

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

  if (geometry.frame() != gjxl::Extent2D{17, 9} ||
      geometry.padded_frame() != gjxl::Extent2D{24, 16} ||
      geometry.block_grid().blocks != gjxl::Extent2D{3, 2}) {

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

  constexpr gjxl::Extent2D kPaddedExtent{24, 16};
  if (!gjxl::BlockGrid::IsPaddedPixelExtent(kPaddedExtent) ||
      gjxl::BlockGrid::IsPaddedPixelExtent({17, 9}) ||
      gjxl::BlockGrid::FromPaddedPixelExtent(kPaddedExtent).blocks !=
        gjxl::Extent2D{3, 2}) {
    std::cerr << "Padded block-grid geometry is incorrect\n";
    return false;
  }

  gjxl::Extent2D padded_extent;
  if (!gjxl::BlockGrid{{3, 2}}.try_padded_pixel_extent(&padded_extent) ||
      padded_extent != kPaddedExtent ||
      gjxl::BlockGrid{{std::numeric_limits<size_t>::max(), 1}}
        .try_padded_pixel_extent(&padded_extent)) {
    std::cerr << "Checked padded extent calculation is incorrect\n";
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

bool CheckMirrorCoordinate() {
  constexpr std::array<size_t, 14> kExpected = {
    1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2,
  };
  for (ptrdiff_t coordinate = -5; coordinate <= 8; ++coordinate) {
    if (gjxl::MirrorCoordinate(coordinate, 3) !=
        kExpected[static_cast<size_t>(coordinate + 5)]) {
      std::cerr << "Mirrored coordinate is incorrect\n";
      return false;
    }
  }

  if (gjxl::MirrorCoordinate(
        std::numeric_limits<ptrdiff_t>::min(), 1) != 0 ||
      gjxl::MirrorCoordinate(
        std::numeric_limits<ptrdiff_t>::max(), 1) != 0) {
    std::cerr << "Mirroring failed at the coordinate limits\n";
    return false;
  }

  return true;
}

bool CheckImageBufferAndCopy() {
  gjxl::Image3FBuffer buffer({3, 2});
  const gjxl::Image3FView buffer_view = buffer.view();
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < 2; ++y) {
      for (size_t x = 0; x < 3; ++x) {
        buffer_view.plane[channel].Row(y)[x] =
          static_cast<float>(100 * channel + 10 * y + x);
      }
    }
  }

  constexpr size_t kStride = 5;
  std::array<std::array<float, kStride * 2>, 3> destination;
  for (auto& plane : destination) {
    plane.fill(-1.0f);
  }
  const gjxl::Image3FView destination_view{{
    gjxl::PlaneF32View{destination[0].data(), {3, 2}, kStride},
    gjxl::PlaneF32View{destination[1].data(), {3, 2}, kStride},
    gjxl::PlaneF32View{destination[2].data(), {3, 2}, kStride},
  }};
  gjxl::CopyImage(buffer.const_view(), destination_view);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < 2; ++y) {
      for (size_t x = 0; x < 3; ++x) {
        if (destination_view.plane[channel].Row(y)[x] !=
            buffer_view.plane[channel].Row(y)[x]) {
          std::cerr << "Image copy changed active pixels\n";
          return false;
        }
      }
      if (destination[channel][y * kStride + 3] != -1.0f ||
          destination[channel][y * kStride + 4] != -1.0f) {
        std::cerr << "Image copy overwrote row padding\n";
        return false;
      }
    }
  }

  if (buffer.cropped_view({2, 1}).extent() != gjxl::Extent2D{2, 1}) {
    std::cerr << "Cropped image-buffer view is incorrect\n";
    return false;
  }

  try {
    buffer.resize({std::numeric_limits<size_t>::max(), 2});
    std::cerr << "Image buffer accepted an overflowing extent\n";
    return false;
  } catch (const std::length_error&) {
    // Expected: the failed resize must leave the prior buffer intact.
  }
  if (buffer.extent() != gjxl::Extent2D{3, 2}) {
    std::cerr << "Failed image-buffer resize changed the image\n";
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
        info.coefficient_extent().width <
          info.coefficient_extent().height ||
        info.low_frequency_extent().width <
          info.low_frequency_extent().height ||
        info.coefficient_count() !=
          info.pixel_extent().width * info.pixel_extent().height) {

      std::cerr << "Invalid AC strategy metadata at index " << i << '\n';
      return false;
    }
  }

  const gjxl::AcStrategyInfo* dct16x8 =
    gjxl::GetAcStrategyInfo(gjxl::AcStrategyType::kDct16x8);
  const gjxl::AcStrategyInfo* dct8x16 =
    gjxl::GetAcStrategyInfo(gjxl::AcStrategyType::kDct8x16);
  const gjxl::AcStrategyInfo* dct8 =
    gjxl::GetAcStrategyInfo(gjxl::AcStrategyType::kDct8);
  const gjxl::AcStrategyInfo* identity =
    gjxl::GetAcStrategyInfo(gjxl::AcStrategyType::kIdentity);

  if (dct16x8 == nullptr ||
      dct16x8->covered_blocks != gjxl::Extent2D{1, 2} ||
      dct16x8->pixel_extent() != gjxl::Extent2D{8, 16} ||
      dct16x8->coefficient_extent() != gjxl::Extent2D{16, 8} ||
      dct16x8->low_frequency_extent() != gjxl::Extent2D{2, 1} ||
      dct16x8->coefficient_index(3, 5) != 83 ||
      dct16x8->coefficient_count() != 128 ||
      dct8x16 == nullptr ||
      dct8x16->covered_blocks != gjxl::Extent2D{2, 1} ||
      dct8x16->pixel_extent() != gjxl::Extent2D{16, 8} ||
      dct8x16->coefficient_index(3, 5) != 53 ||
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
  if (!CheckExtentArea() ||
      !CheckFrameGeometry() ||
      !CheckImageView() ||
      !CheckMirrorCoordinate() ||
      !CheckImageBufferAndCopy() ||
      !CheckAcStrategies()) {

    return EXIT_FAILURE;
  }

  std::cout << "All core geometry tests passed.\n";
  return EXIT_SUCCESS;
}
