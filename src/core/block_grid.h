// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <limits>

#include "core/geometry.h"
#include "core/status.h"

namespace gjxl {

inline constexpr size_t kJxlBlockDimension = 8;
inline constexpr size_t kJxlBlockArea = kJxlBlockDimension * kJxlBlockDimension;

// Dimensions of a regular JPEG XL 8x8 base-block grid.
struct BlockGrid {
  Extent2D blocks;

  [[nodiscard]] static Status Create(
    Extent2D pixel_extent,
    BlockGrid* out) {

    if (out == nullptr) {
      return Status::InvalidArgument(
        "BlockGrid output pointer is null");
    }

    if (pixel_extent.empty()) {
      return Status::InvalidArgument(
        "Pixel dimensions must be non-zero");
    }

    constexpr size_t kPadding = kJxlBlockDimension - 1;

    if (pixel_extent.width >
          std::numeric_limits<size_t>::max() - kPadding ||
        pixel_extent.height >
          std::numeric_limits<size_t>::max() - kPadding) {

      return Status::InvalidArgument(
        "Pixel dimensions are too large");
    }

    out->blocks = {
      .width =
        (pixel_extent.width + kPadding) / kJxlBlockDimension,
      .height =
        (pixel_extent.height + kPadding) / kJxlBlockDimension,
    };

    return Status::Ok();
  }

  [[nodiscard]] constexpr Extent2D padded_pixel_extent() const noexcept {
    return {
      .width = blocks.width * kJxlBlockDimension,
      .height = blocks.height * kJxlBlockDimension,
    };
  }
};

}  // namespace gjxl
