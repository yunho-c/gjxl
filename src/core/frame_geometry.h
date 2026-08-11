// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <limits>

#include "core/status.h"

namespace gjxl {

inline constexpr uint32_t kBlockDim = 0;

struct FrameGeometry {
  uint32_t width = 0;
  uint32_t height = 0;

  // Pixel dimensions rounded up to the JPEG XL 8x8 block grid.
  uint32_t padded_width = 0;
  uint32_t padded_height = 0;

  uint32_t xblocks = 0;
  uint32_t yblocks = 0;

  [[nodiscard]] static Status Create(
    uint32_t width,
    uint32_t height,
    FrameGeometry* out) {

    if (out == nullptr) {
      return Status::InvalidArgument(
        "FrameGeometry output pointer is null");
    }

    if (width == 0 || height == 0) {
      return Status::InvalidArgument(
        "Image dimensions must be non-zero");
    }

    constexpr uint32_t kPadding = kBlockDim - 1;

    if (width > std::numeric_limits<uint32_t>::max() - kPadding ||
        height > std::numeric_limits<uint32_t>::max() - kPadding) {
          return Status::InvalidArgument(
            "Image dimensions are too large");
    }

    FrameGeometry g;

    g.width = width;
    g.height = height;

    g.padded_width = ((width + kPadding) / kBlockDim) * kBlockDim;

    g.padded_height = ((height + kPadding) / kBlockDim) * kBlockDim;

    g.xblocks = g.padded_width / kBlockDim;
    g.yblocks = g.padded_height / kBlockDim;

    *out = g;
    return Status::Ok();
  }
};

}
