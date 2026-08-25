// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

#include "core/block_grid.h"
#include "core/geometry.h"
#include "core/status.h"

namespace gjxl {

struct FrameGeometry {
  Extent2D frame;
  Extent2D padded_frame;
  BlockGrid block_grid;

  [[nodiscard]] static Status Create(
    Extent2D frame,
    FrameGeometry* out) {

    if (out == nullptr) {
      return Status::InvalidArgument(
        "FrameGeometry output pointer is null");
    }

    BlockGrid block_grid;
    Status status = BlockGrid::Create(frame, &block_grid);

    if (!status.ok()) {
      return status;
    }

    const FrameGeometry geometry{
      .frame = frame,
      .padded_frame = block_grid.padded_pixel_extent(),
      .block_grid = block_grid,
    };

    *out = geometry;
    return Status::Ok();
  }

  [[nodiscard]] static Status Create(
    size_t width,
    size_t height,
    FrameGeometry* out) {

    return Create({width, height}, out);
  }
};

}  // namespace gjxl
