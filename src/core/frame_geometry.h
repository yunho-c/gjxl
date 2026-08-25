// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

#include "core/block_grid.h"
#include "core/geometry.h"
#include "core/status.h"

namespace gjxl {

class FrameGeometry {
public:
  // Default construction provides storage for Create(). Accessors describe a
  // valid geometry only after Create() succeeds.
  FrameGeometry() = default;

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

    out->frame_ = frame;
    out->block_grid_ = block_grid;
    return Status::Ok();
  }

  [[nodiscard]] static Status Create(
    size_t width,
    size_t height,
    FrameGeometry* out) {

    return Create({width, height}, out);
  }

  [[nodiscard]] constexpr Extent2D frame() const noexcept {
    return frame_;
  }

  [[nodiscard]] constexpr BlockGrid block_grid() const noexcept {
    return block_grid_;
  }

  [[nodiscard]] constexpr Extent2D padded_frame() const noexcept {
    return block_grid_.padded_pixel_extent();
  }

private:
  Extent2D frame_;
  BlockGrid block_grid_;
};

}  // namespace gjxl
