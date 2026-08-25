// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

namespace gjxl {

// Backend-independent spatial extent. Width is the horizontal/X dimension;
// height is the vertical/Y dimension.
struct Extent2D {
  size_t width = 0;
  size_t height = 0;

  [[nodiscard]] constexpr bool empty() const noexcept {
    return width == 0 || height == 0;
  }

  friend constexpr bool operator==(
    const Extent2D&,
    const Extent2D&) = default;
};

}  // namespace gjxl
