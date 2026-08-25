// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "core/geometry.h"

namespace gjxl {

template <typename T>
struct PlaneView {
  T* data = nullptr;
  Extent2D extent;

  // Number of T elements between rows, not bytes.
  size_t stride = 0;

  [[nodiscard]] bool valid() const noexcept {
    return data != nullptr &&
           !extent.empty() &&
           stride >= extent.width;
  }

  [[nodiscard]] T* Row(size_t y) const noexcept {
    return data + y * stride;
  }
};

using PlaneF32View = PlaneView<float>;
using ConstPlaneF32View = PlaneView<const float>;
using PlaneI32View = PlaneView<int32_t>;
using ConstPlaneI32View = PlaneView<const int32_t>;

template <typename T>
struct Image3View {
  std::array<PlaneView<T>, 3> plane;

  [[nodiscard]] bool valid() const noexcept {
    return plane[0].valid() &&
           plane[1].valid() &&
           plane[2].valid() &&
           plane[0].extent == plane[1].extent &&
           plane[0].extent == plane[2].extent;
  }

  [[nodiscard]] Extent2D extent() const noexcept {
    return plane[0].extent;
  }

  [[nodiscard]] size_t width() const noexcept {
    return extent().width;
  }

  [[nodiscard]] size_t height() const noexcept {
    return extent().height;
  }
};

using Image3FView = Image3View<float>;
using ConstImage3FView = Image3View<const float>;

}  // namespace gjxl
