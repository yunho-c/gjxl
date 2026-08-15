// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>

namespace gjxl {

template <typename T>
struct PlaneView {
  T* data = nullptr;

  size_t width = 0;
  size_t height = 0;

  // Number of T elements between rows, not bytes.
  size_t stride = 0;

  [[nodiscard]] bool valid() const noexcept {
    return data != nullptr &&
           width != 0 &&
           height != 0 &&
           stride >= width;
  }

  [[nodiscard]] T* Row(size_t y) const noexcept {
    return data + y * stride;
  }
};

using PlaneF32View = PlaneView<float>;
using ConstPlaneF32View = PlaneView<const float>;

template <typename T>
struct Image3View {
  std::array<PlaneView<T>, 3> plane;

  [[nodiscard]] bool valid() const noexcept {
    return plane[0].valid() &&
           plane[1].valid() &&
           plane[2].valid() &&
           plane[0].width == plane[1].width &&
           plane[0].width == plane[2].width &&
           plane[0].height == plane[1].height &&
           plane[0].height == plane[2].height;
  }

  [[nodiscard]] size_t width() const noexcept {
    return plane[0].width;
  }

  [[nodiscard]] size_t height() const noexcept {
    return plane[0].height;
  }
};

using Image3FView = Image3View<float>;
using ConstImage3FView = Image3View<const float>;

} // namespace gjxl
