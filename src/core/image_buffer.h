// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/image.h"

namespace gjxl {

/// Contiguous owning storage for a three-plane floating-point image.
class Image3FBuffer {
public:
  Image3FBuffer() = default;

  explicit Image3FBuffer(Extent2D extent) {
    resize(extent);
  }

  /// Replaces the image while preserving the previous value on failure.
  void resize(Extent2D extent) {
    size_t pixel_count = 0;
    if (extent.empty() || !extent.try_area(&pixel_count)) {
      throw std::length_error("Image3FBuffer extent is invalid");
    }

    std::array<std::vector<float>, 3> planes;
    for (std::vector<float>& plane : planes) {
      plane.resize(pixel_count);
    }
    planes_ = std::move(planes);
    extent_ = extent;
  }

  [[nodiscard]] Extent2D extent() const noexcept {
    return extent_;
  }

  [[nodiscard]] Image3FView view() noexcept {
    return {{
      PlaneF32View{planes_[0].data(), extent_, extent_.width},
      PlaneF32View{planes_[1].data(), extent_, extent_.width},
      PlaneF32View{planes_[2].data(), extent_, extent_.width},
    }};
  }

  [[nodiscard]] ConstImage3FView const_view() const noexcept {
    return {{
      ConstPlaneF32View{planes_[0].data(), extent_, extent_.width},
      ConstPlaneF32View{planes_[1].data(), extent_, extent_.width},
      ConstPlaneF32View{planes_[2].data(), extent_, extent_.width},
    }};
  }

  [[nodiscard]] ConstImage3FView cropped_view(
    Extent2D extent) const noexcept {

    assert(!extent.empty());
    assert(extent.width <= extent_.width);
    assert(extent.height <= extent_.height);
    return {{
      ConstPlaneF32View{planes_[0].data(), extent, extent_.width},
      ConstPlaneF32View{planes_[1].data(), extent, extent_.width},
      ConstPlaneF32View{planes_[2].data(), extent, extent_.width},
    }};
  }

  [[nodiscard]] std::span<float> plane(size_t channel) noexcept {
    assert(channel < planes_.size());
    return planes_[channel];
  }

  [[nodiscard]] std::span<const float> plane(
    size_t channel) const noexcept {

    assert(channel < planes_.size());
    return planes_[channel];
  }

private:
  Extent2D extent_;
  std::array<std::vector<float>, 3> planes_;
};

}  // namespace gjxl
