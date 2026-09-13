// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

#include "core/image.h"

namespace gjxl::benchmark {

// Deterministic linear-RGB fixtures. Callers own storage and must provide a
// valid view; row padding is preserved. Keep each pattern's float operations
// unchanged: historical measurements and canonical PFM hashes depend on them.
inline void FillBatchTexture(Image3FView image) {
  assert(image.valid());
  const Extent2D extent = image.extent();
  const float width_scale = extent.width > 1
    ? 1.0f / static_cast<float>(extent.width - 1)
    : 0.0f;
  const float height_scale = extent.height > 1
    ? 1.0f / static_cast<float>(extent.height - 1)
    : 0.0f;
  for (size_t y = 0; y < extent.height; ++y) {
    float* red = image.plane[0].Row(y);
    float* green = image.plane[1].Row(y);
    float* blue = image.plane[2].Row(y);
    for (size_t x = 0; x < extent.width; ++x) {
      const float fx = static_cast<float>(x) * width_scale;
      const float fy = static_cast<float>(y) * height_scale;
      const float texture = static_cast<float>(
        (13 * x + 17 * y + (x * y) % 29) % 97) / 1024.0f;
      red[x] = 0.025f + 0.72f * fx + texture;
      green[x] = 0.020f + 0.64f * fy + texture;
      blue[x] = 0.030f + 0.30f * fx + 0.38f * fy + texture;
    }
  }
}

// The encoding benchmark and resident qualification use this distinct pattern:
// gradients with sinusoidal texture and a checkerboard blue channel.
inline void FillEncodingStress(Image3FView image) {
  assert(image.valid());
  const Extent2D extent = image.extent();
  for (size_t y = 0; y < extent.height; ++y) {
    float* red = image.plane[0].Row(y);
    float* green = image.plane[1].Row(y);
    float* blue = image.plane[2].Row(y);
    for (size_t x = 0; x < extent.width; ++x) {
      const float fx = static_cast<float>(x) /
          static_cast<float>(std::max<size_t>(1, extent.width - 1));
      const float fy = static_cast<float>(y) /
          static_cast<float>(std::max<size_t>(1, extent.height - 1));
      red[x] = std::clamp(
          0.08f + 0.72f * fx +
              0.13f * std::sin(0.47f * static_cast<float>(x + y)),
          0.0f, 1.0f);
      green[x] = std::clamp(
          0.1f + 0.68f * fy +
              0.16f * std::cos(0.39f * (2.0f * static_cast<float>(x) -
                                        static_cast<float>(y))),
          0.0f, 1.0f);
      blue[x] = ((x / 7 + y / 5) & 1u) == 0 ? 0.12f : 0.84f;
    }
  }
}

}  // namespace gjxl::benchmark
